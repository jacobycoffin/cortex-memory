"""Core memory: a small always-injected block, served by plain SQL (backlog item #120).

Every recall path so far is *conditional* — it returns something when the query looks like it
needs memory, and the always-on floor (#1) only fires when the planner says nothing is needed.
Neither guarantees that the handful of facts which must never be missing are actually present.

Core memory is that guarantee: pinned facts, open commitments, and the most load-bearing
stable memories, rendered into 1,200-2,000 characters and prepended to every turn.

Three design constraints, each of which rules out a simpler implementation:

- **Plain SQL, no FTS, no embeddings.** This block must be reproducible when the rest of
  recall is degraded: a corrupt index, an unloaded model, an expired budget. It is the
  memory of last resort, so it depends on nothing but the tables.
- **Characters, not tokens.** The budget is enforced as rendered characters because the block
  is a *prompt prefix*: it must have a fixed, inspectable size that does not drift with the
  tokenizer.
- **Deterministic order.** Pinned first, then due commitments, then importance, then use
  count, then id. Two runs over the same store produce the same string, byte for byte.

Off by default. It costs 300-500 tokens on every single turn, which is real money and real
latency headroom, so switching it on is a product decision with a measured price attached —
not something to slip in behind an optimisation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from .core_voice import enabled as voice_enabled
from .core_voice import rewrite_row

MIN_CHARS = 1200
MAX_CHARS = 2000
# One long memory can swallow the whole block: a single pinned note rendered 320 characters
# in the first live run and starved the commitments. Lines are capped, with the cut visible.
MAX_LINE_CHARS = 200
HEADER = "Core memory (always present, may be incomplete):"

# Deliberately plain SQL: see the module docstring. No FTS, no vector search, no ranking
# model — only the tables and their columns.
# `verified_at` is what makes the block honest (#125): every line says when it was last
# confirmed, used or re-stated, and a line that says so nowhere says `unverified`. The model
# can then treat an unverified line as a claim rather than a fact.
# The *latest* of the vouching events, not the first non-null: a fact used in January and
# re-affirmed in November is verified in November, and COALESCE would report January.
# Promotion is deliberately NOT one of these: a fact promoted by the use rule was verified
# when it was last *used*, not when the rule fired, and showing "verified today" for a memory
# that has been untouched for months would be the opposite of the honesty this exists for.
_VERIFIED = (
    "NULLIF(MAX(COALESCE(c.last_confirmed_at, ''), COALESCE(m.last_helpful_at, ''), "
    "COALESCE(m.last_used_at, '')), '') AS verified_at"
)
_VERIFIED_NO_CORE = (
    "NULLIF(MAX(COALESCE(m.last_helpful_at, ''), COALESCE(m.last_used_at, '')), '') "
    "AS verified_at"
)
# `core_members` only exists once promotion (#122) has run, so the join is conditional: a
# fresh store must still render a block.
_CORE_JOIN = "LEFT JOIN core_members c ON c.memory_id = m.id"

_PINNED_SQL = """
    SELECT m.id, m.kind, m.content, m.pinned, m.importance, m.confirmed_count,
           m.used_count, {verified}
    FROM memories m
    {core_join}
    WHERE m.state = 'active' AND m.pinned = 1
    ORDER BY m.importance DESC, m.confirmed_count DESC, m.used_count DESC, m.id ASC
"""

# Promoted members (#122) ride ahead of the general importance fill: promotion is a recorded
# decision, not a sort order, and the block should honour it even when importance is low.
_CORE_SQL = f"""
    SELECT m.id, m.kind, m.content, m.pinned, m.importance, m.confirmed_count,
           m.used_count, {_VERIFIED}
    FROM memories m
    JOIN core_members c ON c.memory_id = m.id
    WHERE m.state = 'active'
    ORDER BY COALESCE(c.last_confirmed_at, c.promoted_at) DESC, m.importance DESC, m.id ASC
"""

_IMPORTANT_SQL = """
    SELECT m.id, m.kind, m.content, m.pinned, m.importance, m.confirmed_count,
           m.used_count, {verified}
    FROM memories m
    {core_join}
    WHERE m.state = 'active' AND m.pinned = 0 {recent_exclusion}
    ORDER BY m.importance DESC, m.confirmed_count DESC, m.used_count DESC, m.id ASC
"""

# `prospective_items` keys on `memory_id` and carries no content of its own — the note lives
# on the row, the fact in `memories`. The first version selected `id, content` and the
# failure was swallowed by a bare `except`, so the block silently contained zero commitments
# while the store held 52 open ones.
_COMMITMENTS_SQL = """
    SELECT p.memory_id AS id, COALESCE(p.note, m.content) AS content,
           p.due_at, COALESCE(m.kind, 'commitment') AS kind
    FROM prospective_items p
    LEFT JOIN memories m ON m.id = p.memory_id
    WHERE p.status = 'open'
    ORDER BY COALESCE(p.due_at, '9999') ASC, p.memory_id ASC
"""


@dataclass(frozen=True)
class CoreBlock:
    text: str
    memory_ids: tuple[str, ...]
    pinned: int
    commitments: int
    chars: int
    recent: int = 0

    @property
    def estimated_tokens(self) -> int:
        # The renderer's own rule of thumb (4 chars/token) so the block's cost can be
        # compared with `rendered_tokens` in the ledger without a second definition.
        return (self.chars + 3) // 4


def enabled() -> bool:
    """Off unless asked for: this is a per-turn token cost, not a free optimisation."""
    return os.environ.get("CORTEX_CORE_MEMORY", "0") != "0"


def verified_label(row: dict[str, Any]) -> str:
    """`[verified 2026-09-24]` or `[unverified]` — never silence.

    A block where some lines say when they were last checked and others say nothing invites
    the reader to assume the silent ones are equally fresh. They are not, so silence is made
    explicit.
    """
    stamp = str(row.get("verified_at") or "").strip()
    if not stamp:
        return "[unverified]"
    return f"[verified {stamp[:10]}]"


def _line(row: dict[str, Any], *, suffix: str = "") -> str:
    body = " ".join(str(row.get("content") or "").split())
    room = MAX_LINE_CHARS - len(suffix)
    if len(body) > room:
        body = body[: max(0, room - 1)].rstrip() + "\u2026"
    label = f" [M:{str(row.get('id') or '')[:8]} {row.get('kind') or 'fact'}]"
    suffix = f"{suffix} {verified_label(row)}".strip()
    return f"-{label} {body} {suffix}".strip()


SECTIONS = ("commitments", "identity", "people", "projects", "preferences", "constraints", "general")
SECTION_TITLES = {
    "commitments": "Open commitments",
    "identity": "Identity",
    "people": "People",
    "projects": "Projects",
    "preferences": "Preferences",
    "constraints": "Constraints",
    "general": "Other",
}
# Fixed slots (#127): one topic cannot crowd the others out. They sum above 1.0 so the block
# can still fill up when a section is thin — pass two hands unused capacity to whoever has
# more to say, but nobody's *guaranteed* slot is spent on a single topic.
SECTION_SHARES = {
    # Commitments lead here only because they are time-sensitive; they get 15% rather than 25%
    # after the live store showed 52 "prospective" rows (many of them project notes wearing a
    # commitment's kind) taking a quarter of the block ahead of pinned rules.
    "commitments": 0.15,
    "identity": 0.15,
    "people": 0.10,
    "projects": 0.18,
    "preferences": 0.20,
    "constraints": 0.15,
    "general": 0.07,
}
# Warm vs cold (#129): the profile above is the *stable* part and changes slowly; the last
# day of context is appended separately. Two reasons to keep them apart rather than mingling
# them: the stable part is a promise (it should not churn), and the recent part is the one
# that legitimately changes hour to hour — mixing them makes every change look like the
# profile changed.
RECENT_SHARE = 0.25
RECENT_DAYS = 1
RECENT_TITLE = "Recent (last 24h)"

KIND_SECTIONS = {
    "prospective": "commitments",
    "commitment": "commitments",
    "preference": "preferences",
    "identity": "identity",
    "constraint": "constraints",
    "person": "people",
    "project": "projects",
}
_IDENTITY = re.compile(r"\b(?:jacoby|i am|my name|birthday|timezone|lives? in|works? as|pronouns?)\b", re.I)
_PEOPLE = re.compile(r"\b(?:jamie|jarvis|ledger|coby|wife|husband|partner|mother|father|sister|brother|boss|colleague|friend)\b", re.I)
_PREFERENCES = re.compile(r"\b(?:prefers?|likes?|wants?|favou?rite|rather|would prefer|style)\b", re.I)
_CONSTRAINTS = re.compile(r"\b(?:never|must not|do not|don't|avoid|only|always|constraint|required)\b", re.I)
_PROJECTS = re.compile(r"\b(?:project|repo|branch|deploy|release|build|codebase)\b", re.I)

_RECENT_SQL = """
    SELECT m.id, m.kind, m.content, m.pinned, m.importance, m.confirmed_count,
           m.used_count, {verified}
    FROM memories m
    {core_join}
    WHERE m.state = 'active'
      AND COALESCE(m.last_used_at, m.created_at) >= ?
      AND m.pinned = 0
    ORDER BY COALESCE(m.last_used_at, m.created_at) DESC, m.id ASC
"""


def classify_section(row: dict[str, Any]) -> str:
    """Which slot a memory belongs to.

    Deliberately a keyword classifier over metadata plus content: the block is the memory of
    last resort and must render with no model available (#120). A wrong bucket costs a line's
    placement, not its content, so cheap and inspectable beats clever.
    """
    # The memory's own `kind` is the strongest signal and it is already curated: a memory
    # captured as a preference is one, whatever words it happens to use. The store's kind
    # vocabulary is singular ("preference", "prospective") while the sections are plural, so
    # the mapping is explicit rather than a string match.
    kind = str(row.get("kind") or "").strip().lower()
    if kind in KIND_SECTIONS:
        return KIND_SECTIONS[kind]
    if kind in SECTIONS:
        return kind
    scope = row.get("scope_json") or row.get("scope")
    if isinstance(scope, str) and "active_project" in scope:
        return "projects"
    if isinstance(scope, dict) and scope.get("active_project"):
        return "projects"
    content = str(row.get("content") or "")
    if _CONSTRAINTS.search(content):
        return "constraints"
    if _PREFERENCES.search(content):
        return "preferences"
    if _IDENTITY.search(content):
        return "identity"
    if _PEOPLE.search(content):
        return "people"
    if _PROJECTS.search(content):
        return "projects"
    return "general"


def _bucket(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in SECTIONS}
    seen: set[str] = set()
    for row in rows:
        memory_id = str(row.get("id") or "")
        if not memory_id or memory_id in seen:
            continue
        seen.add(memory_id)
        buckets["commitments" if row.get("__commitment") else classify_section(row)].append(row)
    return buckets


def build(
    store: Any,
    *,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    include_recent: bool = True,
    now: Any = None,
) -> CoreBlock:
    """Render the core block for this store. Cheap, no ranking model.

    Two parts (#129): the stable profile, and the last day's context. `now` exists so the
    recent window can be pinned in tests — the rest of the block does not depend on the clock.
    """
    from datetime import datetime, timedelta, timezone

    reference = now or datetime.now(timezone.utc)
    cutoff = (reference - timedelta(days=RECENT_DAYS)).isoformat()
    conn = store._conn
    has_core = bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='core_members'"
        ).fetchone()
    )
    fill = {
        "verified": _VERIFIED if has_core else _VERIFIED_NO_CORE,
        "core_join": _CORE_JOIN if has_core else "",
    }
    pinned = [dict(row) for row in conn.execute(_PINNED_SQL.format(**fill)).fetchall()]
    core_members: list[dict[str, Any]] = []
    try:
        core_members = [dict(row) for row in conn.execute(_CORE_SQL).fetchall()]
    except Exception:  # noqa: BLE001 - the table is absent until promotion first runs
        core_members = []
    commitments: list[dict[str, Any]] = []
    try:
        commitments = [dict(row) for row in conn.execute(_COMMITMENTS_SQL).fetchall()]
        for row in commitments:
            row["__commitment"] = True
    except Exception as error:  # noqa: BLE001 - a missing table must not break the block
        import warnings

        warnings.warn(f"core memory: commitments query failed: {error}", RuntimeWarning, stacklevel=2)
    # A memory touched inside the recent window belongs in *that* section, not in the general
    # fill: otherwise the fill swallows it first and the warm section renders empty.
    fill_important = dict(fill)
    fill_important["recent_exclusion"] = (
        "AND (m.last_used_at IS NULL OR m.last_used_at < ?)" if include_recent else ""
    )
    important_params: tuple[Any, ...] = (cutoff,) if include_recent else ()
    important = [
        dict(row)
        for row in conn.execute(
            _IMPORTANT_SQL.format(**fill_important), important_params
        ).fetchall()
    ]

    recent: list[dict[str, Any]] = []
    if include_recent:
        try:
            recent = [
                dict(row)
                for row in conn.execute(_RECENT_SQL.format(**fill), (cutoff,)).fetchall()
            ]
        except Exception as error:  # noqa: BLE001 - a missing column must not break the block
            import warnings

            warnings.warn(f"core memory: recent query failed: {error}", RuntimeWarning, stacklevel=2)
            recent = []

    # Priority inside a section: pinned, then promoted, then commitments, then importance.
    candidates = [*pinned, *core_members, *commitments, *important]
    buckets = _bucket(candidates)
    tags = {str(row["id"]): "[pinned]" for row in pinned}
    for row in core_members:
        tags.setdefault(str(row["id"]), "[core]")

    def line_for(row: dict[str, Any]) -> str:
        # Model voice (#128): rewriting happens at render time, so classification and
        # de-duplication still see the memory as it was written.
        if voice_enabled():
            row = rewrite_row(row)
        suffix = tags.get(str(row.get("id")), "")
        if row.get("__commitment"):
            due = row.get("due_at")
            suffix = f"[commitment, due {str(due)[:10]}]" if due else "[commitment]"
        return _line(row, suffix=suffix)

    # The stable profile may not spend the recent window's share: the recent part is the only
    # one that is allowed to churn, and it must not be crowded out by a fat profile.
    stable_limit = int(max_chars * (1.0 - RECENT_SHARE)) if recent else max_chars
    rendered: dict[str, list[tuple[str, str]]] = {name: [] for name in SECTIONS}
    used = len(HEADER) + 1
    for name in SECTIONS:
        slot = min(int(max_chars * SECTION_SHARES.get(name, 0.05)), max(0, stable_limit - used))
        spent = 0
        title = SECTION_TITLES[name]
        for row in buckets[name]:
            line = line_for(row)
            cost = len(line) + 1
            if spent + cost > slot or used + cost > stable_limit:
                break
            rendered[name].append((line, str(row["id"])))
            spent += cost
            used += cost
        if rendered[name]:
            used += len(title) + 1  # the header itself

    # Pass two: unused capacity goes to whatever still has candidates, in section order, so a
    # thin section never wastes the block while another overflows.
    for name in SECTIONS:
        if used >= stable_limit:
            break
        taken = {memory_id for _, memory_id in rendered[name]}
        for row in buckets[name]:
            memory_id = str(row.get("id"))
            if memory_id in taken:
                continue
            line = line_for(row)
            cost = len(line) + 1
            if used + cost > stable_limit:
                break
            rendered[name].append((line, memory_id))
            used += cost

    lines: list[str] = [HEADER]
    ids: list[str] = []
    for name in SECTIONS:
        entries = rendered.get(name) or []
        if not entries:
            continue
        lines.append(f"{SECTION_TITLES[name]}:")
        for line, memory_id in entries:
            lines.append(line)
            ids.append(memory_id)

    # Warm appendage (#129), last so it never displaces the profile.
    recent_lines: list[str] = []
    rendered_ids = set(ids)
    for row in recent:
        memory_id = str(row.get("id") or "")
        if not memory_id or memory_id in rendered_ids:
            continue
        line = line_for(row)
        if used + len(line) + 1 > max_chars:
            break
        recent_lines.append(line)
        ids.append(memory_id)
        rendered_ids.add(memory_id)
        used += len(line) + 1
    if recent_lines:
        lines.append(f"{RECENT_TITLE}:")
        lines.extend(recent_lines)

    text = "\n".join(lines)
    return CoreBlock(
        text=text,
        memory_ids=tuple(ids),
        pinned=len(pinned),
        commitments=len(commitments),
        chars=len(text),
        recent=len(recent_lines),
    )


def dedupe_against(text: str, recalled_ids: set[str] | frozenset[str]) -> str:
    """Drop core lines whose memory the turn's recall already surfaced.

    Two copies of one fact cost tokens and teach the model nothing. Matching is on the
    rendered `M:` prefix, so this works on the text rather than needing the ids again.
    """
    if not recalled_ids:
        return text
    keep = []
    for line in text.splitlines():
        marker = line.find("[M:")
        if marker == -1:
            keep.append(line)
            continue
        short = line[marker + 3 : marker + 11]
        if any(str(memory_id).startswith(short) for memory_id in recalled_ids):
            continue
        keep.append(line)
    return "\n".join(keep)


def stats(store: Any) -> dict[str, int]:
    """What the block would contain right now, without rendering it."""
    conn = store._conn
    pinned = int(
        conn.execute("SELECT COUNT(*) FROM memories WHERE state='active' AND pinned=1").fetchone()[0]
    )
    active = int(conn.execute("SELECT COUNT(*) FROM memories WHERE state='active'").fetchone()[0])
    try:
        commitments = int(
            conn.execute("SELECT COUNT(*) FROM prospective_items WHERE status='open'").fetchone()[0]
        )
    except Exception:  # noqa: BLE001
        commitments = 0
    block = build(store)
    return {
        "active": active,
        "pinned": pinned,
        "commitments": commitments,
        "chars": block.chars,
        "estimated_tokens": block.estimated_tokens,
        "lines": block.text.count("\n"),
    }
