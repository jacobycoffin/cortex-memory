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
from dataclasses import dataclass
from typing import Any

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
    WHERE m.state = 'active' AND m.pinned = 0
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


def build(
    store: Any,
    *,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
) -> CoreBlock:
    """Render the core block for this store. Cheap, deterministic, no ranking model."""
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
    commitments: list[dict[str, Any]] = []
    try:
        commitments = [dict(row) for row in conn.execute(_COMMITMENTS_SQL).fetchall()]
    except Exception as error:  # noqa: BLE001 - a missing table must not break the block
        # Loud rather than silent: a swallowed schema error is how the block "worked" while
        # containing none of the 52 open commitments.
        commitments = []
        import warnings

        warnings.warn(f"core memory: commitments query failed: {error}", RuntimeWarning, stacklevel=2)
    core_members: list[dict[str, Any]] = []
    try:
        core_members = [dict(row) for row in conn.execute(_CORE_SQL).fetchall()]
    except Exception:  # noqa: BLE001 - the table is absent until promotion first runs
        core_members = []
    important = [
        dict(row) for row in conn.execute(_IMPORTANT_SQL.format(**fill)).fetchall()
    ]

    lines = [HEADER]
    ids: list[str] = []
    size = len(HEADER) + 1

    def room(line: str) -> bool:
        return size + len(line) + 1 <= max_chars

    # Section budgets, not first-come-first-served. The first live run let nine pinned notes
    # take the whole block and rendered **one** of the 52 open commitments; a block that drops
    # every commitment is worse than a smaller block carrying some of each.
    pinned_cap = int(max_chars * 0.50)
    commitment_cap = int(max_chars * 0.85)

    for row in pinned:
        line = _line(row, suffix=" [pinned]")
        if size + len(line) + 1 > pinned_cap or not room(line):
            continue
        lines.append(line)
        ids.append(str(row["id"]))
        size += len(line) + 1

    seen = {str(row["id"]) for row in pinned}
    for row in core_members:
        if str(row["id"]) in seen:
            continue
        line = _line(row, suffix=" [core]")
        if size + len(line) + 1 > pinned_cap or not room(line):
            continue
        lines.append(line)
        ids.append(str(row["id"]))
        seen.add(str(row["id"]))
        size += len(line) + 1

    for row in commitments:
        due = row.get("due_at")
        suffix = f" [commitment, due {str(due)[:10]}]" if due else " [commitment]"
        line = _line(row, suffix=suffix)
        if size + len(line) + 1 > commitment_cap or not room(line):
            continue
        lines.append(line)
        ids.append(str(row["id"]))
        size += len(line) + 1

    # The band is a floor as well as a ceiling: a block that renders 200 characters is not
    # doing the job it exists for, so keep adding the next most load-bearing fact until the
    # minimum is met or there is nothing left worth adding.
    for row in important:
        if size >= min_chars and size >= max_chars * 0.9:
            break
        line = _line(row)
        if not room(line):
            continue
        lines.append(line)
        ids.append(str(row["id"]))
        size += len(line) + 1

    text = "\n".join(lines)
    return CoreBlock(
        text=text,
        memory_ids=tuple(ids),
        pinned=len(pinned),
        commitments=len(commitments),
        chars=len(text),
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
