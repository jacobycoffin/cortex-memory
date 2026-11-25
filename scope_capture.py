"""Record who, where and when a fact applies, at write time (#30).

A fact written without scope is a fact that will be applied to everything. *"Backups run at 3am"* is
true of one host; *"the vault is unencrypted"* is true of one machine on one date; *"he prefers terse
answers"* is true of one person. Stored without those qualifiers, each becomes a claim about the
world, and the first time it is wrong it looks like the store lied rather than over-generalised.

So the qualifiers are captured at write, from the sentence itself, in three narrow dimensions:

| key | examples |
|---|---|
| `who` | jacoby, hermes, ledger, kaya, teknium |
| `where` | a project name, or a host/system (proxmox, r630, ct116, vps, t3code) |
| `when` | an ISO date, "as of March 2026", "since Tuesday", "until Friday", "on weekdays" |

Two rules keep it honest:

- **fill only what is missing.** An existing `active_project` or an explicit `scope` written by the
  caller is more authoritative than anything inferred from words, and is never overwritten.
- **absence is recorded as absence.** A fact with none of the three is marked `global`, and
  `coverage()` counts how much of the store is global — because the fix for over-generalisation starts
  with knowing its size, not with guessing at scopes.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from . import project_detect

# A small, deliberate lexicon. Recognising people by capitalisation is a bug factory ("Monday" is
# not a person); an unknown name is better captured later by an explicit scope than guessed now.
WHO = ("jacoby", "hermes", "ledger", "teknium", "kaya", "jarvis")
WHERE_HOSTS = (
    "proxmox", "r630", "ct116", "ct117", "vps", "t3code", "selfbox", "unraid", "synology",
    "iphone", "ipad", "macbook", "duo",
)
WHEN_PATTERNS = (
    re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b"),
    re.compile(r"\b(?:as of|since|until|before|after)\s+([A-Z][a-z]+(?:\s+\d{1,2})?(?:\s+20\d{2})?)"),
    re.compile(r"\b(on|every)\s+(weekdays?|weekends?|mondays?|tuesdays?|wednesdays?|thursdays?|"
               r"fridays?|saturdays?|sundays?)\b", re.IGNORECASE),
    re.compile(r"\b(this|next|last)\s+(week|month|year|quarter)\b", re.IGNORECASE),
)


def who(text: str) -> list[str]:
    body = str(text or "").lower()
    return [name for name in WHO if re.search(rf"\b{name}\b", body)]


def where(text: str, store: Any | None = None) -> list[str]:
    """Hosts named outright, plus the project the text points at (when a store is available).

    Project inference needs the store — it scores against known project names — so it is optional
    here: a caller without one still gets the host evidence.
    """
    body = str(text or "").lower()
    found = [host for host in WHERE_HOSTS if re.search(rf"\b{re.escape(host)}\b", body)]
    if store is not None:
        try:
            from . import project_scope

            project = project_scope.infer_project(store, str(text or "")).project
        except Exception:  # noqa: BLE001 - project inference is optional
            project = None
        if project and project.lower() not in found:
            found.insert(0, project.lower())
    return found


def when(text: str) -> list[str]:
    body = str(text or "")
    found: list[str] = []
    for pattern in WHEN_PATTERNS:
        for match in pattern.finditer(body):
            value = " ".join(part for part in match.groups() if part).strip()
            if value and value.lower() not in [item.lower() for item in found]:
                found.append(value)
    return found


def infer(text: str, store: Any | None = None) -> dict[str, Any]:
    """The scope a sentence states about itself. Empty lists mean "not stated"."""
    captured = {
        "who": who(text),
        "where": where(text, store),
        "when": when(text),
    }
    return {key: value for key, value in captured.items() if value}


def is_global(scope: dict[str, Any] | None) -> bool:
    """No audience, no environment, no date — true of everything, therefore checked for nothing."""
    if not scope:
        return True
    return not any(scope.get(key) for key in ("who", "where", "when", "project", "active_project"))


def _scope_of(store: Any, memory_id: str) -> dict[str, Any]:
    try:
        row = store._conn.execute(
            "SELECT scope_json FROM memories WHERE id = ?", (str(memory_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001
        return {}
    if not row or not row["scope_json"]:
        return {}
    try:
        loaded = json.loads(row["scope_json"])
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def capture(store: Any, memory_id: str, content: str | None = None, *,
            at: datetime | None = None) -> dict[str, Any]:
    """Fill in missing scope keys from the text. Existing values always win."""
    existing = _scope_of(store, memory_id)
    text = content
    if text is None:
        try:
            row = store._conn.execute(
                "SELECT content FROM memories WHERE id = ?", (str(memory_id),)
            ).fetchone()
        except Exception:  # noqa: BLE001
            return existing
        text = str(row["content"]) if row else ""
    inferred = infer(str(text or ""), store)
    merged = dict(existing)
    for key, value in inferred.items():
        if not merged.get(key):
            merged[key] = value
    if merged == existing:
        return existing
    merged.setdefault("captured_at", (at or datetime.now(timezone.utc)).isoformat())
    try:
        with store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET scope_json = ? WHERE id = ?",
                (json.dumps(merged, sort_keys=True), str(memory_id)),
            )
    except Exception:  # noqa: BLE001 - scope is an improvement, never a reason to fail a write
        return existing
    return merged


EXPLICIT = "explicit"
DETECTED = "detected"
NONE = "none"


@dataclass(frozen=True)
class Decision:
    """Which project a turn belongs to, and how that was decided."""

    project: str | None
    how: str
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def scope(self) -> dict[str, Any]:
        if not self.project:
            return {}
        return {"project": self.project, "project_source": self.how}

    def render(self) -> str:
        if not self.project:
            return "no project: nothing in the turn identified one"
        return f"project {self.project} ({self.how})"


def for_turn(
    *,
    query: str = "",
    cwd: str | None = None,
    files: Iterable[str] = (),
    messages: Sequence[str] = (),
    candidates: Iterable[str] = (),
    explicit: str | None = None,
) -> Decision:
    """The project a memory written from this turn belongs to. Writes nothing.

    The sentence-level `infer()` above reads the text of one fact; this reads the turn - what was asked, where the
    agent was working, which files it touched - because "the nginx timeout is 60s" does not name its project even when
    the turn obviously has one.
    """
    if explicit and str(explicit).strip():
        return Decision(project=str(explicit).strip(), how=EXPLICIT)
    project, scores = project_detect.detect(
        query=query, cwd=cwd, files=files, messages=messages, candidates=candidates,
    )
    if not project:
        return Decision(project=None, how=NONE, scores=scores)
    return Decision(project=project, how=DETECTED, scores=scores)


def capture_turn(store: Any, memory_id: str, decision: Decision, *,
                 at: datetime | None = None) -> dict[str, Any]:
    """Write the turn's project into a memory's scope, never overwriting an existing one."""
    existing = _scope_of(store, memory_id)
    if not decision.project or str(existing.get("project") or "").strip():
        return existing
    merged = dict(existing)
    merged.update(decision.scope)
    merged.setdefault("captured_at", (at or datetime.now(timezone.utc)).isoformat())
    try:
        with store.transaction() as conn:
            conn.execute("UPDATE memories SET scope_json = ? WHERE id = ?",
                         (json.dumps(merged, sort_keys=True), str(memory_id)))
    except Exception:  # noqa: BLE001 - scope is an improvement, never a reason to fail a write
        return existing
    return merged


@dataclass
class BackfillPlan:
    session_id: str
    project: str
    memory_ids: list[str] = field(default_factory=list)
    skipped_already_scoped: int = 0
    readable: bool = True
    reason: str = ""

    def render(self) -> list[str]:
        if not self.readable:
            return [f"the store could not be read: {self.reason}"]
        return [f"session {self.session_id}: {len(self.memory_ids)} memory(ies) would gain project "
                f"{self.project!r}, {self.skipped_already_scoped} already scoped"]

    def as_dict(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "project": self.project,
                "memory_ids": list(self.memory_ids),
                "skipped_already_scoped": self.skipped_already_scoped,
                "readable": self.readable, "reason": self.reason}


def plan_session_backfill(store: Any, *, session_id: str, project: str,
                          limit: int = 200) -> BackfillPlan:
    """Which memories of a session have no project yet. Writes nothing."""
    plan = BackfillPlan(session_id=str(session_id or ""), project=str(project or ""))
    if not plan.session_id.strip() or not plan.project.strip():
        plan.readable = False
        plan.reason = "a backfill needs a session and a project"
        return plan
    try:
        rows = store._conn.execute(
            "SELECT id, scope_json FROM memories WHERE session_id = ? AND state IN ('active','cold') "
            "ORDER BY created_at DESC LIMIT ?",
            (plan.session_id, max(1, int(limit))),
        ).fetchall()
    except Exception:  # noqa: BLE001
        plan.readable = False
        plan.reason = "no memories table, or the store refused the query"
        return plan
    for row in rows:
        try:
            scope = json.loads(row["scope_json"] or "{}")
        except (TypeError, ValueError):
            scope = {}
        if isinstance(scope, dict) and str(scope.get("project") or "").strip():
            plan.skipped_already_scoped += 1
            continue
        plan.memory_ids.append(str(row["id"]))
    return plan


def apply_session_backfill(store: Any, plan: BackfillPlan, *, apply: bool = False,
                           consented_by: str | None = None) -> dict[str, Any]:
    """Write the project into the planned rows. Dry run unless `apply` and a named consent are given."""
    if not plan.readable:
        return {"ok": False, "reason": plan.reason, "applied": 0, "dry_run": not apply}
    if not apply:
        return {"ok": True, "dry_run": True, "would_apply": len(plan.memory_ids), "applied": 0}
    if not consented_by:
        return {"ok": False, "reason": "re-scoping written memories needs a named consenting person",
                "applied": 0, "dry_run": False}
    applied = 0
    for memory_id in plan.memory_ids:
        merged = capture_turn(store, memory_id,
                              Decision(project=plan.project, how="backfill"))
        if str(merged.get("project") or "") == plan.project:
            applied += 1
    return {"ok": True, "dry_run": False, "applied": applied, "consented_by": str(consented_by)}


def coverage(store: Any, *, limit: int = 5000) -> dict[str, Any]:
    """How much of the store is scoped, and in which dimension."""
    try:
        rows = store._conn.execute(
            """SELECT scope_json FROM memories
               WHERE state NOT IN ('tombstoned','deleted') LIMIT ?""",
            (int(limit),),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {"memories": 0, "global": 0, "who": 0, "where": 0, "when": 0}
    counts = {"memories": 0, "global": 0, "who": 0, "where": 0, "when": 0}
    for row in rows:
        counts["memories"] += 1
        try:
            scope = json.loads(row["scope_json"] or "{}")
        except (TypeError, ValueError):
            scope = {}
        if not isinstance(scope, dict):
            scope = {}
        if is_global(scope):
            counts["global"] += 1
        for key in ("who", "where", "when"):
            if scope.get(key):
                counts[key] += 1
    return counts


def backfill(store: Any, *, limit: int = 500) -> dict[str, int]:
    """Scope existing memories from their own text. Idempotent, and never overwrites."""
    try:
        rows = store._conn.execute(
            """SELECT id, content FROM memories
               WHERE state = 'active' AND (scope_json IS NULL OR scope_json IN ('', '{}'))
               LIMIT ?""",
            (int(limit),),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {"scoped": 0}
    scoped = 0
    for row in rows:
        merged = capture(store, str(row["id"]), str(row["content"]))
        if merged:
            scoped += 1
    return {"scoped": scoped, "scanned": len(rows)}


def report(store: Any) -> dict[str, Any]:
    counts = coverage(store)
    total = counts["memories"] or 1
    return {
        **counts,
        "global_share": round(counts["global"] / total, 4),
        "lexicon": {"who": list(WHO), "where": list(WHERE_HOSTS)},
    }
