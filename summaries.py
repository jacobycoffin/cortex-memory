"""A summary is a new node, never a replacement (#205).

The tempting design is to summarise a group *into* one of its members — pick the oldest, overwrite its
text, and call the group handled. That loses the member, makes the summary un-auditable, and hides the
day it stopped being true.

So a summary here is a **separate memory** with its own id, its own text, and explicit `derived_from`
dependencies on everything it speaks for. Four consequences, each of which is a rule:

- **the members are untouched.** No member's text, state or scope changes because a summary exists;
- **certainty is capped by the weakest member** (inherited from `derived_facts`): a summary of one
  confident fact and one shaky guess is as certain as the guess;
- **staleness is computed, not assumed.** The sources' content hashes are snapshotted at derivation,
  so `stale()` answers with evidence rather than a timestamp comparison — this store stamps
  `updated_at` in bulk, which made every derived row read as stale when timestamps were the test;
- **a refresh supersedes, it does not edit.** Re-summarising writes a new node and retires the old one,
  so the history still shows what the store used to believe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from . import derived_facts

ROLE = "summary"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def create(store: Any, members: Iterable[str], statement: str, *,
           at: datetime | None = None, source_context: str = "") -> dict[str, Any]:
    """Write a summary node over `members`, linked to each of them."""
    ids = [str(item) for item in members if str(item).strip()]
    if not ids:
        return {"created": False, "reason": "a summary needs at least one member"}
    made = derived_facts.derive(
        store, str(statement), ids, kind="summary", at=at,
        source_context=str(source_context) or f"summary of {len(ids)} memory(ies)",
    )
    if not made.get("created"):
        return made
    with store.transaction() as conn:
        # derived_facts marks the row `derived`; a summary is a kind of node in its own right, and the
        # role is what lets `summaries()` tell the two apart later.
        conn.execute("UPDATE memories SET record_role = ? WHERE id = ?", (ROLE, made["memory_id"]))
    return made


def members_of(store: Any, summary_id: str) -> list[str]:
    return derived_facts.sources_of(store, str(summary_id))


def covered_by(store: Any, member_id: str) -> list[str]:
    """Which summaries speak for this memory."""
    _ = store
    return [
        memory_id for memory_id in derived_facts.dependents(store, str(member_id))
    ]


def summaries(store: Any, limit: int = 50) -> list[dict[str, Any]]:
    return _rows(
        store,
        "SELECT id, content, confidence, state, created_at FROM memories "
        "WHERE record_role = ? AND state <> 'superseded' ORDER BY created_at DESC LIMIT ?",
        (ROLE, int(limit)),
    )


@dataclass
class Staleness:
    summary_id: str
    dirty: bool = False
    reason: str = ""
    changed: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.changed is None:
            self.changed = []

    def as_dict(self) -> dict[str, Any]:
        return {"summary_id": self.summary_id, "dirty": self.dirty,
                "reason": self.reason, "changed": self.changed}


def stale(store: Any, summary_id: str) -> Staleness:
    """Has any member changed since the summary was written?"""
    found = derived_facts.check(store, str(summary_id))
    reasons = [str(item) for item in (getattr(found, "reasons", None) or [])]
    return Staleness(summary_id=str(summary_id), dirty=bool(getattr(found, "dirty", False)),
                     reason="; ".join(reasons))



def refresh(store: Any, summary_id: str, statement: str, *,
            at: datetime | None = None) -> dict[str, Any]:
    """Replace a summary with a fresh one — a new node, an audit trail, no editing."""
    source = _rows(store, "SELECT state FROM memories WHERE id = ?", (str(summary_id),))
    if not source:
        return {"created": False, "reason": "no such summary"}
    members = members_of(store, summary_id)
    if not members:
        return {"created": False, "reason": "this memory is not a summary"}
    moment = at or _now()
    made = create(store, members, statement, at=moment,
                  source_context=f"refresh of summary {summary_id}")
    if not made.get("created"):
        return made
    with store.transaction() as conn:
        conn.execute("UPDATE memories SET state = 'superseded' WHERE id = ?", (str(summary_id),))
        conn.execute(
            "UPDATE memories SET supersedes_id = ?, updated_at = ? WHERE id = ?",
            (str(summary_id), moment.isoformat(), made["memory_id"]),
        )
    return {**made, "replaced": str(summary_id)}


def report(store: Any) -> dict[str, Any]:
    rows = summaries(store, limit=500)
    dirty = 0
    for row in rows:
        if stale(store, str(row["id"])).dirty:
            dirty += 1
    return {
        "summaries": len(rows),
        "stale": dirty,
        "note": "a summary is a new node; its members are never rewritten",
    }
