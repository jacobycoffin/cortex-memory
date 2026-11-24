"""A correction is a first-class memory, not an edit (#178)

When the owner says *"no, the port is 8101"*, three things have to be true afterwards, and the live store
currently guarantees none of them:

- **the old belief is retired, not deleted.** It stays readable, with `valid_to` set, so "what did I think
  in October" still has an answer. Live: **0 of 3,843 memories have `valid_to` set** — nothing has ever
  been retired by date, so the store cannot distinguish "wrong" from "old".
- **the correction outranks everything that comes later.** A correction is the strongest thing he can say
  (ceiling 0.97); an agent inference (0.55) may never supersede it. Without that rule, a later
  consolidation pass quietly re-derives the belief he just corrected.
- **it is findable.** "What have you corrected me on" must be one query. Live: **0 memories carry
  `USER_CORRECTION`** — the category exists in the ladder and has never been used, so every correction so
  far is indistinguishable from an ordinary fact.

Two rules about the mechanism itself:

- **no correction without a target.** A "correction" that does not name what it replaces is just a new
  fact, and accepting it silently is how a store ends up with two contradictory beliefs and no history.
- **the chain is walkable and loop-safe.** `supersedes_id` is a chain, not a tree; a cycle would hang
  every "why do you believe this" question, so the walk is depth-bounded and reports a cycle instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

CORRECTION_CATEGORY = "USER_CORRECTION"
CORRECTION_CEILING = 0.97
MAX_CHAIN = 50
# Categories that may never supersede a correction: a belief the owner fixed does not get re-derived.
WEAKER_THAN_CORRECTION = ("AGENT_INFERENCE", "AGENT_PROPOSED", "WEB_CLAIM", "DOCUMENT_EXTRACTED")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _one(store: Any, memory_id: str) -> dict[str, Any] | None:
    rows = _rows(store, "SELECT * FROM memories WHERE id = ?", (str(memory_id),))
    return rows[0] if rows else None


@dataclass
class Result:
    ok: bool = False
    reason: str = ""
    correction_id: str = ""
    superseded_id: str = ""
    retired: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason, "correction_id": self.correction_id,
                "superseded_id": self.superseded_id, "retired": self.retired}


def can_supersede(store: Any, target_id: str, *, source_category: str) -> tuple[bool, str]:
    """A correction outranks later inference; nothing weaker may overwrite it."""
    target = _one(store, target_id)
    if target is None:
        return False, "the memory being corrected does not exist"
    if str(target.get("source_category") or "") == CORRECTION_CATEGORY and str(
            source_category or "") in WEAKER_THAN_CORRECTION:
        return False, ("a correction may not be superseded by "
                       f"{source_category}: he already fixed this")
    return True, ""


def record(store: Any, *, supersedes: str, content: str, kind: str | None = None,
           confidence: float | None = None, now: datetime | None = None) -> Result:
    """Write the correction, link it, and retire what it replaces — in one transaction."""
    body = str(content or "").strip()
    if not body:
        return Result(reason="a correction needs content")
    if not str(supersedes or "").strip():
        return Result(reason="a correction must name what it replaces; otherwise it is a new fact")
    allowed, why = can_supersede(store, supersedes, source_category=CORRECTION_CATEGORY)
    if not allowed:
        return Result(reason=why)
    target = _one(store, supersedes) or {}
    moment = (now or _now()).isoformat()
    ceiling = min(CORRECTION_CEILING, float(confidence)) if confidence is not None else CORRECTION_CEILING
    try:
        memory_id = store.add_memory(
            content=body,
            kind=str(kind or target.get("kind") or "semantic"),
            source_type="user_turn",
            source_category=CORRECTION_CATEGORY,
            confidence=ceiling,
        )[0]
    except Exception as error:  # noqa: BLE001
        return Result(reason=f"the store refused the write: {error}")
    try:
        with store.transaction() as conn:
            conn.execute("UPDATE memories SET supersedes_id = ? WHERE id = ?",
                         (str(supersedes), memory_id))
            conn.execute("UPDATE memories SET valid_to = ?, state = 'superseded' WHERE id = ?",
                         (moment, str(supersedes)))
        return Result(ok=True, correction_id=memory_id, superseded_id=str(supersedes), retired=True)
    except Exception as error:  # noqa: BLE001
        return Result(reason=f"the correction was written but not linked: {error}",
                      correction_id=memory_id)


def chain(store: Any, memory_id: str) -> dict[str, Any]:
    """Walk `supersedes_id` backwards: what this belief replaced, and what replaced it."""
    seen: list[dict[str, Any]] = []
    cursor = str(memory_id)
    cycle = False
    while cursor and len(seen) < MAX_CHAIN:
        row = _one(store, cursor)
        if row is None:
            break
        if any(item["id"] == cursor for item in seen):
            cycle = True
            break
        seen.append({
            "id": str(row.get("id") or ""),
            "content": str(row.get("content") or "")[:120],
            "source_category": str(row.get("source_category") or ""),
            "confidence": round(float(row.get("confidence") or 0.0), 3),
            "valid_to": row.get("valid_to"),
            "created_at": str(row.get("created_at") or "")[:19],
        })
        cursor = str(row.get("supersedes_id") or "")
    return {
        "length": len(seen),
        "cycle": cycle,
        "items": seen,
        "note": "the last item is the oldest belief in the chain" if seen else "no history",
    }


def standing(store: Any, *, limit: int = 50) -> list[dict[str, Any]]:
    """The corrections that are still in force — the answer to "what have you been told to fix"."""
    rows = _rows(
        store,
        "SELECT id, content, kind, confidence, created_at, supersedes_id FROM memories "
        "WHERE source_category = ? AND state <> 'archived' AND valid_to IS NULL "
        "ORDER BY created_at DESC LIMIT ?",
        (CORRECTION_CATEGORY, max(1, int(limit))),
    )
    return [{
        "id": str(row["id"]),
        "content": str(row["content"] or "")[:160],
        "kind": str(row["kind"] or ""),
        "confidence": round(float(row["confidence"] or 0.0), 3),
        "created_at": str(row["created_at"] or "")[:19],
        "supersedes_id": row.get("supersedes_id"),
    } for row in rows]


@dataclass
class Coverage:
    total: int = 0
    corrections: int = 0
    supersessions: int = 0
    retired: int = 0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"total": self.total, "corrections": self.corrections,
                "supersessions": self.supersessions, "retired": self.retired, "note": self.note}


def coverage(store: Any) -> Coverage:
    """Does the store actually use the correction path it documents?"""
    rows = _rows(
        store,
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN source_category = 'USER_CORRECTION' THEN 1 ELSE 0 END) AS corrections, "
        "SUM(CASE WHEN supersedes_id IS NOT NULL THEN 1 ELSE 0 END) AS supersessions, "
        "SUM(CASE WHEN valid_to IS NOT NULL THEN 1 ELSE 0 END) AS retired "
        "FROM memories WHERE state <> 'archived'",
    )
    if not rows:
        return Coverage(note="no memories to examine")
    row = rows[0]
    found = Coverage(
        total=int(row.get("total") or 0),
        corrections=int(row.get("corrections") or 0),
        supersessions=int(row.get("supersessions") or 0),
        retired=int(row.get("retired") or 0),
    )
    gaps = []
    if not found.corrections:
        gaps.append("no memory carries USER_CORRECTION, so a correction is indistinguishable from a fact")
    if not found.retired:
        gaps.append("no memory has valid_to set, so the store cannot tell 'wrong' from 'old'")
    found.note = "; ".join(gaps) if gaps else "the correction path is in use"
    return found


def render(store: Any, *, limit: int = 10) -> list[str]:
    described = coverage(store)
    lines = [f"corrections: {described.corrections} standing of {described.total} memor(ies), "
             f"{described.supersessions} supersession(s), {described.retired} retired"]
    if described.note:
        lines.append(f"  {described.note}")
    for item in standing(store, limit=limit):
        lines.append(f"  · {item['content'][:90]}")
    return lines
