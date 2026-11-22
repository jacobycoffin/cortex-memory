"""Legal hold: records that must never be pruned (#103).

A pin is a preference — anyone can unpin, and the ladder treats it as one protection among
several. A **legal hold** is a different kind of statement: a contract, a medical note, a record
that must survive an audit. So it is enforced differently in two specific ways:

1. **It is a protection the ladder, triage, quota and semantic forgetting all respect**, via the
   same `protections()` map the rest of the system uses — one place to add, nothing to remember.
2. **It survives `allow_protected=True`.** A hard delete can override a pin deliberately; it
   cannot override a hold. If a record is on hold, the only way through is `release()` first, and
   that release is itself recorded with who and why. Otherwise the flag is decorative: every
   destructive path would carry a `force` that quietly means "and ignore the hold too".

Holds carry `review_at` because a hold with no end is how an archive becomes unmanageable — the
report lists holds whose review date has passed, which is the prompt to confirm the record is still
needed rather than a timer that lifts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

_TABLE = "legal_holds"
HOLD = "legal_hold"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
                    memory_id TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    authority TEXT,
                    placed_at TEXT NOT NULL,
                    review_at TEXT,
                    released_at TEXT,
                    released_reason TEXT
                )"""
        )


@dataclass
class Hold:
    memory_id: str
    reason: str
    authority: str
    placed_at: str
    review_at: str | None
    content: str = ""

    def describe(self) -> str:
        due = f"review {self.review_at[:10]}" if self.review_at else "no review date"
        snippet = " ".join(self.content.split())[:70]
        return f"{self.memory_id[:8]} [{self.authority}] {self.reason} ({due}) {snippet}"


@dataclass
class Held:
    placed: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return f"placed {len(self.placed)}, refused {len(self.refused)}"


def place(
    store: Any,
    memory_ids: list[str],
    *,
    reason: str,
    authority: str = "user",
    review_at: datetime | str | None = None,
    at: datetime | None = None,
) -> Held:
    """Put memories on hold. Unknown ids are refused, not invented."""
    _ensure(store)
    stamp = (at or _utc_now()).isoformat()
    review = review_at.isoformat() if isinstance(review_at, datetime) else review_at
    result = Held()
    with store.transaction() as conn:
        for memory_id in memory_ids:
            resolved = store.resolve_id(str(memory_id)) or str(memory_id)
            if not store.get_memory(resolved):
                result.refused.append(str(memory_id))
                continue
            conn.execute(
                f"""INSERT INTO {_TABLE}(memory_id, reason, authority, placed_at, review_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        reason = excluded.reason,
                        authority = excluded.authority,
                        review_at = excluded.review_at,
                        released_at = NULL,
                        released_reason = NULL""",
                (resolved, reason, authority, stamp, review),
            )
            # A held record is also pinned: other protections and the UI already understand pins,
            # and a hold that is not visible in the profile is a hold nobody can see.
            store.set_pinned(resolved, True)
            result.placed.append(resolved)
    return result


def release(store: Any, memory_id: str, *, reason: str = "released", at: datetime | None = None) -> bool:
    """Take a record off hold. The release is recorded; the pin is left alone deliberately."""
    _ensure(store)
    resolved = store.resolve_id(memory_id) or memory_id
    row = store._conn.execute(
        f"SELECT memory_id FROM {_TABLE} WHERE memory_id = ? AND released_at IS NULL",
        (resolved,),
    ).fetchone()
    if row is None:
        return False
    with store.transaction() as conn:
        conn.execute(
            f"UPDATE {_TABLE} SET released_at = ?, released_reason = ? WHERE memory_id = ?",
            ((at or _utc_now()).isoformat(), reason, resolved),
        )
    return True


def is_held(store: Any, memory_id: str) -> bool:
    _ensure(store)
    resolved = store.resolve_id(memory_id) or memory_id
    row = store._conn.execute(
        f"SELECT 1 FROM {_TABLE} WHERE memory_id = ? AND released_at IS NULL", (resolved,)
    ).fetchone()
    return row is not None


def held_ids(store: Any) -> set[str]:
    """Every memory currently on hold — the map `protections()` consumes."""
    _ensure(store)
    try:
        rows = store._conn.execute(
            f"SELECT memory_id FROM {_TABLE} WHERE released_at IS NULL"
        ).fetchall()
    except Exception:  # noqa: BLE001 - an unreadable hold table must not block a sweep
        return set()
    return {str(row["memory_id"]) for row in rows}


def holds(store: Any, *, include_released: bool = False, limit: int = 100) -> list[Hold]:
    _ensure(store)
    clause = "" if include_released else "WHERE h.released_at IS NULL"
    rows = store._conn.execute(
        f"""SELECT h.memory_id, h.reason, h.authority, h.placed_at, h.review_at, m.content
            FROM {_TABLE} h LEFT JOIN memories m ON m.id = h.memory_id
            {clause} ORDER BY h.placed_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        Hold(
            memory_id=str(row["memory_id"]),
            reason=str(row["reason"]),
            authority=str(row["authority"] or "user"),
            placed_at=str(row["placed_at"]),
            review_at=row["review_at"],
            content=str(row["content"] or ""),
        )
        for row in rows
    ]


def due_for_review(store: Any, *, at: datetime | None = None) -> list[Hold]:
    now = (at or _utc_now()).isoformat()
    return [
        hold
        for hold in holds(store)
        if hold.review_at and str(hold.review_at) <= now
    ]


def report(store: Any, *, at: datetime | None = None) -> dict[str, Any]:
    current = holds(store)
    due = due_for_review(store, at=at)
    return {
        "held": len(current),
        "due_for_review": len(due),
        "holds": [hold.describe() for hold in current[:20]],
    }


def review_in(days: int, *, at: datetime | None = None) -> str:
    return ((at or _utc_now()) + timedelta(days=days)).isoformat()
