"""Every fact carries its own review date (#64).

Time-based expiry is wrong in both directions. A blanket "forget after 90 days" deletes the port
number; a blanket "keep forever" keeps last month's rota. What actually varies is **how often a fact
deserves a second look**, and that is knowable at write:

| class | for | review |
|---|---|---|
| `permanent` | identity, preferences, procedures, durable facts | never — but a legal hold or a human can still ask |
| `review_90` | seasonal facts and ordinary semantics | in 90 days |
| `review_30` | ephemeral facts, episodes, transient operational state | in 30 days |

"Review" is not "delete". A due record is a **question**, not a sentence: `acknowledge()` pushes the
date out because somebody confirmed it is still true, and the record stays. That is the difference
between a store that keeps itself honest and one that quietly rots.

The class is derived from volatility (#31) and kind (#87), so it inherits their decisions rather
than inventing a parallel policy — if a fact is known to go stale in days, its review window is
short for the same reason its forgetting window is.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

PERMANENT = "permanent"
REVIEW_90 = "review_90"
REVIEW_30 = "review_30"
CLASSES = (PERMANENT, REVIEW_90, REVIEW_30)

DAYS = {PERMANENT: None, REVIEW_90: 90, REVIEW_30: 30}
TABLE = "memory_expiry"

# Kinds nobody has to re-check: who someone is, what they prefer, how a procedure goes.
_PERMANENT_KINDS = frozenset(
    {"identity", "preference", "procedure", "procedural", "commitment", "correction", "schema"}
)
_SHORT_KINDS = frozenset({"episode", "episodic", "operational", "tool"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def classify(kind: str | None, volatility_class: str | None = None) -> str:
    """Which review class this fact belongs to. Volatility wins; the kind is the fallback."""
    name = str(kind or "").strip().lower()
    flow = str(volatility_class or "").strip().lower()
    if flow == "ephemeral" and name not in _PERMANENT_KINDS:
        return REVIEW_30
    if flow == "durable":
        return PERMANENT
    if flow == "seasonal":
        return REVIEW_90
    if name in _PERMANENT_KINDS:
        return PERMANENT
    if name in _SHORT_KINDS:
        return REVIEW_30
    return REVIEW_90


def review_date(expiry_class: str, *, at: datetime | None = None) -> datetime | None:
    days = DAYS.get(str(expiry_class), 90)
    if days is None:
        return None
    return (at or _now()) + timedelta(days=days)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    memory_id TEXT PRIMARY KEY,
                    expiry_class TEXT NOT NULL,
                    review_after TEXT,
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT
                )"""
        )


@dataclass
class Expiry:
    memory_id: str
    expiry_class: str
    review_after: str | None

    def describe(self) -> str:
        if self.review_after is None:
            return f"{self.expiry_class} (no review date)"
        return f"{self.expiry_class} (review by {self.review_after[:10]})"


def assign(
    store: Any,
    memory_id: str,
    *,
    kind: str | None = None,
    volatility_class: str | None = None,
    at: datetime | None = None,
) -> Expiry | None:
    """Bake the class and the review date into the record. Idempotent per memory."""
    if not memory_id:
        return None
    expiry_class = classify(kind, volatility_class)
    when = review_date(expiry_class, at=at)
    stamp = (at or _now()).isoformat()
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE} (memory_id, expiry_class, review_after, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    expiry_class = excluded.expiry_class,
                    review_after = excluded.review_after""",
            (str(memory_id), expiry_class, when.isoformat() if when else None, stamp),
        )
    return Expiry(str(memory_id), expiry_class, when.isoformat() if when else None)


def of(store: Any, memory_id: str) -> Expiry | None:
    try:
        row = store._conn.execute(
            f"SELECT memory_id, expiry_class, review_after FROM {TABLE} WHERE memory_id = ?",
            (str(memory_id),),
        ).fetchone()
    except Exception:  # noqa: BLE001 - a missing table means "not assigned yet"
        return None
    if not row:
        return None
    return Expiry(str(row["memory_id"]), str(row["expiry_class"]),
                  row["review_after"] and str(row["review_after"]))


def due(store: Any, *, at: datetime | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Records whose review date has passed. A question, never a deletion."""
    _ensure(store)
    moment = (at or _now()).isoformat()
    rows = store._conn.execute(
        f"""SELECT e.memory_id, e.expiry_class, e.review_after, m.content, m.kind, m.state
            FROM {TABLE} e JOIN memories m ON m.id = e.memory_id
            WHERE e.review_after IS NOT NULL AND e.review_after <= ?
              AND m.state IN ('active','cold')
            ORDER BY e.review_after ASC LIMIT ?""",
        (moment, int(limit)),
    ).fetchall()
    return [
        {
            "memory_id": str(row["memory_id"]),
            "expiry_class": str(row["expiry_class"]),
            "review_after": str(row["review_after"]),
            "content": str(row["content"]),
            "kind": str(row["kind"]),
        }
        for row in rows
    ]


def acknowledge(
    store: Any, memory_id: str, *, at: datetime | None = None, extend: bool = True
) -> bool:
    """Somebody confirmed it is still true: push the review date out, keep the record."""
    expiry = of(store, memory_id)
    if expiry is None:
        return False
    moment = at or _now()
    when = review_date(expiry.expiry_class, at=moment) if extend else None
    _ensure(store)
    with store.transaction() as conn:
        cursor = conn.execute(
            f"""UPDATE {TABLE} SET review_after = ?, reviewed_at = ? WHERE memory_id = ?""",
            (when.isoformat() if when else None, moment.isoformat(), str(memory_id)),
        )
        return bool(cursor.rowcount)


def report(store: Any, *, at: datetime | None = None) -> dict[str, Any]:
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT expiry_class, COUNT(*) AS n FROM {TABLE}
            WHERE review_after IS NOT NULL AND review_after <= ?
            GROUP BY expiry_class""",
        ((at or _now()).isoformat(),),
    ).fetchall()
    overdue = {str(row["expiry_class"]): int(row["n"]) for row in rows}
    total = store._conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    return {
        "assigned": int(total or 0),
        "due": sum(overdue.values()),
        "by_class": overdue,
    }
