"""No memory may hold more than N edges; the overflow waits for a topic node (#73).

A hub is not a lie, it is a routing problem. One memory that co-occurs with everything (a
"current state" row, a session summary) accumulates thousands of edges, and every graph walk then
has to score them all — while the edges themselves carry almost no information, because *connected
to everything* means *connected to nothing in particular*.

So the cap is on degree, not on truth:

- an edge is still admitted while the memory is under the cap;
- past the cap the edge is **recorded as overflow**, not dropped, not merged into a fake edge;
- overflow is grouped by hub and relation so a later topic node (#74) can adopt the group wholesale.

The alternative — silently refusing — is what makes people distrust a graph, because the link they
just watched the system use is missing afterwards with no explanation. Here the refusal is a row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

DEFAULT_CAP = 24
TABLE = "edge_overflow"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    memory_id TEXT NOT NULL,
                    other_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL NOT NULL,
                    refused_at TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    PRIMARY KEY (memory_id, other_id, relation)
                )"""
        )


def cap(store: Any | None = None) -> int:
    import os

    raw = os.environ.get("CORTEX_MAX_DEGREE", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = DEFAULT_CAP
        if value > 0:
            return value
    return DEFAULT_CAP


def degree(store: Any, memory_id: str) -> int:
    try:
        row = store._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE src_id = ? OR dst_id = ?",
            (str(memory_id), str(memory_id)),
        ).fetchone()
    except Exception:  # noqa: BLE001 - no edges table means degree zero
        return 0
    return int(row[0] or 0)


@dataclass
class Admission:
    admitted: bool
    memory_id: str = ""
    reason: str = ""
    degree: int = 0

    def describe(self) -> str:
        if self.admitted:
            return "admitted"
        return f"refused: {self.reason} ({self.memory_id[:8]} at {self.degree})"


def admit(
    store: Any,
    src_id: str,
    dst_id: str,
    relation: str,
    *,
    weight: float = 0.25,
    maximum: int | None = None,
    at: datetime | None = None,
) -> Admission:
    """Is this edge allowed? A refusal is recorded, never silent."""
    limit = maximum or cap()
    for memory_id in (str(src_id), str(dst_id)):
        current = degree(store, memory_id)
        if current >= limit:
            record_overflow(
                store, memory_id, str(src_id) if memory_id == str(dst_id) else str(dst_id),
                relation, weight, at=at,
                reason=f"degree {current} at cap {limit}",
            )
            return Admission(False, memory_id, f"degree {current} at cap {limit}", current)
    return Admission(True, "", "", 0)


def record_overflow(
    store: Any,
    memory_id: str,
    other_id: str,
    relation: str,
    weight: float,
    *,
    at: datetime | None = None,
    reason: str = "over cap",
) -> None:
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE} (memory_id, other_id, relation, weight, refused_at, reason)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, other_id, relation) DO UPDATE SET
                    weight = excluded.weight, refused_at = excluded.refused_at""",
            (str(memory_id), str(other_id), str(relation), float(weight),
             (at or _now()).isoformat(), str(reason)),
        )


def overflow(store: Any, memory_id: str | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
    """What is waiting, grouped for a future topic node to adopt."""
    try:
        _ensure(store)
        if memory_id:
            rows = store._conn.execute(
                f"SELECT * FROM {TABLE} WHERE memory_id = ? ORDER BY weight DESC LIMIT ?",
                (str(memory_id), int(limit)),
            ).fetchall()
        else:
            rows = store._conn.execute(
                f"SELECT * FROM {TABLE} ORDER BY weight DESC LIMIT ?", (int(limit),)
            ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [dict(row) for row in rows]


def groups(store: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    """Overflow by hub — the natural unit a topic node would adopt."""
    try:
        _ensure(store)
        rows = store._conn.execute(
            f"""SELECT memory_id, relation, COUNT(*) AS n, SUM(weight) AS mass
                FROM {TABLE} GROUP BY memory_id, relation
                ORDER BY n DESC LIMIT ?""",
            (int(limit),),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [
        {"memory_id": str(row["memory_id"]), "relation": str(row["relation"]),
         "pending": int(row["n"]), "mass": round(float(row["mass"] or 0.0), 3)}
        for row in rows
    ]


def release(store: Any, memory_id: str, *, keep: int = 0) -> int:
    """Drop overflow rows for a hub (a topic node adopted them, or the hub was pruned)."""
    try:
        _ensure(store)
        with store.transaction() as conn:
            cursor = conn.execute(
                f"DELETE FROM {TABLE} WHERE memory_id = ?", (str(memory_id),)
            )
            return int(cursor.rowcount or 0)
    except Exception:  # noqa: BLE001
        return 0


def hubs(store: Any, *, maximum: int | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """Memories already over the cap — the hubs that exist before any enforcement."""
    limit_value = maximum or cap()
    try:
        rows = store._conn.execute(
            """SELECT m.id, m.kind, m.content,
                      (SELECT COUNT(*) FROM edges e
                        WHERE e.src_id = m.id OR e.dst_id = m.id) AS n
               FROM memories m
               WHERE m.state NOT IN ('tombstoned','deleted')
               ORDER BY n DESC LIMIT ?""",
            (int(limit),),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [
        {"memory_id": str(row["id"]), "kind": str(row["kind"]), "degree": int(row["n"] or 0),
         "over": max(0, int(row["n"] or 0) - limit_value), "content": str(row["content"])[:80]}
        for row in rows
        if int(row["n"] or 0) > 0
    ]


def report(store: Any, *, maximum: int | None = None) -> dict[str, Any]:
    limit_value = maximum or cap()
    ranked = hubs(store, maximum=limit_value, limit=200)
    over = [item for item in ranked if item["degree"] > limit_value]
    total = 0
    try:
        total = int(store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] or 0)
    except Exception:  # noqa: BLE001
        pass
    return {
        "cap": limit_value,
        "edges": total,
        "over_cap": len(over),
        "pending": len(overflow(store, limit=10000)),
        "worst": [(item["degree"], item["content"]) for item in ranked[:5]],
    }
