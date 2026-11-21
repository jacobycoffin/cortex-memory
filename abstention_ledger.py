"""An abstention ledger that is useful while abstention is switched off (#15).

Abstention (#14) shipped disabled because no signal separated good recalls from bad ones on
the frozen set. The way out is not a cleverer cut-off, it is real turn data: record what the
policy *would* have done on every live recall, then check later whether the memory it would
have withheld was actually used.

So this is a **shadow ledger**. It records the top candidate's score and whether the
configured shadow threshold would have abstained — always, regardless of whether the real
policy is enabled — plus the ids needed to join against `access_log` for a `used` event
afterwards. That makes "would abstaining here have been wrong?" a query, not an opinion.

Additive table, created on first use, written inside the caller's transaction.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Sequence

TABLE = "abstention_ledger"
DEFAULT_SHADOW_THRESHOLD = 0.55

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    task_id TEXT NOT NULL,
    session_id TEXT,
    threshold REAL NOT NULL,
    top_score REAL NOT NULL,
    top_memory_id TEXT,
    would_abstain INTEGER NOT NULL DEFAULT 0,
    abstained INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (task_id, threshold)
)
"""


def _conn(store: Any) -> sqlite3.Connection | None:
    return getattr(store, "_conn", None)


def record(
    store: Any,
    task_id: str,
    *,
    top_score: float,
    top_memory_id: str | None,
    threshold: float = DEFAULT_SHADOW_THRESHOLD,
    abstained: bool = False,
    session_id: str | None = None,
) -> int:
    """Record one shadow decision. Returns rows written (0 or 1). Never raises."""
    conn = _conn(store)
    if conn is None or not task_id or threshold is None:
        return 0
    try:
        conn.execute(SCHEMA)
        conn.execute(
            f"""INSERT OR REPLACE INTO {TABLE}
                (task_id, session_id, threshold, top_score, top_memory_id,
                 would_abstain, abstained, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (
                str(task_id),
                session_id,
                float(threshold),
                round(float(top_score), 6),
                top_memory_id,
                1 if float(top_score) < float(threshold) else 0,
                1 if abstained else 0,
            ),
        )
        return 1
    except sqlite3.Error:
        return 0


def summary(store: Any, *, days: int = 30) -> dict[str, Any]:
    """Counts for the last `days`, including how often withholding would have been wrong.

    "Wrong" means the memory the policy would have withheld was later recorded as `used`,
    which is the only outcome signal the store keeps.
    """
    conn = _conn(store)
    empty = {
        "turns": 0,
        "would_abstain": 0,
        "abstained": 0,
        "would_abstain_then_used": 0,
        "would_abstain_precision": 0.0,
        "min_threshold": None,
        "max_threshold": None,
    }
    if conn is None:
        return empty
    try:
        rows = conn.execute(
            f"""
            SELECT l.would_abstain, l.abstained, l.threshold,
                   EXISTS (
                       SELECT 1 FROM access_log a
                       WHERE a.memory_id = l.top_memory_id
                         AND a.event = 'used'
                         AND a.created_at >= l.created_at
                   ) AS later_used
            FROM {TABLE} l
            WHERE l.created_at >= datetime('now', ?)
            """,
            (f"-{int(days)} days",),
        ).fetchall()
    except sqlite3.Error:
        return empty

    turns = len(rows)
    would = [r for r in rows if r[0]]
    hits_lost = [r for r in would if r[3]]
    thresholds = [float(r[2]) for r in rows]
    return {
        "turns": turns,
        "would_abstain": len(would),
        "abstained": sum(1 for r in rows if r[1]),
        "would_abstain_then_used": len(hits_lost),
        "would_abstain_precision": (
            round(1 - len(hits_lost) / len(would), 4) if would else 0.0
        ),
        "min_threshold": min(thresholds) if thresholds else None,
        "max_threshold": max(thresholds) if thresholds else None,
    }


def decisions(store: Any, *, limit: int = 20, days: int = 30) -> Sequence[Sequence[Any]]:
    conn = _conn(store)
    if conn is None:
        return []
    try:
        return conn.execute(
            f"""SELECT task_id, threshold, top_score, would_abstain, abstained, created_at
                FROM {TABLE}
                WHERE created_at >= datetime('now', ?)
                  AND would_abstain = 1
                ORDER BY created_at DESC LIMIT ?""",
            (f"-{int(days)} days", int(limit)),
        ).fetchall()
    except sqlite3.Error:
        return []
