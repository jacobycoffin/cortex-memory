"""Keep the top-5 score breakdown for every recall, so a bad answer can be explained.

When recall picks the wrong memory, the useful question is not "what did it return" but
"what did each signal contribute to that choice". The trace records which memories were
selected; it does not keep the arithmetic. This does, in an additive table that is created
on first use so no migration is needed:

    recall_score_breakdown(task_id, rank, memory_id, score, components_json, created_at)

Only the top few results are kept — enough to explain the decision, small enough that it
costs nothing.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable

MAX_ROWS = 5
TABLE = "recall_score_breakdown"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    task_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    memory_id TEXT NOT NULL,
    score REAL NOT NULL,
    components_json TEXT NOT NULL DEFAULT '{{}}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (task_id, rank)
)
"""

# Raw vectors and internal bookkeeping are noise in an explanation.
_SKIP = {"shadow_score", "shadow_score_delta"}


def _components(result: Any) -> dict[str, float]:
    raw = dict(getattr(result, "components", {}) or {})
    out: dict[str, float] = {}
    for key, value in raw.items():
        if key in _SKIP or not isinstance(value, (int, float)):
            continue
        out[key] = round(float(value), 6)
    return out


def record_breakdown(store: Any, task_id: str, results: Iterable[Any]) -> int:
    """Persist the top MAX_ROWS results' scores and components. Returns rows written."""
    rows = []
    for rank, result in enumerate(list(results)[:MAX_ROWS], 1):
        memory_id = str((getattr(result, "memory", {}) or {}).get("id") or "")
        if not memory_id:
            continue
        rows.append(
            (
                str(task_id),
                rank,
                memory_id,
                round(float(getattr(result, "score", 0.0) or 0.0), 6),
                json.dumps(_components(result), sort_keys=True),
            )
        )
    if not rows:
        return 0

    conn = getattr(store, "_conn", None)
    lock = getattr(store, "_lock", None)
    if conn is None:
        return 0
    try:
        if lock is not None:
            with lock:
                conn.execute(SCHEMA)
                conn.execute(f"DELETE FROM {TABLE} WHERE task_id = ?", (str(task_id),))
                conn.executemany(
                    f"""INSERT INTO {TABLE}
                        (task_id, rank, memory_id, score, components_json, created_at)
                        VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                    rows,
                )
                # No commit: this participates in the caller's transaction so a
                # rolled-back turn leaves no breakdown behind either.
                return len(rows)
        conn.execute(SCHEMA)
        conn.execute(f"DELETE FROM {TABLE} WHERE task_id = ?", (str(task_id),))
        conn.executemany(
            f"""INSERT INTO {TABLE}
                (task_id, rank, memory_id, score, components_json, created_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))""",
            rows,
        )
        return len(rows)
    except sqlite3.Error:
        return 0


def load_breakdown(store: Any, task_id: str) -> list[dict[str, Any]]:
    conn = getattr(store, "_conn", None)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            f"""SELECT rank, memory_id, score, components_json FROM {TABLE}
                WHERE task_id = ? ORDER BY rank""",
            (str(task_id),),
        ).fetchall()
    except sqlite3.Error:
        return []
    out = []
    for rank, memory_id, score, components_json in rows:
        try:
            components = json.loads(components_json)
        except ValueError:
            components = {}
        out.append(
            {"rank": rank, "memory_id": memory_id, "score": score, "components": components}
        )
    return out
