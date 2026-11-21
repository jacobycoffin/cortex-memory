"""Recall replay: keep the exact block that was injected, so any answer can be re-derived (#24).

Every complaint about memory starts the same way — "why did it say that?". The score
breakdown (#2) explains the *ranking*; this keeps the *text*, because the block is what the
model actually saw. Given a task id you can read back the exact rendered memory block,
the ids it carried, and a SHA-256 of it, and prove that re-rendering the same ids today
produces the same bytes.

Storage is bounded by pruning rather than by hope: content is stored once per turn and
`prune()` drops anything older than the retention window (default 30 days). Additive table,
created on first use, written inside the caller's transaction.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Sequence

TABLE = "recall_replay"
DEFAULT_RETENTION_DAYS = 30

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    task_id TEXT PRIMARY KEY,
    session_id TEXT,
    block_sha256 TEXT NOT NULL,
    block_text TEXT NOT NULL,
    memory_ids_json TEXT NOT NULL DEFAULT '[]',
    tokens INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
)
"""


def digest(block: str) -> str:
    return hashlib.sha256(str(block or "").encode("utf-8")).hexdigest()


def record(
    store: Any,
    task_id: str,
    block: str,
    *,
    memory_ids: Sequence[str] = (),
    session_id: str | None = None,
) -> int:
    """Store the injected block. Returns 1 when written, 0 when skipped. Never raises."""
    conn = getattr(store, "_conn", None)
    if conn is None or not task_id or not block:
        return 0
    try:
        conn.execute(SCHEMA)
        conn.execute(
            f"""INSERT OR REPLACE INTO {TABLE}
                (task_id, session_id, block_sha256, block_text, memory_ids_json, tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
            (
                str(task_id),
                session_id,
                digest(block),
                str(block),
                json.dumps([str(i) for i in memory_ids]),
                (len(block) + 3) // 4,
            ),
        )
        return 1
    except sqlite3.Error:
        return 0


def load(store: Any, task_id: str) -> dict[str, Any] | None:
    conn = getattr(store, "_conn", None)
    if conn is None:
        return None
    try:
        row = conn.execute(
            f"""SELECT task_id, session_id, block_sha256, block_text, memory_ids_json,
                       tokens, created_at
                FROM {TABLE} WHERE task_id = ?""",
            (str(task_id),),
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return {
        "task_id": row[0],
        "session_id": row[1],
        "sha256": row[2],
        "block": row[3],
        "memory_ids": json.loads(row[4] or "[]"),
        "tokens": row[5],
        "created_at": row[6],
    }


def latest(store: Any, *, session_id: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
    conn = getattr(store, "_conn", None)
    if conn is None:
        return []
    try:
        if session_id:
            rows = conn.execute(
                f"SELECT task_id FROM {TABLE} WHERE session_id = ? "
                f"ORDER BY created_at DESC LIMIT ?",
                (str(session_id), int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT task_id FROM {TABLE} ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
    except sqlite3.Error:
        return []
    out = []
    for (task_id,) in rows:
        entry = load(store, task_id)
        if entry:
            out.append(entry)
    return out


def verify(store: Any, task_id: str, replayed_block: str) -> bool:
    """True when a re-rendered block is byte-identical to what was injected."""
    entry = load(store, task_id)
    if entry is None:
        return False
    return entry["sha256"] == digest(replayed_block)


def prune(store: Any, *, days: int = DEFAULT_RETENTION_DAYS) -> int:
    """Drop replay rows older than `days`. Returns rows removed."""
    conn = getattr(store, "_conn", None)
    if conn is None:
        return 0
    try:
        cursor = conn.execute(
            f"DELETE FROM {TABLE} WHERE created_at < datetime('now', ?)", (f"-{int(days)} days",)
        )
        return int(cursor.rowcount or 0)
    except sqlite3.Error:
        return 0
