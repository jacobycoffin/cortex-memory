"""Read-only queries that never issue COMMIT.

`CortexStore.transaction()` is a write transaction: it opens BEGIN IMMEDIATE and commits
on exit. Using it for a plain SELECT inside a caller that already guarantees atomicity adds
an extra COMMIT and breaks that guarantee — a real regression caught by
`test_core_ledger_is_atomic_and_uses_one_commit`.

Reads go through the connection under the store lock instead, so they observe a consistent
view without changing anyone's transaction state.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Sequence


def select(store: Any, sql: str, params: Sequence[Any] = ()) -> list[tuple]:
    """Execute a read without opening or committing a transaction."""
    lock = getattr(store, "_lock", None)
    conn = getattr(store, "_conn", None)
    if conn is None:
        return []
    try:
        if lock is not None:
            with lock:
                return conn.execute(sql, tuple(params)).fetchall()
        return conn.execute(sql, tuple(params)).fetchall()
    except sqlite3.Error:
        return []
