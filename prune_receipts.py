"""A receipt for every prune, and receipts that are themselves pruned (#100).

Every destructive or semi-destructive operation in this repo already writes *something* —
`lifecycle_events` for state moves, `deletions` for hard deletes, `forget_previews` for envelopes,
`triage_batches` for sweeps. What none of them answer is the question a person actually asks:

    "What did this system do to my memory this week, and can I check that it happened?"

A receipt is that answer: one row per operation, with the ids it touched, so it can be **verified**
later (are those memories actually in the state the receipt claims?) rather than taken on trust.

And because receipts accumulate, they expire: `prune()` removes receipts older than a retention
window while always keeping the newest few per kind. The pruning is counted, not itself receipted —
a log of pruning the log is how audit systems grow without bound.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

_TABLE = "prune_receipts"
_STATS = "prune_receipt_stats"
RETENTION_DAYS = 90
KEEP_LAST = 10


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _retention_days() -> int:
    raw = os.environ.get("CORTEX_RECEIPT_RETENTION_DAYS", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return RETENTION_DAYS
    return value if value >= 0 else RETENTION_DAYS


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
                    receipt_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    action TEXT NOT NULL,
                    item_count INTEGER NOT NULL,
                    ids_json TEXT NOT NULL,
                    summary TEXT,
                    created_at TEXT NOT NULL
                )"""
        )
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {_STATS} (
                    key TEXT PRIMARY KEY,
                    value INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )"""
        )


def record(
    store: Any,
    kind: str,
    action: str,
    ids: list[str],
    *,
    summary: str = "",
    at: datetime | None = None,
) -> str:
    """Record what was pruned. Returns the receipt id."""
    _ensure(store)
    stamp = (at or _utc_now()).isoformat()
    memory_ids = [str(item) for item in ids]
    receipt_id = hashlib.sha256(
        f"{kind}:{action}:{stamp}:{sorted(memory_ids)}".encode("utf-8")
    ).hexdigest()[:16]
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT OR REPLACE INTO {_TABLE}
                (receipt_id, kind, action, item_count, ids_json, summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (receipt_id, kind, action, len(memory_ids), json.dumps(memory_ids),
             summary, stamp),
        )
    return receipt_id


@dataclass
class Receipt:
    receipt_id: str
    kind: str
    action: str
    created_at: str
    ids: list[str] = field(default_factory=list)
    summary: str = ""

    @property
    def count(self) -> int:
        return len(self.ids)

    def describe(self) -> str:
        return (
            f"{self.receipt_id} {self.created_at[:16]} {self.kind}/{self.action} "
            f"{self.count} memories{(' — ' + self.summary) if self.summary else ''}"
        )


def get(store: Any, receipt_id: str) -> Receipt | None:
    _ensure(store)
    row = store._conn.execute(
        f"""SELECT receipt_id, kind, action, ids_json, summary, created_at
            FROM {_TABLE} WHERE receipt_id = ?""",
        (receipt_id,),
    ).fetchone()
    if row is None:
        return None
    return Receipt(
        receipt_id=str(row["receipt_id"]),
        kind=str(row["kind"]),
        action=str(row["action"]),
        created_at=str(row["created_at"]),
        ids=[str(item) for item in json.loads(row["ids_json"])],
        summary=str(row["summary"] or ""),
    )


def recent(store: Any, *, limit: int = 20, kind: str | None = None) -> list[Receipt]:
    _ensure(store)
    if kind:
        rows = store._conn.execute(
            f"""SELECT receipt_id FROM {_TABLE} WHERE kind = ?
                ORDER BY created_at DESC LIMIT ?""",
            (kind, limit),
        ).fetchall()
    else:
        rows = store._conn.execute(
            f"SELECT receipt_id FROM {_TABLE} ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    found = []
    for row in rows:
        receipt = get(store, str(row["receipt_id"]))
        if receipt:
            found.append(receipt)
    return found


def verify(store: Any, receipt_id: str) -> dict[str, Any]:
    """Check the receipt against the store: did this actually happen?"""
    receipt = get(store, receipt_id)
    if receipt is None:
        return {"receipt_id": receipt_id, "found": False}
    present = 0
    missing = []
    for memory_id in receipt.ids:
        if store.get_memory(memory_id):
            present += 1
        else:
            missing.append(memory_id)
    return {
        "receipt_id": receipt_id,
        "found": True,
        "kind": receipt.kind,
        "action": receipt.action,
        "claimed": receipt.count,
        "still_present": present,
        "gone": len(missing),
        "gone_ids": missing[:10],
    }


def prune(
    store: Any,
    *,
    older_than_days: int | None = None,
    keep_last: int = KEEP_LAST,
    at: datetime | None = None,
) -> dict[str, int]:
    """Drop old receipts, always keeping the newest `keep_last` per kind."""
    _ensure(store)
    days = _retention_days() if older_than_days is None else older_than_days
    now = at or _utc_now()
    cutoff = (now - timedelta(days=days)).isoformat()
    kinds = [
        str(row["kind"])
        for row in store._conn.execute(f"SELECT DISTINCT kind FROM {_TABLE}").fetchall()
    ]
    deleted = 0
    kept = 0
    with store.transaction() as conn:
        for kind in kinds:
            rows = conn.execute(
                f"""SELECT receipt_id, created_at FROM {_TABLE} WHERE kind = ?
                    ORDER BY created_at DESC""",
                (kind,),
            ).fetchall()
            for index, row in enumerate(rows):
                if index < keep_last or str(row["created_at"]) >= cutoff:
                    kept += 1
                    continue
                conn.execute(f"DELETE FROM {_TABLE} WHERE receipt_id = ?",
                             (str(row["receipt_id"]),))
                deleted += 1
        conn.execute(
            f"""INSERT INTO {_STATS}(key, value, updated_at) VALUES ('pruned_receipts', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = value + excluded.value, updated_at = excluded.updated_at""",
            (deleted, now.isoformat()),
        )
    return {"deleted": deleted, "kept": kept, "retention_days": days, "keep_last": keep_last}


def stats(store: Any) -> dict[str, Any]:
    _ensure(store)
    total = store._conn.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0]
    row = store._conn.execute(
        f"SELECT value, updated_at FROM {_STATS} WHERE key = 'pruned_receipts'"
    ).fetchone()
    by_kind = {
        str(item["kind"]): int(item["n"])
        for item in store._conn.execute(
            f"SELECT kind, COUNT(*) AS n FROM {_TABLE} GROUP BY kind ORDER BY n DESC"
        ).fetchall()
    }
    return {
        "receipts": int(total or 0),
        "by_kind": by_kind,
        "pruned_total": int(row["value"]) if row else 0,
        "pruned_at": row["updated_at"] if row else None,
    }


def new_receipt_id() -> str:
    return uuid.uuid4().hex[:16]
