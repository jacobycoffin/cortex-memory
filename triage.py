"""Batch triage for memories that were never recalled (#92).

The live store holds ~1,250 active memories that have never been retrieved once. They are not
wrong — they are unread: written by an import, by a vault sync, or by a capture that nobody ever
needed. They cost recall budget every turn they compete for, and no single one of them is worth a
decision on its own. That is a batch problem.

Three decisions shape this:

1. **Archive, not tombstone.** The default action is the gentlest one that removes a memory from
   recall. Triage exists to reduce noise, not to destroy anything.
2. **One tap undo, by batch.** An undo restores exactly the memories that batch archived — not
   "everything archived recently", which would undo decisions a person made by hand.
3. **Keep is a decision too.** Without a record of what a person looked at and kept, the same
   batch comes back next run, and triage becomes a chore that gets abandoned. `keep()` records the
   judgement so the queue shrinks monotonically.

A memory counts as never recalled when `retrieved_count = 0` **and** there is no `access_log` row
of any kind: an event of any type is evidence somebody touched it, and the counter alone can lag.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

BATCH_SIZE = 25
MIN_AGE_DAYS = 14
# Anything above this is a fact somebody thought was important when it was written; it belongs in
# a human's review, not in an automated batch. The first live batch proved the point: sorted by
# importance it led with the dashboard URL, the gaming PC's MAC address and a rollback path —
# unread is not the same as worthless.
HIGH_IMPORTANCE = 0.75
_BATCHES = "triage_batches"
_KEPT = "triage_kept"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _min_age_days() -> int:
    raw = os.environ.get("CORTEX_TRIAGE_MIN_AGE_DAYS", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return MIN_AGE_DAYS
    return value if value >= 0 else MIN_AGE_DAYS


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {_BATCHES} (
                    batch_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    ids_json TEXT NOT NULL,
                    applied_at TEXT,
                    undone_at TEXT
                )"""
        )
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {_KEPT} (
                    memory_id TEXT PRIMARY KEY,
                    kept_at TEXT NOT NULL
                )"""
        )


@dataclass
class Batch:
    batch_id: str
    created_at: str
    action: str
    items: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        return [str(item["memory_id"]) for item in self.items]

    def describe(self, width: int = 8, snippet: int = 110) -> str:
        lines = [f"batch {self.batch_id} · {len(self.items)} memories · default: {self.action}"]
        for item in self.items:
            content = " ".join(str(item.get("content") or "").split())
            lines.append(
                f"  {str(item['memory_id'])[:width]} [{item.get('kind')}] "
                f"{content[:snippet]}"
            )
        return "\n".join(lines)


def candidates(store: Any, *, limit: int = BATCH_SIZE, at: datetime | None = None) -> list[dict[str, Any]]:
    """Active memories that have never been recalled and are not otherwise protected."""
    from .forget_ladder import NEVER_WALK, protections

    _ensure(store)
    now = at or _utc_now()
    cutoff = (now - timedelta(days=_min_age_days())).isoformat()
    held = protections(store)
    # The kinds the ladder refuses to forget are also refused here: if a stale preference is
    # "still the preference", an unread one is still the preference. Triaging them would be the
    # same mistake with a different mechanism.
    never = tuple(sorted(NEVER_WALK))
    marks = ",".join("?" for _ in never)
    sql = (
        "SELECT m.id, m.kind, m.content, m.importance, m.created_at "
        "FROM memories m "
        "WHERE m.state = 'active' "
        "  AND COALESCE(m.pinned, 0) = 0 "
        "  AND COALESCE(m.retrieved_count, 0) = 0 "
        "  AND COALESCE(m.created_at, '') <= ? "
        "  AND COALESCE(m.importance, 0) < ? "
        "  AND NOT EXISTS (SELECT 1 FROM access_log a WHERE a.memory_id = m.id) "
        f"  AND m.id NOT IN (SELECT memory_id FROM {_KEPT}) "
        f"  AND COALESCE(m.kind, '') NOT IN ({marks}) "
        "ORDER BY COALESCE(m.importance, 0) ASC, m.id ASC "
        "LIMIT ?"
    )
    rows = store._conn.execute(
        sql, (cutoff, HIGH_IMPORTANCE, *never, max(limit * 4, 100))
    ).fetchall()
    items = []
    for row in rows:
        memory_id = str(row["id"])
        if memory_id in held:
            continue
        items.append(
            {
                "memory_id": memory_id,
                "kind": row["kind"],
                "content": row["content"],
                "importance": row["importance"],
                "created_at": row["created_at"],
            }
        )
        if len(items) >= limit:
            break
    return items


def count_candidates(store: Any, *, at: datetime | None = None) -> int:
    """How big the queue is, without building a batch."""
    return len(candidates(store, limit=100_000, at=at))


def batch(
    store: Any,
    *,
    size: int = BATCH_SIZE,
    action: str = "archived",
    at: datetime | None = None,
) -> Batch:
    """Build (but do not apply) the next batch. Recorded so the undo can find it later."""
    if action not in {"archived", "cold"}:
        raise ValueError("triage only archives or cools; use the forget ladder to go further")
    items = candidates(store, limit=size, at=at)
    stamp = (at or _utc_now()).isoformat()
    payload = json.dumps([item["memory_id"] for item in items], sort_keys=True)
    batch_id = hashlib.sha256(f"{stamp}:{payload}".encode("utf-8")).hexdigest()[:16]
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {_BATCHES}(batch_id, created_at, action, ids_json) "
            "VALUES (?, ?, ?, ?)",
            (batch_id, stamp, action, payload),
        )
    return Batch(batch_id=batch_id, created_at=stamp, action=action, items=items)


def apply(store: Any, batch: Batch | str, *, at: datetime | None = None) -> list[str]:
    """Archive the batch. Returns the ids actually moved."""
    record = _record(store, batch)
    if record is None:
        return []
    if record["applied_at"]:
        return []
    moved = []
    for memory_id in json.loads(record["ids_json"]):
        try:
            if store.set_state(memory_id, record["action"],
                               reason=f"cold sweep triage batch {record['batch_id']}"):
                moved.append(str(memory_id))
        except Exception:  # noqa: BLE001 - one bad row must not abort a reviewed batch
            continue
    with store.transaction() as conn:
        conn.execute(
            f"UPDATE {_BATCHES} SET applied_at = ? WHERE batch_id = ?",
            ((at or _utc_now()).isoformat(), record["batch_id"]),
        )
    if moved:
        from . import prune_receipts

        prune_receipts.record(
            store, "triage", record["action"], moved,
            summary=f"batch {record['batch_id']}",
        )
    return moved


def undo(store: Any, batch: Batch | str, *, to: str = "active") -> list[str]:
    """One tap: restore exactly the memories this batch moved, and nothing else."""
    record = _record(store, batch)
    if record is None or not record["applied_at"] or record["undone_at"]:
        return []
    restored = []
    for memory_id in json.loads(record["ids_json"]):
        try:
            if store.set_state(str(memory_id), to,
                               reason=f"triage undo batch {record['batch_id']}"):
                restored.append(str(memory_id))
        except Exception:  # noqa: BLE001
            continue
    with store.transaction() as conn:
        conn.execute(
            f"UPDATE {_BATCHES} SET undone_at = ? WHERE batch_id = ?",
            (_utc_now().isoformat(), record["batch_id"]),
        )
    return restored


def keep(store: Any, memory_ids: list[str]) -> int:
    """Record "I looked at these and they stay", so the queue shrinks monotonically."""
    _ensure(store)
    stamp = _utc_now().isoformat()
    kept = 0
    with store.transaction() as conn:
        for memory_id in memory_ids:
            conn.execute(
                f"INSERT OR IGNORE INTO {_KEPT}(memory_id, kept_at) VALUES (?, ?)",
                (str(memory_id), stamp),
            )
            kept += 1
    return kept


def kept(store: Any, limit: int = 500) -> list[str]:
    _ensure(store)
    rows = store._conn.execute(
        f"SELECT memory_id FROM {_KEPT} ORDER BY kept_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [str(row["memory_id"]) for row in rows]


def batches(store: Any, limit: int = 20) -> list[dict[str, Any]]:
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT batch_id, created_at, action, ids_json, applied_at, undone_at
            FROM {_BATCHES} ORDER BY created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        {
            "batch_id": str(row["batch_id"]),
            "created_at": str(row["created_at"]),
            "action": str(row["action"]),
            "size": len(json.loads(row["ids_json"])),
            "applied_at": row["applied_at"],
            "undone_at": row["undone_at"],
        }
        for row in rows
    ]


def _record(store: Any, batch: Batch | str) -> dict[str, Any] | None:
    _ensure(store)
    batch_id = batch.batch_id if isinstance(batch, Batch) else str(batch)
    row = store._conn.execute(
        f"""SELECT batch_id, created_at, action, ids_json, applied_at, undone_at
            FROM {_BATCHES} WHERE batch_id = ?""",
        (batch_id,),
    ).fetchone()
    return dict(row) if row else None


def new_batch_id() -> str:
    return uuid.uuid4().hex[:16]
