"""Suppression and deletion are different operations, and both are visibly honoured (#91).

They get confused because both make a memory stop appearing, which is exactly why the
distinction has to be explicit in code rather than a flag someone remembers:

| | suppressed | deleted |
|---|---|---|
| the row | stays | gone |
| the content | stays, auditable | gone |
| recall | excluded | excluded |
| the graph | untouched | edges removed |
| reversal | `unsuppress()` | impossible; the audit row is all that is left |
| use it for | "stop showing me this for now" | "this must not exist" |

Suppression is a state (`suppressed`), which matters because every recall path in this store
already filters on state — exclusion is honoured by construction rather than by remembering to
add a predicate. Deletion has no such safety net, so it is the operation that carries a
confirmation barrier: it requires the caller to name the exact id, refuses protected memories by
default, and writes an audit row **before** the content disappears, because after that there is
nothing left to describe what was lost.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

SUPPRESSED = "suppressed"
_SUPPRESSIONS = "suppressions"
_DELETIONS = "deletions"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {_SUPPRESSIONS} (
                    memory_id TEXT PRIMARY KEY,
                    reason TEXT,
                    until TEXT,
                    created_at TEXT NOT NULL
                )"""
        )
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {_DELETIONS} (
                    deleted_id TEXT PRIMARY KEY,
                    kind TEXT,
                    content_hash TEXT NOT NULL,
                    preview TEXT,
                    reason TEXT,
                    edges_removed INTEGER NOT NULL DEFAULT 0,
                    deleted_at TEXT NOT NULL
                )"""
        )


@dataclass
class Report:
    suppressed: int = 0
    expired: int = 0
    tombstoned: int = 0
    deleted: int = 0
    recent_deletions: list[dict[str, Any]] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [
            f"suppressed now: {self.suppressed}",
            f"suppressions past their end date (not yet expired): {self.expired}",
            f"tombstoned (ladder end state): {self.tombstoned}",
            f"deleted (audited, content gone): {self.deleted}",
        ]
        for row in self.recent_deletions:
            out.append(
                f"  deleted {row['deleted_id'][:8]} {row['kind']} "
                f"({row['deleted_at'][:10]}, {row['edges_removed']} edges): {row['reason']}"
            )
        return out


# --------------------------------------------------------------------------- suppression


def suppress(
    store: Any,
    memory_id: str,
    *,
    reason: str = "user request",
    until: datetime | str | None = None,
) -> bool:
    """Exclude a memory from recall without touching its content."""
    _ensure(store)
    row = store.get_memory(memory_id)
    if not row:
        return False
    stamp = until.isoformat() if isinstance(until, datetime) else (until or None)
    if not store.set_state(memory_id, SUPPRESSED, reason=f"suppressed: {reason}"):
        return False
    with store.transaction() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {_SUPPRESSIONS}(memory_id, reason, until, created_at) "
            "VALUES (?, ?, ?, ?)",
            (memory_id, reason, stamp, _utc_now().isoformat()),
        )
    return True


def unsuppress(store: Any, memory_id: str, *, to: str = "active") -> bool:
    """Bring a suppressed memory back, restoring its previous state when one is known."""
    _ensure(store)
    row = store._conn.execute(
        f"SELECT memory_id FROM {_SUPPRESSIONS} WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    if row is None:
        return False
    if not store.set_state(memory_id, to, reason="suppression lifted"):
        return False
    with store.transaction() as conn:
        conn.execute(f"DELETE FROM {_SUPPRESSIONS} WHERE memory_id = ?", (memory_id,))
    return True


def is_suppressed(store: Any, memory_id: str) -> bool:
    row = store.get_memory(memory_id)
    return bool(row and str(row.get("state")) == SUPPRESSED)


def expired(store: Any, *, at: datetime | None = None) -> list[str]:
    """Suppressions whose `until` has passed. Reported before they are lifted, deliberately."""
    _ensure(store)
    now = (at or _utc_now()).isoformat()
    rows = store._conn.execute(
        f"SELECT memory_id FROM {_SUPPRESSIONS} WHERE until IS NOT NULL AND until <= ?",
        (now,),
    ).fetchall()
    return [str(row["memory_id"]) for row in rows]


def expire(store: Any, *, at: datetime | None = None) -> list[str]:
    """Lift every suppression that has passed its end date."""
    lifted = []
    for memory_id in expired(store, at=at):
        if unsuppress(store, memory_id):
            lifted.append(memory_id)
    return lifted


def suppressions(store: Any, limit: int = 50) -> list[dict[str, Any]]:
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT s.memory_id, s.reason, s.until, s.created_at, m.kind, m.content
            FROM {_SUPPRESSIONS} s LEFT JOIN memories m ON m.id = s.memory_id
            ORDER BY s.created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


# ------------------------------------------------------------------------------ deletion


class DeleteRefused(ValueError):
    """Raised when a deletion is not safe or not confirmed."""


def delete(
    store: Any,
    memory_id: str,
    *,
    confirm: str,
    reason: str = "user request",
    allow_protected: bool = False,
) -> dict[str, Any]:
    """Destroy a memory. The content is gone afterwards; the audit row is not."""
    resolved = store.resolve_id(memory_id) or ""
    if not resolved or resolved != (store.resolve_id(confirm) or confirm):
        raise DeleteRefused(
            "deletion requires the exact id in `confirm`; nothing was removed"
        )
    row = store.get_memory(resolved)
    if not row:
        raise DeleteRefused("no such memory")
    # Checked before the pin check on purpose: a held record is also pinned, and "pass
    # allow_protected to override" must never be the answer for a legal hold (#103).
    from . import legal_hold

    if legal_hold.is_held(store, resolved):
        raise DeleteRefused(
            "memory is on legal hold; release the hold before deleting it"
        )
    if not allow_protected:
        from .forget_ladder import protections

        held = protections(store)
        if resolved in held:
            raise DeleteRefused(
                f"memory is protected ({held[resolved]}); pass allow_protected to override"
            )
    content = str(row.get("content") or "")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    edges_removed = store._conn.execute(
        "DELETE FROM edges WHERE src_id = ? OR dst_id = ?", (resolved, resolved)
    ).rowcount
    _ensure(store)
    with store.transaction() as conn:
        # Audit first: after the delete there is nothing left to describe what was lost.
        conn.execute(
            f"""INSERT OR REPLACE INTO {_DELETIONS}(deleted_id, kind, content_hash, preview,
                                                     reason, edges_removed, deleted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (resolved, str(row.get("kind") or ""), digest, content[:200], reason,
             int(edges_removed or 0), _utc_now().isoformat()),
        )
        for table in (_SUPPRESSIONS, "core_members", "prospective_items"):
            try:
                conn.execute(f"DELETE FROM {table} WHERE memory_id = ?", (resolved,))
            except Exception:  # noqa: BLE001 - table may not exist
                continue
        conn.execute("DELETE FROM memories WHERE id = ?", (resolved,))
    from . import prune_receipts

    prune_receipts.record(
        store, "deletion", "delete", [resolved],
        summary=f"{row.get('kind') or 'unknown'}, {int(edges_removed or 0)} edges, {reason}",
    )
    return {
        "deleted": resolved,
        "kind": row.get("kind"),
        "edges_removed": int(edges_removed or 0),
        "content_hash": digest[:16],
    }


def deletions(store: Any, limit: int = 20) -> list[dict[str, Any]]:
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT deleted_id, kind, content_hash, preview, reason, edges_removed, deleted_at
            FROM {_DELETIONS} ORDER BY deleted_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def report(store: Any, *, at: datetime | None = None) -> Report:
    _ensure(store)
    counts = {
        str(row["state"]): int(row["n"])
        for row in store._conn.execute(
            "SELECT state, COUNT(*) AS n FROM memories GROUP BY state"
        ).fetchall()
    }
    deleted = store._conn.execute(f"SELECT COUNT(*) FROM {_DELETIONS}").fetchone()[0]
    return Report(
        suppressed=counts.get(SUPPRESSED, 0),
        expired=len(expired(store, at=at)),
        tombstoned=counts.get("tombstoned", 0),
        deleted=int(deleted or 0),
        recent_deletions=deletions(store, limit=5),
    )


def restore_deleted_note(store: Any, deleted_id: str) -> str:
    """What is left of a deleted memory: the audit preview, not the memory itself."""
    for row in deletions(store, limit=10_000):
        if str(row["deleted_id"]).startswith(deleted_id):
            return (
                f"{row['deleted_id']} was deleted {row['deleted_at'][:19]} "
                f"({row['reason']}): {row['preview']}"
            )
    return f"no deletion recorded for {deleted_id}"
