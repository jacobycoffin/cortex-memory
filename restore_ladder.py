"""One call to bring a memory back, and the second time it is protected for good (#99).

Restoring is evidence: the ladder moved a memory down and something said no. One restore is a
judgement call; the same memory restored twice is a policy that is wrong about it. So the restore
path counts itself and, at the threshold, pins the memory permanently — no timer will walk it
again.

Counting comes from `lifecycle_events`, not a new table: every restore already writes a row
(`to_state = 'active'`, `from_state` in the ladder's down-states), and a separate counter would be
a second source of truth that can drift from the history it is supposed to summarise.

The pin is a real pin (`pinned = 1`), so every existing protection treats it like one: the ladder,
triage, dormancy and semantic forgetting all already refuse pinned memories. That is the point of
using the mechanism that exists rather than a new flag only this module knows about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PROTECT_AT = 2
_DOWNGRADES = ("cold", "archived", "tombstoned", "suppressed")


@dataclass
class Restored:
    memory_id: str
    restored: bool = False
    count: int = 0
    protected: bool = False

    @property
    def summary(self) -> str:
        if not self.restored:
            return f"{self.memory_id[:8]}: nothing to restore"
        if self.protected:
            return (
                f"{self.memory_id[:8]}: restored (restore #{self.count}) and pinned — "
                "no timer will walk it again"
            )
        remaining = max(0, PROTECT_AT - self.count)
        return (
            f"{self.memory_id[:8]}: restored (restore #{self.count}); "
            f"one more restore pins it permanently"
            if remaining == 1
            else f"{self.memory_id[:8]}: restored (restore #{self.count})"
        )


def restore_count(store: Any, memory_id: str) -> int:
    """How many times this memory has come back up the ladder."""
    try:
        row = store._conn.execute(
            f"""SELECT COUNT(*) FROM lifecycle_events
                WHERE memory_id = ? AND to_state = 'active'
                  AND from_state IN ({','.join('?' for _ in _DOWNGRADES)})""",
            (memory_id, *_DOWNGRADES),
        ).fetchone()
    except Exception:  # noqa: BLE001 - a store without history has no restores
        return 0
    return int(row[0] or 0) if row else 0


def is_protected(store: Any, memory_id: str) -> bool:
    row = store.get_memory(memory_id)
    return bool(row and row.get("pinned"))


def restore(
    store: Any,
    memory_id: str,
    *,
    to: str = "active",
    protect_at: int = PROTECT_AT,
    reason: str = "restored",
) -> Restored:
    """Bring a memory back, count it, and pin it once it has been restored `protect_at` times."""
    result = Restored(memory_id=memory_id)
    row = store.get_memory(memory_id)
    if not row:
        return result
    if str(row.get("state")) != to:
        result.restored = bool(store.set_state(memory_id, to, reason=f"restore ladder: {reason}"))
    else:
        result.restored = True  # already where it was asked to be; counting still matters
    # A suppression row is a different mechanism saying "stay hidden"; restoring lifts it too.
    try:
        with store.transaction() as conn:
            conn.execute("DELETE FROM suppressions WHERE memory_id = ?", (memory_id,))
    except Exception:  # noqa: BLE001 - table only exists once suppression has been used
        pass
    # Compaction deletes the derived feature rows of retired memories (#98), and those rows are how
    # the retriever finds a memory at all. Restoring therefore re-indexes, or "restored" would mean
    # "present but invisible" — the worst of both states.
    try:
        store._backfill_memory_features()
    except Exception:  # noqa: BLE001 - a restore must not fail on a re-index attempt
        pass
    result.count = restore_count(store, memory_id)
    if protect_at > 0 and result.count >= protect_at and not is_protected(store, memory_id):
        store.set_pinned(memory_id, True)
        result.protected = True
    elif is_protected(store, memory_id):
        result.protected = True
    return result


def protect(store: Any, memory_id: str, *, reason: str = "restored twice") -> bool:
    """Pin explicitly, for callers that want the second half without a restore."""
    if not store.get_memory(memory_id):
        return False
    store.set_pinned(memory_id, True)
    return True


def report(store: Any, *, limit: int = 20) -> dict[str, Any]:
    """Which memories have been restored, how often, and which are now pinned because of it."""
    rows = store._conn.execute(
        f"""SELECT memory_id, COUNT(*) AS n FROM lifecycle_events
            WHERE to_state = 'active' AND from_state IN ({','.join('?' for _ in _DOWNGRADES)})
            GROUP BY memory_id ORDER BY n DESC, memory_id ASC LIMIT ?""",
        (*_DOWNGRADES, limit),
    ).fetchall()
    restored = []
    for row in rows:
        memory_id = str(row["memory_id"])
        memory = store.get_memory(memory_id)
        restored.append(
            {
                "memory_id": memory_id,
                "count": int(row["n"]),
                "pinned": bool(memory and memory.get("pinned")),
                "state": (memory or {}).get("state"),
                "content": " ".join(str((memory or {}).get("content") or "").split())[:80],
            }
        )
    return {
        "protect_at": PROTECT_AT,
        "restored": restored,
        "would_be_pinned": [item["memory_id"] for item in restored
                            if item["count"] >= PROTECT_AT and not item["pinned"]],
    }
