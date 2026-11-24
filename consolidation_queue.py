"""Two speeds: a small drain while working, the full sweep when quiet (#209).

Consolidation done only in one big pass has two failure modes: run it while somebody is talking and
it competes with the answer; run it only at night and a memory written this morning is still
unconsolidated at lunchtime, so the store answers from raw captures for hours.

So the work is **queued** and drained at two speeds:

- **the fast drain** runs during turns but is strictly bounded — a handful of items per turn, and a
  hard ceiling on how long it may take. Its job is to keep the queue from growing, not to finish it.
- **the slow sweep** runs when nobody is talking (via `idle_window`, so it inherits the lock and the
  staleness rules) and drains without a per-turn bound.

Three rules keep a queue from becoming a second source of truth:

- **the queue holds references, not content.** A row is a memory id and a reason; the memory itself is
  read when the work runs, so a stale queue entry cannot resurrect an old version of anything.
- **an item that fails is retried, not dropped**, and after `MAX_ATTEMPTS` it is parked with its error
  so a poisoned entry cannot spin forever;
- **draining is bounded by time, not by hope.** The fast drain checks a deadline before each item, so a
  slow item ends the batch instead of overrunning the turn's budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from . import idle_window

TABLE = "consolidation_queue"
MAX_ATTEMPTS = 3
DRAIN_BUDGET_MS = 150
DRAIN_MAX_ITEMS = 5


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    memory_id TEXT PRIMARY KEY,
                    reason TEXT NOT NULL DEFAULT '',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'queued',
                    last_error TEXT NOT NULL DEFAULT '',
                    queued_at TEXT NOT NULL,
                    drained_at TEXT NOT NULL DEFAULT ''
                )"""
        )


def enqueue(store: Any, memory_ids: Iterable[str], *, reason: str = "",
            at: datetime | None = None) -> int:
    """Queue memories for consolidation. Re-queuing a queued item is a no-op, not a duplicate."""
    ids = [str(item) for item in memory_ids if str(item).strip()]
    if not ids:
        return 0
    moment = (at or _now()).isoformat()
    _ensure(store)
    added = 0
    with store.transaction() as conn:
        for memory_id in ids:
            cursor = conn.execute(
                f"""INSERT OR IGNORE INTO {TABLE} (memory_id, reason, queued_at)
                    VALUES (?, ?, ?)""",
                (memory_id, str(reason), moment),
            )
            if cursor.rowcount:
                added += 1
                continue
            # A parked item that is written again deserves another chance: the new write is new
            # evidence, and leaving it parked would silently drop it.
            conn.execute(
                f"""UPDATE {TABLE} SET state = 'queued', attempts = 0, last_error = '',
                        reason = ?, queued_at = ?
                    WHERE memory_id = ? AND state = 'parked'""",
                (str(reason), moment, memory_id),
            )
    return added


def depth(store: Any) -> dict[str, int]:
    _ensure(store)
    rows = _rows(store, f"SELECT state, COUNT(*) AS count FROM {TABLE} GROUP BY state")
    return {str(row["state"]): int(row["count"] or 0) for row in rows}


@dataclass
class Drain:
    items: int = 0
    done: int = 0
    failed: int = 0
    parked: int = 0
    stopped_on_budget: bool = False
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "done": self.done,
            "failed": self.failed,
            "parked": self.parked,
            "stopped_on_budget": self.stopped_on_budget,
            "errors": self.errors,
            "sentence": self.describe(),
        }

    def describe(self) -> str:
        if not self.items:
            return "nothing queued"
        tail = " (stopped on the time budget)" if self.stopped_on_budget else ""
        return (f"drained {self.done}/{self.items}, {self.failed} failed, {self.parked} parked{tail}")


def _take(store: Any, limit: int) -> list[dict[str, Any]]:
    return _rows(
        store,
        f"SELECT memory_id, attempts FROM {TABLE} WHERE state = 'queued' "
        f"ORDER BY queued_at, memory_id LIMIT ?",
        (max(1, int(limit)),),
    )


def _finish(store: Any, memory_id: str, *, state: str, error: str = "",
            attempts: int = 0, at: datetime | None = None) -> None:
    moment = (at or _now()).isoformat()
    with store.transaction() as conn:
        conn.execute(
            f"""UPDATE {TABLE} SET state = ?, attempts = ?, last_error = ?, drained_at = ?
                WHERE memory_id = ?""",
            (state, int(attempts), str(error)[:300], moment, memory_id),
        )


def drain(store: Any, work: Callable[[str], Any], *, max_items: int = DRAIN_MAX_ITEMS,
          budget_ms: int = DRAIN_BUDGET_MS, now: datetime | None = None) -> Drain:
    """Drain a bounded batch. `work(memory_id)` does the consolidation for one memory."""
    import time

    started = time.monotonic()
    _ensure(store)
    found = Drain()
    for row in _take(store, max_items):
        if (time.monotonic() - started) * 1000 > float(budget_ms):
            found.stopped_on_budget = True
            break
        memory_id = str(row["memory_id"])
        attempts = int(row["attempts"] or 0)
        found.items += 1
        try:
            work(memory_id)
        except Exception as error:  # noqa: BLE001 - one bad item must not end the batch
            found.failed += 1
            found.errors.append(f"{memory_id}: {error}"[:160])
            attempts += 1
            if attempts >= MAX_ATTEMPTS:
                found.parked += 1
                _finish(store, memory_id, state="parked", error=str(error), attempts=attempts,
                        at=now)
            else:
                _finish(store, memory_id, state="queued", error=str(error), attempts=attempts,
                        at=now)
            continue
        found.done += 1
        _finish(store, memory_id, state="done", attempts=attempts, at=now)
    return found


def sweep(store: Any, work: Callable[[str], Any], *, min_idle_seconds: int = 600,
          limit: int = 500, now: datetime | None = None) -> dict[str, Any]:
    """The slow pass: only when nobody is talking, and it holds the idle lock while it runs."""
    verdict = idle_window.should_run(store, min_idle_seconds=min_idle_seconds, now=now)
    if not verdict.run:
        return {"ran": False, "reason": verdict.reason}
    result = idle_window.run_once(
        store, lambda: drain(store, work, max_items=limit, budget_ms=10 ** 9, now=now),
        min_idle_seconds=min_idle_seconds, now=now, note="consolidation sweep",
    )
    drained = result.get("result")
    return {
        "ran": bool(result.get("ran")),
        "status": result.get("status", ""),
        "drain": drained.as_dict() if isinstance(drained, Drain) else None,
    }


def queued(store: Any, limit: int = 50) -> list[dict[str, Any]]:
    _ensure(store)
    return _rows(
        store,
        f"SELECT * FROM {TABLE} WHERE state = 'queued' ORDER BY queued_at LIMIT ?",
        (int(limit),),
    )


def report(store: Any) -> dict[str, Any]:
    counts = depth(store)
    return {
        "queued": counts.get("queued", 0),
        "done": counts.get("done", 0),
        "parked": counts.get("parked", 0),
        "limits": {"max_items": DRAIN_MAX_ITEMS, "budget_ms": DRAIN_BUDGET_MS,
                   "max_attempts": MAX_ATTEMPTS},
        "note": "the queue holds references, not content",
    }
