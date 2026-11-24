"""Consolidate when the box is quiet, not while a turn is waiting (#212).

The expensive passes — consolidation, pruning, the nightly proposal build — all compete with the same
CPU as the turn that is about to answer a person. Running them on a timer means they sometimes run at
the worst possible moment, and the cost shows up as latency somebody notices rather than as a number
in a report.

So the trigger is **activity**, not the clock:

- **idle means nobody is talking.** Activity is read from what the store already records — access log,
  traces, episodes — and the newest of them decides. No new heartbeat to keep alive.
- **a run holds a lock.** Two schedulers, or a slow run overlapping the next tick, must not consolidate
  twice; the lock is a row, so it survives a process restart, and a stale lock **expires** rather than
  wedging the pass forever.
- **a busy store gets a verdict, not a silent skip.** `should_run()` returns its reasons, so a pass
  that never fires is visible instead of looking like a scheduler that was never installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

TABLE = "idle_runs"
DEFAULT_IDLE_SECONDS = 600
STALE_LOCK_SECONDS = 3600

# Where a turn leaves a trace. The newest record across these is the last time anybody was talking.
ACTIVITY_SOURCES = (
    ("access_log", "created_at"),
    ("memory_traces", "created_at"),
    ("episodes", "created_at"),
    ("tool_executions", "created_at"),
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 - a trigger that cannot read must not run the pass
        return []


def _parse(stamp: Any) -> datetime | None:
    text = str(stamp or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'running',
                    note TEXT NOT NULL DEFAULT ''
                )"""
        )


@dataclass
class Activity:
    last_seen: str = ""
    idle_seconds: float = 0.0
    sources: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "last_seen": self.last_seen,
            "idle_seconds": round(self.idle_seconds, 3),
            "sources": self.sources,
        }


def activity(store: Any, *, now: datetime | None = None) -> Activity:
    """The newest recorded activity across every source the store already keeps."""
    moment = now or _now()
    newest: datetime | None = None
    sources: dict[str, str] = {}
    for table, column in ACTIVITY_SOURCES:
        rows = _rows(store, f"SELECT MAX({column}) AS last FROM {table}")
        stamp = _parse((rows[0] if rows else {}).get("last"))
        if stamp is None:
            continue
        sources[table] = stamp.isoformat()
        if newest is None or stamp > newest:
            newest = stamp
    if newest is None:
        # Nothing has ever happened: the store cannot be "busy", and pretending it is would keep the
        # first consolidation from ever running on a fresh install.
        return Activity(last_seen="", idle_seconds=float("inf"), sources=sources)
    return Activity(last_seen=newest.isoformat(),
                    idle_seconds=(moment - newest).total_seconds(), sources=sources)


@dataclass
class Verdict:
    run: bool
    reason: str = ""
    activity: Activity = field(default_factory=Activity)
    lock_holder: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "reason": self.reason,
            "activity": self.activity.as_dict(),
            "lock_holder": self.lock_holder,
        }


def _lock_holder(store: Any, *, now: datetime, stale_seconds: int = STALE_LOCK_SECONDS) -> str:
    """The run currently holding the lock, or "" when it is free or stale."""
    _ensure(store)
    rows = _rows(
        store,
        f"SELECT run_id, started_at FROM {TABLE} WHERE status = 'running' "
        f"ORDER BY started_at DESC LIMIT 1",
    )
    if not rows:
        return ""
    started = _parse(rows[0]["started_at"])
    if started is None:
        return ""
    if (now - started).total_seconds() > stale_seconds:
        # A crashed run must not wedge the pass forever; the stale lock is reported, not deleted, so
        # the history keeps showing that something died holding it.
        return ""
    return str(rows[0]["run_id"])


def should_run(store: Any, *, min_idle_seconds: int = DEFAULT_IDLE_SECONDS,
               now: datetime | None = None) -> Verdict:
    moment = now or _now()
    found = activity(store, now=moment)
    holder = _lock_holder(store, now=moment)
    if holder:
        return Verdict(run=False, reason=f"another run holds the lock ({holder})",
                       activity=found, lock_holder=holder)
    if found.idle_seconds < float(min_idle_seconds):
        return Verdict(run=False,
                       reason=f"busy: last activity {found.idle_seconds:.0f}s ago, "
                              f"need {int(min_idle_seconds)}s",
                       activity=found)
    return Verdict(run=True, reason="idle and unlocked", activity=found)


def run_once(store: Any, work: Callable[[], Any], *, min_idle_seconds: int = DEFAULT_IDLE_SECONDS,
             now: datetime | None = None, note: str = "") -> dict[str, Any]:
    """Run `work` only when the pass is allowed, holding a lock for its duration."""
    moment = now or _now()
    verdict = should_run(store, min_idle_seconds=min_idle_seconds, now=moment)
    if not verdict.run:
        return {"ran": False, "reason": verdict.reason,
                "activity": verdict.activity.as_dict()}
    _ensure(store)
    run_id = f"idle-{int(moment.timestamp())}-{abs(hash(note)) % 10_000:04d}"
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT OR REPLACE INTO {TABLE} (run_id, started_at, finished_at, status, note)
                VALUES (?, ?, '', 'running', ?)""",
            (run_id, moment.isoformat(), note),
        )
    try:
        result = work()
        status, error = "done", ""
    except Exception as error_:  # noqa: BLE001 - the lock must be released whatever happened
        result, status, error = None, "failed", str(error_)[:200]
    finished = (now or _now()).isoformat()
    with store.transaction() as conn:
        conn.execute(
            f"UPDATE {TABLE} SET finished_at = ?, status = ?, note = ? WHERE run_id = ?",
            (finished, status, (note if not error else f"{note} | {error}").strip(" |"), run_id),
        )
    return {"ran": True, "run_id": run_id, "status": status, "result": result, "error": error}


def history(store: Any, limit: int = 20) -> list[dict[str, Any]]:
    _ensure(store)
    return _rows(store, f"SELECT * FROM {TABLE} ORDER BY started_at DESC LIMIT ?", (int(limit),))


def report(store: Any, *, min_idle_seconds: int = DEFAULT_IDLE_SECONDS,
           now: datetime | None = None) -> dict[str, Any]:
    verdict = should_run(store, min_idle_seconds=min_idle_seconds, now=now)
    runs = history(store)
    counts: dict[str, int] = {}
    for row in runs:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    return {
        "run": verdict.run,
        "reason": verdict.reason,
        "activity": verdict.activity.as_dict(),
        "runs": len(runs),
        "by_status": counts,
        "min_idle_seconds": int(min_idle_seconds),
        "note": "a busy store gets a verdict, not a silent skip",
    }
