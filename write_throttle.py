"""Cap automatic writes per turn and per session (#36).

An extraction storm looks like this: one turn mentions twelve things, the extractor proposes twelve
memories, the judge accepts most of them, and the store gains an hour's worth of records in one
minute. Nothing in that chain is individually wrong — which is why it needs a cap rather than a
better rule: the failure is *rate*, and rates need limits.

Three limits, each with a job:

- **per turn** (default 5) — a turn is a moment; more than a handful of durable facts in one moment
  is a sign the extractor is scraping rather than listening;
- **per session** (default 40) — a long conversation still only teaches so much;
- **per hour** (default 25) — catches the storm that spreads itself over several turns.

A throttled write is **deferred, never dropped**: the caller keeps the proposal staged, so nothing is
lost and the next turn can take it. And the throttle only ever sees automatic writes — an explicit
user statement is not rate-limited by anything, because the user asking is the strongest evidence
the store will ever get.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

TABLE = "write_throttle"

TURN_CAP = 5
SESSION_CAP = 40
HOUR_CAP = 25


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Decision:
    allowed: bool
    scope: str
    used: int
    cap: int
    reason: str = ""

    def describe(self) -> str:
        if self.allowed:
            return f"allowed ({self.scope} {self.used}/{self.cap})"
        return f"throttled: {self.reason} ({self.scope} {self.used}/{self.cap})"


def caps() -> tuple[int, int, int]:
    import os

    def read(name: str, default: int) -> int:
        try:
            value = int(str(os.environ.get(name, default)).strip())
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    return (
        read("CORTEX_WRITE_CAP_TURN", TURN_CAP),
        read("CORTEX_WRITE_CAP_SESSION", SESSION_CAP),
        read("CORTEX_WRITE_CAP_HOUR", HOUR_CAP),
    )


def _bucket(scope: str, session_id: str, turn_id: str | None, at: datetime) -> str:
    if scope == "turn":
        # No wall clock here. Keying an unidentified turn on the current second means a long turn
        # crosses a boundary and silently gets a second budget — which is how a slow extraction
        # storm walks past the per-turn cap. An unknown turn is stricter, not looser.
        key = str(turn_id or "").strip() or "unidentified"
        return f"turn:{session_id}:{key}"
    if scope == "hour":
        return f"hour:{session_id}:{at.strftime('%Y%m%dT%H')}"
    return f"session:{session_id}"


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    bucket TEXT PRIMARY KEY,
                    session_id TEXT,
                    writes INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )"""
        )


def _count(store: Any, bucket: str) -> int:
    try:
        row = store._conn.execute(
            f"SELECT writes FROM {TABLE} WHERE bucket = ?", (bucket,)
        ).fetchone()
    except Exception:  # noqa: BLE001 - a missing table means "no writes yet"
        return 0
    return int(row["writes"]) if row else 0


_USER_SOURCES = frozenset(
    {"USER_EXPLICIT", "USER_STATED", "USER_CORRECTION", "USER_IMPLICIT"}
)


def is_automatic(source_category: str | None) -> bool:
    """A user's own statement is never rate-limited: the user asking is the strongest evidence."""
    return str(source_category or "").upper() not in _USER_SOURCES


def check(
    store: Any,
    session_id: str,
    *,
    turn_id: str | None = None,
    at: datetime | None = None,
) -> Decision:
    """Would one more automatic write be allowed right now?"""
    _ensure(store)
    moment = at or _now()
    sid = str(session_id or "unknown")
    turn_cap, session_cap, hour_cap = caps()
    for scope, cap in (("turn", turn_cap), ("hour", hour_cap), ("session", session_cap)):
        bucket = _bucket(scope, sid, turn_id, moment)
        used = _count(store, bucket)
        if used >= cap:
            return Decision(
                allowed=False,
                scope=scope,
                used=used,
                cap=cap,
                reason=(
                    f"automatic write cap reached for this {scope} "
                    f"({used}/{cap}); the proposal stays staged for a later turn"
                    if scope != "session"
                    else f"automatic write cap reached for this session ({used}/{cap})"
                ),
            )
    # Report against the tightest remaining scope, so the log shows the real headroom.
    bucket = _bucket("turn", sid, turn_id, moment)
    return Decision(allowed=True, scope="turn", used=_count(store, bucket), cap=turn_cap)


def record(
    store: Any,
    session_id: str,
    *,
    turn_id: str | None = None,
    at: datetime | None = None,
    count: int = 1,
) -> None:
    """Count one (or more) automatic writes against every applicable bucket."""
    _ensure(store)
    moment = at or _now()
    sid = str(session_id or "unknown")
    stamp = moment.isoformat()
    with store.transaction() as conn:
        for scope in ("turn", "hour", "session"):
            bucket = _bucket(scope, sid, turn_id, moment)
            conn.execute(
                f"""INSERT INTO {TABLE} (bucket, session_id, writes, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(bucket) DO UPDATE SET
                        writes = writes + excluded.writes,
                        updated_at = excluded.updated_at""",
                (bucket, sid, int(count), stamp),
            )


def admit(
    store: Any,
    session_id: str,
    *,
    turn_id: str | None = None,
    at: datetime | None = None,
) -> Decision:
    """Check and, if allowed, count in one step — the call site's whole job."""
    _ensure(store)
    decision = check(store, session_id, turn_id=turn_id, at=at)
    if decision.allowed:
        record(store, session_id, turn_id=turn_id, at=at)
    return decision


def reset(store: Any, session_id: str) -> int:
    """Clear the counters for one session (a new session is a new budget anyway)."""
    _ensure(store)
    with store.transaction() as conn:
        cursor = conn.execute(
            f"DELETE FROM {TABLE} WHERE session_id = ?", (str(session_id),)
        )
        return int(cursor.rowcount or 0)


def prune(store: Any, *, days: int = 30, at: datetime | None = None) -> int:
    _ensure(store)
    cutoff = ((at or _now()) - timedelta(days=days)).isoformat()
    with store.transaction() as conn:
        cursor = conn.execute(f"DELETE FROM {TABLE} WHERE updated_at < ?", (cutoff,))
        return int(cursor.rowcount or 0)


def report(store: Any) -> dict[str, Any]:
    _ensure(store)
    rows = store._conn.execute(
        f"SELECT bucket, writes FROM {TABLE} ORDER BY writes DESC LIMIT 20"
    ).fetchall()
    turn_cap, session_cap, hour_cap = caps()
    return {
        "caps": {"turn": turn_cap, "session": session_cap, "hour": hour_cap},
        "buckets": len(rows),
        "top": [
            {
                "bucket": str(row["bucket"]),
                "writes": int(row["writes"]),
                "fingerprint": hashlib.sha256(str(row["bucket"]).encode()).hexdigest()[:8],
            }
            for row in rows
        ],
    }
