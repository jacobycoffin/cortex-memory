"""Promotion rules: what earns a fact a place in core memory (backlog item #122).

Core memory (#120) is expensive — ~460 tokens on every turn — so what goes in it cannot be
"whatever sorted highest at render time". Promotion is the decision, made once and recorded:

- **pin** — an explicit act by a human or by the agent on their behalf. Instant, always
  wins, and is why `pinned=1` alone has always been enough to reach the block.
- **use** — a fact that keeps informing answers. `used_count` is the strongest evidence the
  system has, because it comes from outcome logging rather than retrieval intent.
- **confirmation** — the same fact stated again later (#216). Repetition without use still
  says this matters.
- **explicit "remember this"** — the user asked. Never second-guessed and never requires a
  threshold.

Membership lives in an additive table created on first use, so no migration is needed:

    core_members(memory_id, promoted_at, promoted_by, reason, last_confirmed_at)

`last_confirmed_at` exists for the demotion side (#123): a core fact that goes quiet past its
window has to be able to leave, and that needs to know when it was last refreshed.

The rules are deliberately conservative. Promotion is cheap to do and expensive to get wrong:
a block full of half-relevant facts is worse than a smaller block of certain ones, because it
spends the model's attention on every turn for the life of the memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

TABLE = "core_members"
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    memory_id TEXT PRIMARY KEY,
    promoted_at TEXT NOT NULL,
    promoted_by TEXT NOT NULL DEFAULT 'use',
    reason TEXT NOT NULL DEFAULT '',
    last_confirmed_at TEXT
)
"""

# Measured against the live store: three uses is where a memory stops being an anecdote, and
# two confirmations is where a restatement stops being a coincidence.
USE_THRESHOLD = 3
CONFIRMATION_THRESHOLD = 2


# How long a core fact may go unconfirmed before it leaves (#123). A quarter is long enough
# that a genuinely important fact will be used or restated inside it, and short enough that a
# dead project's notes stop costing ~460 tokens a turn eventually rather than forever.
DEFAULT_WINDOW_DAYS = 90

HISTORY_TABLE = "core_history"
HISTORY_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} (
    memory_id TEXT NOT NULL,
    action TEXT NOT NULL,
    at TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT ''
)
"""


@dataclass(frozen=True)
class Promotion:
    memory_id: str
    promoted_by: str
    reason: str


def _ensure(store: Any) -> None:
    store._conn.execute(SCHEMA)
    store._conn.execute(HISTORY_SCHEMA)


def _history(conn: Any, memory_id: str, action: str, at: str, reason: str) -> None:
    conn.execute(
        f"INSERT INTO {HISTORY_TABLE}(memory_id, action, at, reason) VALUES(?,?,?,?)",
        (str(memory_id), action, at, reason[:200]),
    )


def window_days(default: int = DEFAULT_WINDOW_DAYS) -> int:
    try:
        return max(1, int(os.environ.get("CORTEX_CORE_WINDOW_DAYS", default)))
    except (TypeError, ValueError):
        return default


def members(store: Any) -> list[str]:
    """Memory ids currently in core, most recently promoted first."""
    _ensure(store)
    rows = store._conn.execute(
        f"SELECT memory_id FROM {TABLE} ORDER BY COALESCE(last_confirmed_at, promoted_at) DESC, memory_id"
    ).fetchall()
    return [str(row[0]) for row in rows]


def is_member(store: Any, memory_id: str) -> bool:
    _ensure(store)
    row = store._conn.execute(
        f"SELECT 1 FROM {TABLE} WHERE memory_id = ?", (str(memory_id),)
    ).fetchone()
    return row is not None


def promote(
    store: Any,
    memory_id: str,
    *,
    by: str = "use",
    reason: str = "",
    at: str | None = None,
) -> Promotion | None:
    """Record a memory as core. Idempotent: a second promotion changes nothing."""
    from .store import utc_now

    _ensure(store)
    stamp = at or utc_now()
    with store.transaction() as conn:
        cursor = conn.execute(
            f"INSERT OR IGNORE INTO {TABLE}(memory_id, promoted_at, promoted_by, reason, "
            "last_confirmed_at) VALUES(?,?,?,?,NULL)",
            (str(memory_id), stamp, by, reason[:200]),
        )
        if not cursor.rowcount:
            return None
        _history(conn, memory_id, "promote", stamp, f"{by}: {reason}")
    return Promotion(memory_id=str(memory_id), promoted_by=by, reason=reason)


def demote(store: Any, memory_id: str, *, reason: str = "") -> bool:
    """Remove a memory from core. Returns whether anything changed.

    Demotion never touches the memory itself — only its membership. A demoted fact is still
    recallable; it simply stops costing tokens on every turn.
    """
    from .store import utc_now

    _ensure(store)
    with store.transaction() as conn:
        cursor = conn.execute(f"DELETE FROM {TABLE} WHERE memory_id = ?", (str(memory_id),))
        if cursor.rowcount:
            _history(conn, memory_id, "demote", utc_now(), reason or "manual demotion")
        return bool(cursor.rowcount)


def touch_confirmations(store: Any, memory_ids: Iterable[str], *, at: str | None = None) -> int:
    """Refresh `last_confirmed_at` for core members that were confirmed again."""
    from .store import utc_now

    ids = [str(i) for i in memory_ids]
    if not ids:
        return 0
    _ensure(store)
    stamp = at or utc_now()
    placeholders = ",".join("?" for _ in ids)
    with store.transaction() as conn:
        cursor = conn.execute(
            f"UPDATE {TABLE} SET last_confirmed_at = ? WHERE memory_id IN ({placeholders})",
            (stamp, *ids),
        )
        return cursor.rowcount or 0


def candidates(store: Any) -> list[Promotion]:
    """Memories that meet a rule and are not members yet — the promotion pass's input."""
    _ensure(store)
    rows = store._conn.execute(
        f"""
        SELECT m.id, m.pinned, m.used_count, m.confirmed_count, m.importance
        FROM memories m
        LEFT JOIN {TABLE} c ON c.memory_id = m.id
        WHERE m.state = 'active' AND c.memory_id IS NULL
          AND (m.pinned = 1 OR m.used_count >= ? OR m.confirmed_count >= ?)
        ORDER BY m.pinned DESC, m.used_count DESC, m.confirmed_count DESC, m.id
        """,
        (USE_THRESHOLD, CONFIRMATION_THRESHOLD),
    ).fetchall()
    out: list[Promotion] = []
    for row in rows:
        if int(row["pinned"] or 0):
            out.append(Promotion(str(row["id"]), "pin", "pinned by a human or agent"))
        elif int(row["used_count"] or 0) >= USE_THRESHOLD:
            out.append(
                Promotion(str(row["id"]), "use", f"used {int(row['used_count'])} times")
            )
        else:
            out.append(
                Promotion(
                    str(row["id"]),
                    "confirmation",
                    f"confirmed {int(row['confirmed_count'])} times",
                )
            )
    return out


def run(store: Any, *, limit: int = 200) -> list[Promotion]:
    """Apply the rules, promoting at most `limit` memories. Safe to run repeatedly."""
    promoted: list[Promotion] = []
    for candidate in candidates(store)[:limit]:
        result = promote(
            store, candidate.memory_id, by=candidate.promoted_by, reason=candidate.reason
        )
        if result is not None:
            promoted.append(result)
    return promoted


def stats(store: Any) -> dict[str, int]:
    _ensure(store)
    total = int(store._conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0])
    by_kind = {
        str(row[0]): int(row[1])
        for row in store._conn.execute(
            f"SELECT promoted_by, COUNT(*) FROM {TABLE} GROUP BY promoted_by"
        ).fetchall()
    }
    return {"members": total, **{f"by_{k}": v for k, v in by_kind.items()}}


def stale_members(
    store: Any,
    *,
    days: int | None = None,
    now: datetime | None = None,
) -> list[tuple[str, str]]:
    """Core members that should leave, with the reason: `(memory_id, reason)`.

    Three ways out, in order of certainty:

    - the memory is **archived or superseded** — it is not current any more, so it has no
      business in an always-present block whatever its window says;
    - it has been **pinned** — a human decision outranks a timer, so pinned members are never
      demoted here (unpinning through `cortex_memory` removes the pin and the next pass
      handles it);
    - it has gone **quiet past the window** — no confirmation, no use, no re-promotion.

    "Quiet" is measured from the most recent of `last_confirmed_at`, `promoted_at` and the
    memory's own `last_used_at`: a fact that keeps being used is not stale even if nobody has
    restated it in so many words.
    """
    _ensure(store)
    horizon_days = window_days() if days is None else max(1, int(days))
    reference = now or datetime.now(timezone.utc)
    cutoff = (reference - timedelta(days=horizon_days)).isoformat()
    rows = store._conn.execute(
        f"""
        SELECT c.memory_id,
               c.promoted_at,
               c.last_confirmed_at,
               m.pinned,
               m.state,
               COALESCE(m.last_used_at, '') AS last_used_at
        FROM {TABLE} c
        LEFT JOIN memories m ON m.id = c.memory_id
        """
    ).fetchall()
    out: list[tuple[str, str]] = []
    for row in rows:
        memory_id = str(row["memory_id"])
        if row["state"] is None:
            out.append((memory_id, "memory no longer exists"))
            continue
        if str(row["state"]) in ("archived", "forgotten", "quarantined"):
            out.append((memory_id, f"memory is {row['state']}"))
            continue
        if int(row["pinned"] or 0):
            continue
        recent = max(
            str(row["promoted_at"] or ""),
            str(row["last_confirmed_at"] or ""),
            str(row["last_used_at"] or ""),
        )
        if recent and recent < cutoff:
            out.append((memory_id, f"quiet since {recent[:10]} (window {horizon_days}d)"))
    return out


def demote_stale(
    store: Any,
    *,
    days: int | None = None,
    now: datetime | None = None,
    apply: bool = True,
) -> list[tuple[str, str]]:
    """Demote every stale core member. `apply=False` reports without changing anything."""
    stale = stale_members(store, days=days, now=now)
    if not apply:
        return stale
    for memory_id, reason in stale:
        demote(store, memory_id, reason=reason)
    return stale


def history(store: Any, memory_id: str | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
    """Promotion and demotion events, newest first — the raw material for a diff view (#124)."""
    _ensure(store)
    if memory_id:
        rows = store._conn.execute(
            f"SELECT memory_id, action, at, reason FROM {HISTORY_TABLE} "
            "WHERE memory_id = ? ORDER BY at DESC LIMIT ?",
            (str(memory_id), limit),
        ).fetchall()
    else:
        rows = store._conn.execute(
            f"SELECT memory_id, action, at, reason FROM {HISTORY_TABLE} ORDER BY at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
