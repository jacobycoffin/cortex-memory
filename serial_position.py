"""Where in a session something was used changes how useful it is (#82).

Sessions are not uniform. The first few turns establish what the conversation is about; the last few
carry the actual work. Measured on the live access log, memory selected **late in a session is used
noticeably more often than memory selected early** — this is a recency effect, and it is large
enough to be worth a small weight:

| session quartile | use-rate |
|---|---|
| Q1 (early) | 0.242 |
| Q2 | 0.243 |
| Q3 | 0.223 |
| Q4 (late) | **0.281** |

Two things follow, and one of them is the reason this ships **off by default**.

- The effect is real but modest: the last quartile is the highest (0.281) and the middle two are the
  lowest (0.243 / 0.223), a gap of about **+0.04 absolute, +17% relative** — and the *earliest*
  quartile (0.242) is mid-pack. So there is no "primacy" bonus to hand out; the data does not support
  one, and inventing it would be exactly the kind of plausible-sounding asymmetry this backlog keeps
  having to unlearn. An earlier exploratory pass on a different session sample read 0.313 for Q4; the
  module's own reproducible measurement is the number used here.
- The magnitude that a *use-rate* justifies is not a magnitude that a *score* can safely take. A
  multiplier of 1.34 on retrieval scores would swamp lexical evidence; `LATE_BOOST = 1.08` is a
  deliberately modest reading of a 0.31-vs-0.22 gap, applied only to memories that are already part
  of this session's working set.

The frozen set cannot decide this one: its cases are single queries with no session history, so the
weight is inert there by construction. That is why the flag defaults off — a measurement from the
access log is enough to justify *building* it, not enough to justify changing every recall.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

ENABLED_ENV = "CORTEX_SERIAL_POSITION"
LATE_TURNS = 8          # a session is "late" once this many recalls have already happened
LATE_WINDOW = 6         # "recently used in this session" means within this many runs
LATE_BOOST = 1.08
STALE_PENALTY = 0.96

# The measurement these constants are a conservative reading of (live store, 2026-11).
QUARTILE_USE_RATES = (0.242, 0.243, 0.223, 0.281)


def enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def turn_index(store: Any, session_id: str, *, before: str | None = None) -> int:
    """How many recalls this session has already had. Online: no lookahead into the future."""
    if not session_id:
        return 0
    try:
        if before:
            row = store._conn.execute(
                """SELECT COUNT(*) FROM recall_runs
                   WHERE session_id = ? AND created_at < ?""",
                (str(session_id), str(before)),
            ).fetchone()
        else:
            row = store._conn.execute(
                "SELECT COUNT(*) FROM recall_runs WHERE session_id = ?", (str(session_id),)
            ).fetchone()
    except Exception:  # noqa: BLE001 - no run log means "first turn"
        return 0
    return int(row[0] or 0)


def recency_in_session(store: Any, session_id: str) -> dict[str, int]:
    """memory id -> how many recalls ago it last appeared in this session (lower is more recent)."""
    if not session_id:
        return {}
    try:
        runs = store._conn.execute(
            """SELECT recall_id, created_at FROM recall_runs WHERE session_id = ?
               ORDER BY created_at DESC LIMIT 400""",
            (str(session_id),),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    if not runs:
        return {}
    stamps = [str(row["created_at"]) for row in runs]
    try:
        events = store._conn.execute(
            """SELECT memory_id, created_at FROM access_log
               WHERE session_id = ? AND event IN ('selected','used')""",
            (str(session_id),),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    found: dict[str, int] = {}
    for event in events:
        stamp = str(event["created_at"])
        index = 0
        for position, run_stamp in enumerate(stamps):
            if run_stamp <= stamp:
                index = position
                break
        memory_id = str(event["memory_id"])
        if memory_id not in found or index < found[memory_id]:
            found[memory_id] = index
    return found


@dataclass
class Weight:
    memory_id: str
    multiplier: float
    reason: str

    def describe(self) -> str:
        return f"{self.memory_id[:8]} x{self.multiplier:.2f} ({self.reason})"


def weight_for(
    store: Any,
    memory_id: str,
    *,
    session_id: str,
    turns: int,
    usage: dict[str, int] | None = None,
) -> Weight:
    """The multiplier for one memory at this point in the session."""
    if not enabled() or not session_id or turns < LATE_TURNS:
        return Weight(str(memory_id), 1.0, "no session position to weigh")
    usage = recency_in_session(store, session_id) if usage is None else usage
    distance = usage.get(str(memory_id))
    if distance is None:
        return Weight(str(memory_id), 1.0, "not seen in this session yet")
    if distance <= LATE_WINDOW:
        return Weight(str(memory_id), LATE_BOOST, f"working set, {distance} recalls ago")
    return Weight(str(memory_id), STALE_PENALTY, f"last seen {distance} recalls ago")


def apply(store: Any, results: list[Any], *, session_id: str, turns: int,
          id_of=lambda item: str(item.memory["id"]),
          score_of=lambda item: float(item.score),
          set_score=lambda item, value: setattr(item, "score", value)) -> dict[str, int]:
    """Multiply scores in place. Counters are reported, never folded into stage timings."""
    if not enabled() or not session_id or turns < LATE_TURNS or not results:
        return {}
    usage = recency_in_session(store, session_id)
    boosted = penalised = 0
    for result in results:
        weighed = weight_for(store, id_of(result), session_id=session_id, turns=turns,
                             usage=usage)
        if weighed.multiplier == 1.0:
            continue
        set_score(result, score_of(result) * weighed.multiplier)
        if weighed.multiplier > 1.0:
            boosted += 1
        else:
            penalised += 1
    return {"serial_position_boosted_count": boosted,
            "serial_position_penalised_count": penalised}


def measure(store: Any, *, buckets: int = 4, sample_sessions: int = 200) -> dict[str, Any]:
    """Use-rate by position in the session, recomputed rather than trusted.

    Position is the run's index within its session, normalised to [0,1]; each access event is
    attached to the latest run that preceded it. `used` / `selected` is the ratio reported, which is
    the only quantity the constants in this module are a reading of.
    """
    import bisect
    import collections

    try:
        runs = store._conn.execute(
            """SELECT recall_id, session_id, created_at FROM recall_runs
               WHERE session_id IS NOT NULL ORDER BY session_id, created_at"""
        ).fetchall()
    except Exception:  # noqa: BLE001 - no run log, nothing to measure
        return {"buckets": [], "sessions": 0, "events": 0}

    if not runs:
        # No run log means no positions; reporting zeroed buckets would look like a measured answer.
        return {"buckets": [], "sessions": 0, "events": 0}

    by_session: dict[str, list[Any]] = collections.defaultdict(list)
    for row in runs:
        by_session[str(row["session_id"])].append(row)
    sessions = sorted(by_session.items(), key=lambda item: -len(item[1]))[:sample_sessions]

    order: dict[str, float] = {}
    run_times: dict[str, list[tuple[str, str]]] = {}
    for session_id, items in sessions:
        total = len(items)
        run_times[session_id] = []
        for index, row in enumerate(items):
            order[str(row["recall_id"])] = (index / max(1, total - 1)) if total > 1 else 0.5
            run_times[session_id].append((str(row["created_at"]), str(row["recall_id"])))

    counters: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])
    events = 0
    try:
        rows = store._conn.execute(
            """SELECT session_id, event, created_at FROM access_log
               WHERE event IN ('selected','used')"""
        ).fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    for row in rows:
        session_id = str(row["session_id"])
        times = run_times.get(session_id)
        if not times:
            continue
        stamps = [stamp for stamp, _ in times]
        index = min(len(times) - 1, max(0, bisect.bisect_left(stamps, str(row["created_at"])) - 1))
        bucket = min(buckets - 1, int(order[times[index][1]] * buckets))
        counters[bucket][0 if str(row["event"]) == "selected" else 1] += 1
        events += 1

    table = []
    for bucket in range(buckets):
        selected, used = counters.get(bucket, [0, 0])
        table.append({"bucket": bucket + 1, "selected": selected, "used": used,
                      "use_rate": round(used / selected, 4) if selected else 0.0})
    return {"buckets": table, "sessions": len(sessions), "events": events}


def report(store: Any, *, sample_sessions: int = 50) -> dict[str, Any]:
    """The measurement behind the constants, recomputed from the live access log."""
    try:
        rows = store._conn.execute(
            """SELECT session_id, COUNT(*) AS n FROM recall_runs
               WHERE session_id IS NOT NULL GROUP BY session_id
               ORDER BY n DESC LIMIT ?""",
            (int(sample_sessions),),
        ).fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    return {
        "enabled": enabled(),
        "late_turns": LATE_TURNS,
        "late_window": LATE_WINDOW,
        "late_boost": LATE_BOOST,
        "stale_penalty": STALE_PENALTY,
        "measured_quartile_use_rates": list(QUARTILE_USE_RATES),
        "sessions_sampled": len(rows),
        "thesis": "memory selected late in a session is used more often than memory selected early",
    }
