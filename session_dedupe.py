"""One chatty session must never manufacture an edge on its own (#204).

Co-occurrence is the cheapest evidence in the system, and it is the easiest to fake by accident: a
single long conversation mentions the same two things forty times. Counted as "uses", that pair is
the strongest association in the store; counted as **sessions**, it is one afternoon.

`co_retrieval` already requires three distinct sessions before it materialises anything, which is
the right shape. The problem is that the rule lived only inside that one caller — any future writer
could count turns instead of sessions and nobody would notice. So the rule becomes a function:

- `sessions()` asks the observation table for the **distinct** session ids behind a pair;
- `admit()` answers yes or no, with the count, and refuses anything below the bar;
- `audit()` reads the whole store and reports the pairs that are currently one-session artifacts.

The audit is the part that matters on a live store. A guard only protects what it is wired into; the
audit finds what already slipped in before the guard existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

OBSERVATIONS = "co_retrieval_observations"
MIN_SESSIONS = 2
STRICT_SESSIONS = 3


def _table_exists(store: Any) -> bool:
    try:
        row = store._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
            (OBSERVATIONS,),
        ).fetchone()
    except Exception:  # noqa: BLE001 - no connection, no table
        return False
    return bool(row and int(row[0]))


def sessions(store: Any, left_id: str, right_id: str) -> set[str]:
    """Distinct sessions that observed this pair together. Order-insensitive."""
    if not _table_exists(store):
        return set()
    try:
        rows = store._conn.execute(
            f"""SELECT DISTINCT session_id FROM {OBSERVATIONS}
                WHERE (left_id = ? AND right_id = ?) OR (left_id = ? AND right_id = ?)""",
            (str(left_id), str(right_id), str(right_id), str(left_id)),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return set()
    return {str(row["session_id"]) for row in rows if row["session_id"]}


@dataclass
class Admission:
    allowed: bool
    sessions: int
    reason: str

    def describe(self) -> str:
        return f"{'admitted' if self.allowed else 'refused'}: {self.reason}"


def admit(store: Any, left_id: str, right_id: str, *, minimum: int = MIN_SESSIONS) -> Admission:
    """Is this pair allowed to become an edge? Turns in one session never are."""
    found = sessions(store, left_id, right_id)
    if len(found) >= minimum:
        return Admission(True, len(found), f"{len(found)} distinct sessions")
    return Admission(
        False, len(found),
        f"{len(found)} distinct session(s), {minimum} required — one session cannot make a link",
    )


def audit(store: Any, *, limit: int = 50) -> dict[str, Any]:
    """How much of the recorded co-occurrence rests on a single session."""
    if not _table_exists(store):
        return {"pairs": 0, "single_session": 0, "multi_session": 0, "examples": []}
    try:
        rows = store._conn.execute(
            f"""SELECT left_id, right_id, COUNT(DISTINCT session_id) AS n,
                       COUNT(*) AS turns
                FROM {OBSERVATIONS}
                GROUP BY left_id, right_id
                ORDER BY n ASC, turns DESC"""
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {"pairs": 0, "single_session": 0, "multi_session": 0, "examples": []}
    single = [row for row in rows if int(row["n"]) < MIN_SESSIONS]
    examples = [
        {"left": str(row["left_id"])[:8], "right": str(row["right_id"])[:8],
         "sessions": int(row["n"]), "turns": int(row["turns"])}
        for row in single[:limit]
    ]
    return {
        "pairs": len(rows),
        "single_session": len(single),
        "multi_session": len(rows) - len(single),
        "busiest_single": examples,
    }


def weight_cap_for(session_count: int, *, cap: float) -> float:
    """The weight a pair may hold given how many sessions vouch for it.

    Zero below the bar, full cap at it: a pair with two sessions is as trustworthy as one with ten,
    because the second session is the thing that distinguishes evidence from repetition.
    """
    if int(session_count) < MIN_SESSIONS:
        return 0.0
    return float(cap)


def report(store: Any) -> dict[str, Any]:
    described = audit(store)
    return {
        "min_sessions": MIN_SESSIONS,
        "strict_sessions": STRICT_SESSIONS,
        "observations_table": _table_exists(store),
        **described,
    }
