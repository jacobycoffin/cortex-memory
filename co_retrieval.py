"""Co-retrieval edges, gated on agreement across sessions (#67).

Two memories surfacing for the same query is weak evidence that they are related: a broad query
retrieves a dozen things, and any two of them co-occur by accident. So the gate is agreement, not
co-occurrence:

- a pair is *observed* whenever both memories are in one turn's retrieved set;
- an **edge is created only after 3 distinct sessions** have observed that pair. Three separate
  conversations worth of evidence is a different claim from three retrievals inside one.
- weight is `min(cap, log1p(sessions))`, decayed by age — so the tenth session adds little and an
  association nobody has confirmed for months fades instead of calcifying.

`log1p` and the cap exist because similarity graphs fail in one direction: unbounded reinforcement
turns a single busy week into a permanent hub, and every later query routes through it.

Nothing here writes to a memory. Observations are their own table so a pair that never reaches the
gate costs three rows and no edge, which is the cheap half of the trade.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

RELATION = "co_retrieved"
SESSIONS_REQUIRED = 3
WEIGHT_CAP = 0.45
HALF_LIFE_DAYS = 90.0
_OBSERVATIONS = "co_retrieval_observations"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {_OBSERVATIONS} (
                    left_id TEXT NOT NULL,
                    right_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    turn_id TEXT,
                    observed_at TEXT NOT NULL,
                    PRIMARY KEY (left_id, right_id, session_id)
                )"""
        )


def _pair(left: str, right: str) -> tuple[str, str]:
    first, second = str(left), str(right)
    return (first, second) if first <= second else (second, first)


def observations(store: Any, *, limit: int = 200) -> list[dict[str, Any]]:
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT left_id, right_id, COUNT(DISTINCT session_id) AS sessions,
                   MAX(observed_at) AS last_seen
            FROM {_OBSERVATIONS} GROUP BY left_id, right_id
            ORDER BY sessions DESC, last_seen DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        {
            "left_id": str(row["left_id"]),
            "right_id": str(row["right_id"]),
            "sessions": int(row["sessions"]),
            "last_seen": str(row["last_seen"]),
        }
        for row in rows
    ]


def observe(
    store: Any,
    memory_ids: Iterable[str],
    *,
    session_id: str,
    turn_id: str | None = None,
    at: datetime | None = None,
) -> int:
    """Record that these memories were retrieved together in one session. Returns rows written."""
    _ensure(store)
    ids = sorted({str(memory_id) for memory_id in memory_ids if str(memory_id).strip()})
    if len(ids) < 2:
        return 0
    stamp = (at or _utc_now()).isoformat()
    written = 0
    with store.transaction() as conn:
        for index, left in enumerate(ids):
            for right in ids[index + 1:]:
                first, second = _pair(left, right)
                cursor = conn.execute(
                    f"""INSERT OR IGNORE INTO {_OBSERVATIONS}
                        (left_id, right_id, session_id, turn_id, observed_at)
                        VALUES (?, ?, ?, ?, ?)""",
                    (first, second, str(session_id), turn_id, stamp),
                )
                written += int(cursor.rowcount or 0)
    return written


def _age_days(stamp: str, now: datetime) -> float:
    try:
        seen = datetime.fromisoformat(str(stamp))
    except ValueError:
        return float(HALF_LIFE_DAYS)
    if not seen.tzinfo:
        seen = seen.replace(tzinfo=timezone.utc)
    return max(0.0, (now - seen).total_seconds() / 86400.0)


def weight_for(sessions: int, *, age_days: float = 0.0) -> float:
    """Capped, logarithmic, decayed. The tenth session adds far less than the third."""
    if sessions < SESSIONS_REQUIRED:
        return 0.0
    base = min(WEIGHT_CAP, 0.15 * math.log1p(sessions))
    decay = 0.5 ** (age_days / HALF_LIFE_DAYS)
    return round(base * decay, 4)


@dataclass
class Gate:
    pair: tuple[str, str]
    sessions: int
    weight: float

    @property
    def eligible(self) -> bool:
        return self.sessions >= SESSIONS_REQUIRED and self.weight > 0.0


def plan(store: Any, *, at: datetime | None = None, limit: int = 500) -> list[Gate]:
    """Which pairs would produce an edge right now. Read-only."""
    now = at or _utc_now()
    gates: list[Gate] = []
    for row in observations(store, limit=limit):
        sessions = int(row["sessions"])
        weight = weight_for(sessions, age_days=_age_days(row["last_seen"], now))
        gates.append(Gate(pair=(row["left_id"], row["right_id"]), sessions=sessions,
                          weight=weight))
    return gates


def materialize(store: Any, *, at: datetime | None = None, limit: int = 500) -> dict[str, int]:
    """Create or reinforce edges for pairs that have earned them. Idempotent."""
    _ensure(store)
    created = 0
    reinforced = 0
    skipped = 0
    for gate in plan(store, at=at, limit=limit):
        if not gate.eligible:
            skipped += 1
            continue
        left, right = gate.pair
        try:
            existed = bool(
                store._conn.execute(
                    "SELECT 1 FROM edges WHERE src_id = ? AND dst_id = ? AND relation = ?",
                    (left, right, RELATION),
                ).fetchone()
            )
            store.add_edge(
                left, right, RELATION,
                weight=gate.weight,
                evidence_type="co_retrieval",
                evidence_key=f"co_retrieval:{left}:{right}",
                explanation=(
                    f"{gate.sessions} independent sessions retrieved these together; "
                    f"weight {gate.weight} (cap {WEIGHT_CAP}, log1p, "
                    f"half-life {HALF_LIFE_DAYS:.0f}d)"
                ),
                metadata={"sessions": gate.sessions},
            )
            if existed:
                reinforced += 1
            else:
                created += 1
        except Exception:  # noqa: BLE001 - one pair must not abort the batch
            skipped += 1
    return {"created": created, "reinforced": reinforced, "skipped": skipped}


def report(store: Any, *, at: datetime | None = None) -> dict[str, Any]:
    gates = plan(store, at=at)
    eligible = [gate for gate in gates if gate.eligible]
    return {
        "pairs_observed": len(gates),
        "eligible": len(eligible),
        "sessions_required": SESSIONS_REQUIRED,
        "weight_cap": WEIGHT_CAP,
        "half_life_days": HALF_LIFE_DAYS,
        "top": [
            {
                "pair": f"{gate.pair[0][:8]} <> {gate.pair[1][:8]}",
                "sessions": gate.sessions,
                "weight": gate.weight,
            }
            for gate in sorted(eligible, key=lambda gate: -gate.sessions)[:10]
        ],
    }


def prune_stale(store: Any, *, days: int = 365, at: datetime | None = None) -> int:
    """Drop observations nobody has re-confirmed in `days`, so the table cannot grow forever."""
    _ensure(store)
    cutoff = ((at or _utc_now()) - timedelta(days=days)).isoformat()
    with store.transaction() as conn:
        cursor = conn.execute(
            f"DELETE FROM {_OBSERVATIONS} WHERE observed_at < ?", (cutoff,)
        )
        return int(cursor.rowcount or 0)
