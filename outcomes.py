"""A memory's track record: whether the tasks it informed actually worked (#71).

`tool_links` records that a memory was in context for a call. That still leaves the question a person
actually asks about a procedure: *has this ever worked?* This module keeps the answer per memory — a
small, boring tally of tasks it informed that succeeded and tasks it informed that failed.

Three rules keep the tally honest:

- **an outcome is a verdict, not a vibe.** `success`, `failure` or `neutral`, recorded against a task
  reference that already exists (a run, a turn, a job). No inferred mood, no "probably worked".
- **a failure is not an accusation.** One failure says nothing; the flag for a memory that keeps
  failing needs several, and it is a flag — reported, never a silent deletion.
- **the weighting is measurement-gated.** `trust()` returns a multiplier, and `ranking_hint()` only
  applies it when `CORTEX_OUTCOME_WEIGHT` is set. A store that re-ranks on a handful of outcomes is a
  store optimising noise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

TABLE = "memory_outcomes"
ENV = "CORTEX_OUTCOME_WEIGHT"
OUTCOMES = ("success", "failure", "neutral")

# A memory needs both a few failures and a bad ratio before it is called harmful.
MIN_JUDGED = 3
HARMFUL_SHARE = 0.5
# The multiplier is deliberately timid: the difference between 0.9 and 1.1 cannot reorder a good
# result past an irrelevant one, it only breaks near-ties.
FLOOR = 0.9
CEILING = 1.1


def _now() -> datetime:
    return datetime.now(timezone.utc)


def enabled(environ: Any = None) -> bool:
    env = environ if environ is not None else os.environ
    return str(env.get(ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    memory_id TEXT NOT NULL,
                    task_ref TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (memory_id, task_ref)
                )"""
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_memory_idx ON {TABLE} (memory_id)")


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 - a report that cannot read is a report of nothing
        return []


def _memory_exists(store: Any, memory_id: str) -> bool:
    return bool(
        _rows(store, "SELECT 1 AS found FROM memories WHERE id = ? LIMIT 1", (str(memory_id),))
    )


@dataclass
class Record:
    memory_id: str
    successes: int = 0
    failures: int = 0
    neutral: int = 0
    last_outcome: str = ""
    last_at: str = ""

    @property
    def judged(self) -> int:
        return self.successes + self.failures

    @property
    def success_share(self) -> float:
        return (self.successes / self.judged) if self.judged else 0.0

    @property
    def harmful(self) -> bool:
        return self.judged >= MIN_JUDGED and self.success_share < HARMFUL_SHARE

    @property
    def helpful(self) -> bool:
        return self.judged >= MIN_JUDGED and self.success_share >= 0.75

    def trust(self) -> float:
        """A multiplier in [0.9, 1.1]; 1.0 until there is a record worth noticing."""
        if not self.judged:
            return 1.0
        return round(FLOOR + (CEILING - FLOOR) * self.success_share, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "successes": self.successes,
            "failures": self.failures,
            "neutral": self.neutral,
            "judged": self.judged,
            "success_share": round(self.success_share, 4),
            "harmful": self.harmful,
            "helpful": self.helpful,
            "trust": self.trust(),
            "last_outcome": self.last_outcome,
        }

    def describe(self) -> str:
        if not self.judged:
            return f"{self.memory_id[:8]}: no track record"
        verdict = "harmful" if self.harmful else ("helpful" if self.helpful else "mixed")
        return (f"{self.memory_id[:8]}: {self.successes} success / {self.failures} failure "
                f"({self.success_share:.0%}) — {verdict}")


def record(store: Any, memory_id: str, task_ref: str, outcome: str, *,
           detail: str = "", at: datetime | None = None) -> dict[str, Any]:
    """One verdict about one task a memory informed. Idempotent per (memory, task)."""
    resolved = str(outcome or "").strip().lower()
    if resolved not in OUTCOMES:
        return {"recorded": False, "reason": f"outcome must be one of {', '.join(OUTCOMES)}"}
    memory = str(memory_id or "").strip()
    task = str(task_ref or "").strip()
    if not memory or not task:
        return {"recorded": False, "reason": "an outcome needs a memory and a task reference"}
    if not _memory_exists(store, memory):
        return {"recorded": False, "reason": f"unknown memory {memory[:12]}"}
    _ensure(store)
    moment = (at or _now()).isoformat()
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE} (memory_id, task_ref, outcome, detail, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, task_ref) DO UPDATE SET
                    outcome = excluded.outcome,
                    detail = excluded.detail,
                    created_at = excluded.created_at""",
            (memory, task, resolved, str(detail)[:300], moment),
        )
    return {"recorded": True, "memory_id": memory, "task_ref": task, "outcome": resolved}


def record_many(store: Any, memory_ids: Iterable[str], task_ref: str, outcome: str, *,
                detail: str = "", at: datetime | None = None) -> dict[str, Any]:
    """A task used several memories; the verdict applies to each of them."""
    recorded = 0
    refused: list[str] = []
    for memory_id in memory_ids:
        outcome_result = record(store, memory_id, task_ref, outcome, detail=detail, at=at)
        if outcome_result.get("recorded"):
            recorded += 1
        else:
            refused.append(str(outcome_result.get("reason") or ""))
    return {"recorded": recorded, "refused": refused}


def track_record(store: Any, memory_id: str) -> Record:
    _ensure(store)
    rows = _rows(
        store,
        f"""SELECT outcome, COUNT(*) AS count, MAX(created_at) AS last_at
            FROM {TABLE} WHERE memory_id = ? GROUP BY outcome""",
        (str(memory_id),),
    )
    summary = Record(memory_id=str(memory_id))
    for row in rows:
        outcome = str(row["outcome"])
        count = int(row["count"] or 0)
        if outcome == "success":
            summary.successes = count
        elif outcome == "failure":
            summary.failures = count
        else:
            summary.neutral = count
        if str(row["last_at"] or "") > summary.last_at:
            summary.last_at = str(row["last_at"])
            summary.last_outcome = outcome
    return summary


def ranking_hint(store: Any, memory_ids: Iterable[str], *,
                 environ: Any = None) -> dict[str, float]:
    """Per-memory multipliers. Empty unless the weighting is switched on."""
    if not enabled(environ):
        return {}
    return {str(memory_id): track_record(store, str(memory_id)).trust()
            for memory_id in memory_ids}


def harmful(store: Any, *, limit: int = 50) -> list[Record]:
    """Memories with several judged outcomes and a bad ratio. Reported, never removed."""
    _ensure(store)
    rows = _rows(
        store,
        f"""SELECT memory_id FROM {TABLE} GROUP BY memory_id
            HAVING SUM(outcome IN ('success','failure')) >= ? LIMIT ?""",
        (MIN_JUDGED, int(limit)),
    )
    records = [track_record(store, str(row["memory_id"])) for row in rows]
    return [item for item in records if item.harmful]


def report(store: Any) -> dict[str, Any]:
    _ensure(store)
    totals = _rows(
        store,
        f"""SELECT COUNT(*) AS outcomes, COUNT(DISTINCT memory_id) AS memories,
                   SUM(outcome = 'success') AS successes, SUM(outcome = 'failure') AS failures
            FROM {TABLE}""",
    )
    row = totals[0] if totals else {}
    return {
        "outcomes": int(row.get("outcomes") or 0),
        "memories": int(row.get("memories") or 0),
        "successes": int(row.get("successes") or 0),
        "failures": int(row.get("failures") or 0),
        "weighting": enabled(),
        "note": "a failure is a data point, not an accusation: the harmful flag needs several",
    }
