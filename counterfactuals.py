"""What a memory was chosen over (#77).

A retrieval decision is a choice, and only half of a choice is recorded. The store keeps what was
selected; the alternatives it beat disappear into the trace. That makes two useful questions
unanswerable after the fact:

- *why is this memory here?* — it may be here because it narrowly beat something better;
- *what did we lose?* — a memory that keeps losing to the same competitor is evidence the competitor
  should have been the one stored.

This keeps the loser. For each recorded trace it pairs every selected memory with the **nearest**
rejected candidates — nearest by the rank the trace recorded, not by a re-score, so the pairing does
not depend on a model that has since changed — and stores the pair with the gap that decided it.

Three boundaries:

- **a counterfactual is evidence of decisiveness, not of correctness.** Nothing here says the chosen
  memory was right, only that it was chosen over specific others.
- **the nearest few, not all.** A trace rejects dozens of candidates; pairing every one of them would
  bury the near-misses (the only interesting ones) under the noise.
- **no re-scoring.** Ranks and scores are copied from the trace, because recomputing them under a
  current ranker would answer a question nobody asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

TABLE = "counterfactual_links"
DEFAULT_ALTERNATIVES = 3


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 - a report that cannot read reports nothing
        return []


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    trace_id TEXT NOT NULL,
                    chosen_id TEXT NOT NULL,
                    rejected_id TEXT NOT NULL,
                    chosen_rank INTEGER NOT NULL DEFAULT 0,
                    rejected_rank INTEGER NOT NULL DEFAULT 0,
                    score_gap REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (trace_id, chosen_id, rejected_id)
                )"""
        )


@dataclass
class Pair:
    chosen_id: str
    rejected_id: str
    chosen_rank: int = 0
    rejected_rank: int = 0
    score_gap: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "chosen_id": self.chosen_id,
            "rejected_id": self.rejected_id,
            "chosen_rank": self.chosen_rank,
            "rejected_rank": self.rejected_rank,
            "score_gap": round(self.score_gap, 6),
        }


def _index(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates = [dict(item) for item in (trace.get("candidate_memories") or [])
                  if isinstance(item, dict) and item.get("memory_id")]
    return candidates, {str(item["memory_id"]): item for item in candidates}


def pairs_for(trace: dict[str, Any], *, alternatives: int = DEFAULT_ALTERNATIVES) -> list[Pair]:
    """The selected memories paired with the nearest rejected ones in the same trace."""
    _candidates, by_id = _index(trace)
    chosen_ids = [str(item) for item in (trace.get("selected_memory_ids") or [])]
    rejected_ids = [str(item) for item in (trace.get("rejected_memory_ids") or [])]
    if not chosen_ids or not rejected_ids:
        return []
    # Nearest by recorded rank: a rejection that scored far below says nothing interesting.
    ordered_rejected = sorted(
        (item for item in rejected_ids if item in by_id),
        key=lambda item: int(by_id[item].get("rank") or 0),
    )[: max(1, int(alternatives))]
    found: list[Pair] = []
    for chosen in chosen_ids:
        chosen_row = by_id.get(chosen)
        if chosen_row is None:
            continue
        for rejected in ordered_rejected:
            if rejected == chosen:
                continue
            rejected_row = by_id[rejected]
            found.append(Pair(
                chosen_id=chosen,
                rejected_id=rejected,
                chosen_rank=int(chosen_row.get("rank") or 0),
                rejected_rank=int(rejected_row.get("rank") or 0),
                score_gap=float(chosen_row.get("score") or 0.0)
                - float(rejected_row.get("score") or 0.0),
            ))
    return found


@dataclass
class Recorded:
    traces: int = 0
    paired: int = 0
    skipped_no_rejections: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"traces": self.traces, "paired": self.paired,
                "skipped_no_rejections": self.skipped_no_rejections,
                "sentence": self.describe()}

    def describe(self) -> str:
        return (f"{self.paired} pair(s) kept from {self.traces} trace(s); "
                f"{self.skipped_no_rejections} trace(s) recorded no rejections")


def record(store: Any, *, limit: int = 200, alternatives: int = DEFAULT_ALTERNATIVES,
           at: datetime | None = None) -> Recorded:
    """Persist the nearest rejected alternative for every selected memory, idempotently."""
    moment = (at or _now()).isoformat()
    traces = []
    try:
        traces = store.memory_traces(limit=int(limit))
    except Exception:  # noqa: BLE001
        traces = []
    _ensure(store)
    found = Recorded(traces=len(traces))
    with store.transaction() as conn:
        for trace in traces:
            pairs = pairs_for(trace, alternatives=alternatives)
            if not pairs:
                found.skipped_no_rejections += 1
                continue
            trace_id = str(trace.get("trace_id") or trace.get("task_id") or "")
            for pair in pairs:
                conn.execute(
                    f"""INSERT OR IGNORE INTO {TABLE}
                        (trace_id, chosen_id, rejected_id, chosen_rank, rejected_rank, score_gap,
                         created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (trace_id, pair.chosen_id, pair.rejected_id, pair.chosen_rank,
                     pair.rejected_rank, pair.score_gap, moment),
                )
                found.paired += 1
    return found


def alternatives(store: Any, memory_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """What this memory was chosen over, nearest first."""
    _ensure(store)
    return _rows(
        store,
        f"""SELECT * FROM {TABLE} WHERE chosen_id = ? ORDER BY ABS(score_gap) ASC LIMIT ?""",
        (str(memory_id), int(limit)),
    )


def beaten_by(store: Any, memory_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """What this memory lost to — the mirror question."""
    _ensure(store)
    return _rows(
        store,
        f"""SELECT * FROM {TABLE} WHERE rejected_id = ? ORDER BY ABS(score_gap) ASC LIMIT ?""",
        (str(memory_id), int(limit)),
    )


def narrowest(store: Any, limit: int = 10) -> list[dict[str, Any]]:
    """The closest calls in the store: pairs decided by the smallest gap.

    These are the decisions a small ranking change would flip, so they are the ones worth reading
    before trusting a score.
    """
    _ensure(store)
    return _rows(
        store,
        f"SELECT * FROM {TABLE} ORDER BY ABS(score_gap) ASC LIMIT ?",
        (int(limit),),
    )


def report(store: Any) -> dict[str, Any]:
    _ensure(store)
    totals = _rows(store, f"SELECT COUNT(*) AS pairs, COUNT(DISTINCT trace_id) AS traces, "
                          f"COUNT(DISTINCT chosen_id) AS chosen, COUNT(DISTINCT rejected_id) AS "
                          f"rejected FROM {TABLE}")
    row = totals[0] if totals else {}
    gaps = _rows(store, f"SELECT AVG(ABS(score_gap)) AS mean FROM {TABLE}")
    return {
        "pairs": int(row.get("pairs") or 0),
        "traces": int(row.get("traces") or 0),
        "chosen": int(row.get("chosen") or 0),
        "rejected": int(row.get("rejected") or 0),
        "mean_gap": round(float((gaps[0] if gaps else {}).get("mean") or 0.0), 6),
        "note": "a counterfactual is evidence of decisiveness, not of correctness",
    }
