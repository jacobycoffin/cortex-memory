"""Private, source-backed evaluation primitives for the Brain dashboard.

The dashboard passes already-stored task labels into this module. Queries and
memory IDs remain inside a disposable database snapshot; returned reports are
sanitized aggregates that are safe to persist in the operational ledger.
"""

from __future__ import annotations

import platform
import sqlite3
import statistics
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .benchmarks.core import approximate_tokens, summarize_latencies
from .cognition import plan_recall
from .retrieval import MemoryRetriever
from .store import CortexStore


REAL_HISTORY_SUITE = "cortex-private-real-history"
REAL_HISTORY_VERSION = "1"
REAL_HISTORY_MIN_CASES = 8
ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class HistoryCase:
    query: str
    relevant_memory_ids: tuple[str, ...]
    task_type: str = "general"


def compare_real_history(
    db_path: str | Path,
    cases: Sequence[HistoryCase],
    *,
    top_k: int = 6,
    token_budget: int = 700,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Compare adaptive and fixed recall on the same private labeled cases."""

    if len(cases) < REAL_HISTORY_MIN_CASES:
        raise ValueError(f"at least {REAL_HISTORY_MIN_CASES} labeled cases are required")
    _emit(progress_callback, phase="snapshot", progress=5, message="Creating a disposable private snapshot.")
    with private_database_snapshot(Path(db_path)) as snapshot_path:
        store = CortexStore(snapshot_path)
        try:
            _validate_targets_exist(store, cases)
            _emit(progress_callback, phase="fixed", progress=18, message="Testing the fixed recall policy.")
            fixed = evaluate_history(store, cases, top_k=top_k, token_budget=token_budget, policy="fixed")
            _emit(progress_callback, phase="adaptive", progress=55, message="Testing the adaptive recall policy.")
            adaptive = evaluate_history(store, cases, top_k=top_k, token_budget=token_budget, policy="adaptive")
        finally:
            store.close()
    _emit(progress_callback, phase="scoring", progress=92, message="Comparing quality, context, and latency.")
    report = {
        "schema_version": 1,
        "evaluation": REAL_HISTORY_SUITE,
        "suite_version": REAL_HISTORY_VERSION,
        "evidence_type": "private_offline_retrieval_mechanics",
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_count": len(cases),
        "reproducibility": {
            "python_version": platform.python_version(),
            "top_k": top_k,
            "token_budget_approx": token_budget,
            "conditions": ["fixed", "adaptive"],
            "same_cases": True,
        },
        "conditions": {"fixed": fixed, "adaptive": adaptive},
        "adaptive_minus_fixed": _condition_delta(adaptive["summary"], fixed["summary"]),
        "privacy": {
            "raw_private_text_omitted": True,
            "omitted": ["queries", "memory_ids", "memory_content", "source_references", "database_path"],
            "operator_review_required": True,
        },
        "claim_boundary": (
            "This paired private evaluation measures retrieval mechanics on operator-labeled history. "
            "It does not measure complete answer correctness, model inference speed, or task success."
        ),
    }
    _emit(progress_callback, phase="completed", progress=100, message="Private retrieval comparison complete.")
    return report


def evaluate_history(
    store: CortexStore,
    cases: Sequence[HistoryCase],
    *,
    top_k: int,
    token_budget: int,
    policy: str,
) -> dict[str, Any]:
    if policy not in {"fixed", "adaptive"}:
        raise ValueError("policy must be fixed or adaptive")
    retriever = MemoryRetriever(store)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        limit = top_k
        budget = token_budget
        search_kwargs: dict[str, Any] = {}
        if policy == "adaptive":
            plan = plan_recall(case.query, max_limit=top_k, max_token_budget=token_budget)
            limit = plan.limit
            budget = plan.token_budget
            search_kwargs = {
                "temporal_mode": plan.temporal_mode,
                "as_of": plan.as_of,
                "graph_depth": plan.graph_depth,
                "threshold": plan.threshold,
            }
        started = time.perf_counter_ns()
        results = retriever.search(case.query, limit=limit, token_budget=budget, **search_kwargs)
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        result_ids = [str(result.memory["id"]) for result in results]
        relevant = set(case.relevant_memory_ids)
        ranks = [rank for rank, memory_id in enumerate(result_ids, start=1) if memory_id in relevant]
        relevant_returned = len(ranks)
        context_text = "\n".join(str(result.memory.get("content", "")) for result in results)
        rows.append(
            {
                "case_index": index,
                "hit_at_k": bool(ranks),
                "first_relevant_rank": min(ranks) if ranks else None,
                "recall_at_k": round(relevant_returned / len(relevant), 6),
                "precision_at_k": round(relevant_returned / len(result_ids), 6) if result_ids else 0.0,
                "returned_memories": len(result_ids),
                "approximate_context_tokens": approximate_tokens(context_text),
                "retrieval_latency_ms": round(latency_ms, 6),
                "effective_top_k": limit,
                "effective_token_budget": budget,
            }
        )
    return {"policy": policy, "summary": _summarize_rows(rows), "cases": rows}


def _summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ranks = [row["first_relevant_rank"] for row in rows]
    return {
        "cases": len(rows),
        "hit_at_k": round(statistics.fmean(row["hit_at_k"] for row in rows), 6),
        "mean_recall_at_k": round(statistics.fmean(row["recall_at_k"] for row in rows), 6),
        "mean_precision_at_k": round(statistics.fmean(row["precision_at_k"] for row in rows), 6),
        "mrr": round(statistics.fmean(1.0 / rank if rank else 0.0 for rank in ranks), 6),
        "zero_result_rate": round(statistics.fmean(row["returned_memories"] == 0 for row in rows), 6),
        "retrieval_latency_ms": asdict(summarize_latencies([row["retrieval_latency_ms"] for row in rows])),
        "approximate_context_tokens": asdict(
            summarize_latencies([float(row["approximate_context_tokens"]) for row in rows])
        ),
    }


def _condition_delta(adaptive: dict[str, Any], fixed: dict[str, Any]) -> dict[str, float]:
    return {
        "hit_at_k": round(float(adaptive["hit_at_k"]) - float(fixed["hit_at_k"]), 6),
        "mean_recall_at_k": round(
            float(adaptive["mean_recall_at_k"]) - float(fixed["mean_recall_at_k"]), 6
        ),
        "mrr": round(float(adaptive["mrr"]) - float(fixed["mrr"]), 6),
        "context_tokens_p50": round(
            float(adaptive["approximate_context_tokens"]["p50_ms"])
            - float(fixed["approximate_context_tokens"]["p50_ms"]),
            3,
        ),
        "retrieval_p95_ms": round(
            float(adaptive["retrieval_latency_ms"]["p95_ms"])
            - float(fixed["retrieval_latency_ms"]["p95_ms"]),
            3,
        ),
    }


def _validate_targets_exist(store: CortexStore, cases: Sequence[HistoryCase]) -> None:
    for index, case in enumerate(cases, start=1):
        missing = sum(store.get_memory(memory_id) is None for memory_id in case.relevant_memory_ids)
        if missing:
            raise ValueError(f"case {index} references {missing} missing memory target(s)")


@contextmanager
def private_database_snapshot(db_path: Path) -> Iterable[Path]:
    resolved = db_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Cortex database does not exist: {db_path}")
    with tempfile.TemporaryDirectory(prefix="cortex-private-evaluation-") as tmp:
        snapshot_path = Path(tmp) / "evaluation.db"
        source = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True, timeout=5.0)
        destination = sqlite3.connect(snapshot_path)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        yield snapshot_path


def _emit(callback: ProgressCallback | None, **payload: object) -> None:
    if callback:
        callback(dict(payload))
