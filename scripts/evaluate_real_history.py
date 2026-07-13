#!/usr/bin/env python3
"""Score Cortex retrieval against private, operator-authored relevance labels.

The input JSONL stays local. The generated report deliberately omits queries,
memory IDs, memory content, source references, case names, and database paths.
It measures retrieval mechanics only; it does not call or grade a language
model.
"""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

try:
    from Brain.benchmarks.core import approximate_tokens, summarize_latencies
    from Brain.cognition import plan_recall
    from Brain.retrieval import MemoryRetriever
    from Brain.store import CortexStore
except ModuleNotFoundError:
    from cortex.benchmarks.core import approximate_tokens, summarize_latencies
    from cortex.cognition import plan_recall
    from cortex.retrieval import MemoryRetriever
    from cortex.store import CortexStore


REPORT_SCHEMA_VERSION = 1
LABEL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RetrievalLabel:
    case_id: str
    query: str
    relevant_memory_ids: tuple[str, ...]
    group: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Cortex retrieval on private operator labels. The output contains metrics only and "
            "never copies queries, memory IDs, memory content, case names, or database paths."
        )
    )
    parser.add_argument("--db", type=Path, required=True, help="Private Cortex database to evaluate.")
    parser.add_argument("--labels", type=Path, required=True, help="Private JSONL relevance labels.")
    parser.add_argument("--output", type=Path, required=True, help="Sanitized JSON report path.")
    parser.add_argument("--top-k", type=int, default=6, help="Maximum returned memories.")
    parser.add_argument("--token-budget", type=int, default=700, help="Maximum approximate memory tokens.")
    parser.add_argument(
        "--policy",
        choices=("fixed", "adaptive"),
        default="adaptive",
        help="Use fixed limits or Cortex's query-dependent recall planner within the supplied maxima.",
    )
    parser.add_argument(
        "--include-group-summary",
        action="store_true",
        help="Include aggregate group names/counts. Group labels may reveal operator metadata.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output report.")
    return parser.parse_args()


def load_labels(path: Path) -> list[RetrievalLabel]:
    labels: list[RetrievalLabel] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"labels line {line_number} is not valid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"labels line {line_number} must be an object")
            if row.get("schema_version", LABEL_SCHEMA_VERSION) != LABEL_SCHEMA_VERSION:
                raise ValueError(f"labels line {line_number} has an unsupported schema_version")
            case_id = _required_text(row, "case_id", line_number)
            query = _required_text(row, "query", line_number)
            if case_id in seen:
                raise ValueError(f"labels line {line_number} repeats a case_id")
            seen.add(case_id)
            raw_ids = row.get("relevant_memory_ids")
            if (
                not isinstance(raw_ids, list)
                or not raw_ids
                or not all(isinstance(value, str) and value for value in raw_ids)
            ):
                raise ValueError(f"labels line {line_number} needs a non-empty relevant_memory_ids string list")
            unique_ids = tuple(dict.fromkeys(raw_ids))
            group = row.get("group")
            if group is not None and (not isinstance(group, str) or not group.strip()):
                raise ValueError(f"labels line {line_number} has an invalid group")
            labels.append(RetrievalLabel(case_id, query, unique_ids, group.strip() if group else None))
    if not labels:
        raise ValueError("labels file has no evaluation cases")
    return labels


def evaluate(
    store: CortexStore,
    labels: Sequence[RetrievalLabel],
    *,
    top_k: int,
    token_budget: int,
    policy: str,
    include_group_summary: bool = False,
) -> dict[str, Any]:
    if top_k < 1 or token_budget < 1:
        raise ValueError("top_k and token_budget must be positive")
    _validate_targets_exist(store, labels)
    retriever = MemoryRetriever(store)
    private_rows: list[tuple[RetrievalLabel, dict[str, Any]]] = []

    for index, label in enumerate(labels, start=1):
        limit = top_k
        budget = token_budget
        search_kwargs: dict[str, Any] = {}
        if policy == "adaptive":
            plan = plan_recall(label.query, max_limit=top_k, max_token_budget=token_budget)
            limit = plan.limit
            budget = plan.token_budget
            search_kwargs = {
                "temporal_mode": plan.temporal_mode,
                "as_of": plan.as_of,
                "graph_depth": plan.graph_depth,
                "threshold": plan.threshold,
            }
        start = time.perf_counter_ns()
        results = retriever.search(label.query, limit=limit, token_budget=budget, **search_kwargs)
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000
        result_ids = [str(result.memory["id"]) for result in results]
        relevant = set(label.relevant_memory_ids)
        relevant_ranks = [rank for rank, memory_id in enumerate(result_ids, start=1) if memory_id in relevant]
        relevant_returned = len(relevant_ranks)
        context_text = "\n".join(str(result.memory.get("content", "")) for result in results)
        row = {
            "case_index": index,
            "hit_at_k": bool(relevant_ranks),
            "first_relevant_rank": min(relevant_ranks) if relevant_ranks else None,
            "recall_at_k": round(relevant_returned / len(relevant), 6),
            "precision_at_k": round(relevant_returned / len(result_ids), 6) if result_ids else 0.0,
            "returned_memories": len(result_ids),
            "approximate_context_tokens": approximate_tokens(context_text),
            "retrieval_latency_ms": round(latency_ms, 6),
            "effective_top_k": limit,
            "effective_token_budget": budget,
        }
        private_rows.append((label, row))

    rows = [row for _, row in private_rows]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation": "cortex-private-real-history-retrieval",
        "evidence_type": "offline_retrieval_mechanics",
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reproducibility": {
            "runner_version": 1,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "python_version": platform.python_version(),
            "case_count": len(rows),
            "policy": policy,
            "max_top_k": top_k,
            "max_token_budget": token_budget,
        },
        "summary": _summarize_rows(rows),
        "cases": rows,
        "privacy": {
            "raw_private_text_omitted": True,
            "operator_review_required": True,
            "omitted": [
                "queries",
                "case_ids",
                "memory_ids",
                "memory_content",
                "source_references",
                "database_path",
                "labels_path",
            ],
            "warning": "Counts, timing, and optional group labels can still reveal operational metadata.",
        },
        "claim_boundary": (
            "This report measures labeled retrieval mechanics only. It does not measure model accuracy, "
            "agent task success, tool reliability, inference latency, or billed tokens."
        ),
    }
    if include_group_summary:
        report["groups"] = _group_summaries(private_rows)
        report["privacy"]["includes_operator_authored_group_names"] = True
    return report


def _summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    first_ranks = [row["first_relevant_rank"] for row in rows]
    return {
        "cases": len(rows),
        "hit_at_k": round(statistics.fmean(row["hit_at_k"] for row in rows), 6),
        "mean_recall_at_k": round(statistics.fmean(row["recall_at_k"] for row in rows), 6),
        "mean_precision_at_k": round(statistics.fmean(row["precision_at_k"] for row in rows), 6),
        "mrr": round(statistics.fmean(1.0 / rank if rank else 0.0 for rank in first_ranks), 6),
        "zero_result_rate": round(statistics.fmean(row["returned_memories"] == 0 for row in rows), 6),
        "retrieval_latency_ms": asdict(summarize_latencies([row["retrieval_latency_ms"] for row in rows])),
        "approximate_context_tokens": asdict(
            summarize_latencies([float(row["approximate_context_tokens"]) for row in rows])
        ),
    }


def _group_summaries(private_rows: Iterable[tuple[RetrievalLabel, dict[str, Any]]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for label, row in private_rows:
        grouped.setdefault(label.group or "ungrouped", []).append(row)
    return {name: _summarize_rows(rows) for name, rows in sorted(grouped.items())}


def _validate_targets_exist(store: CortexStore, labels: Sequence[RetrievalLabel]) -> None:
    for case_index, label in enumerate(labels, start=1):
        missing_count = sum(store.get_memory(memory_id) is None for memory_id in label.relevant_memory_ids)
        if missing_count:
            raise ValueError(
                f"case {case_index} references {missing_count} memory/memories not present in this database; "
                "the report was not generated"
            )


def _required_text(row: dict[str, Any], field: str, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"labels line {line_number} needs a non-empty {field}")
    return value.strip()


def write_report(report: dict[str, Any], output: Path, *, overwrite: bool = False) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@contextmanager
def private_database_snapshot(db_path: Path) -> Iterable[Path]:
    """Yield a consistent disposable snapshot without migrating the live DB."""

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


def main() -> int:
    args = parse_args()
    try:
        labels = load_labels(args.labels)
        with private_database_snapshot(args.db) as snapshot_path:
            store = CortexStore(snapshot_path)
            try:
                report = evaluate(
                    store,
                    labels,
                    top_k=args.top_k,
                    token_budget=args.token_budget,
                    policy=args.policy,
                    include_group_summary=args.include_group_summary,
                )
            finally:
                store.close()
        write_report(report, args.output, overwrite=args.overwrite)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(
        f"Wrote sanitized retrieval report with {report['summary']['cases']} cases to {args.output}. "
        "Review metadata before publishing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
