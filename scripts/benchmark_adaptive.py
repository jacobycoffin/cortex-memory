#!/usr/bin/env python3
"""Ablate fixed, compact, and attention-gated Cortex context construction.

This is a deterministic local prompt-preparation benchmark. It does not call a
model and therefore must not be described as inference or TTFT evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

try:
    from Brain import CortexMemoryProvider
    from Brain.benchmarks.core import approximate_tokens, generate_facts, select_queries, summarize_latencies
    from Brain.store import CortexStore
except ModuleNotFoundError:
    from cortex import CortexMemoryProvider
    from cortex.benchmarks.core import approximate_tokens, generate_facts, select_queries, summarize_latencies
    from cortex.store import CortexStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fixed and adaptive Cortex prompt preparation.")
    parser.add_argument("--size", type=int, default=500, help="Synthetic indexed memories.")
    parser.add_argument("--memory-queries", type=int, default=30, help="Labeled durable-memory questions.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    facts = generate_facts(args.size, seed=args.seed)
    memory_facts = select_queries(facts, min(args.memory_queries, args.size), seed=args.seed)
    workload: list[tuple[str, str, str | None]] = [("memory", fact.query, fact.value) for fact in memory_facts]
    workload.extend(
        ("no_memory", query, None)
        for query in (
            "Hi!",
            "Hey.",
            "Hello!",
            "Thanks!",
            "Thank you.",
            "Good morning!",
            "Good night.",
            "Okay.",
            "Sounds good!",
            "Translate this into French: the door is open.",
            "Rewrite this: Cortex is a local memory provider.",
            "Proofread this: The tests is complete.",
        )
    )
    workload.extend(
        ("procedural", f"Run the test command for {fact.project} and inspect the result.", None)
        for fact in memory_facts[:10]
    )
    workload.extend(
        (
            "deep",
            f"Why is the current {fact.attribute} for {fact.project} related to its deployment history?",
            None,
        )
        for fact in memory_facts[:10]
    )

    conditions = {
        "fixed_verbose": {"adaptive_recall": False, "compact_context": False},
        "fixed_compact": {"adaptive_recall": False, "compact_context": True},
        "adaptive_compact": {"adaptive_recall": True, "compact_context": True},
    }
    report_conditions: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="cortex-adaptive-ablation-") as tmp:
        base_path = Path(tmp) / "base.db"
        base_store = CortexStore(base_path)
        for fact in facts:
            base_store.add_memory(
                fact.content,
                kind="decision",
                source_type="benchmark",
                source_category="TOOL_VERIFIED",
                source_ref=fact.label,
                confidence=0.95,
                currentness_confidence=0.95,
                importance=0.7,
                volatility=0.2,
                trust=0.95,
                subject=fact.project,
                predicate=fact.attribute,
                object_value=fact.value,
                extraction_method="synthetic_benchmark",
            )
        base_store.close()
        for name, config in conditions.items():
            home = Path(tmp) / name
            db_path = home / "cortex" / "benchmark.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base_path, db_path)
            provider = CortexMemoryProvider(
                {
                    "db_path": "$HERMES_HOME/cortex/benchmark.db",
                    "retrieval_threshold": 0.16,
                    "top_k": 6,
                    "token_budget": 700,
                    **config,
                }
            )
            provider.initialize(name, hermes_home=home, agent_context="primary")
            rows: list[dict[str, Any]] = []
            for index, (task_class, query, expected) in enumerate(workload):
                start = time.perf_counter()
                context = provider.prefetch(query, session_id=f"{name}-{index}")
                elapsed_ms = (time.perf_counter() - start) * 1000
                rows.append(
                    {
                        "task_class": task_class,
                        "context_tokens_approx": approximate_tokens(context),
                        "prepare_ms": elapsed_ms,
                        "abstained": not bool(context),
                        "answer_available": expected is not None and expected in context,
                        "has_labeled_answer": expected is not None,
                    }
                )
            provider.shutdown()
            labeled = [row for row in rows if row["has_labeled_answer"]]
            report_conditions[name] = {
                "queries": len(rows),
                "context_tokens_total_approx": sum(row["context_tokens_approx"] for row in rows),
                "context_tokens_median_approx": round(
                    statistics.median(row["context_tokens_approx"] for row in rows), 3
                ),
                "zero_context_rate": round(sum(row["abstained"] for row in rows) / len(rows), 6),
                "labeled_answer_context_recall": round(
                    sum(row["answer_available"] for row in labeled) / max(1, len(labeled)), 6
                ),
                "prepare_latency_ms": summarize_latencies([row["prepare_ms"] for row in rows]).__dict__,
                "by_task_class": _by_class(rows),
            }

    fixed_tokens = report_conditions["fixed_verbose"]["context_tokens_total_approx"]
    adaptive_tokens = report_conditions["adaptive_compact"]["context_tokens_total_approx"]
    report = {
        "benchmark": "cortex_adaptive_prompt_preparation",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_memories": args.size,
        "workload": {"memory_queries": len(memory_facts), "total_queries": len(workload), "seed": args.seed},
        "conditions": report_conditions,
        "adaptive_vs_fixed_verbose": {
            "approx_context_token_reduction": round(1.0 - adaptive_tokens / max(1, fixed_tokens), 6),
            "labeled_answer_context_recall_delta": round(
                report_conditions["adaptive_compact"]["labeled_answer_context_recall"]
                - report_conditions["fixed_verbose"]["labeled_answer_context_recall"],
                6,
            ),
        },
        "claim_boundary": (
            "Local prompt-preparation ablation only. Approximate context tokens are characters/4. "
            "This does not measure model inference, TTFT, total response latency, or task completion."
        ),
    }
    markdown = _markdown(report)
    print(markdown)
    if not args.no_write:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = args.output or ROOT / "benchmark-results" / f"cortex-adaptive-{timestamp}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output.with_suffix(".md").write_text(markdown, encoding="utf-8")
        print(f"JSON: {output}")
        print(f"Report: {output.with_suffix('.md')}")
    return 0


def _by_class(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task_class in sorted({row["task_class"] for row in rows}):
        subset = [row for row in rows if row["task_class"] == task_class]
        result[task_class] = {
            "queries": len(subset),
            "context_tokens_total_approx": sum(row["context_tokens_approx"] for row in subset),
            "zero_context_rate": round(sum(row["abstained"] for row in subset) / len(subset), 6),
        }
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cortex adaptive prompt-preparation ablation",
        "",
        f"Corpus: {report['corpus_memories']:,} synthetic memories · {report['workload']['total_queries']} mixed queries.",
        "",
        "| Condition | Approx total context tokens | Zero-context turns | Labeled answer in context | p50 prep | p95 prep |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["conditions"].items():
        latency = row["prepare_latency_ms"]
        lines.append(
            f"| {name} | {row['context_tokens_total_approx']:,} | {row['zero_context_rate']:.1%} | "
            f"{row['labeled_answer_context_recall']:.1%} | {latency['p50_ms']:.2f} ms | {latency['p95_ms']:.2f} ms |"
        )
    delta = report["adaptive_vs_fixed_verbose"]
    lines.extend(
        [
            "",
            f"Adaptive compact context reduced approximate memory-context tokens by **{delta['approx_context_token_reduction']:.1%}** "
            f"versus fixed verbose context in this mixed workload; labeled answer-context recall changed by "
            f"**{delta['labeled_answer_context_recall_delta']:+.1%}**.",
            "",
            f"> {report['claim_boundary']}",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
