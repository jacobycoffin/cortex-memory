#!/usr/bin/env python3
"""Measure Cortex's exact-query cache without claiming model inference speed."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

try:
    from Brain import CortexMemoryProvider
    from Brain.benchmarks.core import generate_facts, summarize_latencies
    from Brain.store import CortexStore
except ModuleNotFoundError:
    from cortex import CortexMemoryProvider
    from cortex.benchmarks.core import generate_facts, summarize_latencies
    from cortex.store import CortexStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare cache-disabled and cache-enabled preparation for an identical repeated prefetch. "
            "This does not measure model inference or normal varied-query traffic."
        )
    )
    parser.add_argument("--size", type=int, default=2000, help="Synthetic memory count.")
    parser.add_argument("--repetitions", type=int, default=80, help="Warm repeated prefetches per condition.")
    parser.add_argument("--seed", type=int, default=7, help="Synthetic corpus seed.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output report.")
    return parser.parse_args()


def run_benchmark(*, size: int = 2000, repetitions: int = 80, seed: int = 7) -> dict[str, Any]:
    if size < 1 or repetitions < 2:
        raise ValueError("size must be positive and repetitions must be at least 2")
    facts = generate_facts(size, seed=seed)
    query = facts[len(facts) // 2].query
    with tempfile.TemporaryDirectory(prefix="cortex-cache-benchmark-") as tmp:
        root = Path(tmp)
        base_path = root / "base.db"
        store = CortexStore(base_path)
        try:
            for fact in facts:
                store.add_memory(
                    fact.content,
                    kind="decision",
                    source_type="synthetic_benchmark",
                    confidence=0.95,
                    importance=0.7,
                )
        finally:
            store.close()

        disabled = _run_condition(base_path, root / "disabled.db", query, ttl_seconds=0, repetitions=repetitions)
        enabled = _run_condition(base_path, root / "enabled.db", query, ttl_seconds=300, repetitions=repetitions)

    disabled_p50 = disabled["warm_latency_ms"]["p50_ms"]
    enabled_p50 = enabled["warm_latency_ms"]["p50_ms"]
    disabled_p95 = disabled["warm_latency_ms"]["p95_ms"]
    enabled_p95 = enabled["warm_latency_ms"]["p95_ms"]
    return {
        "schema_version": 1,
        "evaluation": "cortex-exact-query-cache",
        "evidence_type": "synthetic_repeated_prefetch_preparation",
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reproducibility": {
            "python_version": platform.python_version(),
            "memory_count": size,
            "warm_repetitions_per_condition": repetitions,
            "seed": seed,
        },
        "conditions": {"cache_disabled": disabled, "cache_enabled": enabled},
        "delta": {
            "warm_p50_reduction_percent": _reduction(disabled_p50, enabled_p50),
            "warm_p95_reduction_percent": _reduction(disabled_p95, enabled_p95),
        },
        "correctness": {
            "cache_enabled_context_stable": enabled["stable_context"],
            "cache_disabled_context_stable": disabled["stable_context"],
            "initial_conditions_returned_same_context": (
                disabled["context_digest"] == enabled["context_digest"]
            ),
        },
        "claim_boundary": (
            "This synthetic microbenchmark measures repeated identical prefetch preparation inside the cache TTL, "
            "before outcome sync. It does not measure varied-query hit rate, language-model inference, time to first "
            "token, total agent latency, answer accuracy, or production task success."
        ),
    }


def _run_condition(
    base_path: Path,
    condition_path: Path,
    query: str,
    *,
    ttl_seconds: int,
    repetitions: int,
) -> dict[str, Any]:
    shutil.copy2(base_path, condition_path)
    provider = CortexMemoryProvider(
        {
            "db_path": str(condition_path),
            "auto_capture": False,
            "adaptive_budget_learning": False,
            "query_cache_ttl_seconds": ttl_seconds,
            "retrieval_threshold": 0.01,
        }
    )
    provider.initialize(f"cache-benchmark-{ttl_seconds}", hermes_home=condition_path.parent)
    try:
        cold_start = time.perf_counter_ns()
        cold_context = provider.prefetch(query)
        cold_latency_ms = (time.perf_counter_ns() - cold_start) / 1_000_000
        contexts: list[str] = []
        warm_latency_ms: list[float] = []
        for _index in range(repetitions):
            start = time.perf_counter_ns()
            contexts.append(provider.prefetch(query))
            warm_latency_ms.append((time.perf_counter_ns() - start) / 1_000_000)
    finally:
        provider.shutdown()
    context_digest = hashlib.sha256(cold_context.encode("utf-8")).hexdigest()
    return {
        "cache_ttl_seconds": ttl_seconds,
        "cold_latency_ms": round(cold_latency_ms, 6),
        "warm_latency_ms": asdict(summarize_latencies(warm_latency_ms)),
        "stable_context": bool(cold_context) and all(context == cold_context for context in contexts),
        "context_digest": context_digest,
    }


def _reduction(baseline: float, candidate: float) -> float | None:
    if baseline <= 0:
        return None
    return round((baseline - candidate) / baseline * 100.0, 3)


def write_report(report: dict[str, Any], output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        report = run_benchmark(size=args.size, repetitions=args.repetitions, seed=args.seed)
        if args.output:
            write_report(report, args.output, overwrite=args.overwrite)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
