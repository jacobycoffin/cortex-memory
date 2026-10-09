#!/usr/bin/env python3
"""Measure synthetic recall latency, correctness, and explicit-hint warming.

Reports contain aggregates only. Warming gets the upcoming query as an explicit
harness hint: its speedup is an upper bound, not a prediction-hit-rate claim.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    spec = importlib.util.spec_from_file_location("cortex", ROOT / "__init__.py",
                                                submodule_search_locations=[str(ROOT)])
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load checkout")
    package = importlib.util.module_from_spec(spec)
    sys.modules["cortex"] = package
    spec.loader.exec_module(package)

from cortex.benchmarks.core import generate_facts
from cortex.client import CortexMemory
from cortex.hermes_provider import CortexMemoryProvider
from cortex.retrieval import MemoryRetriever
from cortex.store import CortexStore


def percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"p50_ms": round(statistics.median(ordered), 3),
            "p95_ms": round(ordered[max(0, (len(ordered) * 95 + 99) // 100 - 1)], 3)}


def prepare_fixture(path: Path, *, size: int, seed: int) -> None:
    facts = generate_facts(size, seed=seed)
    if path.exists():
        # Validate existing fixtures read-only before Cortex can migrate them.
        # An accidentally supplied operator database must remain untouched.
        try:
            with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
                existing = [row[0] for row in conn.execute("SELECT content FROM memories")]
        except sqlite3.Error as error:
            raise ValueError("fixture must be a generated synthetic Cortex database") from error
        if sorted(existing) != sorted(fact.content for fact in facts):
            raise ValueError("fixture must contain exactly the generated synthetic corpus")
        return
    store = CortexStore(path)
    try:
        with store.transaction():
            for fact in facts:
                store.add_memory(fact.content, kind="decision", source_type="synthetic_benchmark",
                                 confidence=0.95, importance=0.7)
    finally:
        store.close()


def run_benchmark(*, size: int = 2000, repetitions: int = 30, seed: int = 7,
                  fixture: Path | None = None, preload: bool = True) -> dict[str, Any]:
    if size < 1 or repetitions < 2:
        raise ValueError("size must be positive and repetitions must be at least two")
    facts = generate_facts(size, seed=seed)
    cases = [facts[(index * 47) % size] for index in range(repetitions)]
    conditions: dict[str, Any] = {}
    selected_by_condition: dict[str, list[tuple[str, ...]]] = {}
    with tempfile.TemporaryDirectory(prefix="cortex-recall-benchmark-") as tmp:
        root = Path(tmp)
        base = fixture or root / "synthetic.db"
        prepare_fixture(base, size=size, seed=seed)
        names = ["raw", "core", "hermes"]
        if preload:
            names += ["preloaded_core", "preloaded_hermes"]
        for name in names:
            path = root / (name + ".db")
            # Include committed WAL pages without modifying the source fixture.
            with sqlite3.connect(base.resolve().as_uri() + "?mode=ro", uri=True) as source:
                with sqlite3.connect(path) as destination:
                    source.backup(destination)
            latencies: list[float] = []
            warming: list[float] = []
            commits: list[int] = []
            stages: dict[str, list[float]] = {}
            selections: list[tuple[str, ...]] = []
            hits = 0
            warm = name.startswith("preloaded_")
            if "hermes" in name:
                owner = CortexMemoryProvider(dict(db_path=str(path), auto_capture=False,
                    adaptive_budget_learning=False, background_preload=warm,
                    query_cache_ttl_seconds=45 if warm else 0))
                owner.initialize("benchmark", hermes_home=root)
                store = owner._store
            elif "core" in name:
                owner = CortexMemory(path, cache_ttl_seconds=45 if warm else 0)
                store = owner.store
            else:
                owner = store = CortexStore(path)
                retriever = MemoryRetriever(store)
            try:
                for fact in cases:
                    if warm:
                        started = time.perf_counter()
                        if "hermes" in name:
                            owner.queue_prefetch(fact.query, session_id="benchmark")
                        else:
                            owner.preload(fact.query, session_id="benchmark")
                        if not owner._preloader.wait_idle(timeout=30):
                            raise RuntimeError("background warming did not finish")
                        warming.append((time.perf_counter() - started) * 1000)
                    count = [0]
                    # Count durable transaction boundaries, not FTS internal
                    # trace callbacks (which inflate apparent SQL query counts).
                    store._conn.set_trace_callback(
                        lambda statement: count.__setitem__(0, count[0] + int(statement == "COMMIT"))
                    )
                    started = time.perf_counter()
                    batch = None
                    if "hermes" in name:
                        owner.prefetch(fact.query, session_id="benchmark")
                    elif "core" in name:
                        batch = owner.recall(fact.query, session_id="benchmark")
                        batch.context()
                    else:
                        results, diagnostics = retriever.search_detailed(fact.query)
                    latencies.append((time.perf_counter() - started) * 1000)
                    store._conn.set_trace_callback(None)
                    commits.append(count[0])
                    if "hermes" in name or "core" in name:
                        row = store._conn.execute(
                            "SELECT stage_ms_json FROM recall_runs ORDER BY rowid DESC LIMIT 1"
                        ).fetchone()
                        stage_values = json.loads(row[0])
                        ids = [str(item["memory_id"]) for item in store.memory_traces(limit=1)[0]["candidate_memories"]
                               if item.get("selected")]
                        content = tuple(store.get_memory(memory_id)["content"] for memory_id in ids)
                    else:
                        stage_values = diagnostics.stage_ms
                        content = tuple(result.memory["content"] for result in results)
                    for key, value in stage_values.items():
                        stages.setdefault(key, []).append(float(value))
                    selections.append(content)
                    hits += int(fact.content in content)
                    if batch is not None:
                        batch.finish()
                    elif "hermes" in name:
                        owner.sync_turn(fact.query, "Synthetic response with no attributed memory use.",
                                        session_id="benchmark")
                        # After-turn speculation is outside foreground timing.
                        if warm and not owner._preloader.wait_idle(timeout=30):
                            raise RuntimeError("after-turn warming did not finish")
                metrics = {**percentiles(latencies), "mean_commits": round(statistics.mean(commits), 2),
                           "synthetic_hit_at_k": hits / repetitions,
                           "stages": {key: percentiles(value) for key, value in stages.items()}}
                if warm:
                    metrics["warming_ms"] = percentiles(warming)
                    metrics["preload"] = owner.preload_stats()
                conditions[name] = metrics
                selected_by_condition[name] = selections
            finally:
                store._conn.set_trace_callback(None)
                if "hermes" in name:
                    owner.shutdown()
                else:
                    owner.close()
    parity = {
        kind: selected_by_condition[kind] == selected_by_condition["preloaded_" + kind]
        for kind in ("core", "hermes") if "preloaded_" + kind in selected_by_condition
    }
    return dict(schema_version=1, evaluation="synthetic-recall-performance",
                python_version=platform.python_version(), memory_count=size,
                repetitions=repetitions, seed=seed, conditions=conditions,
                correctness={"preloaded_selection_matches_uncached": parity},
                claim_boundary=("Synthetic retrieval and preparation only; no model inference or answer grading. "
                    "Each warm condition receives the upcoming query explicitly and completes warming before recall. "
                    "Warming consumes CPU and is reported separately; this is not a future-query prediction claim."))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fixture", type=Path, help="Reusable synthetic-only database outside the repository.")
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output and args.output.exists() and not args.overwrite:
        parser.error("output exists; pass --overwrite to replace it")
    report = run_benchmark(size=args.size, repetitions=args.repetitions, seed=args.seed,
                           fixture=args.fixture, preload=not args.no_preload)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if all(report["correctness"]["preloaded_selection_matches_uncached"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
