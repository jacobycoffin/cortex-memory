#!/usr/bin/env python3
"""Local deterministic ablation baseline for Cortex adaptive features.

Compares recall behavior with adaptive mechanisms enabled versus disabled,
using only synthetic temp-DB corpora. No network, no LLM, no live database.

Conditions (all through public APIs):
  baseline         — defaults, Sleep shadow (no-op preview)
  sleep_apply      — one Sleep apply cycle before measuring (consolidation)
  attention_policy — attention observations recorded + resolved before
                     measuring (salience-weight refresh)
  combined         — sleep_apply + attention_policy

Cases per condition: exact query, paraphrase, corrected-fact wording,
wrong-project scope, and a no-memory greeting. Metrics are sanitized
aggregates only (counts, rates, token estimates, milliseconds) — the report
never contains memory content, queries, or IDs.

Local embeddings are out of scope (a future optional experiment, not part
of this baseline).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import cortex  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - import shim, not behavior
    spec = importlib.util.spec_from_file_location(
        "cortex", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the Cortex package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cortex"] = module
    spec.loader.exec_module(module)

from cortex.client import CortexMemory  # noqa: E402
from cortex.sleep import SleepConfig  # noqa: E402

# Fixed synthetic corpus. Distinctive tokens per fact keep the assertions
# deterministic without any randomness.
CORPUS: list[dict[str, Any]] = [
    {"key": "deploy-key", "project": "acorn",
     "text": "The acorn staging deploy key rotates every Sunday at midnight UTC."},
    {"key": "backup", "project": "acorn",
     "text": "Acorn production deploys require a verified backup before traffic shifts."},
    {"key": "gateway", "project": "acorn",
     "text": "Acorn gateway rotation happens after the deploy train clears staging."},
    {"key": "quota", "project": "beacon",
     "text": "Beacon API quota resets on the first of each month at 00:00 UTC."},
    {"key": "quota-new", "project": "beacon",
     "text": "Correction: Beacon API quota now resets every Monday, not monthly."},
]

CASES: list[dict[str, Any]] = [
    {"name": "exact", "query": "When does the acorn staging deploy key rotate?",
     "expect_key": "deploy-key", "needs_memory": True},
    {"name": "paraphrase", "query": "Tell me about the Sunday midnight UTC rotation for acorn staging.",
     "expect_key": "deploy-key", "needs_memory": True},
    {"name": "corrected", "query": "How often does the Beacon API quota reset now?",
     "expect_key": "quota-new", "needs_memory": True},
    {"name": "wrong_project", "query": "acorn deploy key rotation schedule",
     "project": "beacon", "expect_key": None, "needs_memory": False},
    {"name": "greeting", "query": "hello there",
     "expect_key": None, "needs_memory": False},
]


def _seed(memory: CortexMemory) -> dict[str, str]:
    ids: dict[str, str] = {}
    for item in CORPUS:
        memory_id, _ = memory.remember(
            item["text"], kind="semantic",
            source_category="USER_EXPLICIT",
            scope={"project": item["project"]},
        )
        ids[item["key"]] = memory_id
    return ids


def _apply_condition(memory: CortexMemory, condition: str) -> None:
    if condition in {"sleep_apply", "combined"}:
        memory.sleep(SleepConfig(mode="apply"))
    if condition in {"attention_policy", "combined"}:
        for index, topic in enumerate(("acorn", "beacon")):
            task_id = f"ablation-attention-{topic}"
            memory.store.record_attention_observation(
                task_id, task_type="deployment", topics=[topic],
                live_mode="adaptive", live_budget=700, max_budget=4000,
                selected_count=2,
            )
            memory.store.resolve_usage(task_id, {})


def _measure(memory: CortexMemory, ids: dict[str, str], reps: int) -> dict[str, Any]:
    latencies: list[float] = []
    correct = 0
    answerable = 0
    irrelevant_selected = 0
    total_selected = 0
    rendered_tokens = 0
    for case in CASES:
        for _ in range(reps):
            started = time.perf_counter()
            batch = memory.recall(
                case["query"],
                task_type="deployment",
                active_project=case.get("project"),
                scope={"project": case["project"]} if case.get("project") else None,
            )
            latencies.append((time.perf_counter() - started) * 1000.0)
            selected = [str(item["id"]) for item in batch.memories]
            total_selected += len(selected)
            text = batch.context()
            rendered_tokens += batch.context_tokens()
            assert text is not None
            expected = case["expect_key"]
            if case["needs_memory"]:
                answerable += 1
                if expected is not None and ids.get(expected) in selected[:3]:
                    correct += 1
            elif selected:
                irrelevant_selected += 1
            batch.finish([], outcome=None)
    latencies.sort()
    mid = len(latencies) // 2
    p50 = latencies[mid] if latencies else 0.0
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0
    return {
        "cases": len(CASES) * reps,
        "accuracy": round(correct / answerable, 4) if answerable else None,
        "irrelevant_recall_rate": (
            round(irrelevant_selected / (len(CASES) * reps), 4) if CASES else None
        ),
        "mean_selected_per_recall": round(total_selected / (len(CASES) * reps), 3),
        "mean_rendered_tokens": round(rendered_tokens / (len(CASES) * reps), 1),
        "prepare_p50_ms": round(p50, 3),
        "prepare_p95_ms": round(p95, 3),
    }


def run(*, reps: int = 3) -> dict[str, Any]:
    conditions: dict[str, dict[str, Any]] = {}
    for condition in ("baseline", "sleep_apply", "attention_policy", "combined"):
        with tempfile.TemporaryDirectory() as tmp:
            with CortexMemory(Path(tmp) / "cortex.db") as memory:
                ids = _seed(memory)
                _apply_condition(memory, condition)
                conditions[condition] = _measure(memory, ids, reps)
    return {
        "benchmark": "cortex_adaptive_ablation",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reps_per_case": reps,
        "cases_per_condition": len(CASES) * reps,
        "local_only": True,
        "conditions": conditions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Cortex adaptive-feature ablation.")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(reps=max(1, args.reps))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    header = (
        f"{'condition':<18}{'accuracy':>10}{'irrelevant':>12}"
        f"{'sel/recall':>11}{'tok/recall':>11}{'p50_ms':>9}{'p95_ms':>9}"
    )
    print(header)
    for name, metrics in report["conditions"].items():
        print(
            f"{name:<18}{metrics['accuracy']:>10}{metrics['irrelevant_recall_rate']:>12}"
            f"{metrics['mean_selected_per_recall']:>11}{metrics['mean_rendered_tokens']:>11}"
            f"{metrics['prepare_p50_ms']:>9}{metrics['prepare_p95_ms']:>9}"
        )
    stats = statistics.mean(
        metrics["prepare_p50_ms"] for metrics in report["conditions"].values()
    )
    print(f"mean p50 across conditions: {stats:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
