#!/usr/bin/env python3
"""Local deterministic ablation baseline for Cortex adaptive features.

Compares recall behavior with adaptive mechanisms enabled versus disabled,
using only synthetic temp-DB corpora. No network, no LLM, no live database.

Tier: smoke-baseline. Conditions:
  baseline         — defaults, Sleep shadow (no-op preview)
  sleep_apply      — training co-use signals + one Sleep apply cycle, with
                     an assertion that usage replay actually ran
  attention_policy — training attention observations, with an assertion that
                     the shadow recommendation actually moved
  combined         — sleep_apply + attention_policy

Measurement recalls never resolve usage: eval data stays out of the
training tables (no auto-ignored labels). Cases per condition: exact
query, paraphrase, corrected-fact wording, wrong-project scope, and a
no-memory greeting. Metrics are sanitized aggregates only (counts, rates,
token estimates, milliseconds) — the report never contains memory
content, queries, or IDs.

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


def _seed_sleep_training(memory: CortexMemory, ids: dict[str, str]) -> int:
    """Write resolved helpful co-use tasks Sleep usage-replay can consume.

    Store-level and fully deterministic (no retrieval variance): two tasks
    where the deploy-key and backup memories were used together with a
    helpful outcome. Returns the training task count.
    """
    pair = [ids["deploy-key"], ids["backup"]]
    for index in range(2):
        task_id = memory.store.create_usage_batch(
            [(pair[0], 0.9), (pair[1], 0.8)],
            query="training co-use",
            session_id=f"ablation-sleep-train-{index}",
            task_type="deployment",
            recall_mode="focused",
            requested_budget=700,
            estimated_tokens=600,
        )
        memory.store.resolve_usage(task_id, {pair[0]: 1.0, pair[1]: 1.0})
        memory.store.apply_task_outcome(task_id, "helpful")
    return 2


def _seed_attention_training(memory: CortexMemory, ids: dict[str, str]) -> int:
    """Write resolved attention observations that move the shadow policy.

    Four used samples on one topic/mode (two helpful) clear the weight
    threshold, so the shadow recommendation for a lean live mode must flip
    to the previously useful procedural mode. Public API only — the same
    shape as the adaptive-efficiency tests. Returns the sample count.
    """
    anchor = ids["deploy-key"]
    for index in range(4):
        task_id = memory.store.create_usage_batch(
            [(anchor, 0.9)],
            query="training attention",
            session_id="ablation-attention",
            task_type="deployment",
            recall_mode="procedural",
            requested_budget=600,
            estimated_tokens=520,
        )
        memory.store.record_attention_observation(
            task_id, task_type="deployment", topics=["acorn"],
            live_mode="procedural", live_budget=600, max_budget=700,
            selected_count=1,
        )
        memory.store.resolve_usage(task_id, {anchor: 1.0})
        if index < 2:
            memory.store.apply_task_outcome(task_id, "helpful")
    return 4


def _apply_condition(memory: CortexMemory, ids: dict[str, str], condition: str) -> dict[str, Any]:
    """Engage the condition's mechanism and prove it ran (fail fast)."""
    activation: dict[str, Any] = {
        "sleep_usage_tasks_replayed": 0,
        "sleep_evidence_added": 0,
        "attention_used_samples": 0,
        "attention_shadow_mode": "lean",
    }
    if condition in {"sleep_apply", "combined"}:
        _seed_sleep_training(memory, ids)
        report = memory.sleep(SleepConfig(mode="apply"))
        activation["sleep_usage_tasks_replayed"] = int(report["usage_tasks_replayed"])
        activation["sleep_evidence_added"] = int(report["evidence_added"])
        assert activation["sleep_usage_tasks_replayed"] >= 1, (
            f"{condition}: Sleep usage replay ran zero tasks — fixture broken"
        )
        assert activation["sleep_evidence_added"] >= 1, (
            f"{condition}: Sleep added zero evidence — fixture broken"
        )
    if condition in {"attention_policy", "combined"}:
        pre = memory.store.attention_recommendation(
            "deployment", ("acorn",), "lean", 600, max_budget=700
        )
        assert pre["shadow_mode"] == "lean", pre
        _seed_attention_training(memory, ids)
        post = memory.store.attention_recommendation(
            "deployment", ("acorn",), "lean", 600, max_budget=700
        )
        weights = memory.store._conn.execute(
            "SELECT used_count FROM attention_weights"
            " WHERE task_type='deployment' AND topic_key='acorn'"
        ).fetchall()
        activation["attention_used_samples"] = sum(int(row["used_count"]) for row in weights)
        activation["attention_shadow_mode"] = str(post["shadow_mode"])
        assert activation["attention_used_samples"] >= 4, (
            f"{condition}: attention training wrote no used samples — fixture broken"
        )
        assert post["shadow_mode"] == "procedural", (
            f"{condition}: shadow policy did not move — fixture broken: {post}"
        )
    return activation


def _measure(memory: CortexMemory, ids: dict[str, str], reps: int) -> dict[str, Any]:
    latencies: list[float] = []
    retrieval_hits = 0
    rendered_hits = 0
    answerable = 0
    false_positives = 0
    no_memory_cases = 0
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
            rendered = set(batch.rendered_memory_ids)
            expected = case["expect_key"]
            if case["needs_memory"]:
                answerable += 1
                if expected is not None and ids.get(expected) in selected[:3]:
                    retrieval_hits += 1
                if expected is not None and ids.get(expected) in rendered:
                    rendered_hits += 1
            else:
                no_memory_cases += 1
                if selected:
                    false_positives += 1
            # Deliberately unresolved: measurement recalls must not train.
            # finish() would label every measured batch used/ignored and
            # pollute the learning tables (and move the very weights under
            # test). Temp DBs are discarded with rows still pending.
    latencies.sort()
    mid = len(latencies) // 2
    p50 = latencies[mid] if latencies else 0.0
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0
    return {
        "cases": len(CASES) * reps,
        "retrieval_hit_at_3": round(retrieval_hits / answerable, 4) if answerable else None,
        "rendered_evidence_hit_rate": round(rendered_hits / answerable, 4) if answerable else None,
        "false_positive_rate": (
            round(false_positives / no_memory_cases, 4) if no_memory_cases else None
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
                activation = _apply_condition(memory, ids, condition)
                metrics = _measure(memory, ids, reps)
                metrics["activation"] = activation
                conditions[condition] = metrics
    return {
        "benchmark": "cortex_adaptive_ablation",
        "tier": "smoke-baseline",
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
        f"{'condition':<18}{'retr_hit@3':>10}{'rend_evid':>10}{'fp_rate':>9}"
        f"{'sel/recall':>11}{'tok/recall':>11}{'p50_ms':>9}{'p95_ms':>9}"
    )
    print(header)
    for name, metrics in report["conditions"].items():
        print(
            f"{name:<18}{metrics['retrieval_hit_at_3']:>10}{metrics['rendered_evidence_hit_rate']:>10}"
            f"{metrics['false_positive_rate']:>9}"
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
