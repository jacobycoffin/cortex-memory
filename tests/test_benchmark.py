from __future__ import annotations

import unittest
from pathlib import Path


from tests._bootstrap import ROOT

from cortex.benchmarks.core import (
    approximate_tokens,
    benchmark_scale,
    build_default_snapshot,
    generate_facts,
    render_markdown,
    run_benchmark,
    summarize_latencies,
)
from cortex.scripts.benchmark_aggregate import aggregate_reports


class CortexBenchmarkTests(unittest.TestCase):
    def test_fixture_is_deterministic_and_uniquely_labeled(self) -> None:
        first = generate_facts(40, seed=11)
        second = generate_facts(40, seed=11)
        self.assertEqual(first, second)
        self.assertEqual(len({fact.label for fact in first}), 40)
        self.assertEqual(len({fact.value for fact in first}), 40)

    def test_default_snapshot_never_exceeds_configured_content_limit(self) -> None:
        facts = generate_facts(100)
        snapshot, stored = build_default_snapshot(facts, char_limit=500)
        content = "\n§\n".join(fact.content for fact in stored)
        self.assertLessEqual(len(content), 500)
        self.assertLess(len(stored), len(facts))
        self.assertTrue(all(fact.value in snapshot for fact in stored))

    def test_metric_helpers_use_interpolation_and_clear_token_estimate(self) -> None:
        summary = summarize_latencies([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary.p50_ms, 2.5)
        self.assertEqual(approximate_tokens("a" * 9), 3)

    def test_small_comparison_has_expected_schema_and_recall(self) -> None:
        row = benchmark_scale(48, query_count=24, seed=5, top_k=6)
        self.assertEqual(row["corpus_memories"], 48)
        self.assertLess(row["default_built_in"]["stored_memories"], 48)
        self.assertGreaterEqual(row["cortex"]["recall_at_k"], 0.75)
        self.assertGreaterEqual(row["cortex"]["mrr"], 0.65)
        self.assertGreater(row["cortex"]["query_latency"]["p95_ms"], 0)

    def test_markdown_forbids_inference_claim_from_offline_run(self) -> None:
        report = run_benchmark([24], query_count=12)
        markdown = render_markdown(report)
        self.assertIn("does **not** measure LLM inference speed", markdown)
        self.assertIn("Required caveats", markdown)

    def test_live_aggregate_accepts_independent_query_seeds(self) -> None:
        def live_report(seed: int) -> dict:
            observations = []
            for condition, correct, ttft in (
                ("default_built_in", False, 100.0),
                ("cortex", True, 105.0),
            ):
                observations.append(
                    {
                        "condition": condition,
                        "label": f"fact-{seed}",
                        "correct": correct,
                        "answer_available_in_context": correct,
                        "memory_prepare_ms": 5.0 if condition == "cortex" else 0.0,
                        "agent_ttft_ms": ttft,
                        "agent_total_ms": ttft + 25.0,
                        "reported_prompt_tokens": 100,
                    }
                )
            return {
                "schema_version": 2,
                "model": "example/model",
                "corpus_memories": 500,
                "paired_queries": 1,
                "seed": seed,
                "cortex_mode": "additive",
                "run_at": "2026-07-13T00:00:00+00:00",
                "observations": observations,
                "conditions": {
                    "default_built_in": {
                        "accuracy": 0.0,
                        "reported_prompt_tokens_median": 100,
                    },
                    "cortex": {
                        "accuracy": 1.0,
                        "reported_prompt_tokens_median": 100,
                    },
                },
            }

        aggregate = aggregate_reports([live_report(7), live_report(17)], samples=100)
        self.assertEqual(aggregate["seeds"], [7, 17])
        self.assertEqual(aggregate["total_pairs"], 2)
        self.assertIn(
            "agent_ttft_ms_hierarchical_bootstrap_median_95_ci",
            aggregate["paired_deltas_cortex_minus_default"],
        )


if __name__ == "__main__":
    unittest.main()
