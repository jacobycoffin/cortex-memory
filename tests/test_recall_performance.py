"""Synthetic quality and bounded-query checks, without brittle timing assertions."""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401
from cortex.scripts.benchmark_recall import prepare_fixture, run_benchmark
from cortex.scripts.compare_retrieval_reports import compare_reports
from cortex.store import CortexStore, _retrieval_context_key


class BatchedContextFeedbackTests(unittest.TestCase):
    def test_aggregates_match_existing_formula_with_missing_ids_and_chunking(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CortexStore(Path(tmp) / "memory.db")
            try:
                mid, _ = store.add_memory("Synthetic release channel is green.")
                context = {"active_project": "synthetic", "scope": {"environment": "test"}}
                key, normalized = _retrieval_context_key(context)
                with store.transaction() as conn:
                    for index, outcome in enumerate(("helpful", "validated", "harmful", "corrected", "ignored")):
                        conn.execute("""INSERT INTO memory_context_outcomes
                            (task_id,memory_id,context_key,context_json,selected,used,outcome,updated_at)
                            VALUES(?,?,?,?,1,?,?,?)""",
                            (str(index), mid, key, json.dumps(normalized), int(index < 2), outcome,
                             "2026-01-01T00:00:00+00:00"))
                statements = []
                store._conn.set_trace_callback(statements.append)
                result = store.context_feedback_many([mid, mid, *[f"missing-{i}" for i in range(405)]], context)
                store._conn.set_trace_callback(None)
                selects = [sql for sql in statements if sql.startswith("SELECT memory_id, COUNT(*) observations")]
                self.assertEqual(len(selects), 2)
                self.assertEqual(len(result), 406)
                self.assertEqual(result[mid]["observations"], 5)
                self.assertEqual(result[mid]["positive_count"], 2)
                self.assertEqual(result[mid]["negative_count"], 2)
                self.assertEqual(result[mid]["used_count"], 2)
                expected = round((2 + 1.5) / (5 + 3) + .15 * 2 / 5 - .45 * 2 / 5 - .25 / 5, 6)
                self.assertEqual(result[mid]["usefulness"], expected)
                self.assertEqual(result["missing-0"]["usefulness"], .5)
                self.assertEqual(result[mid], store.context_feedback(mid, context))
                self.assertEqual(store.context_feedback_many([], context), {})
            finally:
                store.close()


    def test_v33_posting_index_upgrades_without_losing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            store = CortexStore(path)
            mid, _ = store.add_memory("Synthetic release gateway address is 192.0.2.20.")
            expected = [(row["id"], row["feature_score"]) for row in store.feature_search("release gateway address")]
            count = store._conn.execute("SELECT COUNT(*) FROM memory_features").fetchone()[0]
            with store.transaction() as conn:
                conn.execute("DROP INDEX idx_memory_features_feature")
                conn.execute("CREATE INDEX idx_memory_features_feature ON memory_features(feature,memory_id)")
                conn.execute("UPDATE meta SET value='33' WHERE key='schema_version'")
            store.close()
            upgraded = CortexStore(path)
            try:
                self.assertEqual([row[2] for row in upgraded._conn.execute("PRAGMA index_info(idx_memory_features_feature)")],
                                 ["feature", "memory_id", "weight"])
                self.assertEqual(upgraded._conn.execute("SELECT COUNT(*) FROM memory_features").fetchone()[0], count)
                self.assertEqual(upgraded._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0], "34")
                self.assertEqual([(row["id"], row["feature_score"]) for row in upgraded.feature_search("release gateway address")], expected)
                self.assertIsNotNone(upgraded.get_memory(mid))
                self.assertTrue(upgraded.audit()["ok"])
            finally:
                upgraded.close()


class PerformanceReportTests(unittest.TestCase):
    def report(self):
        return dict(evaluation="synthetic-recall-performance", schema_version=1,
                    memory_count=2000, seed=7, repetitions=30,
                    conditions={name: dict(synthetic_hit_at_k=.8, p50_ms=50, p95_ms=70)
                                for name in ("raw", "core", "hermes")},
                    correctness={"preloaded_selection_matches_uncached": {"core": True, "hermes": True}})

    def test_quality_and_latency_regressions_fail_independently(self):
        before = self.report()
        after = copy.deepcopy(before)
        self.assertTrue(compare_reports(before, after)["passed"])
        after["conditions"]["core"]["synthetic_hit_at_k"] = .7
        self.assertFalse(compare_reports(before, after)["passed"])
        after = copy.deepcopy(before)
        after["conditions"]["hermes"]["p95_ms"] = 100
        self.assertFalse(compare_reports(before, after)["passed"])
        after = copy.deepcopy(before)
        after["correctness"]["preloaded_selection_matches_uncached"]["core"] = False
        self.assertFalse(compare_reports(before, after)["passed"])

    def test_incompatible_and_tiny_samples_are_rejected(self):
        before = self.report()
        after = copy.deepcopy(before)
        after["seed"] = 8
        with self.assertRaises(ValueError):
            compare_reports(before, after)
        before["repetitions"] = after["repetitions"] = 3
        after["seed"] = 7
        with self.assertRaises(ValueError):
            compare_reports(before, after)

    def test_private_reports_require_privacy_and_guard_quality(self):
        report = dict(evaluation="cortex-private-real-history-retrieval",
            reproducibility=dict(runner_version=1, label_schema_version=1, case_count=8,
                                 policy="adaptive", max_top_k=6, max_token_budget=700),
            privacy={"raw_private_text_omitted": True},
            summary=dict(cases=8, hit_at_k=.9, mean_recall_at_k=.8, mean_precision_at_k=.7, mrr=.8,
                         retrieval_latency_ms={"p50_ms": 50, "p95_ms": 80}))
        candidate = copy.deepcopy(report)
        self.assertTrue(compare_reports(report, candidate)["passed"])
        candidate["summary"]["mrr"] = .6
        self.assertFalse(compare_reports(report, candidate)["passed"])
        candidate["privacy"]["raw_private_text_omitted"] = False
        with self.assertRaises(ValueError):
            compare_reports(report, candidate)

    def test_mismatched_fixture_is_rejected_before_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mismatch.db"
            store = CortexStore(path)
            store.add_memory("An invented fixture that is not the benchmark corpus.")
            with store.transaction() as conn:
                conn.execute("UPDATE meta SET value='33' WHERE key='schema_version'")
            store.close()
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                prepare_fixture(path, size=20, seed=7)
            self.assertEqual(path.read_bytes(), original)

    def test_small_benchmark_reports_selection_parity_and_separate_warming_cost(self):
        report = run_benchmark(size=20, repetitions=3)
        self.assertTrue(all(report["correctness"]["preloaded_selection_matches_uncached"].values()))
        self.assertEqual(report["conditions"]["hermes"]["mean_commits"], 1)
        self.assertGreater(report["conditions"]["preloaded_core"]["warming_ms"]["p50_ms"], 0)
        self.assertEqual(report["conditions"]["preloaded_core"]["preload"]["cache"]["preload_hits"], 3)
        self.assertNotIn("192.0.2", json.dumps(report))
        self.assertIn("not a future-query prediction", report["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
