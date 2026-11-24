"""Which latency number is the real one? (#224)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import latency_reconcile as lr
from cortex.store import CortexStore


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def run_with(self, prepare_ms: float, stages: dict) -> None:
        self.store.record_recall_run(
            session_id="s", query="q", mode="auto", reason="test", requested_limit=6,
            token_budget=700, candidate_count=1, selected_count=1, estimated_tokens=10,
            prepare_ms=prepare_ms, abstained=False, task_id=None,
            stage_ms=stages,
        )

    def test_a_profile_ranks_the_heaviest_stage_first(self) -> None:
        self.run_with(500.0, {"select": 200.0, "semantic": 250.0, "fts": 20.0})
        profile = lr.stage_profile(self.store)
        self.assertEqual(profile.stages[0]["stage"], "semantic")
        self.assertEqual(profile.stages[1]["stage"], "select")

    def test_counts_are_not_treated_as_timings(self) -> None:
        """`stage_ms` carries counters with a _count suffix; they are not milliseconds."""
        self.run_with(500.0, {"select": 100.0, "entity_boosted_count": 999.0})
        stages = {item["stage"] for item in lr.stage_profile(self.store).stages}
        self.assertNotIn("entity_boosted_count", stages)

    def test_a_lexical_profile_says_so(self) -> None:
        self.run_with(150.0, {"select": 30.0, "fts": 12.0})
        profile = lr.stage_profile(self.store)
        self.assertFalse(profile.semantic_seen)
        self.assertIn("lexical fallback", profile.note)

    def test_a_semantic_profile_is_recognised(self) -> None:
        self.run_with(1200.0, {"select": 275.0, "semantic": 263.0})
        self.assertTrue(lr.stage_profile(self.store).semantic_seen)

    def test_the_percentiles_are_reported(self) -> None:
        self.run_with(100.0, {"select": 10.0})
        self.run_with(900.0, {"select": 10.0})
        profile = lr.stage_profile(self.store)
        self.assertGreater(profile.prepare_p95, profile.prepare_p50)

    def test_no_runs_is_reported_not_zero(self) -> None:
        profile = lr.stage_profile(self.store)
        self.assertEqual(profile.runs, 0)
        self.assertIn("no timed runs", profile.note)

    def test_availability_is_asked_of_the_embedder(self) -> None:
        flag = lr.availability()
        self.assertIn("semantic path", flag.describe())

    def test_an_unavailable_embedder_explains_the_fallback(self) -> None:
        flag = lr.Availability(available=False, model="m", reason="not importable")
        self.assertIn("lexical fallback", flag.describe())

    def test_the_sentence_carries_the_configuration(self) -> None:
        self.run_with(1200.0, {"select": 275.0, "semantic": 263.0})
        described = lr.describe(self.store, available=lr.Availability(True, "bge-small"))
        self.assertIn("semantic path live", described["sentence"])
        self.assertIn("heaviest stage select", described["sentence"])

    def test_the_sentence_carries_the_rule(self) -> None:
        self.assertIn("not a fact", lr.describe(self.store)["rule"])

    def test_the_compare_names_the_cause_and_the_check(self) -> None:
        compared = lr.compare(141.0, 1209.0)
        self.assertAlmostEqual(compared["ratio"], 8.6, places=1)
        self.assertIn("semantic", compared["sentence"])
        self.assertIn("get_embedder", compared["check"])

    def test_compare_without_numbers_says_so(self) -> None:
        self.assertIsNone(lr.compare(0, 100)["ratio"])

    def test_a_broken_store_reports_nothing_rather_than_raising(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(lr.stage_profile(Broken()).runs, 0)


if __name__ == "__main__":
    unittest.main()
