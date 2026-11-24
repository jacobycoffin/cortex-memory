"""A memory's track record: whether the tasks it informed actually worked (#71)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import outcomes
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str = "The dashboard listens on port 8100.") -> str:
        return self.store.add_memory(
            content=content, kind="procedure", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def test_an_outcome_is_recorded(self) -> None:
        memory_id = self.memory()
        made = outcomes.record(self.store, memory_id, "task-1", "success", at=NOW)
        self.assertTrue(made["recorded"])
        self.assertEqual(outcomes.track_record(self.store, memory_id).successes, 1)

    def test_an_unknown_outcome_is_refused(self) -> None:
        memory_id = self.memory()
        made = outcomes.record(self.store, memory_id, "task-1", "probably fine")
        self.assertFalse(made["recorded"])
        self.assertIn("must be one of", made["reason"])

    def test_an_outcome_needs_a_task_reference(self) -> None:
        memory_id = self.memory()
        self.assertFalse(outcomes.record(self.store, memory_id, "", "success")["recorded"])

    def test_an_unknown_memory_is_refused(self) -> None:
        made = outcomes.record(self.store, "no-such-memory", "task-1", "success")
        self.assertFalse(made["recorded"])
        self.assertIn("unknown memory", made["reason"])

    def test_recording_the_same_task_twice_updates_it(self) -> None:
        memory_id = self.memory()
        outcomes.record(self.store, memory_id, "task-1", "failure", at=NOW)
        outcomes.record(self.store, memory_id, "task-1", "success", at=NOW)
        record = outcomes.track_record(self.store, memory_id)
        self.assertEqual((record.successes, record.failures), (1, 0))

    def test_the_track_record_tallies_both_ways(self) -> None:
        memory_id = self.memory()
        for index, verdict in enumerate(("success", "success", "failure", "neutral")):
            outcomes.record(self.store, memory_id, f"task-{index}", verdict, at=NOW)
        record = outcomes.track_record(self.store, memory_id)
        self.assertEqual((record.successes, record.failures, record.neutral), (2, 1, 1))
        self.assertEqual(record.judged, 3)

    def test_a_neutral_outcome_is_not_judged(self) -> None:
        memory_id = self.memory()
        outcomes.record(self.store, memory_id, "task-1", "neutral", at=NOW)
        record = outcomes.track_record(self.store, memory_id)
        self.assertEqual(record.judged, 0)
        self.assertEqual(record.trust(), 1.0)

    def test_one_failure_is_not_harmful(self) -> None:
        memory_id = self.memory()
        outcomes.record(self.store, memory_id, "task-1", "failure", at=NOW)
        self.assertFalse(outcomes.track_record(self.store, memory_id).harmful)

    def test_several_failures_with_no_successes_are_harmful(self) -> None:
        memory_id = self.memory()
        for index in range(3):
            outcomes.record(self.store, memory_id, f"task-{index}", "failure", at=NOW)
        self.assertTrue(outcomes.track_record(self.store, memory_id).harmful)

    def test_a_good_ratio_is_helpful(self) -> None:
        memory_id = self.memory()
        for index in range(3):
            outcomes.record(self.store, memory_id, f"task-{index}", "success", at=NOW)
        record = outcomes.track_record(self.store, memory_id)
        self.assertTrue(record.helpful)
        self.assertFalse(record.harmful)

    def test_trust_stays_inside_the_timid_band(self) -> None:
        good = self.memory("A procedure that worked.")
        bad = self.memory("A procedure that failed.")
        for index in range(4):
            outcomes.record(self.store, good, f"task-{index}", "success", at=NOW)
            outcomes.record(self.store, bad, f"task-{index}", "failure", at=NOW)
        self.assertAlmostEqual(outcomes.track_record(self.store, good).trust(), outcomes.CEILING,
                               places=4)
        self.assertAlmostEqual(outcomes.track_record(self.store, bad).trust(), outcomes.FLOOR,
                               places=4)

    def test_weighting_is_off_by_default(self) -> None:
        self.assertEqual(outcomes.ranking_hint(self.store, ["a"], environ={}), {})

    def test_weighting_applies_when_switched_on(self) -> None:
        memory_id = self.memory()
        outcomes.record(self.store, memory_id, "task-1", "success", at=NOW)
        hint = outcomes.ranking_hint(self.store, [memory_id], environ={"CORTEX_OUTCOME_WEIGHT": "1"})
        self.assertIn(memory_id, hint)
        self.assertGreater(hint[memory_id], 1.0)

    def test_recording_a_whole_task_marks_every_memory(self) -> None:
        first = self.memory("The dashboard listens on port 8100.")
        second = self.memory("Nginx proxies the dashboard.")
        made = outcomes.record_many(self.store, [first, second, "missing"], "task-1", "success",
                                    at=NOW)
        self.assertEqual(made["recorded"], 2)
        self.assertEqual(len(made["refused"]), 1)

    def test_the_last_outcome_is_reported(self) -> None:
        memory_id = self.memory()
        outcomes.record(self.store, memory_id, "task-1", "success",
                        at=datetime(2026, 11, 20, tzinfo=timezone.utc))
        outcomes.record(self.store, memory_id, "task-2", "failure",
                        at=datetime(2026, 11, 21, tzinfo=timezone.utc))
        self.assertEqual(outcomes.track_record(self.store, memory_id).last_outcome, "failure")

    def test_harmful_lists_only_the_flagged(self) -> None:
        bad = self.memory("A procedure that failed.")
        good = self.memory("A procedure that worked.")
        for index in range(3):
            outcomes.record(self.store, bad, f"t{index}", "failure", at=NOW)
            outcomes.record(self.store, good, f"t{index}", "success", at=NOW)
        flagged = [item.memory_id for item in outcomes.harmful(self.store)]
        self.assertEqual(flagged, [bad])

    def test_a_record_describes_itself(self) -> None:
        memory_id = self.memory()
        self.assertIn("no track record", outcomes.track_record(self.store, memory_id).describe())
        outcomes.record(self.store, memory_id, "task-1", "failure", at=NOW)
        self.assertIn("failure", outcomes.track_record(self.store, memory_id).describe())

    def test_the_report_counts_outcomes(self) -> None:
        memory_id = self.memory()
        outcomes.record(self.store, memory_id, "task-1", "success", at=NOW)
        described = outcomes.report(self.store)
        self.assertEqual(described["outcomes"], 1)
        self.assertEqual(described["successes"], 1)
        self.assertFalse(described["weighting"])

    def test_the_round_trip(self) -> None:
        memory_id = self.memory()
        outcomes.record(self.store, memory_id, "task-1", "success", at=NOW)
        payload = outcomes.track_record(self.store, memory_id).as_dict()
        for key in ("memory_id", "successes", "failures", "judged", "success_share", "harmful",
                    "helpful", "trust"):
            self.assertIn(key, payload)

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(outcomes.track_record(self.store, "nothing").judged, 0)
        self.assertEqual(outcomes.harmful(self.store), [])
        self.assertEqual(outcomes.report(self.store)["outcomes"], 0)


if __name__ == "__main__":
    unittest.main()
