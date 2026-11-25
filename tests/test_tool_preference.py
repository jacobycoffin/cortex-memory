"""A tool is preferred for a task only when this store watched it work (#132)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import procedure_confidence, tool_preference as tp
from cortex.store import CortexStore

BASE = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
SUCCESS = procedure_confidence.SUCCESS
FAILURE = procedure_confidence.FAILURE


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def runs(self, task_type: str, tool: str, outcomes: str, *, start_minute: int = 0) -> None:
        mapping = {"s": SUCCESS, "f": FAILURE}
        for index, letter in enumerate(outcomes):
            result = tp.record_outcome(self.store, task_type, tool, mapping[letter],
                                       at=BASE + timedelta(minutes=start_minute + index))
            self.assertTrue(result["ok"], result.get("reason"))


class IdentifierTests(unittest.TestCase):
    def test_a_preference_id_round_trips(self) -> None:
        self.assertEqual(tp.split_preference_id(tp.preference_id("run tests", "pytest -n 4")),
                         ("run tests", "pytest -n 4"))

    def test_the_task_is_normalised_and_the_tool_is_not(self) -> None:
        self.assertEqual(tp.preference_id("  Run Tests ", "pytest -n 4"), "run tests::pytest -n 4",
                         "the tool keeps its exact shape: `pytest -n 4` and `pytest -n 8` are different tools")


class PreferTests(Base):
    def test_a_validated_tool_is_preferred(self) -> None:
        self.runs("run tests", "pytest -n 2", "sss")
        preference = tp.prefer(self.store, "run tests")
        self.assertEqual(preference.verdict, tp.PREFERRED)
        self.assertEqual(preference.preferred, "pytest -n 2")

    def test_a_tool_with_no_track_record_is_not_preferred(self) -> None:
        self.runs("run tests", "pytest -n 2", "ss")
        preference = tp.prefer(self.store, "run tests")
        self.assertEqual(preference.verdict, tp.NO_VALIDATED_TOOL)
        self.assertEqual(preference.preferred, "")
        self.assertIn("best: pytest -n 2 at 2", preference.reason)

    def test_a_tool_that_fails_more_than_it_succeeds_is_never_preferred(self) -> None:
        """Even when it is the only candidate: "the only tool ever tried" is not a recommendation."""
        self.runs("move files", "mv --force", "ffs")
        preference = tp.prefer(self.store, "move files")
        self.assertEqual(preference.verdict, tp.NO_VALIDATED_TOOL)
        self.assertEqual(preference.preferred, "")

    def test_the_stronger_streak_wins(self) -> None:
        self.runs("run tests", "pytest -n 2", "sss")
        self.runs("run tests", "pytest -n 8", "ssssss")
        preference = tp.prefer(self.store, "run tests")
        self.assertEqual(preference.preferred, "pytest -n 8")

    def test_an_equal_pair_returns_nothing(self) -> None:
        self.runs("run tests", "pytest -n 2", "sss")
        self.runs("run tests", "pytest -n 4", "sss")
        preference = tp.prefer(self.store, "run tests")
        self.assertEqual(preference.verdict, tp.TIE)
        self.assertEqual(preference.preferred, "", "a coin flip dressed as guidance is worse than silence")
        self.assertIn("equally supported", preference.reason)

    def test_a_later_failure_demotes_the_leader(self) -> None:
        self.runs("run tests", "pytest -n 2", "ssss")
        self.runs("run tests", "pytest -n 8", "sss")
        # The failure has to be *newer* than the successes to demote anything: a failure recorded before them is
        # history the streak already absorbed.
        self.runs("run tests", "pytest -n 2", "f", start_minute=10)
        preference = tp.prefer(self.store, "run tests")
        self.assertEqual(preference.preferred, "pytest -n 8")

    def test_the_count_travels_with_the_preference(self) -> None:
        self.runs("run tests", "pytest -n 2", "sssfsss")
        preference = tp.prefer(self.store, "run tests")
        self.assertEqual(preference.preferred, "pytest -n 2")
        self.assertIn("3 in a row", preference.render())
        self.assertIn("1 failure", preference.render())

    def test_no_history_says_so(self) -> None:
        preference = tp.prefer(self.store, "deploy")
        self.assertEqual(preference.verdict, tp.NO_EVIDENCE)
        self.assertIn("no tool has been recorded", preference.reason)

    def test_an_empty_task_type_is_refused(self) -> None:
        self.assertEqual(tp.prefer(self.store, "").verdict, tp.NO_EVIDENCE)

    def test_an_unreadable_store_is_not_no_evidence(self) -> None:
        class Broken:
            _conn = None

        preference = tp.prefer(Broken(), "run tests")
        self.assertEqual(preference.verdict, tp.NO_EVIDENCE, "no history is no history, even when unreadable")

    def test_the_bar_is_configurable(self) -> None:
        self.runs("run tests", "pytest -n 2", "ss")
        self.assertEqual(tp.prefer(self.store, "run tests", required=2).preferred, "pytest -n 2")
        self.assertEqual(tp.prefer(self.store, "run tests", required=5).preferred, "")

    def test_preferring_writes_nothing(self) -> None:
        self.runs("run tests", "pytest -n 2", "sss")
        before = self.store._conn.execute(
            "SELECT COUNT(*) c FROM procedure_outcomes").fetchone()["c"]
        tp.prefer(self.store, "run tests")
        after = self.store._conn.execute(
            "SELECT COUNT(*) c FROM procedure_outcomes").fetchone()["c"]
        self.assertEqual(before, after)

    def test_tasks_do_not_leak_into_each_other(self) -> None:
        self.runs("run tests", "pytest -n 2", "sss")
        self.runs("deploy", "docker compose up", "sss")
        self.assertEqual(tp.prefer(self.store, "run tests").preferred, "pytest -n 2")
        self.assertEqual(tp.prefer(self.store, "deploy").preferred, "docker compose up")


class SurveyTests(Base):
    def test_a_survey_keeps_the_caller_order(self) -> None:
        self.runs("run tests", "pytest -n 2", "sss")
        preferences = tp.survey(self.store, ["run tests", "deploy"])
        self.assertEqual([item.task_type for item in preferences], ["run tests", "deploy"])
        self.assertEqual([item.verdict for item in preferences], [tp.PREFERRED, tp.NO_EVIDENCE])


if __name__ == "__main__":
    unittest.main()
