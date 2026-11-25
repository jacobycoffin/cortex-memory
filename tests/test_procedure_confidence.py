"""Procedural confidence (#142): failures weigh more, four outcomes to be believed, idleness decays."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import procedure_confidence as pc
from cortex.store import CortexStore

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


class ConfidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def ran(self, pid="p", outcome="success", *, days_ago=0.0, minute=0):
        at = NOW - timedelta(days=days_ago, minutes=minute)
        return pc.record(self.store, pid, outcome, at=at)

    def score(self, pid="p"):
        return pc.confidence(self.store, pid, now=NOW)

    # -- the shape of the number -------------------------------------------------------

    def test_an_untested_procedure_is_untested_not_trusted(self) -> None:
        result = self.score("never-run")
        self.assertEqual(result.verdict, "untested")
        self.assertEqual(result.confidence, 0.0)

    def test_one_success_is_not_trust(self) -> None:
        self.ran()
        result = self.score()
        self.assertLess(result.confidence, 0.5, "one lucky run must not read as proof")
        self.assertEqual(result.verdict, "learning")

    def test_four_clean_successes_earn_trust(self) -> None:
        for minute in range(4):
            self.ran(minute=minute)
        result = self.score()
        self.assertGreaterEqual(result.confidence, 0.6)
        self.assertEqual(result.verdict, "trusted")

    def test_a_failure_costs_more_than_a_success_gains(self) -> None:
        for minute in range(4):
            self.ran(minute=minute)
        clean = self.score().confidence
        self.ran("other", minute=0)
        for minute in range(3):
            self.ran("other", minute=minute + 1)
        self.ran("other", "failure", minute=5)
        with_failure = pc.confidence(self.store, "other", now=NOW).confidence
        self.assertLess(with_failure, clean, "a failure is more informative than a success")
        self.assertGreater(with_failure, clean - 0.35, "but it is not an erasure")

    def test_more_failures_than_successes_is_suspect(self) -> None:
        self.ran(outcome="success", minute=0)
        for minute in range(1, 4):
            self.ran(outcome="failure", minute=minute)
        self.assertEqual(self.score().verdict, "suspect")

    def test_the_number_is_bounded(self) -> None:
        for minute in range(50):
            self.ran(minute=minute)
        self.assertLessEqual(self.score().confidence, 1.0)
        self.assertGreaterEqual(self.score().confidence, 0.0)

    # -- decay -------------------------------------------------------------------------

    def test_an_idle_procedure_decays(self) -> None:
        for minute in range(4):
            self.ran(minute=minute)
        fresh = self.score().confidence
        for minute in range(4):
            self.ran("old", minute=minute)
        aged = pc.confidence(self.store, "old", now=NOW + timedelta(days=90)).confidence
        self.assertLess(aged, fresh * 0.5, "half-life is 45 days")
        self.assertGreater(aged, 0.0, "decay is not deletion")

    def test_a_year_idle_leaves_almost_nothing(self) -> None:
        for minute in range(4):
            self.ran(minute=minute)
        year = pc.confidence(self.store, "p", now=NOW + timedelta(days=365)).confidence
        self.assertLess(year, 0.01)

    def test_a_recent_run_resets_the_idle_clock_but_keeps_the_older_evidence(self) -> None:
        self.ran("spread", outcome="success", days_ago=300, minute=1)
        aged = pc.confidence(self.store, "spread", now=NOW)
        self.ran("spread", outcome="success", minute=0)
        refreshed = pc.confidence(self.store, "spread", now=NOW)
        self.assertGreater(aged.idle_days, 200)
        self.assertEqual(refreshed.idle_days, 0.0)
        self.assertGreater(refreshed.confidence, aged.confidence)
        self.assertEqual(refreshed.successes, 2)

    # -- what counts as evidence -------------------------------------------------------

    def test_recording_is_idempotent_on_the_same_moment(self) -> None:
        at = NOW
        pc.record(self.store, "p", "success", at=at)
        pc.record(self.store, "p", "success", at=at)
        self.assertEqual(self.score().successes, 1)

    def test_reading_confidence_changes_nothing(self) -> None:
        for minute in range(3):
            self.ran(minute=minute)
        before = self.store._conn.execute(f"SELECT COUNT(*) FROM {pc.OUTCOME_TABLE}").fetchone()[0]
        pc.confidence(self.store, "p", now=NOW)
        pc.rank(self.store, now=NOW)
        after = self.store._conn.execute(f"SELECT COUNT(*) FROM {pc.OUTCOME_TABLE}").fetchone()[0]
        self.assertEqual(before, after, "retrieval is not evidence: reading must not raise the score")

    def test_an_unknown_outcome_is_refused(self) -> None:
        self.assertFalse(pc.record(self.store, "p", "maybe")["ok"])

    def test_an_empty_procedure_id_is_refused(self) -> None:
        self.assertFalse(pc.record(self.store, "  ", "success")["ok"])

    # -- reporting ---------------------------------------------------------------------

    def test_the_score_reports_its_parts(self) -> None:
        for minute in range(4):
            self.ran(minute=minute)
        parts = self.score().parts()
        self.assertEqual(set(parts), {"smoothed_success", "evidence", "decay"})
        rendered = self.score().render()
        self.assertIn("4 ok", rendered)
        self.assertIn("0 failed", rendered)
        self.assertIn("d idle", rendered)

    def test_ranking_puts_the_most_trustworthy_first(self) -> None:
        for minute in range(4):
            self.ran("good", minute=minute)
        self.ran("bad", "failure", minute=0)
        order = [item.procedure_id for item in pc.rank(self.store, now=NOW)]
        self.assertEqual(order[0], "good")
        self.assertEqual(order[-1], "bad")

    def test_an_empty_store_says_no_procedure_has_run(self) -> None:
        self.assertEqual(pc.render(pc.rank(self.store, now=NOW)), ["No procedure has been run yet."])

    def test_a_broken_store_is_unreadable_rather_than_untested(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(pc.confidence(Broken(), "p").verdict, "unreadable")
        self.assertEqual(pc.rank(Broken()), [])


if __name__ == "__main__":
    unittest.main()
