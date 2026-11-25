"""The nightly budget (#110): per-night, persistent, soft stop, and a ceiling that cannot be raised."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import sleep_budget
from cortex.store import CortexStore

NIGHT = datetime(2026, 11, 24, 23, 50, tzinfo=timezone.utc)


class BudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    # -- the night --------------------------------------------------------------------

    def test_a_cycle_across_midnight_is_one_night(self) -> None:
        before = sleep_budget.night_of(datetime(2026, 11, 24, 23, 50, tzinfo=timezone.utc))
        after = sleep_budget.night_of(datetime(2026, 11, 25, 0, 20, tzinfo=timezone.utc))
        self.assertEqual(before, after, "splitting an overnight run across two budgets halves both")

    def test_a_morning_run_belongs_to_the_previous_night(self) -> None:
        evening = sleep_budget.night_of(datetime(2026, 11, 24, 22, 0, tzinfo=timezone.utc))
        morning = sleep_budget.night_of(datetime(2026, 11, 25, 5, 0, tzinfo=timezone.utc))
        self.assertEqual(evening, morning)

    def test_the_boundary_is_stated_and_tested(self) -> None:
        late = sleep_budget.night_of(datetime(2026, 11, 25, 5, 59, tzinfo=timezone.utc))
        next_day = sleep_budget.night_of(datetime(2026, 11, 25, 6, 0, tzinfo=timezone.utc))
        self.assertNotEqual(late, next_day)

    # -- persistence ------------------------------------------------------------------

    def test_spend_survives_a_reload(self) -> None:
        sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        sleep_budget.consume(self.store, seconds=30, tokens=200, at=NIGHT)
        reloaded = sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        self.assertEqual(reloaded.spent_seconds, 30)
        self.assertEqual(reloaded.spent_tokens, 200)

    def test_a_restart_does_not_hand_the_cycle_a_new_ceiling(self) -> None:
        sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        sleep_budget.consume(self.store, seconds=95, tokens=10, at=NIGHT)
        restarted = sleep_budget.open_night(self.store, seconds=600, tokens=200_000, at=NIGHT)
        self.assertEqual(restarted.seconds_limit, 100, "a restart must not reset the night's limits")
        self.assertFalse(restarted.may_start())

    # -- the soft stop ----------------------------------------------------------------

    def test_work_may_start_below_ninety_percent(self) -> None:
        sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        budget = sleep_budget.consume(self.store, seconds=80, tokens=100, at=NIGHT)
        self.assertTrue(budget.may_start())
        self.assertFalse(budget.exhausted)

    def test_the_soft_stop_stops_starting_new_work_but_not_finishing(self) -> None:
        sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        budget = sleep_budget.consume(self.store, seconds=91, tokens=100, at=NIGHT)
        self.assertFalse(budget.may_start(), "past the soft stop nothing new may start")
        self.assertFalse(budget.exhausted, "but the unit in flight can still finish")
        self.assertIn(sleep_budget.REASON_SOFT, budget.stop_reason)
        self.assertGreater(budget.remaining_seconds(), 0)

    def test_the_ceiling_is_recorded_with_its_reason(self) -> None:
        sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        budget = sleep_budget.consume(self.store, seconds=100, tokens=10, at=NIGHT)
        self.assertTrue(budget.exhausted)
        self.assertIn(sleep_budget.REASON_CEILING, budget.stop_reason)
        self.assertTrue(budget.stopped_at)

    def test_tokens_can_be_the_resource_that_runs_out(self) -> None:
        sleep_budget.open_night(self.store, seconds=1000, tokens=100, at=NIGHT)
        budget = sleep_budget.consume(self.store, seconds=1, tokens=100, at=NIGHT)
        self.assertTrue(budget.exhausted, "the worst of the two decides")

    # -- stopping and resuming --------------------------------------------------------

    def test_a_manual_stop_keeps_its_reason(self) -> None:
        sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        budget = sleep_budget.stop(self.store, "the user asked it to stop", at=NIGHT)
        self.assertFalse(budget.may_start())
        self.assertIn("user asked", budget.stop_reason)
        self.assertIn("user asked", budget.statement())

    def test_resuming_inside_the_budget_works_and_keeps_the_spend(self) -> None:
        sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        sleep_budget.consume(self.store, seconds=50, tokens=10, at=NIGHT)
        sleep_budget.stop(self.store, "paused", at=NIGHT)
        budget, resumed = sleep_budget.resume(self.store, at=NIGHT)
        self.assertTrue(resumed)
        self.assertTrue(budget.may_start())
        self.assertEqual(budget.spent_seconds, 50)

    def test_resuming_an_exhausted_night_is_refused(self) -> None:
        sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        sleep_budget.consume(self.store, seconds=100, tokens=10, at=NIGHT)
        budget, resumed = sleep_budget.resume(self.store, at=NIGHT)
        self.assertFalse(resumed)
        self.assertFalse(budget.may_start())

    # -- the ceiling cannot be raised -------------------------------------------------

    def test_lowering_the_ceiling_is_allowed(self) -> None:
        sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        budget, refused = sleep_budget.lower(self.store, seconds=40, at=NIGHT)
        self.assertEqual(refused, "")
        self.assertEqual(budget.seconds_limit, 40)

    def test_raising_the_ceiling_is_refused_with_the_current_limit(self) -> None:
        sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        budget, refused = sleep_budget.lower(self.store, seconds=500, at=NIGHT)
        self.assertIn("cannot be raised", refused)
        self.assertIn("100", refused)
        self.assertEqual(budget.seconds_limit, 100)

    def test_lowering_below_the_spend_stops_the_night(self) -> None:
        sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        sleep_budget.consume(self.store, seconds=60, tokens=10, at=NIGHT)
        budget, _ = sleep_budget.lower(self.store, seconds=50, at=NIGHT)
        self.assertFalse(budget.may_start())
        self.assertTrue(budget.stopped_at)

    # -- honesty about measurement ----------------------------------------------------

    def test_an_unmeasured_unit_is_recorded_as_estimated_not_free(self) -> None:
        sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        budget = sleep_budget.consume(self.store, at=NIGHT)
        self.assertTrue(budget.estimated)
        self.assertIn("[estimated]", budget.statement())
        self.assertEqual(budget.spent_seconds, 0.0)

    def test_the_statement_names_the_night_and_the_share(self) -> None:
        sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        budget = sleep_budget.consume(self.store, seconds=50, tokens=500, at=NIGHT)
        statement = budget.statement()
        self.assertIn(budget.night, statement)
        self.assertIn("50%", statement)

    def test_history_lists_the_nights(self) -> None:
        sleep_budget.open_night(self.store, seconds=100, tokens=1000, at=NIGHT)
        sleep_budget.consume(self.store, seconds=10, tokens=10, at=NIGHT)
        nights = sleep_budget.history(self.store)
        self.assertEqual(len(nights), 1)
        self.assertEqual(nights[0]["night"], sleep_budget.night_of(NIGHT))


if __name__ == "__main__":
    unittest.main()
