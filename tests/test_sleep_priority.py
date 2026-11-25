"""Backlog-aware scheduling (#117): value per measured millisecond, and nothing ranked on a made-up cost."""

from __future__ import annotations

import unittest

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import sleep_priority


class PlanTests(unittest.TestCase):
    # -- order -----------------------------------------------------------------------

    def test_work_is_ordered_by_value_per_millisecond(self) -> None:
        kinds = [
            sleep_priority.Kind("slow_valuable", "a", 0.9, 10_000, 5),
            sleep_priority.Kind("fast_moderate", "b", 0.6, 500, 5),
        ]
        order = [task.kind for task in sleep_priority.plan(kinds, budget_ms=100_000).order]
        self.assertEqual(order, ["fast_moderate", "slow_valuable"],
                         "value per millisecond, not value alone: the cheap useful thing goes first")

    def test_a_higher_value_wins_when_the_cost_is_the_same(self) -> None:
        kinds = [
            sleep_priority.Kind("low", "a", 0.3, 1000, 1),
            sleep_priority.Kind("high", "b", 0.9, 1000, 1),
        ]
        self.assertEqual([t.kind for t in sleep_priority.plan(kinds, budget_ms=10_000).order],
                         ["high", "low"])

    def test_equal_ratios_break_deterministically(self) -> None:
        kinds = [
            sleep_priority.Kind("zeta", "a", 0.5, 1000, 1),
            sleep_priority.Kind("alpha", "b", 0.5, 1000, 1),
        ]
        first = [t.kind for t in sleep_priority.plan(kinds, budget_ms=10_000).order]
        second = [t.kind for t in sleep_priority.plan(list(reversed(kinds)), budget_ms=10_000).order]
        self.assertEqual(first, second)
        self.assertEqual(first, ["alpha", "zeta"])

    # -- unmeasured work -------------------------------------------------------------

    def test_an_unmeasured_kind_is_last_and_named(self) -> None:
        kinds = [
            sleep_priority.Kind("unknown_cost", "a", 0.99, None, 1),
            sleep_priority.Kind("known", "b", 0.10, 1000, 1),
        ]
        plan = sleep_priority.plan(kinds, budget_ms=10_000)
        self.assertEqual([task.kind for task in plan.order], ["known", "unknown_cost"])
        self.assertIn("unknown_cost", plan.unmeasured)
        self.assertIn("nobody knows what they cost", "\n".join(plan.render()))

    def test_an_unmeasured_kind_is_never_given_a_fabricated_ratio(self) -> None:
        kind = sleep_priority.Kind("unknown_cost", "a", 0.9, None, 1)
        self.assertFalse(kind.measured)
        self.assertEqual(kind.ratio(), 0.0)

    def test_a_zero_cost_is_treated_as_unmeasured(self) -> None:
        self.assertFalse(sleep_priority.Kind("free", "a", 0.9, 0.0, 1).measured)

    # -- the budget ------------------------------------------------------------------

    def test_a_unit_that_does_not_fit_is_skipped_with_the_arithmetic(self) -> None:
        kinds = [sleep_priority.Kind("expensive", "a", 0.9, 5_000, 1)]
        plan = sleep_priority.plan(kinds, budget_ms=1_000)
        self.assertEqual(plan.order, [], "half a unit is not a unit: it is skipped, not truncated")
        self.assertIn("5000ms", plan.skipped["expensive"])
        self.assertIn("1000ms", plan.skipped["expensive"])

    def test_available_units_are_capped_by_the_budget_and_the_rest_is_reported(self) -> None:
        kinds = [sleep_priority.Kind("replay", "replay_schedule", 0.6, 1000, 200)]
        plan = sleep_priority.plan(kinds, budget_ms=10_000)
        self.assertAlmostEqual(plan.spent_ms, 10_000)
        self.assertIn("10 of 200", plan.order[0].reason)
        self.assertIn("190 unit(s) did not fit", plan.skipped["replay"])

    def test_the_plan_never_spends_more_than_the_budget(self) -> None:
        kinds = [sleep_priority.Kind(f"kind{index}", "x", 0.5, 700, 50) for index in range(6)]
        plan = sleep_priority.plan(kinds, budget_ms=3_333)
        self.assertLessEqual(plan.spent_ms, 3_333)

    def test_a_zero_budget_does_nothing_but_still_reports(self) -> None:
        kinds = [sleep_priority.Kind("replay", "replay_schedule", 0.6, 1000, 5)]
        plan = sleep_priority.plan(kinds, budget_ms=0)
        self.assertEqual(plan.order, [])
        self.assertIn("replay", plan.skipped)

    def test_nothing_available_is_stated_rather_than_silently_missing(self) -> None:
        plan = sleep_priority.plan([sleep_priority.Kind("idle", "x", 0.5, 1000, 0)], budget_ms=10_000)
        self.assertIn("nothing available", plan.skipped["idle"])

    # -- declared value ---------------------------------------------------------------

    def test_the_value_is_declared_and_must_be_in_range(self) -> None:
        with self.assertRaises(ValueError):
            sleep_priority.Kind("bad", "x", 1.5, 1000, 1)
        with self.assertRaises(ValueError):
            sleep_priority.Kind("bad", "x", 0.0, 1000, 1)

    def test_negative_availability_is_a_programming_error(self) -> None:
        with self.assertRaises(ValueError):
            sleep_priority.Kind("bad", "x", 0.5, 1000, -1)

    # -- measured cost ----------------------------------------------------------------

    def test_the_median_cost_is_used_not_the_mean(self) -> None:
        kind = sleep_priority.measured_from("replay", "replay_schedule", 0.6, 10,
                                            [100, 120, 130, 140, 12_000])
        self.assertEqual(kind.unit_cost_ms, 130, "one 12-second outlier must not set the price")

    def test_no_samples_means_unmeasured_rather_than_zero(self) -> None:
        kind = sleep_priority.measured_from("replay", "replay_schedule", 0.6, 10, [])
        self.assertIsNone(kind.unit_cost_ms)
        self.assertFalse(kind.measured)

    # -- handoff ----------------------------------------------------------------------

    def test_every_task_carries_the_owner_that_does_it(self) -> None:
        kinds = [sleep_priority.Kind("replay", "replay_schedule", 0.6, 1000, 1)]
        task = sleep_priority.plan(kinds, budget_ms=10_000).order[0]
        self.assertEqual(task.owner, "replay_schedule")
        self.assertIn("via replay_schedule", task.render())

    def test_the_plan_renders_everything_it_decided(self) -> None:
        kinds = [
            sleep_priority.Kind("replay", "replay_schedule", 0.6, 1000, 100),
            sleep_priority.Kind("merge", "merge_diff", 0.9, None, 1),
        ]
        rendered = "\n".join(sleep_priority.plan(kinds, budget_ms=5_000).render())
        self.assertIn("replay", rendered)
        self.assertIn("unmeasured", rendered)
        self.assertIn("skipped", rendered)

    def test_planning_is_pure_and_can_be_run_twice_identically(self) -> None:
        kinds = [sleep_priority.Kind("replay", "replay_schedule", 0.6, 1_000, 40)]
        first = sleep_priority.plan(kinds, budget_ms=7_000).as_dict()
        second = sleep_priority.plan(kinds, budget_ms=7_000).as_dict()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
