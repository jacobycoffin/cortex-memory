"""Task-type budgets, and the measurement that made them move more than the number (item #13)."""

from __future__ import annotations

import unittest

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.task_budget import DEFAULT, TASK_BUDGETS, allocation, explain, plan_for


class AllocationTests(unittest.TestCase):
    def test_code_adjacent_tasks_get_more_than_email(self) -> None:
        self.assertGreater(allocation("shell")[0], allocation("email")[0])
        self.assertGreater(allocation("filesystem")[1], allocation("email")[1])

    def test_every_task_type_has_an_allocation(self) -> None:
        for task_type in TASK_BUDGETS:
            limit, budget = allocation(task_type)
            self.assertGreater(limit, 0)
            self.assertGreater(budget, 0)

    def test_unknown_types_fall_back_to_general(self) -> None:
        self.assertEqual(allocation("sorcery"), DEFAULT)
        self.assertEqual(allocation(""), DEFAULT)

    def test_lookup_is_case_insensitive(self) -> None:
        self.assertEqual(allocation("SHELL"), allocation("shell"))

    def test_the_allocation_is_monotonic_with_depth(self) -> None:
        """A deeper task should never get a smaller allocation than a shallow one."""
        self.assertGreaterEqual(allocation("filesystem")[0], allocation("general")[0])
        self.assertGreaterEqual(allocation("general")[0], allocation("calendar")[0])


class PlanTests(unittest.TestCase):
    def test_the_plan_is_clamped_to_the_deployment_ceiling(self) -> None:
        plan = plan_for("shell", max_limit=4, max_token_budget=300)
        self.assertEqual(plan["limit"], 4)
        self.assertLessEqual(plan["token_budget"], 300)

    def test_a_tiny_ceiling_still_yields_a_usable_plan(self) -> None:
        plan = plan_for("shell", max_limit=1, max_token_budget=10, floor_tokens=120)
        self.assertEqual(plan["limit"], 1)
        self.assertGreaterEqual(plan["token_budget"], 1)

    def test_the_reason_names_the_task_type(self) -> None:
        self.assertIn("shell", plan_for("shell")["reason"])

    def test_explain_is_human_readable(self) -> None:
        self.assertEqual(explain("calendar"), "calendar: 4 memories, 420 tokens")


if __name__ == "__main__":
    unittest.main()
