"""Query classes route to different recall strategies (item #5)."""

from __future__ import annotations

import unittest

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.cognition import CLASS_STRATEGIES, plan_recall, query_class


class ClassTests(unittest.TestCase):
    def test_a_lookup_is_a_lookup(self) -> None:
        for query in (
            "what's the rtsp path of the front camera?",
            "where is the cortex database?",
            "which port does t3 serve on",
            "how many memories are active",
        ):
            self.assertEqual(query_class(query), "lookup", query)

    def test_planning_is_planning(self) -> None:
        for query in (
            "what should we do next",
            "should we ship the roadmap or fix recall first",
            "what's the next step after this?",
            "how do we prioritise the remaining backlog items",
        ):
            self.assertEqual(query_class(query), "planning", query)

    def test_planning_wins_over_the_what_prefix(self) -> None:
        """'what should we do next' is not a lookup for a value."""
        self.assertEqual(query_class("what should we do next"), "planning")

    def test_a_bare_thanks_is_chitchat(self) -> None:
        self.assertEqual(query_class("thanks"), "chitchat")
        self.assertEqual(query_class("nice, cool"), "chitchat")

    def test_a_real_question_is_recall(self) -> None:
        for query in (
            "the temple build is green now",
            "compare duelist-road and temple-ios",
            "seerr telegram notifications setup",
        ):
            self.assertEqual(query_class(query), "recall", query)

    def test_empty_is_recall_not_chitchat(self) -> None:
        self.assertEqual(query_class(""), "recall")

    def test_a_long_sentence_mentioning_thanks_is_not_chitchat(self) -> None:
        query = "thanks but the temple workout glance still truncates the elapsed tile on the duo"
        self.assertNotEqual(query_class(query), "chitchat")


class StrategyTests(unittest.TestCase):
    def test_planning_buys_less_context_than_a_lookup(self) -> None:
        planning = CLASS_STRATEGIES["planning"]
        lookup = CLASS_STRATEGIES["lookup"]
        self.assertLess(int(planning["limit"]), int(lookup["limit"]))
        self.assertLess(int(planning["token_budget"]), int(lookup["token_budget"]))

    def test_a_lookup_lowers_the_threshold_and_planning_raises_it(self) -> None:
        self.assertLess(float(CLASS_STRATEGIES["lookup"]["threshold_delta"]), 0.0)
        self.assertGreater(float(CLASS_STRATEGIES["planning"]["threshold_delta"]), 0.0)

    def test_the_plan_carries_the_class(self) -> None:
        lookup = plan_recall("what's the rtsp path of the front camera?")
        self.assertEqual(lookup.mode, "lookup")
        self.assertTrue(lookup.needs_memory)
        self.assertLess(lookup.threshold, 0.16)

    def test_planning_gets_its_own_mode_and_a_tighter_budget(self) -> None:
        plan = plan_recall("how do we prioritise the remaining backlog items")
        self.assertEqual(plan.mode, "planning")
        self.assertLessEqual(plan.limit, 4)
        self.assertLessEqual(plan.token_budget, 520)

    def test_chitchat_needs_no_durable_recall(self) -> None:
        plan = plan_recall("thanks")
        self.assertEqual(plan.mode, "none")
        self.assertFalse(plan.needs_memory)

    def test_a_durable_cue_still_overrides_the_class(self) -> None:
        """'my' or 'remember' is a stronger signal than the shape of the question."""
        plan = plan_recall("what's my usual rtsp path?")
        self.assertIn(plan.mode, {"focused", "deep"})
        self.assertTrue(plan.needs_memory)

    def test_plans_stay_within_the_callers_bounds(self) -> None:
        plan = plan_recall("what's the path", max_limit=2, max_token_budget=200)
        self.assertLessEqual(plan.limit, 2)
        self.assertLessEqual(plan.token_budget, 200)


if __name__ == "__main__":
    unittest.main()
