"""A policy change ships on a confidence interval, or it does not ship (#210)."""

from __future__ import annotations

import unittest

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import bootstrap_gate


class GateTests(unittest.TestCase):
    def test_identical_measurements_are_the_same_policy(self) -> None:
        values = [0.2 + index / 100 for index in range(40)]
        verdict = bootstrap_gate.paired_gate(values, values)
        self.assertEqual(verdict.decision, bootstrap_gate.SAME)
        self.assertIn("identical", verdict.reason)

    def test_a_clear_gain_is_decisioned(self) -> None:
        baseline = [0.0] * 40
        candidate = [1.0] * 40
        verdict = bootstrap_gate.paired_gate(baseline, candidate)
        self.assertEqual(verdict.decision, bootstrap_gate.BETTER)
        self.assertGreater(verdict.lower, 0)

    def test_a_clear_loss_is_decisioned_the_other_way(self) -> None:
        baseline = [1.0] * 40
        candidate = [0.0] * 40
        verdict = bootstrap_gate.paired_gate(baseline, candidate)
        self.assertEqual(verdict.decision, bootstrap_gate.WORSE)
        self.assertLess(verdict.upper, 0)

    def test_noise_is_not_a_win(self) -> None:
        """Half the cases up, half down, by the same amount: an average of zero and no verdict."""
        baseline = [0.0, 0.0, 0.0, 0.0] * 10
        candidate = [0.5, -0.5, 0.4, -0.4] * 10
        verdict = bootstrap_gate.paired_gate(baseline, candidate)
        self.assertEqual(verdict.decision, bootstrap_gate.SAME)
        self.assertFalse(verdict.decided)

    def test_a_tiny_sample_is_refused_rather_than_believed(self) -> None:
        verdict = bootstrap_gate.paired_gate([0.0] * 5, [1.0] * 5)
        self.assertEqual(verdict.decision, bootstrap_gate.SAME)
        self.assertIn("need", verdict.reason)

    def test_the_interval_spans_what_a_mean_hides(self) -> None:
        baseline = [0.0] * 35
        candidate = [1.0] * 18 + [0.0] * 17
        verdict = bootstrap_gate.paired_gate(baseline, candidate)
        self.assertLess(verdict.lower, verdict.mean_delta)
        self.assertGreater(verdict.upper, verdict.mean_delta)

    def test_the_same_numbers_give_the_same_interval(self) -> None:
        """A gate that changes its answer between runs is a lottery somebody re-runs until it passes."""
        baseline = [0.1, 0.4, 0.2, 0.9] * 12
        candidate = [0.2, 0.44, 0.2, 0.86] * 12
        first = bootstrap_gate.paired_gate(baseline, candidate)
        second = bootstrap_gate.paired_gate(baseline, candidate)
        self.assertEqual((first.lower, first.upper), (second.lower, second.upper))

    def test_mismatched_lengths_are_refused(self) -> None:
        verdict = bootstrap_gate.paired_gate([0.0] * 40, [0.0] * 39)
        self.assertEqual(verdict.decision, bootstrap_gate.SAME)
        self.assertIn("same cases", verdict.reason)

    def test_an_empty_measurement_is_refused(self) -> None:
        verdict = bootstrap_gate.paired_gate([], [])
        self.assertEqual(verdict.cases, 0)
        self.assertFalse(verdict.decided)

    def test_a_binary_metric_works_too(self) -> None:
        baseline = [False] * 40
        candidate = [True] * 40
        verdict = bootstrap_gate.from_hits(baseline, candidate)
        self.assertEqual(verdict.decision, bootstrap_gate.BETTER)

    def test_a_small_sample_is_flagged_in_the_verdict(self) -> None:
        verdict = bootstrap_gate.paired_gate([0.0] * 31, [1.0] * 31)
        self.assertTrue(verdict.decided)
        self.assertTrue(verdict.notes)

    def test_a_narrow_alpha_widens_the_interval(self) -> None:
        baseline = [0.0] * 40
        candidate = [1.0 if index % 3 else 0.0 for index in range(40)]
        wide = bootstrap_gate.paired_gate(baseline, candidate, alpha=0.001)
        narrow = bootstrap_gate.paired_gate(baseline, candidate, alpha=0.2)
        self.assertLess(wide.lower, narrow.lower)

    def test_the_report_separates_decided_from_undecided(self) -> None:
        described = bootstrap_gate.report({
            "clear": ([0.0] * 40, [1.0] * 40),
            "noise": ([0.0] * 40, [0.0, 0.0, 0.0, 0.0] * 10),
        })
        self.assertEqual(described["decided"], ["clear"])
        self.assertIn("noise", described["undecided"])

    def test_the_sentence_says_the_interval(self) -> None:
        verdict = bootstrap_gate.paired_gate([0.0] * 40, [1.0] * 40)
        self.assertIn("95% CI", verdict.describe())
        self.assertIn("over 40 case(s)", verdict.describe())

    def test_the_minimum_can_be_lowered_deliberately(self) -> None:
        verdict = bootstrap_gate.paired_gate([0.0] * 8, [1.0] * 8, min_cases=5)
        self.assertEqual(verdict.decision, bootstrap_gate.BETTER)


if __name__ == "__main__":
    unittest.main()
