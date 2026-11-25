"""Consolidation A/B: the worst metric decides, and nothing private leaves the module (#119)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import consolidation_ab as ab

BEFORE = {"hit_at_k": 0.80, "mean_recall_at_k": 0.62, "mrr": 0.55, "zero_result_rate": 0.04}


def metrics(**overrides) -> dict:
    return {**BEFORE, **overrides}


class CompareTests(unittest.TestCase):
    def test_identical_states_are_flat(self) -> None:
        result = ab.compare(BEFORE, metrics(), cases_before=200, cases_after=200)
        self.assertEqual(result.verdict, ab.FLAT)
        self.assertTrue(all(delta.verdict == ab.FLAT for delta in result.deltas))

    def test_a_real_gain_is_improved(self) -> None:
        result = ab.compare(BEFORE, metrics(hit_at_k=0.90), cases_before=200, cases_after=200)
        self.assertEqual(result.verdict, ab.IMPROVED)
        self.assertEqual([d.metric for d in result.deltas if d.verdict == ab.IMPROVED], ["hit_at_k"])

    def test_a_real_loss_is_regressed(self) -> None:
        result = ab.compare(BEFORE, metrics(mrr=0.40), cases_before=200, cases_after=200)
        self.assertEqual(result.verdict, ab.REGRESSED)
        self.assertEqual([d.metric for d in result.regressions], ["mrr"])

    def test_the_worst_metric_decides(self) -> None:
        """One regression outweighs three improvements: a night that breaks a query class is a bad night."""
        result = ab.compare(BEFORE, metrics(hit_at_k=0.95, mean_recall_at_k=0.80, mrr=0.75, zero_result_rate=0.09),
                            cases_before=200, cases_after=200)
        self.assertEqual(result.verdict, ab.REGRESSED)
        self.assertEqual([d.metric for d in result.regressions], ["zero_result_rate"],
                         "a rise in zero-result queries is a regression, and it is the only one")

    def test_a_delta_inside_tolerance_is_flat(self) -> None:
        result = ab.compare(BEFORE, metrics(hit_at_k=0.805), cases_before=200, cases_after=200)
        self.assertEqual(result.verdict, ab.FLAT, "0.005 is noise, not an improvement")

    def test_the_tolerance_is_the_gate_tolerance(self) -> None:
        self.assertEqual(ab.TOLERANCES["mrr"], 0.02, "the A/B must not be more lenient than the release gate")

    def test_different_case_counts_are_refused(self) -> None:
        result = ab.compare(BEFORE, metrics(hit_at_k=0.99), cases_before=200, cases_after=150)
        self.assertEqual(result.verdict, ab.UNMEASURABLE)
        self.assertIn("different case counts", result.reason)

    def test_a_missing_metric_is_unmeasurable_not_zero(self) -> None:
        after = {key: value for key, value in BEFORE.items() if key != "mrr"}
        result = ab.compare(BEFORE, after, cases_before=200, cases_after=200)
        mrr = next(delta for delta in result.deltas if delta.metric == "mrr")
        self.assertEqual(mrr.verdict, ab.UNMEASURABLE)

    def test_every_metric_missing_is_unmeasurable(self) -> None:
        result = ab.compare({}, {}, cases_before=10, cases_after=10)
        self.assertEqual(result.verdict, ab.UNMEASURABLE)

    def test_the_render_states_the_case_count_and_the_verdict(self) -> None:
        lines = ab.compare(BEFORE, metrics(hit_at_k=0.90), cases_before=200, cases_after=200).render()
        self.assertIn("200 case(s)", lines[0])
        self.assertIn(ab.IMPROVED, lines[0])

    def test_the_output_carries_no_query_and_no_memory_id(self) -> None:
        """The frozen set is private; the report must be metric names and numbers only."""
        rendered = " ".join(ab.compare(BEFORE, metrics(mrr=0.10), cases_before=200, cases_after=200).render())
        self.assertNotIn("mem-", rendered)
        self.assertNotIn("?", rendered)
        self.assertIn("mrr", rendered)


class PrivacyTests(unittest.TestCase):
    def test_a_set_inside_the_repository_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as repo:
            inside = Path(repo) / "evals" / "frozen"
            inside.mkdir(parents=True)
            with self.assertRaises(ValueError) as caught:
                ab.assert_set_is_private(inside, repo_root=repo)
            self.assertIn("outside the repository", str(caught.exception))

    def test_a_set_outside_the_repository_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as repo, tempfile.TemporaryDirectory() as elsewhere:
            ab.assert_set_is_private(elsewhere, repo_root=repo)  # must not raise

    def test_the_repository_root_itself_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as repo:
            with self.assertRaises(ValueError):
                ab.assert_set_is_private(repo, repo_root=repo)


class RunPairTests(unittest.TestCase):
    def test_the_evaluator_is_called_for_both_states(self) -> None:
        seen: list[str] = []

        def measure(store_path, set_dir):
            seen.append(str(store_path))
            return {**BEFORE, "cases": 200}

        with tempfile.TemporaryDirectory() as elsewhere:
            result = ab.run_pair(measure, before_store="before.db", after_store="after.db",
                                 set_dir=elsewhere, repo_root="/root/src/hermes-cortex-memory")
        self.assertEqual(seen, ["before.db", "after.db"])
        self.assertEqual(result.verdict, ab.FLAT)

    def test_a_private_set_is_refused_before_any_measurement(self) -> None:
        called: list[str] = []

        def measure(store_path, set_dir):
            called.append(str(store_path))
            return BEFORE

        with tempfile.TemporaryDirectory() as repo:
            inside = Path(repo) / "frozen"
            inside.mkdir()
            with self.assertRaises(ValueError):
                ab.run_pair(measure, before_store="a", after_store="b", set_dir=inside, repo_root=repo)
        self.assertEqual(called, [], "the guard runs before the evaluator, not after")


if __name__ == "__main__":
    unittest.main()
