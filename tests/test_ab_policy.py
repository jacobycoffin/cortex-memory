"""The A/B harness: assignment stability, pairing, and judges (#25)."""

from __future__ import annotations

import unittest

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.ab_policy import (
    POLICIES,
    _bootstrap_ci,
    assign,
    compare,
    coverage_judge,
    model_judge,
)


class AssignmentTests(unittest.TestCase):
    def test_the_same_session_always_gets_the_same_policy(self) -> None:
        first = assign("session-abc")
        for _ in range(20):
            self.assertEqual(assign("session-abc"), first)

    def test_assignment_is_stable_across_processes(self) -> None:
        """Python's `hash()` is salted per process; a session must not flip on restart."""
        import hashlib

        expected = list(POLICIES)[
            int.from_bytes(hashlib.sha256(b"session-abc").digest()[:4], "big") % len(POLICIES)
        ]
        self.assertEqual(assign("session-abc"), expected)

    def test_sessions_spread_over_the_policies(self) -> None:
        seen = {assign(f"session-{i}") for i in range(200)}
        self.assertEqual(seen, set(POLICIES))

    def test_an_empty_policy_list_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            assign("session-abc", policies=[])


class ComparisonTests(unittest.TestCase):
    def _run(self, scores: dict[str, list[float]]):
        def run(policy: str, case: int) -> tuple[float, str, float]:
            return scores[policy][case], "block", 10.0

        return run

    def test_identical_policies_are_all_ties(self) -> None:
        scores = {"baseline": [1, 0, 1, 0], "challenger": [1, 0, 1, 0]}
        result = compare(self._run(scores), [(i, i) for i in range(4)], challenger="challenger")
        self.assertEqual((result.wins, result.losses, result.ties), (0, 0, 4))
        self.assertFalse(result.decisive)

    def test_a_consistent_loss_is_decisive(self) -> None:
        scores = {"baseline": [1.0] * 20, "challenger": [0.0] * 20}
        result = compare(self._run(scores), [(i, i) for i in range(20)], challenger="challenger")
        self.assertEqual(result.losses, 20)
        self.assertTrue(result.decisive)
        self.assertLess(result.ci_high, 0.0)
        self.assertIn("worse", result.summary())

    def test_a_consistent_win_is_decisive(self) -> None:
        scores = {"baseline": [0.0] * 20, "challenger": [1.0] * 20}
        result = compare(self._run(scores), [(i, i) for i in range(20)], challenger="challenger")
        self.assertTrue(result.decisive)
        self.assertGreater(result.ci_low, 0.0)

    def test_noise_is_not_decisive(self) -> None:
        """Alternating wins and losses must not be reported as an improvement."""
        scores = {"baseline": [1.0 if i % 2 else 0.0 for i in range(20)],
                  "challenger": [0.0 if i % 2 else 1.0 for i in range(20)]}
        result = compare(self._run(scores), [(i, i) for i in range(20)], challenger="challenger")
        self.assertFalse(result.decisive)

    def test_an_unknown_policy_is_an_error(self) -> None:
        with self.assertRaises(KeyError):
            compare(
                lambda p, c: (0.0, "", 0.0),
                [],
                challenger="nope",
                policies={"baseline": {}, "challenger": {}},
            )

    def test_the_ci_is_deterministic(self) -> None:
        deltas = [1.0, 0.0, 1.0, 0.0, 1.0]
        self.assertEqual(_bootstrap_ci(deltas), _bootstrap_ci(deltas))


class JudgeTests(unittest.TestCase):
    def test_coverage_accepts_any_relevant_fact(self) -> None:
        judge = coverage_judge(["the first fact", "the second fact"])
        self.assertEqual(judge("context: the second fact is here"), 1.0)
        self.assertEqual(judge("context: nothing relevant"), 0.0)

    def test_coverage_normalises_whitespace(self) -> None:
        judge = coverage_judge("the  maintenance   window is sunday")
        self.assertEqual(judge("context: The maintenance window is Sunday."), 1.0)

    def test_coverage_with_no_gold_is_zero(self) -> None:
        self.assertEqual(coverage_judge([])("anything"), 0.0)

    def test_model_judge_grades_the_answer_not_the_context(self) -> None:
        judge = model_judge(
            "when is maintenance?",
            "sunday 02:00",
            complete=lambda prompt: "It is Sunday 02:00.",
        )
        self.assertEqual(judge("irrelevant context"), 1.0)

    def test_model_judge_rejects_an_answer_that_misses(self) -> None:
        judge = model_judge(
            "when is maintenance?",
            "sunday 02:00",
            complete=lambda prompt: "NOT IN CONTEXT",
        )
        self.assertEqual(judge("sunday 02:00"), 0.0)


if __name__ == "__main__":
    unittest.main()
