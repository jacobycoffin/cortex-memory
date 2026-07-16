from __future__ import annotations

import unittest

from tests._bootstrap import ROOT

from cortex.extraction import extract_candidates


class SelectiveEncodingTests(unittest.TestCase):
    def test_one_turn_requests_are_not_durable_preferences(self) -> None:
        self.assertEqual(extract_candidates("I want you to deploy the service."), [])
        self.assertEqual(extract_candidates("We need to update it today."), [])

    def test_stable_preferences_and_explicit_memory_requests_remain_candidates(self) -> None:
        stable = extract_candidates("I always prefer compact technical explanations.")
        explicit = extract_candidates("Remember that I want the dashboard in amber.")

        self.assertEqual(stable[0].kind, "preference")
        self.assertTrue(explicit)

    def test_assistant_completion_status_is_not_memory(self) -> None:
        self.assertEqual(
            extract_candidates(
                "I successfully deployed the service and the tests passed.", role="assistant"
            ),
            [],
        )

    def test_assistant_fix_explanation_can_still_be_procedural(self) -> None:
        candidates = extract_candidates(
            "The root cause was a stale port; the fix was verified with the health endpoint.",
            role="assistant",
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0].kind, "procedure")


if __name__ == "__main__":
    unittest.main()
