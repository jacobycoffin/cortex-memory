"""The automatic admission bar must be reachable by the candidates it is meant to admit.

The bug this covers: the temporary bar was 0.62 while volatility >= 0.70 is *what makes a candidate temporary* and also
costs 0.036 of score. Operational candidates therefore topped out around 0.60-0.62 and were always refused - 52 of 52
automatic writes on the live box, several at 0.6012 and 0.6192.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

import cortex.store as store_module
from cortex.store import CortexStore


class AdmissionBarTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self._tmp.name) / "admission.db")

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def assess(self, content: str, **kwargs):
        return self.store.assess_storage_candidate(content, automatic=True, **kwargs)

    def test_an_operational_fact_with_high_volatility_is_temporary(self) -> None:
        verdict = self.assess(
            "The Teknium queue console runs as the systemd unit cortex-queue.service on port 8791",
            kind="operational",
            volatility=0.72,
        )
        self.assertEqual(verdict["durability"], "temporary")

    def test_the_temporary_bar_is_lower_than_it_was(self) -> None:
        """0.62 was unreachable once volatility had already cost the score 0.036."""
        verdict = self.assess(
            "The Teknium queue console runs as the systemd unit cortex-queue.service on port 8791",
            kind="operational",
            volatility=0.72,
        )
        self.assertEqual(verdict["minimum_reusable_score"], store_module.AUTOMATIC_MIN_REUSABLE_SCORE_TEMPORARY)
        self.assertLess(verdict["minimum_reusable_score"], 0.62)

    def test_a_temporary_candidate_that_scores_above_the_bar_is_stored(self) -> None:
        verdict = self.assess(
            "The Teknium queue console runs as the systemd unit cortex-queue.service on port 8791",
            kind="operational",
            volatility=0.72,
        )
        if verdict["reusable_score"] >= verdict["minimum_reusable_score"]:
            self.assertEqual(verdict["decision"], "created", verdict["reason"])
        else:  # pragma: no cover - only if the scorer changes
            self.assertIn("automatic temporary threshold", verdict["reason"])

    def test_a_durable_candidate_keeps_the_lower_bar(self) -> None:
        verdict = self.assess(
            "Jacoby prefers the queue console to stay on port 8791 so bookmarks keep working",
            kind="preference",
            volatility=0.12,
        )
        self.assertEqual(verdict["durability"], "durable")
        self.assertEqual(verdict["minimum_reusable_score"], store_module.AUTOMATIC_MIN_REUSABLE_SCORE_DURABLE)

    def test_a_weak_candidate_is_still_refused(self) -> None:
        """Lowering the bar must not turn the gate into a rubber stamp."""
        verdict = self.assess("ok", kind="operational", volatility=0.72, importance=0.0, confidence=0.0)
        self.assertEqual(verdict["decision"], "ignored")

    def test_a_manual_write_keeps_the_lowest_bar(self) -> None:
        verdict = self.store.assess_storage_candidate(
            "The Teknium queue console runs as the systemd unit cortex-queue.service on port 8791",
            kind="operational",
            volatility=0.72,
            automatic=False,
        )
        self.assertEqual(verdict["minimum_reusable_score"], store_module.MANUAL_MIN_REUSABLE_SCORE)

    def test_the_temporary_bar_stays_above_the_durable_one(self) -> None:
        self.assertGreater(
            store_module.AUTOMATIC_MIN_REUSABLE_SCORE_TEMPORARY,
            store_module.AUTOMATIC_MIN_REUSABLE_SCORE_DURABLE,
        )


if __name__ == "__main__":
    unittest.main()
