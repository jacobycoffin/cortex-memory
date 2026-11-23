"""Calibration per kind: a 0.9 is not worth the same everywhere (#53)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import calibration
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, kind: str = "semantic", confidence: float = 0.9,
               _seen: int = 1) -> str:
        memory_id = self.store.add_memory(
            content=content + (" " * _seen), kind=kind, source_type="user_turn",
            source_category="USER_EXPLICIT", confidence=confidence,
        )[0]
        return memory_id

    def use(self, memory_id: str) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO access_log (memory_id, event, created_at) VALUES (?, 'used', ?)",
                (memory_id, NOW.isoformat()),
            )

    def correct(self, memory_id: str, new_content: str = "The corrected fact.") -> None:
        from cortex import self_correction

        self_correction.intercept(self.store, "I meant " + new_content, target_id=memory_id,
                                  at=NOW)

    def judged_population(self, kind: str, *, right: int, wrong: int, confidence: float = 0.9):
        for index in range(right):
            self.use(self.memory(f"{kind} fact {index} that held up.", kind=kind,
                                 confidence=confidence, _seen=index + 2))
        for index in range(wrong):
            memory_id = self.memory(f"{kind} fact {index} that was wrong.", kind=kind,
                                    confidence=confidence, _seen=index + 2)
            self.correct(memory_id)

    def test_a_used_fact_counts_as_right(self) -> None:
        memory_id = self.memory("Identity fact one.", kind="identity")
        self.use(memory_id)
        measured = {item.kind: item for item in calibration.measure(self.store)}
        self.assertEqual(measured["identity"].right, 1)

    def test_a_corrected_fact_counts_as_wrong(self) -> None:
        memory_id = self.memory("Identity fact one.", kind="identity")
        self.correct(memory_id)
        measured = {item.kind: item for item in calibration.measure(self.store)}
        self.assertEqual(measured["identity"].wrong, 1)

    def test_an_unjudged_fact_is_excluded(self) -> None:
        self.memory("Never used, never corrected.", kind="semantic")
        described = calibration.report(self.store)
        self.assertEqual(described["judged"], 0)

    def test_the_brier_score_is_computed_over_judged_facts_only(self) -> None:
        self.judged_population("identity", right=3, wrong=1, confidence=1.0)
        measured = {item.kind: item for item in calibration.measure(self.store)}["identity"]
        # (3 x 0^2 + 1 x 1^2) / 4 = 0.25
        self.assertAlmostEqual(measured.brier, 0.25, places=6)
        self.assertAlmostEqual(measured.observed_accuracy, 0.75, places=6)

    def test_a_kind_with_too_little_evidence_gets_no_ceiling(self) -> None:
        self.judged_population("identity", right=2, wrong=1)
        measured = {item.kind: item for item in calibration.measure(self.store)}["identity"]
        self.assertIsNone(measured.ceiling)
        self.assertIn("judged fact", measured.note)

    def test_a_kind_with_no_negatives_gets_no_ceiling(self) -> None:
        """With no wrong facts, accuracy is perfect by construction and the ceiling is a tautology."""
        self.judged_population("identity", right=25, wrong=0, confidence=0.9)
        measured = {item.kind: item for item in calibration.measure(self.store)}["identity"]
        self.assertIsNone(measured.ceiling)
        self.assertIn("too few negatives", measured.note)

    def test_a_kind_with_enough_evidence_gets_a_ceiling(self) -> None:
        self.judged_population("operational", right=12, wrong=8, confidence=0.9)
        measured = {item.kind: item for item in calibration.measure(self.store)}["operational"]
        self.assertIsNotNone(measured.ceiling)
        self.assertAlmostEqual(measured.ceiling, 0.6, places=6)

    def test_a_stated_confidence_above_the_ceiling_is_overconfident(self) -> None:
        self.judged_population("operational", right=12, wrong=8, confidence=0.9)
        measured = {item.kind: item for item in calibration.measure(self.store)}["operational"]
        self.assertTrue(measured.overconfident)
        self.assertIn("exceeds", measured.note)

    def test_a_well_calibrated_kind_is_not_overconfident(self) -> None:
        self.judged_population("identity", right=19, wrong=1, confidence=0.95)
        measured = {item.kind: item for item in calibration.measure(self.store)}["identity"]
        self.assertFalse(measured.overconfident)

    def test_kinds_are_measured_separately(self) -> None:
        self.judged_population("identity", right=19, wrong=1, confidence=0.95)
        self.judged_population("operational", right=10, wrong=10, confidence=0.9)
        measured = {item.kind: item for item in calibration.measure(self.store)}
        self.assertGreater(measured["identity"].observed_accuracy,
                           measured["operational"].observed_accuracy)

    def test_apply_clamps_only_downward(self) -> None:
        self.judged_population("operational", right=12, wrong=8, confidence=0.9)
        outcome = calibration.apply(self.store, at=NOW)
        self.assertGreater(outcome["changed"], 0)
        ceiling = calibration.measure(self.store)
        operational = [item for item in ceiling if item.kind == "operational"][0]
        self.assertLessEqual(operational.mean_confidence, operational.ceiling + 1e-9)

    def test_apply_is_idempotent(self) -> None:
        self.judged_population("operational", right=12, wrong=8, confidence=0.9)
        calibration.apply(self.store, at=NOW)
        self.assertEqual(calibration.apply(self.store, at=NOW)["changed"], 0)

    def test_apply_never_raises_a_confidence(self) -> None:
        memory_id = self.memory("A modestly stated fact.", kind="identity", confidence=0.2)
        self.use(memory_id)
        calibration.apply(self.store, at=NOW)
        stored = float(self.store._conn.execute(
            "SELECT confidence FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()[0])
        self.assertEqual(stored, 0.2)

    def test_the_report_leads_with_coverage(self) -> None:
        self.memory("Unjudged one.", _seen=3)
        self.judged_population("identity", right=2, wrong=0)
        described = calibration.report(self.store)
        self.assertLess(described["judged_share"], 1.0)
        self.assertIn("excluded", described["note"])

    def test_a_calibration_describes_itself(self) -> None:
        self.judged_population("operational", right=12, wrong=8, confidence=0.9)
        measured = {item.kind: item for item in calibration.measure(self.store)}["operational"]
        self.assertIn("ceiling", measured.describe())

    def test_the_round_trip(self) -> None:
        self.judged_population("identity", right=2, wrong=1)
        payload = calibration.measure(self.store)[0].as_dict()
        for key in ("kind", "judged", "right", "wrong", "brier", "ceiling", "overconfident"):
            self.assertIn(key, payload)

    def test_an_empty_store_is_quiet(self) -> None:
        described = calibration.report(self.store)
        self.assertEqual(described["kinds"], [])
        self.assertEqual(described["judged"], 0)
        self.assertEqual(calibration.apply(self.store)["changed"], 0)


if __name__ == "__main__":
    unittest.main()
