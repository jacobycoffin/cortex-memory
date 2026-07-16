from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT

from cortex.metacognition import assess_retrieval
from cortex.retrieval import MemoryRetriever, RetrievalResult
from cortex.store import CortexStore


class MetacognitiveAssessmentTests(unittest.TestCase):
    def test_new_source_priors_preserve_origin_and_do_not_equate_approval_with_truth(self) -> None:
        def source_prior(source_category: str, *, origin: str | None = None) -> float:
            memory = {
                "id": f"source-{source_category}-{origin or 'none'}",
                "source_category": source_category,
                "confidence": 0.65,
                "trust": 0.65,
                "success_count": 0,
                "confirmed_count": 0,
                "helpful_count": 0,
                "validated_count": 0,
                "harmful_count": 0,
                "false_positive_count": 0,
                "dirty": 0,
                "state": "active",
            }
            if origin:
                memory["origin_source_category"] = origin
            result = RetrievalResult(
                memory=memory,
                score=0.5,
                components={
                    "lexical": 0.5,
                    "phrase": 0.0,
                    "semantic": 0.0,
                    "currentness": 0.7,
                    "utility": 0.5,
                    "stale_risk": 0.0,
                    "wrong_rate": 0.0,
                    "superseded": 0.0,
                },
                estimated_tokens=20,
            )
            return float(assess_retrieval(result).features["source_prior"])

        self.assertGreater(source_prior("USER_STATED"), source_prior("OPERATOR_APPROVED"))
        self.assertGreater(source_prior("OPERATOR_APPROVED"), source_prior("AGENT_PROPOSED"))
        self.assertEqual(
            source_prior("OPERATOR_APPROVED", origin="AGENT_PROPOSED"),
            source_prior("AGENT_PROPOSED"),
        )

    def test_source_monitoring_is_separate_from_retrieval_score(self) -> None:
        memory = {
            "id": "verified-memory",
            "source_category": "TOOL_VERIFIED",
            "confidence": 0.95,
            "trust": 0.95,
            "success_count": 3,
            "confirmed_count": 0,
            "helpful_count": 3,
            "validated_count": 1,
            "harmful_count": 0,
            "false_positive_count": 0,
            "dirty": 0,
            "state": "active",
        }
        result = RetrievalResult(
            memory=memory,
            score=0.55,
            components={
                "lexical": 0.85,
                "phrase": 0.0,
                "semantic": 0.4,
                "currentness": 0.9,
                "utility": 0.85,
                "stale_risk": 0.02,
                "wrong_rate": 0.0,
                "superseded": 0.0,
            },
            estimated_tokens=30,
        )

        assessment = assess_retrieval(result)

        self.assertEqual(assessment.decision, "use")
        self.assertGreaterEqual(assessment.calibrated_probability, 0.70)
        self.assertIn("source provenance", assessment.reason)

    def test_dirty_inferred_memory_is_flagged_for_abstention(self) -> None:
        memory = {
            "id": "weak-memory",
            "source_category": "AGENT_INFERENCE",
            "confidence": 0.35,
            "trust": 0.30,
            "success_count": 0,
            "confirmed_count": 0,
            "helpful_count": 0,
            "validated_count": 0,
            "harmful_count": 2,
            "false_positive_count": 2,
            "dirty": 1,
            "state": "cold",
        }
        result = RetrievalResult(
            memory=memory,
            score=0.72,
            components={
                "lexical": 0.55,
                "phrase": 0.0,
                "semantic": 0.2,
                "currentness": 0.25,
                "utility": 0.1,
                "stale_risk": 0.8,
                "wrong_rate": 0.7,
                "superseded": 0.0,
            },
            estimated_tokens=30,
        )

        assessment = assess_retrieval(result)

        self.assertEqual(assessment.decision, "abstain")
        self.assertLess(assessment.calibrated_probability, 0.48)
        self.assertIn("dependent evidence changed", assessment.reason)


class MetacognitiveOutcomeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.memory_id, _ = self.store.add_memory(
            "The verified deployment channel is testing.",
            source_category="TOOL_VERIFIED",
            confidence=0.9,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _assessment(self, probability: float = 0.6) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "source_category": "TOOL_VERIFIED",
            "raw_probability": probability,
            "calibrated_probability": probability,
            "decision": "verify",
            "reason": "Test assessment.",
            "calibration_scope": "prior",
            "calibration_samples": 0,
            "features": {"source_prior": 0.92},
        }

    def _labeled_prediction(self, outcome: str) -> None:
        task_id = self.store.create_usage_batch(
            [(self.memory_id, 0.8)],
            query="Which deployment channel?",
            session_id="metacognition-test",
            task_type="deployment",
            recall_mode="focused",
            requested_budget=300,
            estimated_tokens=40,
            metacognitive_assessments=[self._assessment()],
            metacognition_mode="shadow",
        )
        self.store.resolve_usage(task_id, {self.memory_id: 1.0})
        self.store.apply_task_outcome(task_id, outcome)

    def test_prediction_follows_usage_and_outcome_lifecycle(self) -> None:
        self._labeled_prediction("helpful")

        row = self.store._conn.execute(
            "SELECT outcome,decision,monitor_mode,applied FROM metacognitive_predictions"
        ).fetchone()
        self.assertEqual(dict(row), {"outcome": "helpful", "decision": "verify", "monitor_mode": "shadow", "applied": 0})
        snapshot = self.store.dashboard_snapshot()["metacognition"]
        self.assertEqual(snapshot["summary"]["prediction_count"], 1)
        self.assertEqual(snapshot["summary"]["labeled_count"], 1)
        self.assertEqual(snapshot["summary"]["positive_count"], 1)
        self.assertEqual(len(snapshot["calibration_bins"]), 1)

    def test_calibration_waits_for_outcomes_then_moves_conservatively(self) -> None:
        before = self.store.calibrate_metacognitive_probability(
            0.6,
            task_type="deployment",
            source_category="TOOL_VERIFIED",
            min_samples=4,
        )
        self.assertEqual(before["scope"], "prior")
        self.assertEqual(before["probability"], 0.6)

        for _index in range(4):
            self._labeled_prediction("helpful")

        after = self.store.calibrate_metacognitive_probability(
            0.6,
            task_type="deployment",
            source_category="TOOL_VERIFIED",
            min_samples=4,
        )
        self.assertEqual(after["scope"], "task_source")
        self.assertEqual(after["sample_count"], 4)
        self.assertGreater(after["probability"], 0.6)
        self.assertLess(after["probability"], 0.75)

    def test_external_recall_records_shadow_assessments(self) -> None:
        result = MemoryRetriever(self.store, threshold=0.0).search(
            "Which verified deployment channel is testing?", limit=1
        )[0]
        prior = assess_retrieval(result)
        task_id = self.store.create_usage_batch(
            [(self.memory_id, result.score)],
            query="Which verified deployment channel is testing?",
            session_id="external",
            task_type="deployment",
            recall_mode="external_adapter",
            metacognitive_assessments=[prior.as_record()],
        )
        row = self.store._conn.execute(
            "SELECT task_id,decision,outcome FROM metacognitive_predictions WHERE task_id=?",
            (task_id,),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertIn(row["decision"], {"use", "verify", "abstain"})
        self.assertEqual(row["outcome"], "pending")


if __name__ == "__main__":
    unittest.main()
