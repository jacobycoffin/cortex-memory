from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401

from cortex.autojudge import AutoJudgeConfig, AutoJudgeError
from cortex.semantic_consolidation import (
    _parse_semantic_decisions,
    run_semantic_consolidation,
)
from cortex.store import CortexStore


class SemanticConsolidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    @staticmethod
    def config(**overrides) -> AutoJudgeConfig:
        values = {
            "enabled": True,
            "endpoint": "http://127.0.0.1:9999/v1/chat/completions",
            "model": "synthetic-consolidation-judge",
            "api_key_env": "",
            "credential_file": None,
            "timeout_seconds": 5.0,
            "max_output_tokens": 1200,
            "max_proposals": 12,
            "minimum_age_seconds": 0,
            "keep_threshold": 0.80,
            "decision_threshold": 0.72,
            "strong_feedback_boost": 0.08,
            "positive_feedback_boost": 0.02,
        }
        values.update(overrides)
        return AutoJudgeConfig(**values)

    def add_related_pair(self) -> tuple[str, str]:
        left, _ = self.store.add_memory(
            "Cortex nightly sleep replays durable memory evidence before maintenance.",
            kind="semantic",
            entities=("Cortex", "Sleep"),
            confidence=0.82,
        )
        right, _ = self.store.add_memory(
            "Cortex nightly sleep replays durable memory evidence and proposes maintenance changes.",
            kind="semantic",
            entities=("Cortex", "Sleep"),
            confidence=0.91,
        )
        return left, right

    def test_candidates_are_bounded_oldest_first_and_exclude_unsafe_sources(self) -> None:
        left, right = self.add_related_pair()
        self.store.add_memory(
            "Cortex nightly sleep replays durable memory evidence for this preference.",
            kind="preference",
        )
        protected, _ = self.store.add_memory(
            "Cortex nightly sleep replays durable memory evidence while pinned.",
            kind="semantic",
            pinned=True,
        )
        first = self.store.semantic_consolidation_candidates(limit=20)
        self.assertLessEqual(len(first), 5)
        self.assertEqual(first[0]["left"]["id"], left)
        self.assertEqual(first[0]["right"]["id"], right)
        observed_ids = {
            memory["id"]
            for candidate in first
            for memory in (candidate["left"], candidate["right"])
        }
        self.assertNotIn(protected, observed_ids)

    def test_shadow_run_records_proposal_without_mutating_sources(self) -> None:
        left, right = self.add_related_pair()

        def provider(_endpoint, _key, payload, _timeout):
            pairs = json.loads(payload["messages"][1]["content"])["pairs"]
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decisions": [
                                        {
                                            "pair_id": pairs[0]["pair_id"],
                                            "action": "merge",
                                            "confidence": 0.93,
                                            "reason": "Both describe the same bounded Sleep mechanism.",
                                            "merged_content": (
                                                "Cortex nightly Sleep replays durable memory evidence "
                                                "before proposing maintenance changes."
                                            ),
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 80, "completion_tokens": 40},
            }

        report = run_semantic_consolidation(
            self.store,
            self.config(),
            provider_call=provider,
        )
        self.assertEqual(report["mode"], "shadow")
        self.assertEqual(report["judged"], 1)
        self.assertEqual(report["applied"], 0)
        self.assertEqual(self.store.get_memory(left)["state"], "active")
        self.assertEqual(self.store.get_memory(right)["state"], "active")
        snapshot = self.store.semantic_consolidation_snapshot()
        self.assertEqual(snapshot["counts"]["proposed"], 1)
        self.assertIsNone(snapshot["decisions"][0]["result_memory_id"])

    def test_structured_conflicts_are_never_candidates(self) -> None:
        first, _ = self.store.add_memory(
            "The Cortex dashboard listens on port 8787.",
            kind="operational",
            subject="cortex:dashboard",
            predicate="port",
            object_value="8787",
        )
        second, _ = self.store.add_memory(
            "The Cortex dashboard listens on port 8788.",
            kind="operational",
            subject="cortex:dashboard",
            predicate="port",
            object_value="8788",
        )
        candidates = self.store.semantic_consolidation_candidates()
        candidate_ids = {
            memory["id"]
            for candidate in candidates
            for memory in (candidate["left"], candidate["right"])
        }
        self.assertNotIn(first, candidate_ids)
        self.assertNotIn(second, candidate_ids)

    def test_changed_source_makes_shadow_proposal_stale_instead_of_applying(self) -> None:
        left, right = self.add_related_pair()

        def provider(_endpoint, _key, payload, _timeout):
            pair_id = json.loads(payload["messages"][1]["content"])["pairs"][0]["pair_id"]
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decisions": [
                                        {
                                            "pair_id": pair_id,
                                            "action": "merge",
                                            "confidence": 0.94,
                                            "reason": "The records appear complementary.",
                                            "merged_content": (
                                                "Cortex nightly Sleep replays durable evidence "
                                                "and proposes reversible maintenance changes."
                                            ),
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        run_semantic_consolidation(
            self.store,
            self.config(),
            provider_call=provider,
        )
        decision_id = self.store.semantic_consolidation_snapshot()["decisions"][0]["decision_id"]
        self.store.correct_memory(
            left,
            "Cortex nightly Sleep replays independently reviewed evidence before maintenance.",
            reason="test source revision",
        )
        result = self.store.apply_semantic_consolidation(decision_id)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(self.store.get_memory(left)["state"], "active")
        self.assertEqual(self.store.get_memory(right)["state"], "active")

    def test_explicit_apply_creates_provenanced_memory_and_undo_restores_sources(self) -> None:
        left, right = self.add_related_pair()

        def provider(_endpoint, _key, payload, _timeout):
            pair_id = json.loads(payload["messages"][1]["content"])["pairs"][0]["pair_id"]
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decisions": [
                                        {
                                            "pair_id": pair_id,
                                            "action": "merge",
                                            "confidence": 0.95,
                                            "reason": "The second source adds the maintenance proposal detail.",
                                            "merged_content": (
                                                "Cortex nightly Sleep replays durable memory evidence "
                                                "before proposing reversible maintenance changes."
                                            ),
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        report = run_semantic_consolidation(
            self.store,
            self.config(),
            provider_call=provider,
            apply=True,
        )
        self.assertEqual(report["applied"], 1)
        snapshot = self.store.semantic_consolidation_snapshot()
        decision = snapshot["decisions"][0]
        result_id = decision["result_memory_id"]
        self.assertEqual(decision["status"], "applied")
        self.assertEqual(self.store.get_memory(left)["state"], "archived")
        self.assertEqual(self.store.get_memory(right)["state"], "archived")
        result = self.store.get_memory(result_id)
        self.assertEqual(result["state"], "active")
        self.assertEqual(result["source_type"], "semantic_consolidation")
        self.assertFalse(result["dirty"])
        with self.store._lock:
            dependencies = self.store._conn.execute(
                """SELECT evidence_id,relation FROM memory_dependencies
                   WHERE memory_id=? AND active=1 ORDER BY evidence_id""",
                (result_id,),
            ).fetchall()
        self.assertEqual(
            {str(row["evidence_id"]) for row in dependencies},
            {left, right},
        )
        self.assertEqual(
            {str(row["relation"]) for row in dependencies},
            {"consolidated_from"},
        )
        self.assertTrue(self.store.audit()["ok"])

        feedback = self.store.record_semantic_consolidation_feedback(
            decision["decision_id"],
            "wrong",
            reason="Human review found the merged statement too broad.",
        )
        self.assertEqual(feedback["undo"]["restored"], 2)
        self.assertEqual(self.store.get_memory(left)["state"], "active")
        self.assertEqual(self.store.get_memory(right)["state"], "active")
        self.assertEqual(self.store.get_memory(result_id)["state"], "archived")
        reviewed = self.store.semantic_consolidation_snapshot()["feedback"]
        self.assertEqual(reviewed["wrong"], 1)
        self.assertEqual(reviewed["correctness"], 0.0)

    def test_parser_rejects_boolean_confidence_and_source_copy_merge(self) -> None:
        pair_id = "left:right"
        candidates = {
            pair_id: {
                "left": {"content": "First source."},
                "right": {"content": "Second source."},
            }
        }
        boolean_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "decisions": [
                                    {
                                        "pair_id": pair_id,
                                        "action": "merge",
                                        "confidence": True,
                                        "reason": "Invalid.",
                                        "merged_content": "Combined source.",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        with self.assertRaises(AutoJudgeError):
            _parse_semantic_decisions(
                boolean_response,
                {pair_id},
                candidates=candidates,
            )

        copied_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "decisions": [
                                    {
                                        "pair_id": pair_id,
                                        "action": "merge",
                                        "confidence": 0.9,
                                        "reason": "Invalid copy.",
                                        "merged_content": "First source.",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        with self.assertRaises(AutoJudgeError):
            _parse_semantic_decisions(
                copied_response,
                {pair_id},
                candidates=candidates,
            )


if __name__ == "__main__":
    unittest.main()
