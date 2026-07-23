from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401

from cortex.adaptive_weights import run_adaptive_weight_learning
from cortex.autojudge import AutoJudgeConfig
from cortex.retrieval import MemoryRetriever, RetrievalContext, SCORE_SIGNAL_WEIGHTS
from cortex.store import CortexStore


class AdaptiveWeightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.memory_id, _ = self.store.add_memory(
            "Use the verified blue deployment gateway."
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    @staticmethod
    def config() -> AutoJudgeConfig:
        return AutoJudgeConfig(
            enabled=True,
            endpoint="http://127.0.0.1:9999/v1/chat/completions",
            model="synthetic-weight-judge",
            api_key_env="",
            credential_file=None,
            timeout_seconds=5.0,
            max_output_tokens=1200,
        )

    def observation(self, index: int, *, used: bool, task_type: str = "deployment") -> None:
        task_id = self.store.create_usage_batch(
            [(self.memory_id, 0.8)],
            query="Which deployment gateway is verified?",
            session_id=f"weights-{index}",
            task_type=task_type,
            recall_mode="focused",
        )
        self.store.record_memory_trace_decision(
            task_id=task_id,
            session_id=f"weights-{index}",
            goal="Choose the deployment gateway",
            context_summary=f"task_type={task_type}",
            task_type=task_type,
            recall_mode="focused",
            retrieval_used=True,
            retrieval_reason="selected for weight evidence",
            queries=["Which deployment gateway is verified?"],
            candidate_memories=[
                {
                    "memory_id": self.memory_id,
                    "kind": "semantic",
                    "selected": True,
                    "score": 0.8,
                    "components": {
                        **{signal: 0.4 for signal in SCORE_SIGNAL_WEIGHTS},
                        "lexical": 0.9 if used else 0.2,
                    },
                    "reason": "selected",
                }
            ],
        )
        self.store.resolve_usage(task_id, {self.memory_id: 1.0} if used else {})

    def staged_proposal(self, *, delta: float = 0.01) -> dict:
        for index in range(8):
            self.observation(index, used=index < 6)

        def provider(_endpoint, _key, payload, _timeout):
            request = json.loads(payload["messages"][1]["content"])
            task = request["task_types"][0]
            weights = dict(task["current_weights"])
            weights["lexical"] += delta
            weights["activation"] -= delta
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "proposals": [
                                        {
                                            "task_type": task["task_type"],
                                            "weights": weights,
                                            "confidence": 0.88,
                                            "reason": "Lexical evidence separates used selections.",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 12},
            }

        report = run_adaptive_weight_learning(
            self.store,
            self.config(),
            provider_call=provider,
        )
        return {
            "report": report,
            "proposal": self.store.scoring_weight_snapshot()["proposals"][0],
        }

    def test_tuning_is_shadow_only_and_records_outcome_evidence(self) -> None:
        result = self.staged_proposal()

        self.assertEqual(result["report"]["proposals"], 1)
        self.assertEqual(result["proposal"]["status"], "proposed")
        self.assertEqual(result["proposal"]["evidence"]["resolved"], 8)
        self.assertEqual(
            self.store.active_scoring_weights("deployment")["source"],
            "factory_defaults",
        )

    def test_explicit_apply_versions_retrieval_and_rollback_restores_baseline(self) -> None:
        result = self.staged_proposal()
        proposal_id = result["proposal"]["proposal_id"]

        applied = self.store.apply_scoring_weight_proposal(
            proposal_id,
            actor="test-operator",
        )
        context = RetrievalContext(scope={"task_type": "deployment"})
        _selected, diagnostics = MemoryRetriever(self.store, threshold=0.0).search_detailed(
            "verified blue deployment gateway",
            limit=3,
            context=context,
        )

        self.assertEqual(applied["status"], "approved")
        self.assertTrue(
            all(
                item["scoring_policy_version"].startswith("adaptive_weights:")
                for item in diagnostics.candidate_decisions
            )
        )
        self.assertAlmostEqual(
            self.store.active_scoring_weights("deployment")["weights"]["lexical"],
            SCORE_SIGNAL_WEIGHTS["lexical"] + 0.01,
        )
        self.assertTrue(self.store.audit()["ok"])
        rolled_back = self.store.rollback_scoring_weights(
            proposal_id,
            reason="synthetic rollback",
        )
        self.assertEqual(rolled_back["status"], "rolled_back")
        self.assertEqual(
            self.store.active_scoring_weights("deployment")["weights"],
            SCORE_SIGNAL_WEIGHTS,
        )

    def test_large_change_requires_separate_confirmation(self) -> None:
        result = self.staged_proposal(delta=0.05)
        proposal_id = result["proposal"]["proposal_id"]

        with self.assertRaisesRegex(ValueError, "explicit confirmation"):
            self.store.apply_scoring_weight_proposal(proposal_id)
        applied = self.store.apply_scoring_weight_proposal(
            proposal_id,
            confirm_large_change=True,
        )

        self.assertEqual(applied["status"], "approved")
        self.assertTrue(result["proposal"]["explicit_confirmation_required"])

    def test_invalid_provider_weight_set_fails_closed(self) -> None:
        for index in range(8):
            self.observation(index, used=True)

        def provider(_endpoint, _key, payload, _timeout):
            task = json.loads(payload["messages"][1]["content"])["task_types"][0]
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "proposals": [
                                        {
                                            "task_type": task["task_type"],
                                            "weights": {"lexical": 0.9},
                                            "confidence": 0.9,
                                            "reason": "Unsafe incomplete profile.",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        with self.assertRaisesRegex(Exception, "every scoring signal"):
            run_adaptive_weight_learning(
                self.store,
                self.config(),
                provider_call=provider,
            )
        self.assertEqual(self.store.scoring_weight_snapshot()["proposals"], [])


if __name__ == "__main__":
    unittest.main()
