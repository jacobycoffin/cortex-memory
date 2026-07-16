from __future__ import annotations

import unittest
from unittest.mock import patch


from tests._bootstrap import ROOT  # noqa: F401

from cortex.review_copilot import (
    ReviewCopilot,
    ReviewCopilotConfig,
    ReviewCopilotError,
)


def connection_item(*, reinforcement: bool = False, typed: bool = False) -> dict:
    return {
        "item_type": "proposal",
        "proposal_id": "proposal-1",
        "proposal_kind": "association_reinforcement" if reinforcement else "association",
        "question": "Two replay witnesses used both memories.",
        "memories": [
            {
                "id": "memory-a",
                "display_title": "Deployment rule",
                "display_summary": "Deploy Cortex through Hermes.",
                "content": "Deploy Cortex through the Hermes host after tests pass.",
                "kind": "procedure",
                "source_category": "DOCUMENT",
                "source_ref": "runbook.md",
            },
            {
                "id": "memory-b",
                "display_title": "Verification rule",
                "display_summary": "Run tests before deployment.",
                "content": "The Cortex release requires the complete test suite.",
                "kind": "procedure",
                "source_category": "DOCUMENT",
                "source_ref": "release.md",
            },
        ],
        "connection_review": {
            "training": {"approved": 1, "denied": 2, "target": 5},
            "existing_edge": {"relation": "supports"} if typed else None,
            "evidence_summary": {"distinct_witnesses": 2, "required_witnesses": 2},
            "shared_signals": [{"label": "Project", "value": "Cortex"}],
        },
    }


class ReviewCopilotTests(unittest.TestCase):
    def config(self) -> ReviewCopilotConfig:
        return ReviewCopilotConfig(
            enabled=True,
            endpoint="http://127.0.0.1:9/v1/chat/completions",
            model="test-model",
            api_key_env="",
        )

    def test_recommendation_is_validated_and_effect_is_deterministic(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": """{
                          "mode":"recommendation",
                          "message":"I understand.",
                          "recommendation":{
                            "action_key":"approve_a_supports_b",
                            "reason_code":"wrong-value-is-ignored",
                            "decision_scope":"item_only",
                            "heard":"A explains why B matters.",
                            "rationale":"The relationship is specific to these release records.",
                            "general_rule":"",
                            "confidence":0.91,
                            "caveat":""
                          }
                        }"""
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }
        copilot = ReviewCopilot(self.config(), provider_call=lambda *_args: response)
        result = copilot.interpret(connection_item(), "A is the procedure that supports B.")
        recommendation = result["recommendation"]
        self.assertEqual(recommendation["api_action"], "approve")
        self.assertEqual(recommendation["reason_code"], "a_supports_b")
        self.assertEqual(recommendation["decision_scope"], "item_only")
        self.assertIn("A → B", recommendation["change_now"])
        self.assertEqual(result["usage"]["total_tokens"], 150)

    def test_broad_scope_requires_explicit_operator_pattern_language(self) -> None:
        def provider(*_args):
            return {
                "choices": [
                    {
                        "message": {
                            "content": """{
                              "mode":"recommendation",
                              "message":"This sounds broad.",
                              "recommendation":{
                                "action_key":"deny",
                                "reason_code":"co_occurrence_only",
                                "decision_scope":"policy_evidence",
                                "heard":"Reject co-occurrence-only links.",
                                "rationale":"Shared appearance is not meaning.",
                                "general_rule":"Similar memories should not be linked from co-occurrence alone.",
                                "confidence":0.88,
                                "caveat":""
                              }
                            }"""
                        }
                    }
                ]
            }

        copilot = ReviewCopilot(self.config(), provider_call=provider)
        one_off = copilot.interpret(connection_item(), "These two should not be connected.")
        self.assertEqual(one_off["recommendation"]["decision_scope"], "item_only")
        self.assertTrue(one_off["recommendation"]["scope_adjustment"])
        broad = copilot.interpret(
            connection_item(),
            "Cortex keeps doing this whenever memories appear together; this pattern should stop.",
        )
        self.assertEqual(broad["recommendation"]["decision_scope"], "policy_evidence")
        self.assertIn("denial 3 of 5", broad["recommendation"]["teaches_cortex"])

    def test_copilot_can_ask_one_question_without_proposing_an_action(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"mode":"clarify","message":"I see two possibilities.",'
                        '"question":"Should this apply only to this release or to similar releases?"}'
                    }
                }
            ]
        }
        result = ReviewCopilot(self.config(), provider_call=lambda *_args: response).interpret(
            connection_item(), "I do not like this connection."
        )
        self.assertEqual(result["mode"], "clarify")
        self.assertIsNone(result["recommendation"])
        self.assertIn("only to this release", result["question"])

    def test_typed_reinforcement_cannot_be_recast_by_provider(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": """{
                          "mode":"recommendation","message":"Use a new type.",
                          "recommendation":{"action_key":"approve_same_subject","reason_code":"same_subject",
                          "decision_scope":"item_only","heard":"same","rationale":"same","general_rule":"",
                          "confidence":0.5,"caveat":""}}
                        """
                    }
                }
            ]
        }
        copilot = ReviewCopilot(self.config(), provider_call=lambda *_args: response)
        with self.assertRaises(ReviewCopilotError):
            copilot.interpret(connection_item(reinforcement=True, typed=True), "Strengthen this edge.")

    def test_remote_provider_is_not_reported_ready_without_credential(self) -> None:
        config = ReviewCopilotConfig(
            enabled=True,
            endpoint="https://provider.test/v1/chat/completions",
            model="test-model",
            api_key_env="CORTEX_TEST_PROVIDER_KEY",
        )
        with patch.dict("os.environ", {"CORTEX_TEST_PROVIDER_KEY": ""}):
            status = config.public_status()
        self.assertFalse(status["enabled"])
        self.assertIn("credential", status["reason"])


if __name__ == "__main__":
    unittest.main()
