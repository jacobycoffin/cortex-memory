"""Only a strong model's 'yes' may write a memory (#26)."""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import model_gate


class ClassifyTests(unittest.TestCase):
    def test_a_strong_family_is_recognised(self) -> None:
        for model in ("gpt-5.1", "gpt-6.1-sol", "claude-opus-4.5", "deepseek-v4.1-flash",
                      "gemini-3-pro", "claude-sonnet-4"):
            self.assertEqual(model_gate.classify(model), "strong", model)

    def test_a_cheap_family_is_weak(self) -> None:
        for model in ("gpt-4o-mini", "claude-haiku-3", "hy4-flash-free", "deepseek-chat",
                      "gemini-flash-lite"):
            self.assertEqual(model_gate.classify(model), "weak", model)

    def test_an_unfamiliar_name_is_unknown_and_allowed_by_default(self) -> None:
        """Self-hosted and synthetic models are real; a gate that blocks anything unfamiliar
        would be switched off within a week."""
        self.assertEqual(model_gate.classify("synthetic-memory-judge"), "unknown")
        self.assertTrue(model_gate.may_auto_write("synthetic-memory-judge"))

    def test_strict_mode_turns_unknown_into_a_refusal(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_STRONG_MODELS_STRICT": "1"}):
            self.assertFalse(model_gate.may_auto_write("synthetic-memory-judge"))
            self.assertTrue(model_gate.may_auto_write("gpt-6.1-sol"))

    def test_matching_is_case_insensitive(self) -> None:
        self.assertEqual(model_gate.classify("GPT-4O-MINI"), "weak")
        self.assertEqual(model_gate.classify("Claude-Opus-4.5"), "strong")

    def test_a_version_bump_does_not_change_the_verdict(self) -> None:
        self.assertEqual(model_gate.classify("gpt-4o-mini-2026"), "weak")
        self.assertTrue(model_gate.may_auto_write("gpt-6.2"))

    def test_the_deny_list_is_configurable(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_WEAK_MODELS": "my-cheap-model"}):
            self.assertFalse(model_gate.may_auto_write("my-cheap-model"))
            self.assertTrue(model_gate.may_auto_write("gpt-4o-mini"))


class GateTests(unittest.TestCase):
    def test_a_strong_model_keeps_its_write(self) -> None:
        action, reason = model_gate.gate_action("gpt-6.1-sol", "remember")
        self.assertEqual(action, "remember")
        self.assertEqual(reason, "")

    def test_a_cheap_model_has_its_write_withheld(self) -> None:
        action, reason = model_gate.gate_action("gpt-4o-mini", "remember")
        self.assertEqual(action, "defer")
        self.assertIn("withheld", reason)
        self.assertIn("gpt-4o-mini", reason)

    def test_a_cheap_model_may_still_refuse(self) -> None:
        """Saying 'no' is always safe; only the 'yes' needs a strong model."""
        for action in ("reject", "needs_context", "defer"):
            gated, reason = model_gate.gate_action("gpt-4o-mini", action)
            self.assertEqual(gated, action)
            self.assertEqual(reason, "")

    def test_every_write_action_is_gated(self) -> None:
        for action in sorted(model_gate.WRITE_ACTIONS):
            gated, _reason = model_gate.gate_action("hy4-flash-free", action)
            self.assertEqual(gated, "defer", action)

    def test_an_unset_model_keeps_working_by_default(self) -> None:
        gated, _reason = model_gate.gate_action(None, "remember")
        self.assertEqual(gated, "remember")

    def test_the_gate_can_be_switched_off(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_STRONG_MODEL_GATE": "0"}):
            self.assertFalse(model_gate.enabled())


class WiringTests(unittest.TestCase):
    def test_the_auto_review_path_consults_the_gate(self) -> None:
        """The gate only matters if the automatic path calls it before writing."""
        source = Path("autojudge.py").read_text()
        self.assertIn("model_gate.gate_action", source)
        self.assertLess(
            source.index("model_gate.gate_action"),
            source.index("result = store.review_memory_creation"),
            "the gate must run before the write",
        )

    def test_a_withheld_write_is_counted_as_a_deferral(self) -> None:
        source = Path("autojudge.py").read_text()
        block = source[source.index("model_gate.gate_action"):]
        self.assertRegex(block[:900], re.compile(r'report\["deferred"\]'))

    def test_the_gate_reports_its_configuration(self) -> None:
        described = model_gate.describe()
        self.assertTrue(described["enabled"])
        self.assertIn("gpt-4o-mini", described["weak_prefixes"])


class BehaviourTests(unittest.TestCase):
    """The real path: a cheap model's 'remember' must not create a memory."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        from cortex.store import CortexStore

        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _run(self, model: str):
        import json

        from cortex.autojudge import AutoJudge, AutoJudgeConfig

        proposal = self.store.propose_memory_creation(
            "Project Acorn deployments require a verified backup checklist.",
            kind="procedure",
            source_type="user_turn",
            source_category="USER_STATED",
            session_id="session-a",
            confidence=0.9,
            importance=0.87,
        )

        def provider_call(endpoint, api_key, payload, timeout):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decisions": [
                                        {
                                            "proposal_id": proposal["proposal_id"],
                                            "action": "remember",
                                            "confidence": 0.95,
                                            "reason": "stable procedure",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

        import dataclasses

        config = dataclasses.replace(
            AutoJudgeConfig.from_env(),
            enabled=True,
            endpoint="https://api.example.com/v1/chat/completions",
            model=model,
            api_key_env="SYNTHETIC_JUDGE_KEY",
            minimum_age_seconds=0,
            keep_threshold=0.80,
        )
        with mock.patch.dict(os.environ, {"SYNTHETIC_JUDGE_KEY": "synthetic-key"}):
            report = AutoJudge(config, provider_call=provider_call).run(self.store)
        return report, self.store.get_memory_creation_proposal(proposal["proposal_id"])

    def test_a_cheap_model_cannot_write_a_memory(self) -> None:
        report, decided = self._run("gpt-4o-mini")
        self.assertEqual(report.get("remembered", 0), 0)
        self.assertGreaterEqual(report["deferred"], 1)
        self.assertNotEqual(decided["status"], "remembered")

    def test_a_strong_model_still_can(self) -> None:
        report, decided = self._run("gpt-6.1-sol")
        self.assertEqual(report.get("remembered", 0), 1)
        self.assertEqual(decided["status"], "remembered")


if __name__ == "__main__":
    unittest.main()
