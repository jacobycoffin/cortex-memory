from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT  # noqa: F401

from cortex.autojudge import AutoJudgeConfig
from cortex.brain_mechanics import brain_mechanics_config, run_due_brain_mechanics
from cortex.store import CortexStore


class BrainMechanicsSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.config = AutoJudgeConfig(
            enabled=True,
            endpoint="http://127.0.0.1:9999/v1/chat/completions",
            model="synthetic-mechanics",
            api_key_env="",
            credential_file=None,
            timeout_seconds=5.0,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_every_mechanic_is_opt_in_by_default(self) -> None:
        names = [
            "CORTEX_AUTO_JUDGE_RECONSOLIDATE",
            "CORTEX_AUTO_JUDGE_PRUNE",
            "CORTEX_AUTO_JUDGE_CONSOLIDATE",
            "CORTEX_AUTO_JUDGE_SCHEMAS",
            "CORTEX_AUTO_JUDGE_TUNE_WEIGHTS",
        ]
        with patch.dict(os.environ, {name: "false" for name in names}, clear=False):
            report = run_due_brain_mechanics(self.store, self.config)

        self.assertTrue(
            all(item["status"] == "disabled" for item in report["passes"].values())
        )

    def test_separate_model_override_is_bounded_to_mechanics(self) -> None:
        with patch.dict(
            os.environ,
            {"CORTEX_BRAIN_MECHANICS_MODEL": "grok-4.5"},
            clear=False,
        ):
            overridden = brain_mechanics_config(self.config)

        self.assertEqual(overridden.model, "grok-4.5")
        self.assertEqual(self.config.model, "synthetic-mechanics")

    def test_due_pass_is_reserved_once_and_always_stays_shadow(self) -> None:
        env = {
            "CORTEX_AUTO_JUDGE_RECONSOLIDATE": "false",
            "CORTEX_AUTO_JUDGE_PRUNE": "true",
            "CORTEX_AUTO_JUDGE_CONSOLIDATE": "false",
            "CORTEX_AUTO_JUDGE_SCHEMAS": "false",
            "CORTEX_AUTO_JUDGE_TUNE_WEIGHTS": "false",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch(
                "cortex.brain_mechanics.run_adaptive_pruning",
                return_value={"mode": "shadow", "proposals": 1},
            ) as pruning,
        ):
            first = run_due_brain_mechanics(self.store, self.config)
            second = run_due_brain_mechanics(self.store, self.config)

        self.assertEqual(first["passes"]["pruning"]["status"], "completed")
        self.assertEqual(first["passes"]["pruning"]["result"]["mode"], "shadow")
        self.assertEqual(second["passes"]["pruning"]["status"], "not_due")
        self.assertEqual(pruning.call_count, 1)
        self.assertFalse(pruning.call_args.kwargs["apply"])

    def test_schema_scheduler_waits_for_a_completed_sleep_cycle(self) -> None:
        env = {
            "CORTEX_AUTO_JUDGE_RECONSOLIDATE": "false",
            "CORTEX_AUTO_JUDGE_PRUNE": "false",
            "CORTEX_AUTO_JUDGE_CONSOLIDATE": "false",
            "CORTEX_AUTO_JUDGE_SCHEMAS": "true",
            "CORTEX_AUTO_JUDGE_TUNE_WEIGHTS": "false",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch("cortex.brain_mechanics.run_schema_formation") as schemas,
        ):
            report = run_due_brain_mechanics(self.store, self.config)

        self.assertEqual(report["passes"]["schemas"]["status"], "awaiting_sleep")
        schemas.assert_not_called()

    def test_failed_pass_releases_claim_for_next_timer_tick(self) -> None:
        env = {
            "CORTEX_AUTO_JUDGE_RECONSOLIDATE": "false",
            "CORTEX_AUTO_JUDGE_PRUNE": "false",
            "CORTEX_AUTO_JUDGE_CONSOLIDATE": "true",
            "CORTEX_AUTO_JUDGE_SCHEMAS": "false",
            "CORTEX_AUTO_JUDGE_TUNE_WEIGHTS": "false",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch(
                "cortex.brain_mechanics.run_semantic_consolidation",
                side_effect=[RuntimeError("synthetic failure"), {"mode": "shadow"}],
            ) as consolidation,
        ):
            first = run_due_brain_mechanics(self.store, self.config)
            second = run_due_brain_mechanics(self.store, self.config)

        self.assertEqual(first["passes"]["consolidation"]["status"], "failed")
        self.assertEqual(second["passes"]["consolidation"]["status"], "completed")
        self.assertEqual(consolidation.call_count, 2)


if __name__ == "__main__":
    unittest.main()
