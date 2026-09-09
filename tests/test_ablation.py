"""Tests for the local adaptive-feature ablation runner."""

from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401

sys.path.insert(0, str(ROOT / "scripts"))

from ablate_adaptive import CORPUS, run  # noqa: E402


class AblationRunnerTests(unittest.TestCase):
    def test_runner_reports_sanitized_aggregates_only(self) -> None:
        report = run(reps=1)
        self.assertEqual(
            set(report),
            {"benchmark", "created_at", "reps_per_case", "cases_per_condition",
             "local_only", "conditions"},
        )
        self.assertTrue(report["local_only"])
        self.assertEqual(
            set(report["conditions"]),
            {"baseline", "sleep_apply", "attention_policy", "combined"},
        )
        for metrics in report["conditions"].values():
            self.assertEqual(
                set(metrics),
                {"cases", "accuracy", "irrelevant_recall_rate",
                 "mean_selected_per_recall", "mean_rendered_tokens",
                 "prepare_p50_ms", "prepare_p95_ms"},
            )
            for value in metrics.values():
                self.assertIsInstance(value, (int, float))
        # Privacy: no seeded content, query text, project name, or ID leaks.
        rendered = json.dumps(report)
        for item in CORPUS:
            self.assertNotIn(item["text"][:24], rendered)
            self.assertNotIn(item["project"], rendered)
        for token in ("acorn staging deploy key", "Beacon API quota", "hello there"):
            self.assertNotIn(token, rendered)

    def test_runner_is_deterministic(self) -> None:
        first = run(reps=1)
        second = run(reps=1)
        for condition in first["conditions"]:
            for metric in ("accuracy", "irrelevant_recall_rate",
                           "mean_selected_per_recall", "mean_rendered_tokens"):
                self.assertEqual(
                    first["conditions"][condition][metric],
                    second["conditions"][condition][metric],
                )


if __name__ == "__main__":
    unittest.main()
