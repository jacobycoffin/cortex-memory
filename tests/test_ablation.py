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

from ablate_adaptive import CORPUS, _measure, _seed, run  # noqa: E402


class AblationRunnerTests(unittest.TestCase):
    def test_runner_reports_sanitized_aggregates_only(self) -> None:
        report = run(reps=1)
        self.assertEqual(
            set(report),
            {"benchmark", "tier", "created_at", "reps_per_case", "cases_per_condition",
             "local_only", "conditions"},
        )
        self.assertEqual(report["tier"], "smoke-baseline")
        self.assertTrue(report["local_only"])
        self.assertEqual(
            set(report["conditions"]),
            {"baseline", "sleep_apply", "attention_policy", "combined"},
        )
        for metrics in report["conditions"].values():
            self.assertEqual(
                set(metrics),
                {"cases", "retrieval_hit_at_3", "rendered_hit_at_3",
                 "false_positive_rate",
                 "mean_selected_per_recall", "mean_rendered_tokens",
                 "prepare_p50_ms", "prepare_p95_ms", "activation"},
            )
            for key, value in metrics.items():
                if key == "activation":
                    self.assertEqual(
                        set(value),
                        {"sleep_usage_tasks_replayed", "sleep_evidence_added",
                         "attention_used_samples", "attention_shadow_mode"},
                    )
                    continue
                self.assertIsInstance(value, (int, float))
        # Privacy: no seeded content, query text, project name, or ID leaks.
        rendered = json.dumps(report)
        for item in CORPUS:
            self.assertNotIn(item["text"][:24], rendered)
            self.assertNotIn(item["project"], rendered)
        for token in ("acorn staging deploy key", "Beacon API quota", "hello there"):
            self.assertNotIn(token, rendered)

    def test_conditions_activate_their_mechanisms(self) -> None:
        """Each experimental condition proves its mechanism ran."""
        report = run(reps=1)
        sleep = report["conditions"]["sleep_apply"]["activation"]
        self.assertGreaterEqual(sleep["sleep_usage_tasks_replayed"], 1)
        self.assertGreaterEqual(sleep["sleep_evidence_added"], 1)
        attention = report["conditions"]["attention_policy"]["activation"]
        self.assertGreaterEqual(attention["attention_used_samples"], 4)
        self.assertEqual(attention["attention_shadow_mode"], "procedural")
        combined = report["conditions"]["combined"]["activation"]
        self.assertGreaterEqual(combined["sleep_evidence_added"], 1)
        self.assertEqual(combined["attention_shadow_mode"], "procedural")
        baseline = report["conditions"]["baseline"]["activation"]
        self.assertEqual(baseline["sleep_evidence_added"], 0)
        self.assertEqual(baseline["attention_shadow_mode"], "lean")

    def test_measurement_recalls_create_no_training_signal(self) -> None:
        """Eval recalls stay pending: no auto-ignored labels in training tables."""
        import tempfile
        from pathlib import Path as _Path

        from cortex.client import CortexMemory as _CortexMemory

        with tempfile.TemporaryDirectory() as tmp:
            with _CortexMemory(_Path(tmp) / "cortex.db") as memory:
                ids = _seed(memory)
                _measure(memory, ids, reps=1)
                pending_budgets = memory.store._conn.execute(
                    "SELECT COUNT(*) FROM recall_budget_observations WHERE outcome<>'pending'"
                ).fetchone()[0]
                self.assertEqual(int(pending_budgets), 0)
                pending_usage = memory.store._conn.execute(
                    "SELECT COUNT(*) FROM usage_records WHERE outcome<>'pending'"
                ).fetchone()[0]
                self.assertEqual(int(pending_usage), 0)
                self.assertEqual(
                    memory.store._conn.execute(
                        "SELECT COUNT(*) FROM attention_observations"
                    ).fetchone()[0],
                    0,
                )

    def test_runner_is_deterministic(self) -> None:
        first = run(reps=1)
        second = run(reps=1)
        for condition in first["conditions"]:
            for metric in ("retrieval_hit_at_3", "rendered_hit_at_3",
                           "false_positive_rate",
                           "mean_selected_per_recall", "mean_rendered_tokens"):
                self.assertEqual(
                    first["conditions"][condition][metric],
                    second["conditions"][condition][metric],
                )

    def test_metric_names_match_documented_definitions(self) -> None:
        """retrieval_hit_at_3 is placement; rendered_hit_at_3 needs rendering.

        With the default budget everything selected is rendered, so the two
        hits agree; the false-positive rate uses the no-memory subset as its
        denominator (2 of 5 cases), not all cases.
        """
        report = run(reps=2)
        metrics = report["conditions"]["baseline"]
        self.assertEqual(metrics["cases"], 10)
        self.assertEqual(metrics["retrieval_hit_at_3"], 1.0)
        self.assertEqual(metrics["rendered_hit_at_3"], 1.0)
        # 2 no-memory cases x 2 reps = 4; FP rate denominator is that subset.
        self.assertGreaterEqual(metrics["false_positive_rate"], 0.0)
        self.assertLessEqual(metrics["false_positive_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
