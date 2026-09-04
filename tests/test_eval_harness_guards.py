from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


from tests._bootstrap import ROOT

from cortex.store import CortexStore


def _load_script(name: str, filename: str):
    import sys

    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses resolve annotations through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


real_history = _load_script("cortex_eval_real_history", "evaluate_real_history.py")
tool_bench = _load_script("cortex_benchmark_tool_calling", "benchmark_tool_calling.py")


def _pair_observations(pair_id: str) -> list[dict]:
    rows = []
    for condition in ("default_built_in", "cortex"):
        rows.append(
            {
                "pair_id": pair_id,
                "condition": condition,
                "provenance": "recorded_live",
                "tool_selected_correctly": condition == "cortex",
                "arguments_valid": True,
                "tool_succeeded": True,
                "task_succeeded": True,
                "provider_latency_ms": 800.0,
                "total_latency_ms": 900.0,
                "prompt_tokens": 700,
                "context_tokens": 100,
            }
        )
    return rows


class RealHistoryGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.memory_ids = [
            self.store.add_memory(
                f"Operator memory number {index} about the approved deploy channel.",
                kind="semantic",
                confidence=0.9,
                importance=0.8,
            )[0]
            for index in range(8)
        ]

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _labels(self, count: int):
        return [
            real_history.RetrievalLabel(
                case_id=f"case-{index:03d}",
                query=f"Which deploy channel does operator memory number {index} approve?",
                relevant_memory_ids=(self.memory_ids[index],),
                group="durable-fact",
            )
            for index in range(count)
        ]

    def test_small_sample_is_refused_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "min-cases"):
            real_history.evaluate(
                self.store,
                self._labels(2),
                top_k=6,
                token_budget=700,
                policy="adaptive",
            )

    def test_small_sample_override_stamps_report_and_passes_privacy(self) -> None:
        report = real_history.evaluate(
            self.store,
            self._labels(2),
            top_k=6,
            token_budget=700,
            policy="adaptive",
            allow_small_sample=True,
        )
        self.assertFalse(report["sample_size"]["representative"])
        self.assertTrue(report["sample_size"]["small_sample_override"])
        self.assertIn("note", report["sample_size"])

    def test_full_sample_reports_representative(self) -> None:
        report = real_history.evaluate(
            self.store,
            self._labels(8),
            top_k=6,
            token_budget=700,
            policy="adaptive",
        )
        self.assertTrue(report["sample_size"]["representative"])
        self.assertEqual(report["sample_size"]["cases"], 8)

    def test_privacy_self_check_catches_leaked_query(self) -> None:
        labels = self._labels(2)
        dirty = {"summary": {"cases": 2}, "echo": labels[0].query}
        with self.assertRaisesRegex(ValueError, "privacy self-check"):
            real_history._assert_report_privacy(
                dirty, [(label, {}) for label in labels]
            )

    def test_privacy_self_check_catches_leaked_memory_id(self) -> None:
        labels = self._labels(2)
        dirty = {"summary": {"cases": 2}, "echo": labels[1].relevant_memory_ids[0]}
        with self.assertRaisesRegex(ValueError, "privacy self-check"):
            real_history._assert_report_privacy(
                dirty, [(label, {}) for label in labels]
            )

    def test_privacy_self_check_allows_opted_in_group_names(self) -> None:
        labels = self._labels(2)
        grouped = {
            "summary": {"cases": 2},
            "groups": {"durable-fact": {"cases": 2}},
        }
        # Without the opt-in flag the group name must fail the check.
        with self.assertRaisesRegex(ValueError, "privacy self-check"):
            real_history._assert_report_privacy(
                grouped, [(label, {}) for label in labels]
            )
        # With the flag the operator-reviewed group summary is permitted.
        real_history._assert_report_privacy(
            grouped, [(label, {}) for label in labels], include_group_summary=True
        )


class ToolBenchGuardTests(unittest.TestCase):
    def test_small_sample_is_refused_by_default(self) -> None:
        observations = _pair_observations("pair-001") + _pair_observations("pair-002")
        with self.assertRaisesRegex(ValueError, "min-pairs"):
            tool_bench.summarize_observations(
                observations, evidence_type="operator_recorded_outcomes", seed=7
            )

    def test_small_sample_override_stamps_report(self) -> None:
        observations = _pair_observations("pair-001") + _pair_observations("pair-002")
        report = tool_bench.summarize_observations(
            observations,
            evidence_type="operator_recorded_outcomes",
            seed=7,
            allow_small_sample=True,
        )
        self.assertFalse(report["sample_size"]["representative"])
        self.assertTrue(report["sample_size"]["small_sample_override"])
        self.assertIn("30 pairs", report["sample_size"]["note"])

    def test_full_sample_reports_representative(self) -> None:
        observations = []
        for index in range(8):
            observations.extend(_pair_observations(f"pair-{index:03d}"))
        report = tool_bench.summarize_observations(
            observations, evidence_type="operator_recorded_outcomes", seed=7
        )
        self.assertTrue(report["sample_size"]["representative"])
        self.assertEqual(report["sample_size"]["paired_cases"], 8)


if __name__ == "__main__":
    unittest.main()
