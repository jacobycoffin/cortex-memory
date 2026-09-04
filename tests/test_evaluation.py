from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


from tests._bootstrap import ROOT

from cortex.evaluation import HistoryCase, compare_real_history
from cortex.scripts.benchmark_tool_calling import (
    load_recorded_observations,
    run_live,
    summarize_observations,
)
from cortex.scripts.evaluate_real_history import RetrievalLabel, evaluate, private_database_snapshot
from cortex.store import CortexStore


class RealHistoryEvaluationTests(unittest.TestCase):
    def test_dashboard_private_comparison_is_paired_and_omits_private_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cortex-private-evaluation-") as tmp:
            database_path = Path(tmp) / "private-kaya-history.db"
            store = CortexStore(database_path)
            cases = []
            try:
                for index in range(8):
                    code = f"orchid-{index}-private"
                    memory_id, _ = store.add_memory(
                        f"Deployment lane {index} uses the private marker {code}.",
                        kind="operational",
                        source_ref=f"private-source-{index}",
                    )
                    cases.append(
                        HistoryCase(
                            query=f"Which marker does deployment lane {index} use?",
                            relevant_memory_ids=(memory_id,),
                            task_type="deployment",
                        )
                    )
                report = compare_real_history(database_path, cases, top_k=6, token_budget=700)
            finally:
                store.close()

            self.assertEqual(report["case_count"], 8)
            self.assertEqual(set(report["conditions"]), {"fixed", "adaptive"})
            self.assertIn("context_tokens_p50", report["adaptive_minus_fixed"])
            self.assertTrue(report["privacy"]["raw_private_text_omitted"])
            serialized = json.dumps(report)
            self.assertNotIn("orchid-0-private", serialized)
            self.assertNotIn("private-kaya-history.db", serialized)
            self.assertNotIn(cases[0].relevant_memory_ids[0], serialized)

    def test_report_scores_private_labels_without_copying_private_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cortex-evaluation-") as tmp:
            private_db_path = Path(tmp) / "operator-private-name.db"
            store = CortexStore(private_db_path)
            try:
                memory_id, _ = store.add_memory(
                    "Project Firefly deploys through the silver private gateway.",
                    kind="decision",
                    source_type="test",
                    source_ref="private-source-reference",
                    subject="Project Firefly",
                    predicate="gateway",
                    object_value="silver private gateway",
                )
                labels = [
                    RetrievalLabel(
                        case_id="private-case-name",
                        query="Which gateway does Project Firefly use?",
                        relevant_memory_ids=(memory_id,),
                    )
                ]
                # Single-case privacy fixture: explicitly opted into the small-sample
                # stamp so this mechanics test is not gated by the sample floor.
                report = evaluate(
                    store, labels, top_k=6, token_budget=700, policy="fixed",
                    allow_small_sample=True,
                )
            finally:
                store.close()

            serialized = json.dumps(report)
            self.assertEqual(report["evidence_type"], "offline_retrieval_mechanics")
            self.assertEqual(report["summary"]["cases"], 1)
            self.assertEqual(report["summary"]["hit_at_k"], 1.0)
            for private_value in (
                "Project Firefly",
                "silver private gateway",
                "private-case-name",
                "private-source-reference",
                memory_id,
                str(private_db_path),
            ):
                self.assertNotIn(private_value, serialized)

    def test_missing_labeled_memory_stops_the_evaluation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cortex-evaluation-") as tmp:
            store = CortexStore(Path(tmp) / "cortex.db")
            try:
                with self.assertRaisesRegex(ValueError, "not present"):
                    evaluate(
                        store,
                        [RetrievalLabel("case", "query", ("missing-memory-id",))],
                        top_k=6,
                        token_budget=700,
                        policy="fixed",
                    )
            finally:
                store.close()

    def test_private_snapshot_contains_live_wal_data_and_is_disposable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cortex-evaluation-") as tmp:
            database_path = Path(tmp) / "cortex.db"
            store = CortexStore(database_path)
            memory_id, _ = store.add_memory("A private memory that is still in the live database.")
            try:
                with private_database_snapshot(database_path) as snapshot_path:
                    self.assertTrue(snapshot_path.exists())
                    snapshot_store = CortexStore(snapshot_path)
                    try:
                        self.assertIsNotNone(snapshot_store.get_memory(memory_id))
                    finally:
                        snapshot_store.close()
                    disposable_path = snapshot_path
                self.assertFalse(disposable_path.exists())
            finally:
                store.close()


class ToolCallingEvaluationTests(unittest.TestCase):
    def test_recorded_report_is_paired_and_omits_pair_ids(self) -> None:
        rows = [
            self._observation("sensitive-deployment-case", "default_built_in", False, False),
            self._observation("sensitive-deployment-case", "cortex", True, True),
            self._observation("sensitive-backup-case", "default_built_in", True, True),
            self._observation("sensitive-backup-case", "cortex", True, True),
        ]
        report = summarize_observations(
            rows, evidence_type="operator_recorded_outcomes", seed=11,
            allow_small_sample=True,  # two-pair pairing fixture, not a claim
        )
        delta = report["paired_deltas_cortex_minus_default"]["tool_selected_correctly"]
        self.assertEqual(report["reproducibility"]["paired_cases"], 2)
        self.assertEqual(delta["cortex_wins"], 1)
        self.assertEqual(delta["cortex_losses"], 0)
        self.assertEqual(delta["rate_delta"], 0.5)
        serialized = json.dumps(report)
        self.assertNotIn("sensitive-deployment-case", serialized)
        self.assertNotIn("sensitive-backup-case", serialized)

    def test_recorded_loader_rejects_an_unpaired_condition(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cortex-tool-evaluation-") as tmp:
            path = Path(tmp) / "observations.jsonl"
            path.write_text(
                json.dumps(self._observation("only-one-side", "cortex", True, True)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly one observation"):
                load_recorded_observations(path)

    def test_live_mode_uses_provider_responses_but_replays_tool_fixture(self) -> None:
        scenario = {
            "case_id": "private-live-case",
            "prompt": "private user prompt",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "deploy_service",
                        "description": "Deploy a service",
                        "parameters": {"type": "object", "properties": {"service": {"type": "string"}}},
                    },
                }
            ],
            "expected_tool": "deploy_service",
            "expected_arguments": {"service": "api"},
            "conditions": {"default_built_in": "old context", "cortex": "current context"},
            "tool_outcome": {"ok": True, "content": {"status": "deployed"}},
            "expected_final_contains": ["deployed"],
        }
        tool_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "deploy_service", "arguments": '{"service":"api"}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 10},
        }
        final_response = {
            "choices": [{"message": {"role": "assistant", "content": "The service was deployed."}}],
            "usage": {"prompt_tokens": 12},
        }
        with patch(
            "cortex.scripts.benchmark_tool_calling._post_chat",
            side_effect=[
                (tool_response, 10.0),
                (final_response, 20.0),
                (tool_response, 11.0),
                (final_response, 21.0),
            ],
        ):
            observations = run_live(
                [scenario],
                endpoint="https://provider.invalid/v1/chat/completions",
                api_key="not-a-real-secret",
                model="example/model",
                seed=7,
                timeout=1.0,
                max_output_tokens=64,
                request_delay_ms=0,
            )

        self.assertEqual(len(observations), 2)
        self.assertTrue(all(row["tool_selected_correctly"] for row in observations))
        self.assertTrue(all(row["arguments_valid"] for row in observations))
        self.assertTrue(all(row["task_succeeded"] for row in observations))
        report = summarize_observations(
            observations,
            evidence_type="live_provider_with_recorded_tool_fixtures",
            seed=7,
            model="example/model",
            allow_small_sample=True,  # single-scenario replay fixture, not a claim
        )
        serialized = json.dumps(report)
        for private_value in ("private-live-case", "private user prompt", "deploy_service", "not-a-real-secret"):
            self.assertNotIn(private_value, serialized)

    @staticmethod
    def _observation(pair_id: str, condition: str, selected: bool, succeeded: bool) -> dict:
        return {
            "schema_version": 1,
            "pair_id": pair_id,
            "condition": condition,
            "provenance": "recorded_live",
            "tool_selected_correctly": selected,
            "arguments_valid": selected,
            "tool_succeeded": succeeded,
            "task_succeeded": succeeded,
            "provider_latency_ms": 100.0,
            "total_latency_ms": 150.0,
            "prompt_tokens": 200,
            "context_tokens": 50,
        }


if __name__ == "__main__":
    unittest.main()
