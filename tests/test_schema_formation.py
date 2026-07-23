from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401

from cortex.autojudge import AutoJudgeConfig
from cortex.retrieval import MemoryRetriever
from cortex.schema_formation import run_schema_formation
from cortex.store import CortexStore


class SchemaFormationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    @staticmethod
    def config() -> AutoJudgeConfig:
        return AutoJudgeConfig(
            enabled=True,
            endpoint="http://127.0.0.1:9999/v1/chat/completions",
            model="synthetic-schema-judge",
            api_key_env="",
            credential_file=None,
            timeout_seconds=5.0,
            max_output_tokens=1600,
        )

    def source_cluster(self, count: int = 3) -> list[str]:
        ids: list[str] = []
        for index in range(count):
            memory_id, _ = self.store.add_memory(
                f"Verified deployment checklist requires health check step {index}.",
                kind="episode" if index < 2 else "procedure",
            )
            task_type = "deployment" if index % 2 == 0 else "debugging"
            task_id = self.store.create_usage_batch(
                [(memory_id, 0.8)],
                query="Which verified deployment checklist step applies?",
                session_id=f"schema-{index}",
                task_type=task_type,
                recall_mode="focused",
            )
            self.store.record_memory_trace_decision(
                task_id=task_id,
                session_id=f"schema-{index}",
                goal="Follow the verified deployment checklist",
                context_summary=f"task_type={task_type}",
                task_type=task_type,
                recall_mode="focused",
                retrieval_used=True,
                retrieval_reason="source influenced the task",
                queries=["Which verified deployment checklist step applies?"],
                candidate_memories=[
                    {
                        "memory_id": memory_id,
                        "kind": "episode" if index < 2 else "procedure",
                        "selected": True,
                        "score": 0.8,
                        "components": {},
                        "reason": "selected",
                    }
                ],
            )
            self.store.resolve_usage(task_id, {memory_id: 1.0})
            ids.append(memory_id)
        return ids

    def stage(self) -> tuple[dict, list[str]]:
        source_ids = self.source_cluster()

        def provider(_endpoint, _key, payload, _timeout):
            cluster = json.loads(payload["messages"][1]["content"])["clusters"][0]
            included = [source["memory_id"] for source in cluster["sources"]]
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "proposals": [
                                        {
                                            "cluster_id": cluster["cluster_id"],
                                            "action": "abstract",
                                            "abstract_content": (
                                                "Verified deployments use a checklist with an explicit "
                                                "health-check step before completion."
                                            ),
                                            "included_source_ids": included,
                                            "confidence": 0.86,
                                            "reason": "Three tasks support the recurring checklist pattern.",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 15},
            }

        report = run_schema_formation(
            self.store,
            self.config(),
            provider_call=provider,
        )
        return {
            "report": report,
            "proposal": self.store.schema_formation_snapshot()["proposals"][0],
        }, source_ids

    def test_candidate_gate_requires_three_tasks_and_multiple_types_or_episodes(self) -> None:
        two_sources = self.source_cluster(count=2)
        self.assertEqual(self.store.schema_formation_candidates(), [])

        third_id, _ = self.store.add_memory(
            "Verified deployment checklist requires health check step 3.",
            kind="procedure",
        )
        task_id = self.store.create_usage_batch(
            [(third_id, 0.8)],
            query="deployment checklist",
            session_id="schema-third",
            task_type="deployment",
            recall_mode="focused",
        )
        self.store.record_memory_trace_decision(
            task_id=task_id,
            session_id="schema-third",
            goal="deployment checklist",
            context_summary="deployment",
            task_type="deployment",
            recall_mode="focused",
            retrieval_used=True,
            retrieval_reason="used",
            queries=["deployment checklist"],
            candidate_memories=[
                {
                    "memory_id": third_id,
                    "kind": "procedure",
                    "selected": True,
                    "score": 0.8,
                    "components": {},
                    "reason": "selected",
                }
            ],
        )
        self.store.resolve_usage(task_id, {third_id: 1.0})

        candidates = self.store.schema_formation_candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["evidence"]["distinct_tasks"], 3)
        self.assertEqual(candidates[0]["evidence"]["episode_count"], 2)
        self.assertTrue(set(two_sources).issubset(candidates[0]["source_ids"]))

    def test_schema_stays_shadow_until_explicit_apply_and_preserves_sources(self) -> None:
        result, source_ids = self.stage()
        proposal_id = result["proposal"]["proposal_id"]

        self.assertEqual(result["proposal"]["status"], "proposed")
        with self.store._lock:
            schema_count = self.store._conn.execute(
                "SELECT COUNT(*) count FROM memories WHERE kind='schema'"
            ).fetchone()["count"]
        self.assertEqual(schema_count, 0)
        applied = self.store.apply_schema_formation(proposal_id)
        schema_id = applied["schema_memory_id"]

        self.assertEqual(self.store.get_memory(schema_id)["kind"], "schema")
        self.assertTrue(all(self.store.get_memory(source_id)["state"] == "active" for source_id in source_ids))
        self.assertEqual(
            {item["evidence_id"] for item in self.store.dependencies(schema_id) if item["active"]},
            set(source_ids),
        )
        self.assertTrue(self.store.audit()["ok"])
        for source_id in source_ids:
            self.assertEqual(len(self.store.edge_evidence(source_id, schema_id, "example_of")), 1)

    def test_source_correction_marks_schema_dirty_and_wrong_feedback_reverses(self) -> None:
        result, source_ids = self.stage()
        proposal_id = result["proposal"]["proposal_id"]
        schema_id = self.store.apply_schema_formation(proposal_id)["schema_memory_id"]

        self.store.correct_memory(
            source_ids[0],
            "Verified deployment checklist health checks are environment-specific.",
            reason="operator correction",
        )
        self.assertTrue(self.store.get_memory(schema_id)["dirty"])
        feedback = self.store.record_schema_formation_feedback(
            proposal_id,
            "wrong",
            reason="abstraction was too broad",
        )

        self.assertTrue(feedback["automatically_undone"])
        self.assertEqual(self.store.get_memory(schema_id)["state"], "archived")
        self.assertEqual(feedback["feedback"]["accuracy"], 0.0)

    def test_applied_schema_marks_examples_for_reduced_generic_recall(self) -> None:
        result, source_ids = self.stage()
        self.store.apply_schema_formation(result["proposal"]["proposal_id"])

        _selected, diagnostics = MemoryRetriever(self.store, threshold=0.0).search_detailed(
            "verified deployment checklist health check",
            limit=10,
        )
        decisions = {item["memory_id"]: item for item in diagnostics.candidate_decisions}

        self.assertTrue(all(decisions[source_id]["components"]["schema_example"] == 1.0 for source_id in source_ids))
        self.assertTrue(
            all(decisions[source_id]["components"]["specificity_request"] == 0.0 for source_id in source_ids)
        )

    def test_wrong_feedback_rejects_unapplied_proposal(self) -> None:
        result, _source_ids = self.stage()
        proposal_id = result["proposal"]["proposal_id"]

        feedback = self.store.record_schema_formation_feedback(
            proposal_id,
            "wrong",
            reason="review found the abstraction too broad",
        )
        proposal = self.store.schema_formation_snapshot()["proposals"][0]

        self.assertFalse(feedback["automatically_undone"])
        self.assertEqual(proposal["status"], "rejected")
        with self.assertRaisesRegex(ValueError, "not proposed"):
            self.store.apply_schema_formation(proposal_id)

    def test_no_schema_discards_echoed_sources_and_content(self) -> None:
        self.source_cluster()

        def provider(_endpoint, _key, payload, _timeout):
            cluster = json.loads(payload["messages"][1]["content"])["clusters"][0]
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "proposals": [
                                        {
                                            "cluster_id": cluster["cluster_id"],
                                            "action": "no_schema",
                                            "abstract_content": "Provider content must be discarded.",
                                            "included_source_ids": [
                                                source["memory_id"]
                                                for source in cluster["sources"]
                                            ],
                                            "confidence": 0.81,
                                            "reason": "Examples are too context-specific.",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        report = run_schema_formation(
            self.store,
            self.config(),
            provider_call=provider,
        )
        proposal = self.store.schema_formation_snapshot()["proposals"][0]

        self.assertEqual(report["actions"]["no_schema"], 1)
        self.assertEqual(proposal["action"], "no_schema")
        self.assertIsNone(proposal["abstract_content"])
        self.assertEqual(proposal["included_source_ids"], [])


if __name__ == "__main__":
    unittest.main()
