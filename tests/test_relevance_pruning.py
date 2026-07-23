from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401

from cortex.autojudge import AutoJudgeConfig
from cortex.relevance_pruning import run_adaptive_pruning
from cortex.store import CortexStore


class AdaptivePruningTests(unittest.TestCase):
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
            model="synthetic-pruning-judge",
            api_key_env="",
            credential_file=None,
            timeout_seconds=5.0,
            max_output_tokens=1200,
            max_proposals=12,
            minimum_age_seconds=0,
            keep_threshold=0.80,
            decision_threshold=0.72,
            strong_feedback_boost=0.08,
            positive_feedback_boost=0.02,
        )

    def old_memory(self, content: str = "Legacy staging note for an unused service.") -> str:
        memory_id, _ = self.store.add_memory(
            content,
            kind="operational",
            confidence=0.45,
            importance=0.15,
            trust=0.5,
        )
        with self.store._lock:
            self.store._conn.execute(
                """UPDATE memories SET created_at='2020-01-01T00:00:00+00:00',
                   updated_at='2020-01-01T00:00:00+00:00',
                   observed_at='2020-01-01T00:00:00+00:00'
                   WHERE id=?""",
                (memory_id,),
            )
            self.store._conn.commit()
        return memory_id

    def test_candidate_score_is_activity_aware_and_excludes_protected_memory(self) -> None:
        old_id = self.old_memory()
        protected_id, _ = self.store.add_memory(
            "Protected recovery procedure.",
            kind="procedure",
            pinned=True,
        )
        candidates = self.store.adaptive_pruning_candidates(
            relevance_threshold=0.8,
            limit=100,
        )
        ids = {item["memory"]["id"] for item in candidates}
        self.assertIn(old_id, ids)
        self.assertNotIn(protected_id, ids)
        old = next(item for item in candidates if item["memory"]["id"] == old_id)
        self.assertGreater(old["score_evidence"]["age_days"], 1000)
        self.assertEqual(old["score_evidence"]["positive_outcomes"], 0)
        self.assertLessEqual(old["relevance_score"], 0.8)

    def test_shadow_pruning_records_proposal_without_lifecycle_change(self) -> None:
        memory_id = self.old_memory()

        def provider(_endpoint, _key, payload, _timeout):
            candidate = json.loads(payload["messages"][1]["content"])["candidates"][0]
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decisions": [
                                        {
                                            "memory_id": candidate["memory_id"],
                                            "action": "cool",
                                            "confidence": 0.9,
                                            "reason": "Old, unused, and low-confidence.",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        report = run_adaptive_pruning(
            self.store,
            self.config(),
            provider_call=provider,
            relevance_threshold=0.8,
            max_candidates=1,
        )
        self.assertEqual(report["mode"], "shadow")
        self.assertEqual(report["judged"], 1)
        self.assertEqual(self.store.get_memory(memory_id)["state"], "active")
        self.assertEqual(self.store.adaptive_pruning_snapshot()["counts"]["proposed"], 1)

    def test_stranding_is_reversible_and_regret_restores_edges(self) -> None:
        memory_id = self.old_memory()
        neighbor_id, _ = self.store.add_memory(
            "Current service inventory.",
            kind="operational",
        )
        self.store.add_edge(
            memory_id,
            neighbor_id,
            "related",
            evidence_type="operator_review",
            evidence_key="pruning-test-edge",
            explanation="The legacy note once belonged to this service inventory.",
        )

        def provider(_endpoint, _key, payload, _timeout):
            candidate = json.loads(payload["messages"][1]["content"])["candidates"][0]
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decisions": [
                                        {
                                            "memory_id": candidate["memory_id"],
                                            "action": "orphan_strand",
                                            "confidence": 0.94,
                                            "reason": "No use evidence; preserve it disconnected.",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        report = run_adaptive_pruning(
            self.store,
            self.config(),
            provider_call=provider,
            relevance_threshold=0.8,
            max_candidates=1,
            apply=True,
        )
        self.assertEqual(report["applied"], 1)
        memory = self.store.get_memory(memory_id)
        self.assertEqual(memory["state"], "cold")
        self.assertTrue(memory["stranded"])
        with self.store._lock:
            edge_count = self.store._conn.execute(
                """SELECT COUNT(*) count FROM edges
                   WHERE src_id=? OR dst_id=?""",
                (memory_id, memory_id),
            ).fetchone()["count"]
        self.assertEqual(edge_count, 0)

        self.assertTrue(
            self.store.record_pruning_regret(
                memory_id,
                query="legacy staging note",
                score=0.9,
                restore=True,
            )
        )
        restored = self.store.get_memory(memory_id)
        self.assertEqual(restored["state"], "active")
        self.assertFalse(restored["stranded"])
        with self.store._lock:
            edge_count = self.store._conn.execute(
                """SELECT COUNT(*) count FROM edges
                   WHERE src_id=? OR dst_id=?""",
                (memory_id, memory_id),
            ).fetchone()["count"]
        self.assertEqual(edge_count, 1)
        self.assertEqual(self.store.adaptive_pruning_snapshot()["counts"]["reversed"], 1)
        self.assertTrue(self.store.audit()["ok"])

    def test_quarantine_requires_harm_evidence(self) -> None:
        memory_id = self.old_memory()
        candidate = self.store.adaptive_pruning_candidates(
            relevance_threshold=0.8,
            limit=1,
        )[0]
        run_id = "synthetic-pruning-run"
        decision_id = "synthetic-pruning-decision"
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO adaptive_pruning_runs(
                   run_id,mode,status,model,relevance_threshold,candidate_count,started_at
                   ) VALUES(?,'shadow','completed','synthetic',0.8,1,'2026-01-01T00:00:00+00:00')""",
                (run_id,),
            )
            conn.execute(
                """INSERT INTO adaptive_pruning_decisions(
                   decision_id,run_id,memory_id,memory_hash,relevance_score,
                   score_evidence_json,action,confidence,reason,status,prior_state,created_at
                   ) VALUES(?,?,?,?,?,?,'quarantine',0.95,'unsafe','proposed','active',
                            '2026-01-01T00:00:00+00:00')""",
                (
                    decision_id,
                    run_id,
                    memory_id,
                    candidate["memory"]["content_hash"],
                    candidate["relevance_score"],
                    json.dumps(candidate["score_evidence"]),
                ),
            )
        result = self.store.apply_adaptive_pruning(decision_id)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(self.store.get_memory(memory_id)["state"], "active")


if __name__ == "__main__":
    unittest.main()
