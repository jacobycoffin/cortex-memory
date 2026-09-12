"""A failed semantic-merge finalize must never leave a live merged memory.

Regression coverage for the audit's pipeline finding: the merged result memory
was created durable and recall-eligible before the decision ledger finalize,
so a failure between those two steps stranded a live memory with the decision
still marked "proposed".
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT  # noqa: F401 - loads the flat ``cortex`` package

from cortex.autojudge import AutoJudgeConfig
from cortex.semantic_consolidation import run_semantic_consolidation
from cortex.store import CortexStore


def _config() -> AutoJudgeConfig:
    return AutoJudgeConfig(
        enabled=True,
        endpoint="http://127.0.0.1:1",
        model="synthetic",
        api_key_env="",
        minimum_age_seconds=0,
    )


def _merge_provider(endpoint: str, api_key: str, payload: dict, timeout: float) -> dict:
    pair = json.loads(payload["messages"][1]["content"])["pairs"][0]
    decision = {
        "pair_id": pair["pair_id"],
        "action": "merge",
        "confidence": 0.99,
        "reason": "Synthetic merge",
        "merged_content": (
            "Synthetic combined Cortex nightly replay reviews durable evidence "
            "before proposing safe maintenance changes."
        ),
    }
    return {
        "choices": [{"message": {"content": json.dumps({"decisions": [decision]})}}],
        "usage": {"total_tokens": 100},
    }


class SemanticConsolidationAtomicityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _seed_and_judge(self) -> str:
        self.store.add_memory(
            "Synthetic Cortex nightly replay reviews durable evidence before maintenance.",
            entities=("Cortex", "Sleep"),
        )
        self.store.add_memory(
            "Synthetic Cortex nightly replay reviews durable evidence and proposes maintenance changes.",
            entities=("Cortex", "Sleep"),
        )
        run_semantic_consolidation(self.store, _config(), provider_call=_merge_provider)
        row = self.store._conn.execute("SELECT decision_id FROM semantic_consolidation_decisions").fetchone()
        assert row is not None
        return str(row["decision_id"])

    def _result_rows(self) -> list[dict]:
        with self.store._lock:
            rows = self.store._conn.execute(
                "SELECT id, state FROM memories WHERE source_type='semantic_consolidation'"
            ).fetchall()
        return [dict(row) for row in rows]

    def test_finalize_failure_leaves_merged_memory_quarantined(self) -> None:
        decision_id = self._seed_and_judge()
        with patch.object(
            self.store,
            "_refinery_state_change_tx",
            side_effect=RuntimeError("synthetic finalize failure"),
        ):
            with self.assertRaises(RuntimeError):
                self.store.apply_semantic_consolidation(decision_id)

        results = self._result_rows()
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["state"], "quarantine")
        self.assertFalse(self.store.is_memory_recall_eligible(result["id"]))
        status = self.store._conn.execute(
            "SELECT status FROM semantic_consolidation_decisions WHERE decision_id=?",
            (decision_id,),
        ).fetchone()["status"]
        self.assertEqual(status, "proposed")

    def test_successful_apply_promotes_merged_memory_to_active(self) -> None:
        decision_id = self._seed_and_judge()
        report = self.store.apply_semantic_consolidation(decision_id)
        self.assertEqual(report["status"], "applied")

        results = self._result_rows()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["state"], "active")
        self.assertTrue(self.store.is_memory_recall_eligible(results[0]["id"]))
        status = self.store._conn.execute(
            "SELECT status FROM semantic_consolidation_decisions WHERE decision_id=?",
            (decision_id,),
        ).fetchone()["status"]
        self.assertEqual(status, "applied")


if __name__ == "__main__":
    unittest.main()