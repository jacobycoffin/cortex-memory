from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.retrieval import MemoryRetriever, RetrievalDiagnostics
from cortex.store import CortexStore


EXPECTED_STAGES = (
    "fts",
    "feature",
    "context",
    "neighborhood",
    "prospective_merge",
    "scoring",
    "graph",
    "select",
)


class RetrievalStageTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "cortex.db"
        self.store = CortexStore(self.db)
        self.store.add_memory(
            "The VPS hosts the Cortex brain dashboard on port 8100.",
            kind="operational",
            confidence=0.9,
            importance=0.8,
        )
        self.store.add_memory(
            "Project Amber deploys require a verified backup before release.",
            kind="procedure",
            confidence=0.8,
            importance=0.7,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_search_detailed_reports_per_stage_timings(self) -> None:
        _results, diagnostics = MemoryRetriever(self.store, threshold=0.0).search_detailed(
            "Where is the Cortex brain dashboard hosted?", limit=3
        )
        self.assertIsInstance(diagnostics, RetrievalDiagnostics)
        for stage in EXPECTED_STAGES:
            self.assertIn(stage, diagnostics.stage_ms)
            self.assertGreaterEqual(diagnostics.stage_ms[stage], 0.0)
        self.assertGreater(sum(diagnostics.stage_ms.values()), 0.0)

    def test_early_return_carries_empty_stage_timings(self) -> None:
        _results, diagnostics = MemoryRetriever(self.store).search_detailed("", limit=3)
        self.assertEqual(diagnostics.stage_ms, {})

    def test_record_recall_run_persists_stage_json(self) -> None:
        _results, diagnostics = MemoryRetriever(self.store, threshold=0.0).search_detailed(
            "Cortex brain dashboard port", limit=2
        )
        recall_id = self.store.record_recall_run(
            session_id="timing-test",
            query="Cortex brain dashboard port",
            mode="focused",
            reason="timing persistence check",
            requested_limit=2,
            token_budget=700,
            candidate_count=diagnostics.candidate_count,
            selected_count=diagnostics.selected_count,
            estimated_tokens=diagnostics.estimated_tokens,
            prepare_ms=12.5,
            abstained=False,
            stage_ms=dict(diagnostics.stage_ms),
        )
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT stage_ms_json FROM recall_runs WHERE recall_id=?", (recall_id,)
            ).fetchone()
        payload = json.loads(row["stage_ms_json"])
        for stage in EXPECTED_STAGES:
            self.assertIn(stage, payload)

    def test_record_recall_run_without_stages_defaults_to_empty_object(self) -> None:
        recall_id = self.store.record_recall_run(
            session_id="timing-test",
            query="hello",
            mode="lean",
            reason="legacy caller without stage data",
            requested_limit=1,
            token_budget=100,
            candidate_count=0,
            selected_count=0,
            estimated_tokens=0,
            prepare_ms=1.0,
            abstained=True,
        )
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT stage_ms_json FROM recall_runs WHERE recall_id=?", (recall_id,)
            ).fetchone()
        self.assertEqual(json.loads(row["stage_ms_json"]), {})

    def test_stage_column_migrates_onto_legacy_recall_runs_table(self) -> None:
        self.store.close()
        with sqlite3.connect(self.db) as conn:
            conn.execute("ALTER TABLE recall_runs DROP COLUMN stage_ms_json")
            conn.commit()
        reopened = CortexStore(self.db)
        try:
            columns = {
                row[1] for row in reopened._conn.execute("PRAGMA table_info(recall_runs)")
            }
            self.assertIn("stage_ms_json", columns)
            recall_id = reopened.record_recall_run(
                session_id="timing-test",
                query="post-migration write",
                mode="lean",
                reason="migration check",
                requested_limit=1,
                token_budget=100,
                candidate_count=0,
                selected_count=0,
                estimated_tokens=0,
                prepare_ms=1.0,
                abstained=True,
                stage_ms={"fts": 1.5},
            )
            with reopened.transaction() as conn:
                row = conn.execute(
                    "SELECT stage_ms_json FROM recall_runs WHERE recall_id=?", (recall_id,)
                ).fetchone()
            self.assertEqual(json.loads(row["stage_ms_json"]), {"fts": 1.5})
        finally:
            reopened.close()
            self.store = CortexStore(self.db)


class DegenerateQueryAbstentionTests(unittest.TestCase):
    """A query with no retrieval signal must abstain, not return the newest rows.

    Regression (2026-09-14): the store's no-token FTS fallback answered a blank,
    punctuation-only, stopword-only or single-character query with the newest
    memories. Those cleared the score threshold, so the model received three
    unrelated memories at ~0.29 presented as relevant context — for a turn that
    said nothing to recall. Measured before the guard: `"   "`, `"?!..."`,
    `",,,"`, `"a"` and `"the and of to"` each selected 3 memories with
    `abstained=False`.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.store.add_memory("The homelab router hands out leases on 10.20.0.0/24.", kind="semantic")
        self.store.add_memory("Plex runs on the r630 host behind the proxy.", kind="semantic")
        self.store.add_memory("Deployments require a verified backup before release.", kind="procedure")
        self.store.add_memory("日本語のメモ は ここ に あります。", kind="semantic")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_no_token_queries_abstain_instead_of_returning_newest_memories(self) -> None:
        for query in ("   ", "?!...", ",,,", "a", "the and of to", "\n\t "):
            with self.subTest(query=query):
                results, diagnostics = MemoryRetriever(self.store, threshold=0.0).search_detailed(
                    query, limit=6, token_budget=700, graph_depth=0
                )
                self.assertEqual(results, [], f"{query!r} must not recall memories")
                self.assertTrue(diagnostics.abstained)
                # Same shape as the empty-query early return: no stage timings.
                self.assertEqual(diagnostics.stage_ms, {})

    def test_queries_with_signal_still_recall(self) -> None:
        retriever = MemoryRetriever(self.store, threshold=0.0)
        for query, expected in (
            ("homelab router leases", "10.20.0.0/24"),
            ("plex r630", "r630"),
            ("verified backup", "backup"),
            ("日本語のメモ", "日本語"),
        ):
            with self.subTest(query=query):
                results, diagnostics = retriever.search_detailed(
                    query, limit=6, token_budget=700, graph_depth=0
                )
                self.assertFalse(diagnostics.abstained, f"{query!r} must still recall")
                self.assertTrue(
                    any(expected in result.memory["content"] for result in results),
                    f"{query!r} did not recall the expected memory",
                )


if __name__ == "__main__":
    unittest.main()
