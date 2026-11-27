"""Per-stage recall latency must be readable from the store, and must never be load-bearing.

The measured problem this covers: the first prefetch of a session took 1,422 ms and the warm ones
1,142 / 726 ms against a 1,500 ms pre-turn deadline, and roughly 800 ms of the first turn was the
one-time ONNX model load - but nothing in the store could say that, because `semantic` bundled the
load with inference and rendering was not timed at all. These tests pin the three properties a
report depends on: the stages are recorded, the counters are not mistaken for stages, and a broken
timer costs at most one measurement.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch


from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import recall_timing
from cortex.latency_budget import Budget
from cortex.recall_timing import (
    STAGE_DURATIONS,
    StageSink,
    add_stage,
    format_report,
    percentile,
    sanitize_stage_ms,
    stage_percentiles,
)
from cortex.retrieval import MemoryRetriever
from cortex.store import CortexStore


class _BadNumber:
    """A value whose conversion raises, to prove a bad measurement is dropped, not raised."""

    def __float__(self) -> float:  # pragma: no cover - the point is that it raises
        raise ValueError("not a number")


class _BrokenMapping(Mapping):
    """A mapping that cannot even be read."""

    def __getitem__(self, key):  # pragma: no cover - unreachable
        raise KeyError(key)

    def __iter__(self):
        raise RuntimeError("mapping is unreadable")

    def __len__(self) -> int:
        return 1


class WritingTests(unittest.TestCase):
    def test_a_sink_keeps_only_finite_non_negative_numbers(self) -> None:
        sink = StageSink()
        sink["fts"] = 12.5
        sink["nested"] = "31.25"
        sink["negative"] = -3.0
        sink["nan"] = float("nan")
        sink["inf"] = float("inf")
        sink["text"] = "slow"
        sink["broken"] = _BadNumber()
        self.assertEqual(dict(sink), {"fts": 12.5, "nested": 31.25})
        self.assertTrue(all(isinstance(value, float) for value in sink.values()))

    def test_a_counter_gets_the_count_suffix(self) -> None:
        sink = StageSink()
        sink.count("entity_boosted", 4)
        sink.count("already_count", 2)
        self.assertEqual(dict(sink), {"entity_boosted_count": 4.0, "already_count": 2.0})

    def test_sanitize_drops_nan_infinity_and_negatives(self) -> None:
        cleaned = sanitize_stage_ms(
            {"select": 25.0, "semantic": float("nan"), "graph": float("-inf"), "x": -1.0}
        )
        self.assertEqual(cleaned, {"select": 25.0})

    def test_sanitize_survives_a_mapping_that_cannot_be_read(self) -> None:
        self.assertEqual(sanitize_stage_ms(_BrokenMapping()), {})
        self.assertEqual(sanitize_stage_ms(None), {})

    def test_add_stage_merges_into_a_copy(self) -> None:
        original = {"fts": 1.0}
        merged = add_stage(original, "render", 4.0)
        self.assertEqual(merged, {"fts": 1.0, "render": 4.0})
        self.assertEqual(original, {"fts": 1.0})
        self.assertEqual(add_stage(original, "render", _BadNumber()), {"fts": 1.0})


class FailsafeTests(unittest.TestCase):
    """A timing failure may cost one measurement. It may never cost the recall."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.store.add_memory(
            "The VPS hosts the Cortex brain dashboard on port 8100.",
            kind="operational",
            confidence=0.9,
            importance=0.8,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_broken_clock_does_not_propagate_out_of_the_helpers(self) -> None:
        with patch("time.perf_counter", side_effect=RuntimeError("clock is gone")):
            started = recall_timing.clock()
            self.assertEqual(started, 0.0)
            self.assertEqual(recall_timing.elapsed_ms(started), 0.0)
            self.assertEqual(add_stage({"fts": 1.0}, "render", recall_timing.elapsed_ms(started)),
                             {"fts": 1.0, "render": 0.0})

    def test_a_broken_clock_does_not_break_a_recall(self) -> None:
        with patch("time.perf_counter", side_effect=RuntimeError("clock is gone")):
            results, diagnostics = MemoryRetriever(
                self.store, threshold=0.0
            ).search_detailed("Cortex brain dashboard port", limit=3)
        self.assertTrue(results, "recall must still return its memories without a clock")
        self.assertEqual(diagnostics.stage_ms["fts"], 0.0)
        self.assertEqual(diagnostics.stage_ms["select"], 0.0)

    def test_a_broken_clock_does_not_break_the_latency_budget(self) -> None:
        with patch("time.perf_counter", side_effect=RuntimeError("clock is gone")):
            budget = Budget(500)
            self.assertFalse(budget.expired())
            self.assertEqual(budget.elapsed_ms(), 0.0)

    def test_a_broken_stage_mapping_does_not_prevent_the_run_being_recorded(self) -> None:
        recall_id = self.store.record_recall_run(
            session_id="timing-test",
            query="Cortex brain dashboard port",
            mode="focused",
            reason="hostile stage mapping",
            requested_limit=2,
            token_budget=700,
            candidate_count=1,
            selected_count=1,
            estimated_tokens=10,
            prepare_ms=12.5,
            abstained=False,
            stage_ms=_BrokenMapping(),
        )
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT prepare_ms, stage_ms_json FROM recall_runs WHERE recall_id=?",
                (recall_id,),
            ).fetchone()
        self.assertIsNotNone(row, "a timing failure must not lose the recall record")
        self.assertEqual(row["prepare_ms"], 12.5)
        self.assertEqual(json.loads(row["stage_ms_json"]), {})


class StoreBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def run_with(self, stages, prepare_ms: float = 100.0) -> str:
        return self.store.record_recall_run(
            session_id="s",
            query="q",
            mode="auto",
            reason="test",
            requested_limit=6,
            token_budget=700,
            candidate_count=1,
            selected_count=1,
            estimated_tokens=10,
            prepare_ms=prepare_ms,
            abstained=False,
            stage_ms=stages,
        )

    def payload(self, recall_id: str) -> dict:
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT stage_ms_json FROM recall_runs WHERE recall_id=?", (recall_id,)
            ).fetchone()
        return json.loads(row["stage_ms_json"])

    def test_a_nan_cannot_reach_the_json_column(self) -> None:
        payload = self.payload(self.run_with({"fts": 2.0, "semantic": float("nan")}))
        self.assertEqual(payload, {"fts": 2.0})
        # And the stored text is valid JSON, not `NaN`.
        with self.store.transaction() as conn:
            raw = conn.execute(
                "SELECT stage_ms_json FROM recall_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()[0]
        self.assertNotIn("NaN", raw)


class ReadSideTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def run_with(self, stages=None, prepare_ms: float = 100.0) -> str:
        return self.store.record_recall_run(
            session_id="s",
            query="q",
            mode="auto",
            reason="test",
            requested_limit=6,
            token_budget=700,
            candidate_count=1,
            selected_count=1,
            estimated_tokens=10,
            prepare_ms=prepare_ms,
            abstained=False,
            stage_ms=stages,
        )

    def test_percentile_matches_the_documented_nearest_rank(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(percentile(values, 0.5), 30.0)
        self.assertEqual(percentile(values, 0.95), 50.0)
        self.assertEqual(percentile([], 0.5), 0.0)

    def test_p50_and_p95_are_reported_per_stage(self) -> None:
        for value in (10.0, 20.0, 30.0, 40.0, 50.0):
            self.run_with({"fts": value, "select": value * 2}, prepare_ms=value * 10)
        report = stage_percentiles(self.store, window=500)
        self.assertEqual(report.runs, 5)
        self.assertEqual(report.stage("fts")["p50"], 30.0)
        self.assertEqual(report.stage("fts")["p95"], 50.0)
        self.assertEqual(report.stage("select")["p50"], 60.0)
        self.assertEqual(report.prepare_p50, 300.0)
        self.assertEqual(report.prepare_p95, 500.0)

    def test_the_window_bounds_the_sample(self) -> None:
        self.run_with({"fts": 1.0})
        self.run_with({"fts": 500.0})
        report = stage_percentiles(self.store, window=1)
        self.assertEqual(report.runs, 1)
        self.assertEqual(report.stage("fts")["p50"], 500.0)

    def test_cache_hits_are_excluded_from_the_percentiles(self) -> None:
        """A cached run stores zeros for stages it never ran; averaging them in is a fiction."""
        self.run_with({"fts": 100.0, "select": 100.0})
        self.run_with({"fts": 0.0, "select": 0.0, "cache_lookup": 1.5})
        report = stage_percentiles(self.store)
        self.assertEqual(report.cached_runs, 1)
        self.assertEqual(report.runs, 1)
        self.assertEqual(report.stage("fts")["p50"], 100.0)

    def test_counters_are_not_reported_as_stages(self) -> None:
        self.run_with(
            {
                "fts": 4.0,
                "entity_boosted_count": 9.0,
                "reranked_count": 3.0,
                # Legacy rows carried counters and even a score without the suffix.
                "abstained": 1.0,
                "abstained_top_score": 0.42,
            }
        )
        report = stage_percentiles(self.store)
        self.assertEqual([item["stage"] for item in report.stages], ["fts"])
        names = {item["counter"] for item in report.counters}
        self.assertEqual(names, {"entity_boosted_count", "reranked_count", "abstained",
                                 "abstained_top_score"})

    def test_every_reported_stage_is_part_of_the_declared_contract(self) -> None:
        self.run_with({name: 1.0 for name in STAGE_DURATIONS})
        report = stage_percentiles(self.store)
        self.assertEqual({item["stage"] for item in report.stages}, set(STAGE_DURATIONS))
        self.assertEqual(report.counters, [])

    def test_one_unreadable_row_does_not_hide_the_others(self) -> None:
        self.run_with({"fts": 10.0})
        self.run_with({"fts": 20.0})
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE recall_runs SET stage_ms_json='not json' WHERE rowid=("
                "SELECT MIN(rowid) FROM recall_runs)"
            )
        report = stage_percentiles(self.store)
        # The row is still a run (it has prepare_ms); only its stages are unreadable.
        self.assertEqual(report.runs, 2)
        self.assertEqual(report.stage("fts")["runs"], 1)
        self.assertEqual(report.stage("fts")["p50"], 20.0)

    def test_an_empty_store_says_so_instead_of_reporting_zero_latency(self) -> None:
        report = stage_percentiles(self.store)
        self.assertEqual(report.runs, 0)
        self.assertEqual(report.stages, [])
        self.assertIn("no timed recalls", report.note)

    def test_a_store_that_cannot_be_read_returns_a_report_not_an_exception(self) -> None:
        broken = types.SimpleNamespace()
        report = stage_percentiles(broken)   # no `_conn` at all: the read fails, the call does not
        self.assertEqual(report.runs, 0)
        self.assertTrue(report.note)
        self.assertEqual(format_report(report)[0].split(":")[0],
                         "recall latency, last 512 run(s)")

    def test_the_report_is_printable_and_json_serialisable(self) -> None:
        self.run_with({"fts": 10.0, "render": 2.0})
        report = stage_percentiles(self.store)
        lines = format_report(report)
        self.assertTrue(any("fts" in line for line in lines))
        self.assertTrue(any(line.startswith("prepare_ms") for line in lines))
        self.assertIn("render", json.dumps(report.as_dict()))


class EmbedderTimingTests(unittest.TestCase):
    def test_an_unavailable_model_reports_zero_and_still_degrades(self) -> None:
        from cortex.embeddings import Embedder

        with tempfile.TemporaryDirectory() as tmp:
            embedder = Embedder(model_dir=Path(tmp) / "absent")
            self.assertEqual(embedder.embed_one("anything"), [])
        self.assertEqual(embedder.last_load_ms, 0.0)
        # No model, so no load and no inference: only the failed file checks were timed.
        self.assertLess(embedder.last_embed_ms, 10.0)

    def test_embedding_time_is_published_even_when_inference_raises(self) -> None:
        """The `finally:` is the point: a failed embed that took 900 ms must not report 0 ms."""
        from cortex.embeddings import Embedder

        with tempfile.TemporaryDirectory() as tmp:
            embedder = Embedder(model_dir=Path(tmp))
            embedder._session = object()
            embedder._tokenizer = object()
            embedder.last_embed_ms = -1.0
            with patch.dict(sys.modules, {"numpy": None}):
                with self.assertRaises(ImportError):
                    embedder.embed(["anything"])
        self.assertGreaterEqual(embedder.last_embed_ms, 0.0)

    def test_a_semantic_recall_carries_embed_stages_into_stage_ms(self) -> None:
        class _FakeEmbedder:
            available = True
            model_id = "fake-small"
            last_load_ms = 812.5
            last_embed_ms = 831.0

            def embed_one(self, text: str) -> list[float]:
                return [0.1] * 8

        with tempfile.TemporaryDirectory() as tmp:
            store = CortexStore(Path(tmp) / "cortex.db")
            try:
                store.add_memory("The Cortex dashboard listens on port 8100.", kind="operational")
                with patch("cortex.embeddings.get_embedder", return_value=_FakeEmbedder()):
                    _results, diagnostics = MemoryRetriever(
                        store, threshold=0.0, semantic_weight=0.3
                    ).search_detailed("which port does the cortex dashboard use", limit=3)
                self.assertEqual(diagnostics.stage_ms["embed_load"], 812.5)
                self.assertEqual(diagnostics.stage_ms["embed"], 831.0)
            finally:
                store.close()


class PrefetchRenderStageTests(unittest.TestCase):
    """End to end: a real prefetch stores a `render` stage beside the retrieval stages."""

    def setUp(self) -> None:
        from cortex import CortexMemoryProvider

        self.tmp = tempfile.TemporaryDirectory()
        self.provider = CortexMemoryProvider(
            {"db_path": "$HERMES_HOME/cortex/test.db", "retrieval_threshold": 0.08}
        )
        self.provider.initialize("render-session", hermes_home=self.tmp.name)
        self.store = self.provider._store
        self.store.add_memory(
            "The VPS hosts the Cortex brain dashboard on port 8100.",
            kind="operational",
            confidence=0.9,
            importance=0.8,
        )

    def tearDown(self) -> None:
        self.provider.shutdown()
        self.tmp.cleanup()

    def test_a_prefetch_stores_retrieval_and_render_stages(self) -> None:
        block = self.provider.prefetch(
            "Where is the Cortex brain dashboard hosted?", session_id="render-session"
        )
        self.assertIn("dashboard", block)
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT stage_ms_json FROM recall_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        payload = json.loads(row["stage_ms_json"])
        self.assertIn("render", payload)
        self.assertGreaterEqual(payload["render"], 0.0)
        self.assertIn("fts", payload)


if __name__ == "__main__":
    unittest.main()
