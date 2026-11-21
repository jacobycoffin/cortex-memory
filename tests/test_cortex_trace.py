"""Phase tracing (``CORTEX_TRACE=1``) and the trace report reader.

Two things are pinned here: tracing writes exactly one JSONL line per phase and
does nothing at all when the flag is off, and ``scripts/cortex_trace_report.py``
computes nearest-rank percentiles and flags only runs strictly over budget.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401

from cortex import cortex_trace
from cortex.autojudge import AutoJudgeConfig
from cortex.relevance_pruning import run_adaptive_pruning
from cortex.retrieval import MemoryRetriever
from cortex.semantic_consolidation import run_semantic_consolidation
from cortex.store import CortexStore
from cortex.scripts.cortex_trace_report import (
    DEFAULT_BUDGETS_MS,
    default_trace_path,
    format_report,
    load_records,
    main,
    parse_budgets,
    percentile,
    summarize,
)


def _read_trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _TraceEnv(unittest.TestCase):
    """Isolate the cached flag/path between tests and restore the environment."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.trace_path = Path(self.tmp.name) / "cortex-trace.jsonl"
        self._saved = {
            key: os.environ.get(key)
            for key in ("CORTEX_TRACE", "CORTEX_TRACE_PATH", "CORTEX_HOME")
        }
        self.addCleanup(self._restore_env)
        os.environ["CORTEX_TRACE_PATH"] = str(self.trace_path)
        cortex_trace.reset()
        self.addCleanup(cortex_trace.reset)

    def _restore_env(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def enable(self) -> None:
        os.environ["CORTEX_TRACE"] = "1"
        cortex_trace.reset()


class TraceModuleTests(_TraceEnv):
    def setUp(self) -> None:
        super().setUp()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.addCleanup(self.store.close)

    def test_disabled_flag_records_nothing_and_touches_no_clock_or_file(self) -> None:
        os.environ.pop("CORTEX_TRACE", None)
        cortex_trace.reset()
        with mock.patch.object(cortex_trace, "new_run_id", side_effect=AssertionError("allocated")), \
                mock.patch.object(cortex_trace, "trace_path", side_effect=AssertionError("resolved path")):
            self.assertIsNone(cortex_trace.record_phase("recall", 12.5))
        self.assertFalse(self.trace_path.exists())

    def test_enabled_emits_one_jsonl_line_with_the_documented_fields(self) -> None:
        self.enable()
        record = cortex_trace.record_phase(
            "recall",
            12.3456,
            counts={"candidates": 40, "returned": 6},
            run_id="run-1",
            session_id="session-1",
        )
        self.assertIsNotNone(record)
        lines = _read_trace(self.trace_path)
        self.assertEqual(len(lines), 1)
        line = lines[0]
        self.assertEqual(line["phase"], "recall")
        self.assertAlmostEqual(line["duration_ms"], 12.346, places=3)
        self.assertEqual(line["counts"], {"candidates": 40, "returned": 6, "written": 0})
        self.assertEqual(line["run_id"], "run-1")
        self.assertEqual(line["session_id"], "session-1")
        self.assertTrue(line["ts"].endswith("Z"))
        self.assertIsInstance(line["pid"], int)

    def test_write_phase_records_written_count_and_session(self) -> None:
        self.enable()
        self.store.add_memory("Pinned lane ORCHID-11 is the release lane.", session_id="sess-w")
        writes = [line for line in _read_trace(self.trace_path) if line["phase"] == "write"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]["counts"]["written"], 1)
        self.assertEqual(writes[0]["session_id"], "sess-w")
        self.assertGreaterEqual(writes[0]["duration_ms"], 0.0)

    def test_recall_phase_records_candidate_and_returned_counts(self) -> None:
        self.store.add_memory(
            "Deployment lane ORCHID-11 ships through the staging ring.",
            kind="operational",
        )
        self.enable()
        retriever = MemoryRetriever(self.store)
        results, diagnostics = retriever.search_detailed(
            "which deployment lane ships through the staging ring ORCHID-11"
        )
        self.assertTrue(results)
        recalls = [line for line in _read_trace(self.trace_path) if line["phase"] == "recall"]
        self.assertEqual(len(recalls), 1)
        self.assertEqual(recalls[0]["counts"]["candidates"], diagnostics.candidate_count)
        self.assertEqual(recalls[0]["counts"]["returned"], diagnostics.selected_count)

    def test_prune_and_consolidate_record_a_phase_line(self) -> None:
        self.enable()
        config = AutoJudgeConfig(enabled=False)
        prune_report = run_adaptive_pruning(self.store, config)
        consolidate_report = run_semantic_consolidation(self.store, config)
        self.assertFalse(prune_report["enabled"])
        self.assertFalse(consolidate_report["enabled"])
        phases = [line["phase"] for line in _read_trace(self.trace_path)]
        self.assertEqual(phases, ["prune", "consolidate"])

    def test_tracing_off_starts_no_clock_on_recall_or_write(self) -> None:
        os.environ.pop("CORTEX_TRACE", None)
        cortex_trace.reset()
        with mock.patch.object(cortex_trace, "phase_start", side_effect=AssertionError("clock read")):
            self.store.add_memory("Untraced lane ORCHID-12 is quiet.")
            results, _ = MemoryRetriever(self.store).search_detailed("Untraced lane ORCHID-12")
        self.assertTrue(results)
        self.assertFalse(self.trace_path.exists())

    def test_write_failure_does_not_break_the_memory_path(self) -> None:
        self.enable()
        # A directory where the trace file should be makes the append fail.
        self.trace_path.mkdir(parents=True)
        memory_id, created = self.store.add_memory("Lane ORCHID-13 survives a broken trace sink.")
        self.assertTrue(created)
        self.assertTrue(memory_id)
        self.assertIsNotNone(self.store.get_memory(memory_id))


class TraceReportTests(_TraceEnv):
    def fixture(self) -> Path:
        path = Path(self.tmp.name) / "fixture.jsonl"
        rows = [
            {"ts": "2026-11-21T10:00:00.000Z", "phase": "recall", "duration_ms": 10, "counts": {"candidates": 5, "returned": 1}, "run_id": "r1"},
            {"ts": "2026-11-21T10:00:01.000Z", "phase": "recall", "duration_ms": 20, "counts": {"candidates": 5, "returned": 2}, "run_id": "r2"},
            {"ts": "2026-11-21T10:00:02.000Z", "phase": "recall", "duration_ms": 30, "counts": {"candidates": 5, "returned": 3}, "run_id": "r3"},
            {"ts": "2026-11-21T10:00:03.000Z", "phase": "recall", "duration_ms": 40, "counts": {"candidates": 5, "returned": 4}, "run_id": "r4"},
            {"ts": "2026-11-21T10:00:04.000Z", "phase": "recall", "duration_ms": 300, "counts": {"candidates": 5, "returned": 5}, "run_id": "slow-recall", "session_id": "s-1"},
            # Exactly at budget: not a breach.
            {"ts": "2026-11-21T10:00:05.000Z", "phase": "write", "duration_ms": 500, "counts": {"written": 1}, "run_id": "w1"},
            {"ts": "2026-11-21T10:00:06.000Z", "phase": "prune", "duration_ms": 6000, "counts": {"candidates": 9, "written": 2}, "run_id": "p1"},
            {"ts": "2026-11-21T10:00:07.000Z", "phase": "consolidate", "duration_ms": 70000, "counts": {"candidates": 4, "written": 1}, "run_id": "c1"},
            "not json at all",
        ]
        path.write_text("\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_percentile_is_nearest_rank(self) -> None:
        self.assertEqual(percentile([10, 20, 30, 40], 50), 20)
        self.assertEqual(percentile([10, 20, 30, 40], 95), 40)
        self.assertEqual(percentile([10, 20, 30, 40, 300], 50), 30)
        self.assertEqual(percentile([7], 95), 7)
        self.assertEqual(percentile([], 95), 0.0)

    def test_summary_computes_p50_p95_and_flags_only_real_breaches(self) -> None:
        records = load_records(self.fixture())
        self.assertEqual(len(records), 8)  # malformed line dropped
        summary = summarize(records, DEFAULT_BUDGETS_MS)
        by_phase = {item["phase"]: item for item in summary["phases"]}
        self.assertEqual(by_phase["recall"]["count"], 5)
        self.assertEqual(by_phase["recall"]["p50_ms"], 30)
        self.assertEqual(by_phase["recall"]["p95_ms"], 300)
        self.assertEqual(by_phase["write"]["p50_ms"], 500)
        breached = {(item["phase"], item["run_id"]) for item in summary["breaches"]}
        self.assertEqual(
            breached,
            {("recall", "slow-recall"), ("prune", "p1"), ("consolidate", "c1")},
        )

    def test_budget_overrides_clear_and_add_breaches(self) -> None:
        records = load_records(self.fixture())
        relaxed = parse_budgets(["recall=1000", "prune=10000", "consolidate=90000"])
        self.assertEqual(summarize(records, relaxed)["breaches"], [])
        strict = parse_budgets(["write=100"])
        summary = summarize(records, strict)
        self.assertIn(("write", "w1"), {(item["phase"], item["run_id"]) for item in summary["breaches"]})

    def test_bad_budget_spec_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_budgets(["recall"])
        with self.assertRaises(ValueError):
            parse_budgets(["recall=fast"])
        with self.assertRaises(ValueError):
            parse_budgets(["recall=-1"])

    def test_cli_exit_codes_and_output(self) -> None:
        path = self.fixture()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--path", str(path)])
        self.assertEqual(code, 1)
        text = out.getvalue()
        self.assertIn("p50_ms", text)
        self.assertIn("breaches: 3 run(s) over budget", text)
        self.assertIn("slow-recall", text)

        clean = io.StringIO()
        with contextlib.redirect_stdout(clean):
            code = main(["--path", str(path), "--budget", "recall=1000",
                         "--budget", "prune=10000", "--budget", "consolidate=90000"])
        self.assertEqual(code, 0)
        self.assertIn("breaches: none", clean.getvalue())

    def test_missing_trace_file_exits_two(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--path", str(Path(self.tmp.name) / "absent.jsonl")]), 2)

    def test_report_default_path_matches_the_trace_module(self) -> None:
        os.environ["CORTEX_TRACE_PATH"] = str(self.trace_path)
        cortex_trace.reset()
        self.assertEqual(default_trace_path(), cortex_trace.trace_path())
        os.environ.pop("CORTEX_TRACE_PATH", None)
        os.environ["CORTEX_HOME"] = self.tmp.name
        cortex_trace.reset()
        self.assertEqual(default_trace_path(), cortex_trace.trace_path())
        self.assertEqual(default_trace_path().name, "cortex-trace.jsonl")
        os.environ.pop("CORTEX_HOME", None)
        cortex_trace.reset()

    def test_report_text_includes_budget_column(self) -> None:
        records = load_records(self.fixture())
        text = format_report(summarize(records, DEFAULT_BUDGETS_MS))
        self.assertIn("budget_ms", text)
        self.assertIn("250", text)


if __name__ == "__main__":
    unittest.main()
