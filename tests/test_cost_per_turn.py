"""Tests for scripts/cost_per_turn.py — arithmetic first, then the schema tolerance."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cost_per_turn.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cost_per_turn", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_db(path: Path, rows: list[tuple], aux: list[tuple] | None = None) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE recall_runs (
               recall_id TEXT, session_id TEXT, query TEXT, mode TEXT, reason TEXT,
               requested_limit INTEGER, token_budget INTEGER, candidate_count INTEGER,
               selected_count INTEGER, estimated_tokens INTEGER, prepare_ms REAL,
               abstained INTEGER, created_at TEXT, task_id TEXT, stage_ms_json TEXT,
               rendered_count INTEGER, rendered_tokens INTEGER)"""
    )
    conn.execute(
        """CREATE TABLE review_copilot_interpretations (
               input_tokens INTEGER, output_tokens INTEGER)"""
    )
    for row in rows:
        conn.execute(
            """INSERT INTO recall_runs (session_id, estimated_tokens, rendered_tokens,
                                        abstained, candidate_count, selected_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            row,
        )
    for row in aux or []:
        conn.execute("INSERT INTO review_copilot_interpretations VALUES (?, ?)", row)
    conn.commit()
    conn.close()


class PercentileTests(unittest.TestCase):
    def test_nearest_rank(self) -> None:
        module = load_module()
        self.assertEqual(module.percentile([10, 20, 30], 0.50), 20)
        self.assertEqual(module.percentile([10, 20, 30], 0.95), 30)
        self.assertEqual(module.percentile([], 0.95), 0.0)

    def test_single_value(self) -> None:
        module = load_module()
        self.assertEqual(module.percentile([7], 0.95), 7)


class CollectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.tmp = Path(self.id().replace(".", "_") + ".db")

    def tearDown(self) -> None:
        self.tmp.unlink(missing_ok=True)

    def test_mean_and_percentiles(self) -> None:
        build_db(self.tmp, [
            ("s1", 100, 0, 0, 10, 3, "2026-11-01T00:00:00+00:00"),
            ("s1", 200, 0, 0, 12, 4, "2026-11-01T00:01:00+00:00"),
            ("s2", 300, 0, 1, 8, 0, "2026-11-02T00:00:00+00:00"),
        ])
        stats = self.module.collect(self.module.connect(self.tmp), None)
        self.assertEqual(stats["turns"], 3)
        self.assertEqual(stats["sessions"], 2)
        self.assertEqual(stats["tokens_total"], 600)
        self.assertEqual(stats["tokens_mean"], 200.0)
        self.assertEqual(stats["tokens_p95"], 300.0)
        self.assertEqual(stats["abstained"], 1)
        self.assertEqual(stats["token_source"], "estimated_tokens")
        self.assertIn("rendered_tokens", stats["warning"])

    def test_prefers_rendered_when_present(self) -> None:
        build_db(self.tmp, [
            ("s1", 999, 120, 0, 10, 3, "2026-11-01T00:00:00+00:00"),
            ("s1", 999, 180, 0, 10, 3, "2026-11-01T00:01:00+00:00"),
        ])
        stats = self.module.collect(self.module.connect(self.tmp), None)
        self.assertEqual(stats["token_source"], "rendered_tokens")
        self.assertEqual(stats["tokens_mean"], 150.0)
        self.assertIsNone(stats["warning"])

    def test_auxiliary_tokens_summed(self) -> None:
        build_db(
            self.tmp,
            [("s1", 100, 0, 0, 10, 3, "2026-11-01T00:00:00+00:00")],
            aux=[(1000, 500), (2000, 250)],
        )
        stats = self.module.collect(self.module.connect(self.tmp), None)
        self.assertEqual(stats["aux_calls"], 2)
        self.assertEqual(stats["aux_input_tokens"], 3000)
        self.assertEqual(stats["aux_output_tokens"], 750)

    def test_empty_window(self) -> None:
        build_db(self.tmp, [])
        stats = self.module.collect(self.module.connect(self.tmp), 30)
        self.assertEqual(stats["turns"], 0)
        self.assertIn("warning", stats)


class PricingTests(unittest.TestCase):
    def test_per_turn_cost_math(self) -> None:
        # 500 tokens/turn at $0.15/M is $0.000075, i.e. $0.225/month at 100 turns/day
        per_turn = 500 / 1_000_000 * 0.15
        self.assertAlmostEqual(per_turn, 0.000075, places=9)
        self.assertAlmostEqual(per_turn * 100 * 30, 0.225, places=6)

    def test_missing_db_exits_2(self) -> None:
        module = load_module()
        with self.assertRaises(SystemExit) as ctx:
            module.connect(Path("/nonexistent/cortex.db"))
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
