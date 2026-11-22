"""Unused links fade; the evidence behind them does not (#72)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import edge_decay
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class FactorTests(unittest.TestCase):
    def test_a_fresh_edge_is_untouched(self) -> None:
        self.assertEqual(edge_decay.factor(0.0), 1.0)

    def test_one_half_life_halves_the_weight(self) -> None:
        self.assertAlmostEqual(edge_decay.factor(edge_decay.HALF_LIFE_DAYS), 0.5, places=3)

    def test_two_half_lives_quarter_it(self) -> None:
        self.assertAlmostEqual(edge_decay.factor(edge_decay.HALF_LIFE_DAYS * 2), 0.25, places=3)

    def test_the_floor_holds(self) -> None:
        self.assertEqual(edge_decay.factor(edge_decay.HALF_LIFE_DAYS * 50),
                         edge_decay.FLOOR_RATIO)

    def test_a_zero_half_life_is_the_floor(self) -> None:
        self.assertEqual(edge_decay.factor(10.0, half_life=0), edge_decay.FLOOR_RATIO)

    def test_decay_is_monotonic(self) -> None:
        values = [edge_decay.factor(day) for day in (0, 30, 60, 120, 240)]
        self.assertEqual(values, sorted(values, reverse=True))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.left = self.memory("The dashboard runs on port 8100.")
        self.right = self.memory("Nginx proxies the dashboard to 127.0.0.1:8100.")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def edge(self, weight: float = 0.6, *, age_days: float = 0.0) -> None:
        self.store.add_edge(self.left, self.right, "related", weight=weight)
        stamp = (NOW - timedelta(days=age_days)).isoformat()
        with self.store.transaction() as conn:
            # add_edge may store the pair in canonical order, so age it from either end.
            conn.execute(
                "UPDATE edges SET last_reinforced_at = ?, created_at = ? "
                "WHERE src_id = ? OR dst_id = ?",
                (stamp, stamp, self.left, self.left),
            )

    def weight(self) -> float:
        row = self.store._conn.execute(
            "SELECT weight FROM edges WHERE src_id = ? OR dst_id = ?", (self.left, self.left)
        ).fetchone()
        self.assertIsNotNone(row, "the edge disappeared")
        return float(row[0])

    def test_a_fresh_edge_is_not_planned_for_fading(self) -> None:
        self.edge()
        self.assertEqual(edge_decay.planned(self.store, at=NOW), [])

    def test_a_stale_edge_is_planned(self) -> None:
        self.edge(age_days=edge_decay.HALF_LIFE_DAYS * 2)
        planned = edge_decay.planned(self.store, at=NOW)
        self.assertEqual(len(planned), 1)
        self.assertLess(planned[0].weight, planned[0].base_weight)

    def test_applying_records_the_base_weight_and_lowers_the_edge(self) -> None:
        self.edge(weight=0.6, age_days=edge_decay.HALF_LIFE_DAYS * 2)
        result = edge_decay.apply(self.store, at=NOW)
        self.assertEqual(result["faded"], 1)
        self.assertLess(self.weight(), 0.6)
        row = self.store._conn.execute("SELECT base_weight FROM edge_decay").fetchone()
        self.assertAlmostEqual(float(row["base_weight"]), 0.6, places=4)

    def test_the_fade_is_reversible(self) -> None:
        self.edge(weight=0.6, age_days=edge_decay.HALF_LIFE_DAYS * 2)
        edge_decay.apply(self.store, at=NOW)
        faded = self.weight()
        self.assertEqual(edge_decay.restore(self.store), 1)
        self.assertAlmostEqual(self.weight(), 0.6, places=4)
        self.assertLess(faded, self.weight())

    def test_the_evidence_survives_the_fade(self) -> None:
        self.edge(age_days=edge_decay.HALF_LIFE_DAYS * 3)
        before = edge_decay.evidence_count(self.store)
        edge_decay.apply(self.store, at=NOW)
        self.assertEqual(edge_decay.evidence_count(self.store), before)

    def test_the_edge_itself_survives(self) -> None:
        self.edge(age_days=edge_decay.HALF_LIFE_DAYS * 3)
        edge_decay.apply(self.store, at=NOW)
        count = self.store._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE src_id = ? OR dst_id = ?",
            (self.left, self.left),
        ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_a_dry_run_changes_nothing(self) -> None:
        self.edge(weight=0.6, age_days=edge_decay.HALF_LIFE_DAYS * 2)
        edge_decay.apply(self.store, at=NOW, dry_run=True)
        self.assertAlmostEqual(self.weight(), 0.6, places=4)

    def test_repeated_runs_do_not_compound_the_fade(self) -> None:
        """The base weight is recorded once, so a second run recomputes from the same anchor."""
        self.edge(weight=0.6, age_days=edge_decay.HALF_LIFE_DAYS * 2)
        edge_decay.apply(self.store, at=NOW)
        first = self.weight()
        edge_decay.apply(self.store, at=NOW)
        self.assertAlmostEqual(self.weight(), first, places=4)

    def test_the_floor_is_respected_over_a_long_idle(self) -> None:
        self.edge(weight=0.8, age_days=edge_decay.HALF_LIFE_DAYS * 100)
        edge_decay.apply(self.store, at=NOW)
        self.assertAlmostEqual(self.weight(), 0.8 * edge_decay.FLOOR_RATIO, places=3)

    def test_the_report_counts_edges_and_evidence(self) -> None:
        self.edge(age_days=edge_decay.HALF_LIFE_DAYS * 2)
        described = edge_decay.report(self.store, at=NOW)
        self.assertEqual(described["edges"], 1)
        self.assertEqual(described["would_fade"], 1)
        self.assertTrue(described["steepest"])

    def test_a_store_with_no_edges_reports_cleanly(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            described = edge_decay.report(other)
            self.assertEqual(described["edges"], 0)
            self.assertEqual(edge_decay.apply(other), {"faded": 0, "planned": 0, "dry_run": False})
        finally:
            other.close()


if __name__ == "__main__":
    unittest.main()
