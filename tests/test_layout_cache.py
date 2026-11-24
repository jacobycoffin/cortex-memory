"""Positions computed once, on the server (#162)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import layout_cache
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str) -> str:
        return self.store.add_memory(
            content=text, kind="semantic", source_type="user_turn", source_category="USER_STATED",
        )[0]

    def link(self, first: str, second: str, weight: float = 0.5) -> None:
        self.store.add_edge(first, second, "co_retrieval", weight=weight)

    def chain(self, count: int) -> list[str]:
        # Unique content per call: identical text is deduplicated by the store, which would silently
        # weld two "separate" chains into one component and make the component assertions meaningless.
        self.chains = getattr(self, "chains", 0) + 1
        ids = [self.memory(f"chain {self.chains} fact {index}") for index in range(count)]
        for left, right in zip(ids, ids[1:]):
            self.link(left, right)
        return ids

    def test_a_chain_is_one_component(self) -> None:
        ids = self.chain(4)
        groups = layout_cache.components(self.store)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0]), sorted(ids))

    def test_two_chains_are_two_components_largest_first(self) -> None:
        small = self.chain(2)
        large = self.chain(5)
        groups = layout_cache.components(self.store)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), len(large))
        self.assertEqual(sorted(groups[1]), sorted(small))

    def test_an_isolated_memory_is_not_placed(self) -> None:
        """A memory with no edges has nothing to be near; a dot for it would mean nothing."""
        lone = self.memory("nobody links to me")
        self.chain(3)
        layout_cache.compute(self.store, at=NOW)
        placed = {point["id"] for point in layout_cache.positions(self.store, now=NOW)["points"]}
        self.assertNotIn(lone, placed)

    def test_every_connected_memory_gets_a_position(self) -> None:
        ids = self.chain(5)
        described = layout_cache.compute(self.store, at=NOW)
        self.assertEqual(described["placed"], len(ids))

    def test_the_layout_is_deterministic(self) -> None:
        self.chain(6)
        first = layout_cache.compute(self.store, at=NOW)
        one = layout_cache.positions(self.store, now=NOW)["points"]
        second = layout_cache.compute(self.store, at=NOW)
        two = layout_cache.positions(self.store, now=NOW)["points"]
        self.assertEqual(one, two)
        self.assertEqual(first["placed"], second["placed"])

    def test_the_most_connected_memory_sits_at_the_component_centre(self) -> None:
        hub = self.memory("the hub")
        for index in range(4):
            self.link(hub, self.memory(f"spoke {index}"))
        layout_cache.compute(self.store, at=NOW)
        points = {point["id"]: point for point in layout_cache.positions(self.store, now=NOW)["points"]}
        self.assertEqual((points[hub]["x"], points[hub]["y"]), (0.0, 0.0))

    def test_degree_is_reported_as_a_fact(self) -> None:
        hub = self.memory("hub")
        for index in range(3):
            self.link(hub, self.memory(f"spoke {index}"))
        layout_cache.compute(self.store, at=NOW)
        points = {point["id"]: point for point in layout_cache.positions(self.store, now=NOW)["points"]}
        self.assertEqual(points[hub]["degree"], 3)

    def test_components_do_not_overlap(self) -> None:
        self.chain(3)
        self.chain(3)
        layout_cache.compute(self.store, at=NOW)
        by_component: dict[int, list[float]] = {}
        rows = self.store._conn.execute(
            "SELECT component, x FROM graph_layout").fetchall()
        for row in rows:
            by_component.setdefault(int(row["component"]), []).append(float(row["x"]))
        left, right = by_component[0], by_component[1]
        self.assertLess(max(left), min(right))

    def test_a_fresh_layout_is_not_stale(self) -> None:
        self.chain(3)
        layout_cache.compute(self.store, at=NOW)
        self.assertFalse(layout_cache.positions(self.store, now=NOW)["stale"])

    def test_an_old_layout_is_stale(self) -> None:
        self.chain(3)
        layout_cache.compute(self.store, at=NOW - timedelta(days=5))
        self.assertTrue(layout_cache.positions(self.store, now=NOW)["stale"])

    def test_nothing_placed_yet_is_reported_as_stale_and_empty(self) -> None:
        described = layout_cache.positions(self.store, now=NOW)
        self.assertEqual(described["placed"], 0)
        self.assertTrue(described["stale"])

    def test_the_age_is_stated(self) -> None:
        self.chain(2)
        layout_cache.compute(self.store, at=NOW - timedelta(hours=3))
        described = layout_cache.positions(self.store, now=NOW)
        self.assertAlmostEqual(described["age_hours"], 3.0, places=1)

    def test_a_refresh_skips_a_fresh_layout(self) -> None:
        self.chain(3)
        layout_cache.compute(self.store, at=NOW)
        described = layout_cache.refresh_if_stale(self.store, now=NOW)
        self.assertFalse(described["recomputed"])

    def test_a_refresh_recomputes_a_stale_layout(self) -> None:
        self.chain(3)
        layout_cache.compute(self.store, at=NOW - timedelta(days=3))
        described = layout_cache.refresh_if_stale(self.store, now=NOW)
        self.assertTrue(described["recomputed"])

    def test_recomputing_replaces_rather_than_duplicates(self) -> None:
        self.chain(3)
        layout_cache.compute(self.store, at=NOW)
        layout_cache.compute(self.store, at=NOW)
        rows = self.store._conn.execute("SELECT COUNT(*) FROM graph_layout").fetchone()[0]
        self.assertEqual(rows, 3)

    def test_neighbours_are_returned_with_positions(self) -> None:
        first, second = self.memory("a"), self.memory("b")
        self.link(first, second)
        layout_cache.compute(self.store, at=NOW)
        found = layout_cache.neighbours(self.store, first)
        self.assertEqual([item["id"] for item in found], [second])
        self.assertIn("x", found[0])

    def test_a_memory_with_no_neighbours_returns_nothing(self) -> None:
        lone = self.memory("alone")
        layout_cache.compute(self.store, at=NOW)
        self.assertEqual(layout_cache.neighbours(self.store, lone), [])

    def test_the_note_says_a_position_is_not_a_claim(self) -> None:
        self.chain(2)
        described = layout_cache.compute(self.store, at=NOW)
        self.assertIn("not a claim", described["note"])

    def test_a_broken_store_does_not_raise(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(layout_cache.components(Broken()), [])


if __name__ == "__main__":
    unittest.main()
