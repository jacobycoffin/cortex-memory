"""How are these two memories connected? (#165)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import path_finder
from cortex.store import CortexStore


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

    def link(self, first: str, second: str, weight: float = 0.8,
             relation: str = "co_retrieval") -> None:
        self.store.add_edge(first, second, relation, weight=weight)

    def chain(self, count: int, weight: float = 0.8) -> list[str]:
        self.chains = getattr(self, "chains", 0) + 1
        ids = [self.memory(f"chain {self.chains} fact {index}") for index in range(count)]
        for left, right in zip(ids, ids[1:]):
            self.link(left, right, weight)
        return ids

    def test_a_direct_neighbour_is_one_hop(self) -> None:
        first, second = self.memory("a"), self.memory("b")
        self.link(first, second, 0.9)
        found = path_finder.find(self.store, first, second)
        self.assertTrue(found.found)
        self.assertEqual(found.hops, 1)
        self.assertEqual(found.nodes, [first, second])

    def test_the_shortest_chain_wins_over_a_longer_one(self) -> None:
        first, middle, last = self.chain(3)
        far = self.memory("far")
        self.link(first, far)
        self.link(far, middle)
        self.link(far, last)
        found = path_finder.find(self.store, first, last)
        self.assertEqual(found.hops, 2)  # first → middle → last, not the detour through `far`

    def test_the_path_is_deterministic(self) -> None:
        ids = self.chain(5)
        one = path_finder.find(self.store, ids[0], ids[4])
        two = path_finder.find(self.store, ids[0], ids[4])
        self.assertEqual(one.nodes, two.nodes)

    def test_the_same_memory_is_zero_hops(self) -> None:
        memory_id = self.memory("a")
        found = path_finder.find(self.store, memory_id, memory_id)
        self.assertTrue(found.found)
        self.assertEqual(found.hops, 0)
        self.assertEqual(found.describe(), "the same memory")

    def test_no_path_is_not_the_same_as_unrelated(self) -> None:
        first = self.memory("a")
        second = self.memory("b")
        found = path_finder.find(self.store, first, second, max_hops=3)
        self.assertFalse(found.found)
        self.assertIn("not the same as unrelated", found.reason)

    def test_the_hop_limit_is_respected(self) -> None:
        ids = self.chain(6)
        found = path_finder.find(self.store, ids[0], ids[5], max_hops=2)
        self.assertFalse(found.found)
        self.assertIn("within 2 hop", found.reason)

    def test_the_weakest_link_is_named(self) -> None:
        # Built by hand rather than from `chain()`: adding a second edge between the same pair does not
        # lower a stronger one, so a chain helper would report the weight it created first.
        first, second, third = (self.memory(f"weak {index}") for index in range(3))
        self.link(first, second, 0.9)
        self.link(second, third, 0.05)
        found = path_finder.find(self.store, first, third)
        self.assertEqual(found.hops, 2)
        self.assertAlmostEqual(found.weakest, 0.05, places=4)

    def test_the_weakest_link_appears_in_the_sentence(self) -> None:
        ids = self.chain(2, weight=0.11)
        found = path_finder.find(self.store, ids[0], ids[1])
        self.assertIn("weakest link", found.describe())

    def test_a_hub_in_the_middle_is_named(self) -> None:
        hub = self.memory("the hub")
        for index in range(30):
            self.link(hub, self.memory(f"spoke {index}"))
        first, second = self.memory("left"), self.memory("right")
        self.link(first, hub)
        self.link(hub, second)
        found = path_finder.find(self.store, first, second)
        self.assertTrue(found.found)
        self.assertEqual(found.through_hubs, [hub])

    def test_the_render_shows_each_hop(self) -> None:
        ids = self.chain(3)
        lines = path_finder.render(path_finder.find(self.store, ids[0], ids[2]))
        self.assertTrue(any("→" in line for line in lines))
        self.assertIn(ids[0][:8], lines[0])

    def test_the_render_stops_at_the_limit_and_counts_the_rest(self) -> None:
        ids = self.chain(6)
        lines = path_finder.render(path_finder.find(self.store, ids[0], ids[5]), limit=2)
        self.assertTrue(any("more hop" in line for line in lines))

    def test_a_hub_route_is_flagged_in_the_render(self) -> None:
        hub = self.memory("hub")
        for index in range(30):
            self.link(hub, self.memory(f"spoke {index}"))
        left, right = self.memory("l"), self.memory("r")
        self.link(left, hub)
        self.link(hub, right)
        rendered = "\n".join(path_finder.render(path_finder.find(self.store, left, right)))
        self.assertIn("passes through a hub", rendered)

    def test_two_ids_are_required(self) -> None:
        self.assertFalse(path_finder.find(self.store, "", "x").found)

    def test_between_reports_each_consecutive_pair(self) -> None:
        ids = self.chain(3)
        pairs = path_finder.between(self.store, ids)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(pair["found"] for pair in pairs))

    def test_degree_counts_both_directions(self) -> None:
        hub = self.memory("hub")
        self.link(hub, self.memory("a"))
        self.link(self.memory("b"), hub)
        self.assertEqual(path_finder.degree(self.store, hub), 2)

    def test_edges_come_back_in_a_stable_order(self) -> None:
        memory_id = self.memory("hub")
        others = [self.memory(f"other {index}") for index in range(4)]
        for other in others:
            self.link(memory_id, other)
        first = [item["id"] for item in path_finder.neighbours(self.store, memory_id)]
        second = [item["id"] for item in path_finder.neighbours(self.store, memory_id)]
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(others))

    def test_a_broken_store_reports_no_path_rather_than_raising(self) -> None:
        class Broken:
            _conn = None

        self.assertFalse(path_finder.find(Broken(), "a", "b").found)


if __name__ == "__main__":
    unittest.main()
