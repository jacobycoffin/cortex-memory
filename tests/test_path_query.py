"""How are these two related? The hops, with the evidence attached (#75)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import path_query
from cortex.store import CortexStore


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, entities: list[str] | None = None) -> str:
        memory_id = self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        if entities:
            with self.store.transaction() as conn:
                conn.execute("UPDATE memories SET entities_json = ? WHERE id = ?",
                             (json.dumps(entities), memory_id))
        return memory_id

    def link(self, left: str, right: str, relation: str = "related", weight: float = 0.5) -> None:
        self.store.add_edge(left, right, relation, weight=weight)

    def test_a_direct_edge_is_one_hop(self) -> None:
        left = self.memory("The dashboard runs on port 8100.")
        right = self.memory("Nginx proxies the dashboard.")
        self.link(left, right)
        path = path_query.between(self.store, left, right)
        self.assertTrue(path.found)
        self.assertEqual(len(path.hops), 1)
        self.assertEqual(path.hops[0].relation, "related")

    def test_a_two_hop_path_reports_both_hops(self) -> None:
        left = self.memory("A.")
        middle = self.memory("B.")
        right = self.memory("C.")
        self.link(left, middle)
        self.link(middle, right)
        path = path_query.between(self.store, left, right)
        self.assertEqual(len(path.hops), 2)
        self.assertEqual(path.hops[0].dst_id, middle)

    def test_traversal_is_undirected(self) -> None:
        left = self.memory("The old fact.")
        right = self.memory("The corrected fact.")
        self.link(left, right, relation="supersedes")
        self.assertTrue(path_query.between(self.store, right, left).found)

    def test_the_shortest_path_wins(self) -> None:
        left = self.memory("A.")
        direct = self.memory("B.")
        right = self.memory("C.")
        self.link(left, direct)
        self.link(direct, right)
        self.link(left, right)
        self.assertEqual(len(path_query.between(self.store, left, right).hops), 1)

    def test_the_depth_limit_is_respected(self) -> None:
        nodes = [self.memory(f"Node {index}.") for index in range(6)]
        for first, second in zip(nodes, nodes[1:]):
            self.link(first, second)
        self.assertFalse(path_query.between(self.store, nodes[0], nodes[5], depth=2).found)
        self.assertTrue(path_query.between(self.store, nodes[0], nodes[5], depth=5).found)

    def test_no_path_lists_shared_neighbours(self) -> None:
        left = self.memory("A.")
        right = self.memory("B.")
        shared = self.memory("C.")
        self.link(left, shared)
        self.link(right, shared)
        # depth 1 cannot reach B, but the shared neighbour is still reported
        path = path_query.between(self.store, left, right, depth=1)
        self.assertFalse(path.found)
        self.assertEqual(path.shared_neighbours, [shared])

    def test_no_path_lists_shared_entities(self) -> None:
        left = self.memory("A fact about the box.", entities=["proxmox", "r630"])
        right = self.memory("Another fact about the box.", entities=["proxmox", "t3code"])
        path = path_query.between(self.store, left, right)
        self.assertFalse(path.found)
        self.assertEqual(path.shared_entities, ["proxmox"])

    def test_a_phrase_resolves_to_a_memory(self) -> None:
        left = self.memory("The dashboard runs on port 8100.")
        right = self.memory("Nginx proxies the dashboard.")
        self.link(left, right)
        path = path_query.between(self.store, "port 8100", "Nginx proxies")
        self.assertTrue(path.found)

    def test_an_unknown_phrase_is_reported(self) -> None:
        path = path_query.between(self.store, "no such text anywhere", "nor here")
        self.assertFalse(path.found)
        self.assertIn("unknown memory", path.reason)

    def test_the_same_memory_is_not_a_path(self) -> None:
        left = self.memory("A.")
        path = path_query.between(self.store, left, left)
        self.assertFalse(path.found)
        self.assertIn("same memory", path.reason)

    def test_evidence_is_attached_when_it_exists(self) -> None:
        left = self.memory("A.")
        right = self.memory("B.")
        self.store.add_edge(
            left, right, "related", weight=0.5,
            evidence_type="co_retrieval", evidence_key="k1",
            explanation="two sessions retrieved these together",
        )
        path = path_query.between(self.store, left, right)
        self.assertEqual(len(path.hops), 1)
        self.assertTrue(path.hops[0].evidence)
        self.assertIn("two sessions", " ".join(path.hops[0].evidence))

    def test_a_hop_without_evidence_says_so(self) -> None:
        """An edge written straight into the table has no evidence row to quote."""
        left = self.memory("A.")
        right = self.memory("B.")
        stamp = "2026-11-22T12:00:00+00:00"
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO edges (src_id, dst_id, relation, weight, evidence_count,"
                " created_at, last_reinforced_at) VALUES (?, ?, ?, ?, 1, ?, ?)",
                (left, right, "related", 0.5, stamp, stamp),
            )
        hop = path_query.between(self.store, left, right).hops[0]
        self.assertIn("no recorded evidence", hop.describe())

    def test_a_weak_legacy_edge_quotes_its_own_caveat(self) -> None:
        """add_edge() writes an evidence row saying the link is unsupported; the hop repeats it."""
        left = self.memory("A.")
        right = self.memory("B.")
        self.link(left, right)
        hop = path_query.between(self.store, left, right).hops[0]
        self.assertTrue(hop.evidence)
        self.assertIn("without preserved supporting evidence", " ".join(hop.evidence))

    def test_the_explanation_names_the_endpoints(self) -> None:
        left = self.memory("The dashboard runs on port 8100.")
        right = self.memory("Nginx proxies the dashboard.")
        self.link(left, right)
        explanation = path_query.between(self.store, left, right).explain()
        self.assertIn("dashboard", explanation)
        self.assertIn("Nginx", explanation)

    def test_the_result_is_deterministic(self) -> None:
        left = self.memory("A.")
        first = self.memory("B.")
        second = self.memory("C.")
        end = self.memory("D.")
        self.link(left, first, weight=0.5)
        self.link(left, second, weight=0.5)
        self.link(first, end)
        self.link(second, end)
        paths = {tuple(hop.dst_id for hop in path_query.between(self.store, left, end).hops)
                 for _ in range(5)}
        self.assertEqual(len(paths), 1)

    def test_describe_returns_json_ready_hops(self) -> None:
        left = self.memory("A.")
        right = self.memory("B.")
        self.link(left, right, weight=0.42)
        described = path_query.describe(self.store, left, right)
        self.assertTrue(described["found"])
        self.assertAlmostEqual(described["hops"][0]["weight"], 0.42, places=4)
        self.assertIn("src_content", described["hops"][0])

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertFalse(path_query.describe(self.store, "x", "y")["found"])


class ProviderTests(unittest.TestCase):
    """The tool boundary: asking in chat must answer in chat."""

    def setUp(self) -> None:
        from cortex import CortexMemoryProvider

        self.tmp = tempfile.TemporaryDirectory()
        self.provider = CortexMemoryProvider(
            {
                "db_path": str(Path(self.tmp.name) / "cortex" / "test.db"),
                "auto_capture": False,
                "retrieval_threshold": 0.08,
                "pruning_mode": "shadow",
            }
        )
        self.provider.initialize("session-1", hermes_home=self.tmp.name,
                                 agent_context="primary")
        self.store = self.provider._store

    def tearDown(self) -> None:
        self.provider.shutdown()
        self.tmp.cleanup()

    def call(self, **args) -> dict:
        import json as _json

        return _json.loads(self.provider.handle_tool_call("cortex_memory", args))

    def remember(self, content: str) -> str:
        saved = self.call(action="remember", content=content, kind="semantic")
        reviewed = self.store.review_memory_creation(
            saved["proposal_id"], "remember", actor="path-test"
        )
        return str(reviewed["memory_id"])

    def test_the_path_action_answers_from_the_tool(self) -> None:
        left = self.remember("The dashboard runs on port 8100.")
        right = self.remember("Nginx proxies the dashboard to that port.")
        self.store.add_edge(left, right, "related", weight=0.5)
        answered = self.call(action="path", left=left, right=right)
        self.assertTrue(answered["found"])
        self.assertEqual(len(answered["hops"]), 1)

    def test_the_path_action_needs_both_ends(self) -> None:
        answered = self.call(action="path", left="only one")
        self.assertIn("required", str(answered))


if __name__ == "__main__":
    unittest.main()
