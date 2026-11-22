"""Deliberately surface a memory from another project when the match is strong (#78)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import cross_project as cp
from cortex.store import CortexStore


class SimilarityTests(unittest.TestCase):
    def test_identical_text_scores_one(self) -> None:
        self.assertEqual(cp.similarity("nginx timeout on the dashboard", "nginx timeout on the dashboard"), 1.0)

    def test_unrelated_text_scores_zero(self) -> None:
        self.assertEqual(cp.similarity("nginx timeout", "sourdough starter feeding"), 0.0)

    def test_stopwords_do_not_inflate_a_match(self) -> None:
        self.assertEqual(cp.similarity("the of and", "the of and"), 0.0)

    def test_a_contained_match_scores_one(self) -> None:
        """Containment on the shorter side: the memory is *about* the query, word for word."""
        self.assertEqual(
            cp.similarity("nginx proxy timeout dashboard",
                          "nginx proxy timeout dashboard port"), 1.0)

    def test_a_long_question_is_not_punished(self) -> None:
        question = "opencode-go lane requires a real User-Agent header"
        memory = "opencode-go requires a real User-Agent header"
        self.assertGreaterEqual(cp.similarity(question, memory), cp.MIN_SIMILARITY)

    def test_two_shared_words_are_not_a_bridge(self) -> None:
        self.assertEqual(cp.similarity("nginx proxy timeout dashboard port", "nginx proxy"), 0.0)

    def test_project_of_reads_the_scope_json(self) -> None:
        self.assertEqual(cp.project_of({"scope_json": json.dumps({"active_project": "cortex"})}),
                         "cortex")

    def test_project_of_tolerates_missing_and_broken_scope(self) -> None:
        self.assertEqual(cp.project_of({"scope_json": ""}), "")
        self.assertEqual(cp.project_of({"scope_json": "not json"}), "")
        self.assertEqual(cp.project_of({}), "")
        self.assertEqual(cp.project_of(None), "")

    def test_the_flag_is_off_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORTEX_CROSS_PROJECT", None)
            self.assertFalse(cp.enabled())

    def test_the_flag_can_be_turned_on(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_CROSS_PROJECT": "1"}):
            self.assertTrue(cp.enabled())


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, project: str | None = None) -> str:
        memory_id = self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        if project:
            with self.store.transaction() as conn:
                conn.execute("UPDATE memories SET scope_json = ? WHERE id = ?",
                             (json.dumps({"active_project": project}), memory_id))
        return memory_id

    def test_a_strong_foreign_match_is_surfaced(self) -> None:
        home = self.memory("nginx proxy timeout on the dashboard", "cortex")
        foreign = self.memory("nginx proxy timeout on the dashboard resolved by raising proxy_read_timeout",
                              "duelist-road")
        bridges = cp.find(self.store, "nginx proxy timeout on the dashboard",
                          active_project="cortex")
        ids = [bridge.memory_id for bridge in bridges]
        self.assertIn(foreign, ids)
        self.assertNotIn(home, ids)

    def test_the_same_project_is_never_a_bridge(self) -> None:
        self.memory("nginx proxy timeout on the dashboard", "cortex")
        self.assertEqual(cp.find(self.store, "nginx proxy timeout on the dashboard",
                                 active_project="cortex"), [])

    def test_a_weak_match_is_not_a_bridge(self) -> None:
        self.memory("nginx timeout notes", "duelist-road")
        self.assertEqual(cp.find(self.store, "nginx proxy timeout dashboard headers",
                                 active_project="cortex"), [])

    def test_the_cap_is_respected(self) -> None:
        for index in range(6):
            self.memory(f"nginx proxy timeout on the dashboard variant {index}", "duelist-road")
        bridges = cp.find(self.store, "nginx proxy timeout on the dashboard",
                          active_project="cortex")
        self.assertLessEqual(len(bridges), cp.MAX_BRIDGES)

    def test_the_bridge_names_its_project(self) -> None:
        self.memory("nginx proxy timeout on the dashboard", "duelist-road")
        bridge = cp.find(self.store, "nginx proxy timeout on the dashboard",
                         active_project="cortex")[0]
        self.assertEqual(bridge.project, "duelist-road")
        self.assertIn("duelist-road", bridge.render())

    def test_an_unscoped_foreign_memory_is_labelled(self) -> None:
        self.memory("nginx proxy timeout on the dashboard")
        bridge = cp.find(self.store, "nginx proxy timeout on the dashboard",
                         active_project="cortex")[0]
        self.assertEqual(bridge.project, "")
        self.assertIn("unscoped", bridge.render())

    def test_no_active_project_means_any_project_may_answer(self) -> None:
        self.memory("nginx proxy timeout on the dashboard", "duelist-road")
        self.assertTrue(cp.find(self.store, "nginx proxy timeout on the dashboard"))

    def test_an_empty_query_finds_nothing(self) -> None:
        self.memory("nginx proxy timeout on the dashboard", "duelist-road")
        self.assertEqual(cp.find(self.store, "", active_project="cortex"), [])

    def test_the_result_is_deterministic(self) -> None:
        for index in range(4):
            self.memory(f"nginx proxy timeout on the dashboard {index}", "duelist-road")
        first = [b.memory_id for b in cp.find(self.store, "nginx proxy timeout on the dashboard",
                                              active_project="cortex", limit=4)]
        again = [b.memory_id for b in cp.find(self.store, "nginx proxy timeout on the dashboard",
                                              active_project="cortex", limit=4)]
        self.assertEqual(first, again)

    def test_render_is_empty_for_no_bridges(self) -> None:
        self.assertEqual(cp.render([]), "")

    def test_the_report_splits_scoped_from_unscoped(self) -> None:
        self.memory("One fact.", "cortex")
        self.memory("Two facts.", "duelist-road")
        self.memory("Three facts.")
        described = cp.report(self.store)
        self.assertEqual(described["scoped"], 2)
        self.assertEqual(described["unscoped"], 1)
        self.assertEqual(described["projects"]["cortex"], 1)

    def test_an_empty_store_is_quiet(self) -> None:
        described = cp.report(self.store)
        self.assertEqual(described["memories"], 0)
        self.assertEqual(cp.find(self.store, "anything at all"), [])


if __name__ == "__main__":
    unittest.main()
