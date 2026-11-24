"""Context as a retrieval key (#83)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import context_tags as ct
from cortex.store import CortexStore


class CompatibilityTests(unittest.TestCase):
    def tags(self, **kwargs) -> ct.Tags:
        return ct.Tags(**kwargs)

    def test_a_matching_project_boosts(self) -> None:
        match = ct.compatibility(self.tags(project="cortex"), ct.context(project="cortex"))
        self.assertGreater(match.boost, 0)
        self.assertIn("project", match.matched)

    def test_a_different_project_penalises(self) -> None:
        match = ct.compatibility(self.tags(project="temple"), ct.context(project="cortex"))
        self.assertLess(match.boost, 0)
        self.assertIn("project", match.penalised)

    def test_silence_is_neutral(self) -> None:
        match = ct.compatibility(self.tags(project="unknown"), ct.context(project="cortex"))
        self.assertEqual(match.boost, 0.0)
        self.assertIn("project", match.ignored)

    def test_an_empty_context_is_neutral(self) -> None:
        match = ct.compatibility(self.tags(project="cortex"), ct.context())
        self.assertEqual(match.boost, 0.0)

    def test_a_different_kind_is_not_evidence_against(self) -> None:
        """A note of the wrong kind can still be about exactly the right thing."""
        match = ct.compatibility(self.tags(kind="semantic"), ct.context(kind="procedure"))
        self.assertEqual(match.boost, 0.0)
        self.assertEqual(match.penalised, [])

    def test_a_matching_kind_boosts_a_little(self) -> None:
        match = ct.compatibility(self.tags(kind="procedure"), ct.context(kind="procedure"))
        self.assertGreater(match.boost, 0)

    def test_matching_who_boosts(self) -> None:
        match = ct.compatibility(self.tags(who=("jacoby",)), ct.context(who=("jacoby",)))
        self.assertGreater(match.boost, 0)
        self.assertIn("who", match.matched)

    def test_a_different_who_is_a_penalty(self) -> None:
        match = ct.compatibility(self.tags(who=("ledger",)), ct.context(who=("jacoby",)))
        self.assertLess(match.boost, 0)

    def test_empty_who_on_the_memory_is_neutral(self) -> None:
        match = ct.compatibility(self.tags(who=()), ct.context(who=("jacoby",)))
        self.assertEqual(match.boost, 0.0)

    def test_the_boost_is_capped_on_both_ends(self) -> None:
        everything = self.tags(project="cortex", who=("jacoby",), where=("r630",), when="2026-11",
                               kind="procedure")
        agreeing = ct.compatibility(everything, ct.context(
            project="cortex", who=("jacoby",), where=("r630",), when="2026-11", kind="procedure"))
        disagreeing = ct.compatibility(everything, ct.context(
            project="temple", who=("ledger",), where=("vps",), when="2025-01", kind="semantic"))
        self.assertLessEqual(agreeing.boost, ct.MAX_BOOST)
        self.assertGreaterEqual(disagreeing.boost, -ct.MAX_BOOST)

    def test_the_match_describes_itself(self) -> None:
        match = ct.compatibility(self.tags(project="cortex"), ct.context(project="cortex"))
        self.assertIn("matches project", match.describe())
        neutral = ct.compatibility(self.tags(), ct.context(project="cortex"))
        self.assertIn("neutral", neutral.describe())

    def test_the_round_trip(self) -> None:
        payload = ct.compatibility(self.tags(project="cortex"), ct.context(project="cortex"),
                                   memory_id="m1").as_dict()
        for key in ("memory_id", "boost", "multiplier", "matched", "penalised", "ignored", "sentence"):
            self.assertIn(key, payload)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, scope: dict | None = None, kind: str = "semantic") -> str:
        memory_id = self.store.add_memory(
            content=content, kind=kind, source_type="user_turn", source_category="USER_STATED",
        )[0]
        if scope is not None:
            with self.store.transaction() as conn:
                conn.execute("UPDATE memories SET scope_json = ? WHERE id = ?",
                             (json.dumps(scope), memory_id))
        return memory_id

    def test_tags_are_read_from_the_scope(self) -> None:
        memory_id = self.memory("Cortex prefers conservative defaults.",
                                scope={"project": "Cortex", "who": ["Jacoby"],
                                       "where": ["r630"], "when": "2026-11"})
        found = ct.tags_for(self.store, memory_id)
        self.assertEqual(found.project, "cortex")
        self.assertEqual(found.who, ("jacoby",))
        self.assertEqual(found.where, ("r630",))

    def test_an_unscoped_memory_is_unknown_not_wrong(self) -> None:
        memory_id = self.memory("Something old.")
        self.assertEqual(ct.tags_for(self.store, memory_id).project, "unknown")

    def test_a_malformed_scope_is_a_missing_scope(self) -> None:
        memory_id = self.memory("Broken scope.")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET scope_json = 'not json' WHERE id = ?", (memory_id,))
        self.assertEqual(ct.tags_for(self.store, memory_id).project, "unknown")

    def test_an_unknown_memory_has_no_tags(self) -> None:
        self.assertEqual(ct.tags_for(self.store, "nope"), ct.Tags())

    def test_the_boost_uses_the_stored_scope(self) -> None:
        memory_id = self.memory("Cortex prefers conservative defaults.",
                                scope={"project": "cortex"})
        self.assertGreater(ct.boost(self.store, memory_id, ct.context(project="cortex")), 1.0)
        self.assertLess(ct.boost(self.store, memory_id, ct.context(project="temple")), 1.0)

    def test_an_empty_context_means_no_boost_at_all(self) -> None:
        memory_id = self.memory("Cortex prefers conservative defaults.",
                                scope={"project": "cortex"})
        self.assertEqual(ct.boost(self.store, memory_id, ct.context()), 1.0)

    def test_rank_orders_by_compatibility(self) -> None:
        matching = self.memory("Cortex does the thing.", scope={"project": "cortex"})
        other = self.memory("Temple does the thing.", scope={"project": "temple"})
        unscoped = self.memory("Something else entirely.")
        ranked = ct.rank(self.store, [other, unscoped, matching], ct.context(project="cortex"))
        self.assertEqual([row["memory_id"] for row in ranked], [matching, unscoped, other])

    def test_applied_returns_multipliers_a_ranker_can_use(self) -> None:
        memory_id = self.memory("Cortex does the thing.", scope={"project": "cortex"})
        pairs = ct.applied(self.store, [memory_id], ct.context(project="cortex"))
        self.assertEqual(len(pairs), 1)
        self.assertGreater(pairs[0][1], 1.0)

    def test_the_report_counts_coverage(self) -> None:
        self.memory("scoped", scope={"project": "cortex", "who": ["Jacoby"]})
        self.memory("unscoped")
        described = ct.report(self.store)
        self.assertEqual(described["sampled"], 2)
        self.assertEqual(described["unscoped"], 1)
        self.assertEqual(described["scoped"]["project"], 1)

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(ct.report(self.store)["sampled"], 0)
        self.assertEqual(ct.rank(self.store, [], ct.context(project="x")), [])


if __name__ == "__main__":
    unittest.main()
