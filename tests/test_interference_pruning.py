"""Interference pruning (#108): settle exact duplicates in batch, queue everything that needs a person."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import interference_pruning as ip
from cortex.store import CortexStore


class PruningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, kind: str = "semantic", used: int = 0) -> str:
        """A row that duplicates `content` even though the write path would have refused it.

        `add_memory` dedupes exact content and returns the existing id, so duplicates in a real store come from
        imports and legacy rows that predate the dedupe. The test builds one the same way: write a distinct row,
        then make its content and hash match — which leaves every NOT NULL column valid.
        """
        import hashlib

        unique = f"{content} [copy {abs(hash((content, kind, used))) % 10**8}]"
        memory_id = self.store.add_memory(content=unique, kind=kind, source_type="user_turn",
                                          source_category="USER_STATED")[0]
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET content = ?, content_hash = ?, used_count = ? WHERE id = ?",
                (content, hashlib.sha256(content.encode()).hexdigest(), used, memory_id),
            )
        return memory_id

    # -- what may be automatic ---------------------------------------------------------

    def test_an_exact_duplicate_group_is_automatic(self) -> None:
        first = self.memory("the shelf needs a second coat", used=5)
        second = self.memory("the shelf needs a second coat", used=1)
        plan = ip.plan(self.store, include_contradictions=False)
        self.assertEqual(len(plan.auto_mergeable), 1)
        item = plan.auto_mergeable[0]
        self.assertEqual(item.survivor_id, first, "the more-used row stays")
        self.assertEqual(item.absorbed_id, second)

    def test_whitespace_and_case_are_not_differences(self) -> None:
        self.memory("The shelf needs a second coat", used=3)
        self.memory("  the shelf   needs a second coat ", used=1)
        plan = ip.plan(self.store, include_contradictions=False)
        self.assertEqual(len(plan.auto_mergeable), 1)

    def test_different_kinds_are_not_duplicates(self) -> None:
        self.memory("the shelf needs a second coat", kind="semantic")
        self.memory("the shelf needs a second coat", kind="episodic")
        plan = ip.plan(self.store, include_contradictions=False)
        self.assertEqual(plan.items, [])

    def test_unique_rows_produce_nothing(self) -> None:
        self.memory("one fact about the shelf")
        self.memory("another fact about the shed")
        plan = ip.plan(self.store, include_contradictions=False)
        self.assertEqual(plan.duplicate_groups, 0)
        self.assertEqual(plan.items, [])

    def test_a_group_of_three_keeps_one_and_absorbs_two(self) -> None:
        keeper = self.memory("same claim three times", used=9)
        self.memory("same claim three times", used=2)
        self.memory("same claim three times", used=1)
        plan = ip.plan(self.store, include_contradictions=False)
        self.assertEqual(len(plan.auto_mergeable), 2)
        self.assertTrue(all(item.survivor_id == keeper for item in plan.auto_mergeable))

    # -- what never is ----------------------------------------------------------------

    def test_a_near_duplicate_is_listed_but_never_automatic(self) -> None:
        """Exact duplicates are prevented at write time, so the band is where the real work is."""
        self.memory("the walnut shelf in the garage needs a second coat of paint", used=5)
        self.memory("the walnut shelf in the garage needs a second coat", used=1)
        plan = ip.plan(self.store, include_contradictions=False)
        self.assertEqual(len(plan.auto_mergeable), 0, "a near miss is never merged without a person")
        self.assertEqual(len(plan.needs_decision), 1)
        self.assertIn("near duplicate", plan.needs_decision[0].reason)

    def test_rows_that_merely_share_words_are_not_listed(self) -> None:
        self.memory("the shelf needs a second coat and the bracket is loose", used=5)
        self.memory("the shelf needs a second coat", used=1)
        plan = ip.plan(self.store, include_contradictions=False)
        self.assertEqual(plan.needs_decision, [], "0.62 overlap is two different claims, not a duplicate")

    def test_archived_rows_are_not_candidates(self) -> None:
        self.memory("a retired duplicate", used=1)
        other = self.memory("a retired duplicate", used=2)
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'archived' WHERE id = ?", (other,))
        plan = ip.plan(self.store, include_contradictions=False)
        self.assertEqual(plan.items, [])

    # -- applying ---------------------------------------------------------------------

    def test_the_default_is_a_dry_run(self) -> None:
        self.memory("duplicate claim here", used=3)
        self.memory("duplicate claim here", used=1)
        plan = ip.plan(self.store, include_contradictions=False)
        result = ip.apply(self.store, plan)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["would_merge"], 1)
        states = [row[0] for row in self.store._conn.execute(
            "SELECT state FROM memories WHERE content = 'duplicate claim here'")]
        self.assertEqual(states, ["active", "active"], "a dry run changes nothing")

    def test_applying_needs_consent(self) -> None:
        self.memory("duplicate claim here", used=3)
        self.memory("duplicate claim here", used=1)
        plan = ip.plan(self.store, include_contradictions=False)
        result = ip.apply(self.store, plan, apply=True)
        self.assertFalse(result["ok"])
        self.assertIn("consent", result["reason"])

    def test_applying_with_consent_retires_the_absorbed_row(self) -> None:
        keeper = self.memory("duplicate claim here", used=3)
        absorbed = self.memory("duplicate claim here", used=1)
        plan = ip.plan(self.store, include_contradictions=False)
        result = ip.apply(self.store, plan, apply=True, consented_by="teknium")
        self.assertTrue(result["ok"], result.get("failures"))
        self.assertEqual(result["applied"], 1)
        states = dict(self.store._conn.execute(
            "SELECT id, state FROM memories WHERE id IN (?, ?)", (keeper, absorbed)).fetchall())
        self.assertEqual(states[keeper], "active")
        self.assertNotEqual(states[absorbed], "active")

    def test_a_second_run_finds_nothing_left_to_merge(self) -> None:
        self.memory("duplicate claim here", used=3)
        self.memory("duplicate claim here", used=1)
        plan = ip.plan(self.store, include_contradictions=False)
        ip.apply(self.store, plan, apply=True, consented_by="teknium")
        again = ip.plan(self.store, include_contradictions=False)
        self.assertEqual(again.auto_mergeable, [], "idempotent: the settled pair is gone")

    # -- reporting --------------------------------------------------------------------

    def test_the_plan_is_bounded_and_counts_the_skips(self) -> None:
        for index in range(6):
            self.memory(f"repeated claim number {index}", used=2)
            self.memory(f"repeated claim number {index}", used=1)
        plan = ip.plan(self.store, limit=2, include_contradictions=False)
        self.assertEqual(len(plan.items), 2)
        self.assertGreater(plan.skips.get("over budget", 0), 0)

    def test_the_render_states_both_counts(self) -> None:
        self.memory("duplicate claim here", used=3)
        self.memory("duplicate claim here", used=1)
        lines = ip.plan(self.store, include_contradictions=False).render()
        self.assertIn("mergeable", lines[0])
        self.assertIn("need a decision", lines[0])

    def test_a_broken_store_is_unreadable(self) -> None:
        class Broken:
            _conn = None

        plan = ip.plan(Broken())
        self.assertFalse(plan.readable)
        self.assertIn("could not be read", plan.render()[0])

    def test_planning_writes_nothing(self) -> None:
        self.memory("duplicate claim here", used=3)
        self.memory("duplicate claim here", used=1)
        before = self.store._conn.execute("SELECT COUNT(*), SUM(state = 'active') FROM memories").fetchone()
        ip.plan(self.store, include_contradictions=False)
        after = self.store._conn.execute("SELECT COUNT(*), SUM(state = 'active') FROM memories").fetchone()
        self.assertEqual(tuple(before), tuple(after))


if __name__ == "__main__":
    unittest.main()
