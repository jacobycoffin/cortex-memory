"""Cross-domain transfer (#113): a pattern from one project predicts another, and says so as a proposal."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import cross_domain as cd
from cortex.store import CortexStore


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, project: str, predicate: str, object_value: str,
               subject: str = "thing", *, kind: str = "semantic") -> str:
        memory_id, _ = self.store.add_memory(
            content=f"{subject} {predicate} {object_value}",
            kind=kind, source_type="user_turn", source_category="USER_STATED",
            scope={"project": project} if project else {},
            subject=subject, predicate=predicate, object_value=object_value,
        )
        return memory_id

    def two_alike_projects(self) -> None:
        """Two projects sharing three patterns, one holding a fourth the other lacks."""
        for predicate, value in (("uses", "pytest"), ("runs_on", "debian"), ("deploys_with", "docker")):
            self.memory("duelist-road", predicate, value)
            self.memory("arttools", predicate, value)
        self.memory("duelist-road", "verifies_with", "the mac runner")

    def run_record(self) -> str:
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO sleep_runs(run_id,mode,status,cutoff_at,started_at) "
                "VALUES('run-1','nightly','completed','2026-11-24T06:00:00+00:00',"
                "'2026-11-24T06:00:00+00:00')")


class PatternTests(Base):
    def test_a_pattern_is_predicate_plus_object(self) -> None:
        self.memory("duelist-road", "uses", "pytest")
        patterns, skips = cd.pattern_sets(self.store)
        self.assertIn(("uses", "pytest"), patterns["duelist-road"])

    def test_unstructured_memories_are_skipped_and_counted(self) -> None:
        self.store.add_memory("a note with no structure at all", kind="semantic",
                              source_type="user_turn", source_category="USER_STATED",
                              scope={"project": "duelist-road"})
        patterns, skips = cd.pattern_sets(self.store)
        self.assertEqual(patterns, {})
        self.assertEqual(skips.get("not structured (no predicate/object)"), 1)

    def test_an_unscoped_memory_belongs_to_no_project(self) -> None:
        self.memory("", "uses", "pytest")
        patterns, skips = cd.pattern_sets(self.store)
        self.assertEqual(patterns, {})
        self.assertEqual(skips.get("no project scope"), 1)

    def test_likeness_is_jaccard_over_patterns(self) -> None:
        self.assertAlmostEqual(cd.likeness({("a", "1"), ("b", "2")}, {("a", "1")}), 0.5, places=4)
        self.assertEqual(cd.likeness(set(), {("a", "1")}), 0.0)


class PlanTests(Base):
    def test_alike_projects_produce_a_transfer_for_the_missing_pattern(self) -> None:
        self.two_alike_projects()
        plan = cd.plan(self.store)
        patterns = {(p.source_project, p.target_project, p.pattern) for p in plan.proposals}
        self.assertIn(("duelist-road", "arttools", "verifies_with -> the mac runner"), patterns)

    def test_the_likening_is_stated(self) -> None:
        self.two_alike_projects()
        proposal = cd.plan(self.store).proposals[0]
        self.assertGreaterEqual(len(proposal.shared), cd.MIN_SHARED)
        self.assertGreaterEqual(proposal.likeness, cd.MIN_LIKENESS)
        self.assertIn("uses -> pytest", proposal.shared)

    def test_a_pattern_the_target_already_holds_is_not_proposed(self) -> None:
        self.two_alike_projects()
        plan = cd.plan(self.store)
        proposed = {p.pattern for p in plan.proposals}
        self.assertNotIn("uses -> pytest", proposed, "both projects hold it: duplication is not transfer")

    def test_evidence_is_attached_to_both_sides(self) -> None:
        self.two_alike_projects()
        proposal = next(p for p in cd.plan(self.store).proposals
                        if p.pattern == "verifies_with -> the mac runner")
        self.assertTrue(proposal.support_ids, "the proposal names the memory that established the pattern")
        self.assertTrue(proposal.shared, "and the patterns that made the projects look alike")

    def test_projects_below_the_shared_bar_are_not_compared(self) -> None:
        self.memory("duelist-road", "uses", "pytest")
        self.memory("arttools", "uses", "pytest")
        plan = cd.plan(self.store)
        self.assertEqual(plan.proposals, [], "one shared pattern is a coincidence, not a likeness")

    def test_a_plan_states_its_thresholds(self) -> None:
        self.two_alike_projects()
        lines = cd.plan(self.store).render()
        self.assertIn("likeness", lines[0])
        self.assertIn(str(cd.MIN_SHARED), lines[0])

    def test_planning_writes_nothing(self) -> None:
        self.two_alike_projects()
        before = self.store._conn.execute("SELECT COUNT(*) c FROM sleep_proposals").fetchone()["c"]
        cd.plan(self.store)
        after = self.store._conn.execute("SELECT COUNT(*) c FROM sleep_proposals").fetchone()["c"]
        self.assertEqual(before, after)

    def test_the_plan_is_bounded(self) -> None:
        # A large shared core keeps the two projects alike while both carry several one-sided patterns, which is the
        # only shape that produces more proposals than the limit.
        for index in range(9):
            self.memory("duelist-road", f"core_{index}", f"value_{index}")
            self.memory("arttools", f"core_{index}", f"value_{index}")
        for index in range(3):
            self.memory("duelist-road", f"extra_{index}", f"left_{index}")
            self.memory("arttools", f"other_{index}", f"right_{index}")
        full = cd.plan(self.store, limit=50)
        self.assertGreater(len(full.proposals), 2, "the fixture must produce more than the limit")
        plan = cd.plan(self.store, limit=2)
        self.assertEqual(len(plan.proposals), 2)

    def test_a_broken_store_is_unreadable_not_empty(self) -> None:
        class Broken:
            _conn = None

        plan = cd.plan(Broken())
        self.assertFalse(plan.readable)
        self.assertIn("could not be read", plan.render()[0])


class ProposeTests(Base):
    def test_the_default_is_a_dry_run(self) -> None:
        self.two_alike_projects()
        self.run_record()
        result = cd.propose(self.store, cd.plan(self.store), run_id="run-1")
        self.assertTrue(result["dry_run"])
        self.assertGreater(result["would_write"], 0)
        self.assertEqual(self.store._conn.execute(
            "SELECT COUNT(*) c FROM sleep_proposals").fetchone()["c"], 0)

    def test_a_proposal_needs_a_run(self) -> None:
        self.two_alike_projects()
        result = cd.propose(self.store, cd.plan(self.store), run_id="", apply=True)
        self.assertFalse(result["ok"])
        self.assertIn("run_id", result["reason"])

    def test_proposals_are_written_against_the_run(self) -> None:
        self.two_alike_projects()
        self.run_record()
        result = cd.propose(self.store, cd.plan(self.store), run_id="run-1", apply=True)
        self.assertTrue(result["ok"], result.get("reason"))
        self.assertGreater(result["written"], 0)
        stored = self.store._conn.execute(
            "SELECT kind,status,rationale FROM sleep_proposals").fetchall()
        self.assertTrue(all(row["kind"] == "cross_domain_transfer" for row in stored))
        self.assertTrue(all(row["status"] == "proposed" for row in stored))
        self.assertIn("share", stored[0]["rationale"])

    def test_the_same_suggestion_is_not_written_twice(self) -> None:
        self.two_alike_projects()
        self.run_record()
        plan = cd.plan(self.store)
        cd.propose(self.store, plan, run_id="run-1", apply=True)
        again = cd.propose(self.store, plan, run_id="run-1", apply=True)
        self.assertEqual(again["written"], 0)
        self.assertGreater(again["skipped_duplicates"], 0)

    def test_a_proposal_never_becomes_a_memory(self) -> None:
        self.two_alike_projects()
        self.run_record()
        before = self.store._conn.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"]
        cd.propose(self.store, cd.plan(self.store), run_id="run-1", apply=True)
        after = self.store._conn.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"]
        self.assertEqual(before, after, "a transfer is a proposal; asserting it is a person's decision")


if __name__ == "__main__":
    unittest.main()
