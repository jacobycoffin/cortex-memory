"""Summary regeneration (#109): a summary goes stale when its facts change, and says which ones did."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import research, summary_regen as sr
from cortex.store import CortexStore


class SummaryRegenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    # -- fixtures ---------------------------------------------------------------------

    def memory(self, content: str, *, subject: str = "shelf") -> str:
        memory_id, _ = self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn", source_category="USER_STATED",
            subject=subject, predicate="needs", object_value="paint",
        )
        return memory_id

    def candidate(self, first: str, second: str, *, title: str = "Evidence bundle") -> str:
        """A candidate shaped exactly like `research.generate_summary_candidates` writes one."""
        import uuid

        candidate_id = str(uuid.uuid4())
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO summary_candidates(candidate_id,source_signature,title,summary_text,status,created_at) "
                "VALUES(?,?,?,?,'proposed','2026-11-01T00:00:00+00:00')",
                (candidate_id, f"sig-{candidate_id}", title, "a summary"),
            )
            for ordinal, memory_id in enumerate((first, second), 1):
                content = self.store._conn.execute(
                    "SELECT content FROM memories WHERE id=?", (memory_id,)).fetchone()["content"]
                conn.execute(
                    "INSERT INTO summary_candidate_claims(candidate_id,ordinal,claim_text) VALUES(?,?,?)",
                    (candidate_id, ordinal, str(content)),
                )
                conn.execute(
                    "INSERT INTO summary_candidate_sources(candidate_id,claim_ordinal,memory_id,excerpt) "
                    "VALUES(?,?,?,?)",
                    (candidate_id, ordinal, memory_id, sr._excerpt(content)),
                )
        return candidate_id

    def edit(self, memory_id: str, content: str) -> None:
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET content=? WHERE id=?", (content, memory_id))

    def state(self, memory_id: str, state: str) -> None:
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state=? WHERE id=?", (state, memory_id))

    def candidate_status(self, candidate_id: str) -> str:
        return self.store._conn.execute(
            "SELECT status FROM summary_candidates WHERE candidate_id=?", (candidate_id,)).fetchone()["status"]

    # -- what counts as stale ---------------------------------------------------------

    def test_unchanged_facts_leave_a_summary_current(self) -> None:
        first = self.memory("the shelf needs a second coat of paint")
        second = self.memory("the shelf bracket is loose")
        candidate = self.candidate(first, second)
        plan = sr.plan(self.store)
        self.assertEqual([item.candidate_id for item in plan.current], [candidate])
        self.assertEqual(plan.stale, [])

    def test_an_edited_source_makes_the_summary_stale(self) -> None:
        first = self.memory("the shelf needs a second coat of paint")
        second = self.memory("the shelf bracket is loose")
        candidate = self.candidate(first, second)
        self.edit(first, "the shelf needs a third coat of paint")
        plan = sr.plan(self.store)
        self.assertEqual([item.candidate_id for item in plan.stale], [candidate])
        self.assertIn(sr.CHANGED, plan.stale[0].reasons)
        self.assertEqual(plan.stale[0].changed, (first,))

    def test_age_alone_does_not_make_a_summary_stale(self) -> None:
        """The candidate above is dated 2026-11-01; age is not a reason."""
        first = self.memory("the shelf needs a second coat of paint")
        second = self.memory("the shelf bracket is loose")
        self.candidate(first, second)
        plan = sr.plan(self.store)
        self.assertEqual(plan.stale, [], "staleness is about facts, not about how old the row is")

    def test_a_retired_source_is_a_different_reason_than_an_edit(self) -> None:
        first = self.memory("the shelf needs a second coat of paint")
        second = self.memory("the shelf bracket is loose")
        candidate = self.candidate(first, second)
        self.state(second, "archived")
        item = sr.plan(self.store).stale[0]
        self.assertEqual(item.candidate_id, candidate)
        self.assertIn(sr.RETIRED, item.reasons)
        self.assertNotIn(sr.CHANGED, item.reasons, "a withdrawn fact is not an edited fact")
        self.assertEqual(item.retired, (second,))

    def test_both_reasons_are_reported_together(self) -> None:
        first = self.memory("the shelf needs a second coat of paint")
        second = self.memory("the shelf bracket is loose")
        self.candidate(first, second)
        self.edit(first, "the shelf needs a third coat")
        self.state(second, "tombstoned")
        item = sr.plan(self.store).stale[0]
        self.assertEqual(set(item.reasons), {sr.CHANGED, sr.RETIRED})

    def test_the_reason_names_the_sources(self) -> None:
        first = self.memory("the shelf needs a second coat of paint")
        second = self.memory("the shelf bracket is loose")
        self.candidate(first, second)
        self.edit(second, "the shelf bracket is missing")
        item = sr.plan(self.store).stale[0]
        self.assertEqual(item.changed, (second,))
        self.assertIn("changed", item.render())

    # -- what it refuses --------------------------------------------------------------

    def test_planning_writes_nothing(self) -> None:
        first = self.memory("the shelf needs a second coat of paint")
        second = self.memory("the shelf bracket is loose")
        self.candidate(first, second)
        self.edit(first, "the shelf needs a third coat")
        before = self.store._conn.execute("SELECT status FROM summary_candidates").fetchall()[0]["status"]
        sr.plan(self.store)
        after = self.store._conn.execute("SELECT status FROM summary_candidates").fetchall()[0]["status"]
        self.assertEqual(before, after)

    def test_the_default_is_a_dry_run(self) -> None:
        first = self.memory("the shelf needs a second coat of paint")
        second = self.memory("the shelf bracket is loose")
        candidate = self.candidate(first, second)
        self.edit(first, "the shelf needs a third coat")
        result = sr.apply(self.store, sr.plan(self.store))
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["would_supersede"], 1)
        self.assertEqual(self.candidate_status(candidate), "proposed", "a dry run changes nothing")

    def test_applying_needs_consent(self) -> None:
        first = self.memory("the shelf needs a second coat of paint")
        second = self.memory("the shelf bracket is loose")
        candidate = self.candidate(first, second)
        self.edit(first, "the shelf needs a third coat")
        result = sr.apply(self.store, sr.plan(self.store), apply=True)
        self.assertFalse(result["ok"])
        self.assertIn("consent", result["reason"])
        self.assertEqual(self.candidate_status(candidate), "proposed")

    def test_applying_supersedes_and_keeps_the_record(self) -> None:
        first = self.memory("the shelf needs a second coat of paint")
        second = self.memory("the shelf bracket is loose")
        candidate = self.candidate(first, second)
        self.edit(first, "the shelf needs a third coat")
        result = sr.apply(self.store, sr.plan(self.store), apply=True, consented_by="teknium")
        self.assertTrue(result["ok"], result.get("failures"))
        self.assertEqual(result["applied"], 1)
        self.assertEqual(self.candidate_status(candidate), "superseded")
        kept = self.store._conn.execute(
            "SELECT COUNT(*) c FROM summary_candidate_sources WHERE candidate_id=?", (candidate,)).fetchone()["c"]
        self.assertEqual(kept, 2, "superseded is not deleted: the sources stay for the record")

    def test_current_candidates_are_never_touched(self) -> None:
        first = self.memory("the shelf needs a second coat of paint")
        second = self.memory("the shelf bracket is loose")
        candidate = self.candidate(first, second)
        sr.apply(self.store, sr.plan(self.store), apply=True, consented_by="teknium")
        self.assertEqual(self.candidate_status(candidate), "proposed")

    def test_a_second_run_finds_nothing_left_to_retire(self) -> None:
        first = self.memory("the shelf needs a second coat of paint")
        second = self.memory("the shelf bracket is loose")
        self.candidate(first, second)
        self.edit(first, "the shelf needs a third coat")
        sr.apply(self.store, sr.plan(self.store), apply=True, consented_by="teknium")
        again = sr.plan(self.store)
        self.assertEqual(again.stale, [], "a superseded candidate is out of the proposed set")

    def test_regeneration_proposes_and_never_promotes(self) -> None:
        """The refreshed summary re-enters review; nothing here creates recallable memory."""
        first = self.memory("the shelf needs a second coat of paint")
        second = self.memory("the shelf bracket is loose")
        self.candidate(first, second)
        self.edit(first, "the shelf needs a third coat")
        sr.apply(self.store, sr.plan(self.store), apply=True, consented_by="teknium")
        active_before = self.store._conn.execute(
            "SELECT COUNT(*) c FROM memories WHERE state='active'").fetchone()["c"]
        research.generate_summary_candidates(self.store, limit=4)
        active_after = self.store._conn.execute(
            "SELECT COUNT(*) c FROM memories WHERE state='active'").fetchone()["c"]
        self.assertEqual(active_before, active_after, "a candidate is not a memory until someone approves it")

    # -- reporting --------------------------------------------------------------------

    def test_the_plan_states_both_counts(self) -> None:
        first = self.memory("the shelf needs a second coat of paint")
        second = self.memory("the shelf bracket is loose")
        third = self.memory("the shed door sticks in winter")
        fourth = self.memory("the shed roof needs flashing")
        self.candidate(first, second)
        self.candidate(third, fourth, title="Shed bundle")
        self.edit(first, "the shelf needs a third coat")
        lines = sr.plan(self.store).render()
        self.assertIn("stale", lines[0])
        self.assertIn("current", lines[0])

    def test_the_plan_is_bounded(self) -> None:
        for index in range(4):
            left = self.memory(f"claim {index} about the shelf")
            right = self.memory(f"claim {index} about the bracket")
            self.candidate(left, right, title=f"bundle {index}")
        plan = sr.plan(self.store, limit=2)
        self.assertEqual(plan.considered, 2)

    def test_a_broken_store_is_unreadable_not_empty(self) -> None:
        class Broken:
            _conn = None

        plan = sr.plan(Broken())
        self.assertFalse(plan.readable)
        self.assertIn("could not be read", plan.render()[0])
        self.assertFalse(sr.apply(Broken(), plan, apply=True, consented_by="teknium")["ok"])

    def test_a_candidate_with_no_sources_is_skipped_loudly(self) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO summary_candidates(candidate_id,source_signature,title,summary_text,status,created_at) "
                "VALUES('orphan','sig-orphan','Orphan','text','proposed','2026-11-01T00:00:00+00:00')")
        plan = sr.plan(self.store)
        self.assertEqual(plan.items, [])
        self.assertEqual(plan.skips.get("no cited sources"), 1)


if __name__ == "__main__":
    unittest.main()
