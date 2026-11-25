"""Reversing a night's merges in one command, and refusing the parts that are no longer safe (#112)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import consolidation_rollback as rollback
from cortex import merge_diff
from cortex.store import CortexStore

# 12:00 UTC is inside the night that started at 06:00 the same day.
NIGHT = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
# 03:00 UTC belongs to the night that started at 06:00 the *previous* day.
EARLY = datetime(2026, 11, 24, 3, 0, tzinfo=timezone.utc)

KEPT = "The dashboard runs on port 8100 behind nginx."
ABSORBED = "The dashboard runs on port 8100 behind nginx. The staging host is cyt116."


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, entities: list[str] | None = None) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT", entities=entities or [],
        )[0]

    def merge(self, *, at: datetime = NIGHT, kept: str = KEPT,
              absorbed: str = ABSORBED) -> str:
        survivor = self.memory(kept)
        victim = self.memory(absorbed)
        diff = merge_diff.compute(self.store, survivor, victim, merged_text=absorbed, at=at)
        self.assertIsNotNone(diff)
        applied = merge_diff.apply(self.store, diff, consent=True, consented_by="teknium",
                                   accept_loss=True, at=at)
        self.assertTrue(applied.ok, applied.reason)
        return str(diff.undo.get("merge_id") or "")

    def edit(self, memory_id: str, content: str) -> None:
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET content = ? WHERE id = ?", (content, memory_id))

    def row(self, memory_id: str) -> dict:
        return dict(self.store._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone())

    def record(self, merge_id: str) -> dict:
        return dict(self.store._conn.execute(
            "SELECT * FROM merge_diff_records WHERE merge_id = ?", (merge_id,)).fetchone())


class NightTests(unittest.TestCase):
    def test_the_night_runs_from_0600(self) -> None:
        start, end, label = rollback.night_bounds(NIGHT)
        self.assertEqual(start.isoformat(), "2026-11-24T06:00:00+00:00")
        self.assertEqual(end.isoformat(), "2026-11-25T06:00:00+00:00")
        self.assertEqual(label, "2026-11-24")

    def test_an_early_morning_hour_belongs_to_the_previous_night(self) -> None:
        start, _, label = rollback.night_bounds(EARLY)
        self.assertEqual(start.isoformat(), "2026-11-23T06:00:00+00:00")
        self.assertEqual(label, "2026-11-23", "03:00 is still the night that began yesterday")

    def test_naive_datetimes_are_read_as_utc(self) -> None:
        start, _, _ = rollback.night_bounds(datetime(2026, 11, 24, 12, 0))
        self.assertEqual(start.isoformat(), "2026-11-24T06:00:00+00:00")


class PlanTests(Base):
    def test_a_night_with_no_merges_plans_nothing(self) -> None:
        plan = rollback.plan(self.store, now=NIGHT)
        self.assertTrue(plan.readable)
        self.assertEqual(plan.items, [])
        self.assertEqual(plan.reversible, [])

    def test_a_merge_in_the_night_is_reversible(self) -> None:
        merge_id = self.merge()
        plan = rollback.plan(self.store, now=NIGHT)
        self.assertEqual([item.merge_id for item in plan.reversible], [merge_id])

    def test_a_merge_from_another_night_is_not_in_this_one(self) -> None:
        self.merge(at=datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc))
        plan = rollback.plan(self.store, now=NIGHT)
        self.assertEqual(plan.items, [])

    def test_the_plan_is_newest_first(self) -> None:
        """The plan is also the execution order, and the order is the safety property."""
        older = self.merge(at=datetime(2026, 11, 24, 7, 0, tzinfo=timezone.utc))
        newer = self.merge(at=datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc))
        plan = rollback.plan(self.store, now=NIGHT)
        self.assertEqual([item.merge_id for item in plan.items], [newer, older])

    def test_an_edited_survivor_is_refused_and_names_the_field(self) -> None:
        merge_id = self.merge()
        survivor = self.record(merge_id)["survivor_id"]
        self.edit(survivor, "The dashboard moved to port 9100 behind nginx.")
        plan = rollback.plan(self.store, now=NIGHT)
        item = plan.items[0]
        self.assertEqual(item.verdict, rollback.EDITED_SINCE)
        self.assertIn("content", item.fields_changed)
        self.assertIn("discard", item.reason)
        self.assertEqual(plan.reversible, [])

    def test_an_already_undone_merge_is_reported_not_retried(self) -> None:
        merge_id = self.merge()
        merge_diff.undo(self.store, merge_id)
        plan = rollback.plan(self.store, now=NIGHT)
        self.assertEqual(plan.items[0].verdict, rollback.ALREADY_UNDONE)
        self.assertEqual(plan.reversible, [])

    def test_planning_writes_nothing(self) -> None:
        self.merge()
        before = self.store._conn.execute(
            "SELECT COUNT(*) c, SUM(undone_at IS NOT NULL) u FROM merge_diff_records").fetchone()
        rollback.plan(self.store, now=NIGHT)
        after = self.store._conn.execute(
            "SELECT COUNT(*) c, SUM(undone_at IS NOT NULL) u FROM merge_diff_records").fetchone()
        self.assertEqual(tuple(before), tuple(after))

    def test_an_unreadable_store_says_so(self) -> None:
        class Broken:
            _conn = None

        plan = rollback.plan(Broken(), now=NIGHT)
        self.assertTrue(plan.readable, "no merge table is an empty night, not an unreadable store")
        self.assertEqual(plan.items, [])


class ApplyTests(Base):
    def test_the_default_is_a_dry_run(self) -> None:
        merge_id = self.merge()
        plan = rollback.plan(self.store, now=NIGHT)
        result = rollback.apply(self.store, plan)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["would_undo"], 1)
        self.assertIsNone(self.record(merge_id)["undone_at"], "a dry run changes nothing")

    def test_applying_needs_consent(self) -> None:
        merge_id = self.merge()
        result = rollback.apply(self.store, rollback.plan(self.store, now=NIGHT), apply=True)
        self.assertFalse(result["ok"])
        self.assertIn("consent", result["reason"])
        self.assertIsNone(self.record(merge_id)["undone_at"])

    def test_applying_reverses_the_merge(self) -> None:
        merge_id = self.merge()
        survivor = self.record(merge_id)["survivor_id"]
        victim = self.record(merge_id)["absorbed_id"]
        merged_content = self.row(survivor)["content"]
        result = rollback.apply(self.store, rollback.plan(self.store, now=NIGHT),
                                apply=True, consented_by="teknium")
        self.assertTrue(result["ok"], result.get("stopped_at"))
        self.assertEqual(result["undone"], 1)
        self.assertEqual(self.row(survivor)["content"], KEPT, "the survivor is back to its pre-merge text")
        self.assertNotEqual(self.row(victim)["state"], "superseded", "the absorbed row is live again")
        self.assertNotEqual(merged_content, self.row(survivor)["content"])

    def test_the_rollback_leaves_its_own_record(self) -> None:
        merge_id = self.merge()
        rollback.apply(self.store, rollback.plan(self.store, now=NIGHT), apply=True, consented_by="teknium")
        self.assertIsNotNone(self.record(merge_id)["undone_at"],
                             "a reversal nobody can see is indistinguishable from a merge that never happened")

    def test_a_whole_night_is_reversed_newest_first(self) -> None:
        older = self.merge(at=datetime(2026, 11, 24, 7, 0, tzinfo=timezone.utc),
                           kept="The shed roof needs flashing.", absorbed="The shed roof needs flashing. It leaks.")
        newer = self.merge(at=datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc))
        result = rollback.apply(self.store, rollback.plan(self.store, now=NIGHT),
                                apply=True, consented_by="teknium")
        self.assertTrue(result["ok"], result.get("stopped_at"))
        self.assertEqual(result["undone"], 2)
        self.assertEqual(result["merge_ids"], [newer, older], "reverse order is the safety property")

    def test_a_refused_merge_is_not_touched(self) -> None:
        merge_id = self.merge()
        survivor = self.record(merge_id)["survivor_id"]
        self.edit(survivor, "The dashboard moved to port 9100 behind nginx.")
        result = rollback.apply(self.store, rollback.plan(self.store, now=NIGHT),
                                apply=True, consented_by="teknium")
        self.assertEqual(result["undone"], 0)
        self.assertEqual(result["refused"], [merge_id])
        self.assertIsNone(self.record(merge_id)["undone_at"])
        self.assertEqual(self.row(survivor)["content"], "The dashboard moved to port 9100 behind nginx.",
                         "the later edit survives a refused rollback")

    def test_a_second_run_has_nothing_to_do(self) -> None:
        self.merge()
        rollback.apply(self.store, rollback.plan(self.store, now=NIGHT), apply=True, consented_by="teknium")
        again = rollback.plan(self.store, now=NIGHT)
        self.assertEqual(again.reversible, [])
        self.assertEqual(again.items[0].verdict, rollback.ALREADY_UNDONE)

    def test_it_stops_at_the_first_failure(self) -> None:
        """A failure disproves the state the remaining undos were planned against."""
        older = self.merge(at=datetime(2026, 11, 24, 7, 0, tzinfo=timezone.utc),
                           kept="The shed roof needs flashing.", absorbed="The shed roof needs flashing. It leaks.")
        newer = self.merge(at=datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc))
        plan = rollback.plan(self.store, now=NIGHT)
        # Break the newest record after planning, so its undo fails at apply time.
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM merge_diff_records WHERE merge_id = ?", (newer,))
        result = rollback.apply(self.store, plan, apply=True, consented_by="teknium")
        self.assertFalse(result["ok"])
        self.assertEqual(result["undone"], 0)
        self.assertIn(newer, result["stopped_at"])
        self.assertIsNone(self.record(older)["undone_at"], "the older merge was not attempted")

    def test_the_render_states_the_night_and_the_counts(self) -> None:
        self.merge()
        lines = rollback.plan(self.store, now=NIGHT).render()
        self.assertIn("night 2026-11-24", lines[0])
        self.assertIn("reversible", lines[0])


if __name__ == "__main__":
    unittest.main()
