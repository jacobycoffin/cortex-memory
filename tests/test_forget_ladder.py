"""The forget ladder walks deterministically, and cannot eat something valuable (#86)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import forget_ladder as ladder
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


def days_ago(days: float) -> str:
    return (NOW - timedelta(days=days)).isoformat()


class LadderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str = "A fact.", *, kind: str = "semantic",
               quiet_days: float = 0, state: str = "active", **kwargs) -> str:
        created = self.store.add_memory(
            content=content,
            kind=kind,
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            **{k: v for k, v in kwargs.items() if k in {"pinned", "importance", "subject",
                                                        "predicate", "object_value"}},
        )
        memory_id = created[0] if isinstance(created, tuple) else created
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET created_at = ?, last_used_at = ? WHERE id = ?",
                (days_ago(900), days_ago(quiet_days), memory_id),
            )
        if state != "active":
            self.store.set_state(memory_id, state, reason="fixture")
        return memory_id

    def plan(self, **kwargs) -> ladder.Sweep:
        return ladder.plan(self.store, at=NOW, **kwargs)


class KindSetsTheClockTests(LadderTestCase):
    def test_a_semantic_fact_goes_cold_after_its_window(self) -> None:
        self.memory("semantic fact", kind="semantic", quiet_days=130)
        self.assertEqual([s.to_state for s in self.plan().steps], ["cold"])

    def test_the_same_age_is_not_enough_for_a_semantic_fact(self) -> None:
        self.memory("semantic fact", kind="semantic", quiet_days=100)
        self.assertEqual(self.plan().steps, [])
        self.assertEqual(self.plan().skipped.get("still_warm"), 1)

    def test_an_episode_goes_cold_far_sooner(self) -> None:
        self.memory("episode note", kind="episode", quiet_days=10)
        self.assertEqual([s.to_state for s in self.plan().steps], ["cold"])

    def test_a_procedure_outlives_both(self) -> None:
        self.memory("release procedure", kind="procedural", quiet_days=200)
        self.assertEqual(self.plan().steps, [])

    def test_an_unknown_kind_gets_the_default_window(self) -> None:
        self.memory("mystery", kind="", quiet_days=200)
        self.assertEqual([s.to_state for s in self.plan().steps], ["cold"])


class OneRungPerSweepTests(LadderTestCase):
    def test_a_cold_memory_goes_to_archived_not_past_it(self) -> None:
        self.memory("cold fact", kind="semantic", quiet_days=400, state="cold")
        steps = self.plan().steps
        self.assertEqual(len(steps), 1)
        self.assertEqual((steps[0].from_state, steps[0].to_state), ("cold", "archived"))

    def test_an_archived_memory_can_reach_the_tombstone(self) -> None:
        self.memory("archived fact", kind="semantic", quiet_days=900, state="archived")
        steps = self.plan().steps
        self.assertEqual((steps[0].from_state, steps[0].to_state), ("archived", "tombstoned"))

    def test_a_tombstoned_memory_is_never_planned_again(self) -> None:
        self.memory("gone", kind="semantic", quiet_days=2000, state="tombstoned")
        self.assertEqual(self.plan().steps, [])

    def test_a_transition_is_capped_per_sweep(self) -> None:
        for index in range(6):
            self.memory(f"cold fact {index}", kind="semantic", quiet_days=200 + index)
        sweep = self.plan(max_per_transition=2)
        self.assertEqual(len(sweep.steps), 2)
        self.assertEqual(sweep.skipped.get("capped:active->cold"), 4)


class ProtectionTests(LadderTestCase):
    def test_a_pinned_memory_never_walks(self) -> None:
        self.memory("pinned rule", kind="semantic", quiet_days=2000, pinned=True)
        sweep = self.plan()
        self.assertEqual(sweep.steps, [])
        self.assertEqual(sweep.protected.get("pinned"), 1)

    def test_a_commitment_never_walks(self) -> None:
        self.memory("call the plumber", kind="prospective", quiet_days=2000)
        sweep = self.plan()
        self.assertEqual(sweep.steps, [])
        self.assertEqual(sweep.skipped.get("kind_never_walks"), 1)

    def test_a_core_member_never_walks(self) -> None:
        from cortex.promotion import promote

        memory_id = self.memory("core fact", kind="semantic", quiet_days=2000)
        promote(self.store, memory_id, by="pin")
        sweep = self.plan()
        self.assertEqual(sweep.steps, [])
        self.assertIn("core_member", sweep.protected)

    def test_recent_use_is_evidence_against_forgetting(self) -> None:
        memory_id = self.memory("used fact", kind="semantic", quiet_days=2000)
        self.store.log_access(memory_id, "used", query="q")
        sweep = self.plan()
        self.assertEqual(sweep.steps, [])
        self.assertIn("use_evidence", sweep.protected)

    def test_a_corroborated_edge_protects_a_memory(self) -> None:
        left = self.memory("left fact", kind="semantic", quiet_days=2000)
        right = self.memory("right fact", kind="semantic", quiet_days=2000)
        self.store.add_edge(left, right, "relates_to", evidence_type="fixture")
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE edges SET evidence_count = 3 WHERE src_id = ? AND dst_id = ?",
                (left, right),
            )
        sweep = self.plan()
        self.assertEqual(sweep.steps, [])
        self.assertIn("corroborated_edge", sweep.protected)

    def test_still_used_beats_long_quiet(self) -> None:
        """`last_used_at` outranks `created_at`: a fact still informing answers stays put."""
        memory_id = self.memory("old but used", kind="semantic", quiet_days=1000)
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET last_used_at = ? WHERE id = ?",
                (days_ago(3), memory_id),
            )
        self.assertEqual(self.plan().steps, [])


class ApplyTests(LadderTestCase):
    def test_a_dry_run_writes_nothing(self) -> None:
        memory_id = self.memory("dry fact", kind="semantic", quiet_days=200)
        planned, applied = ladder.sweep(self.store, at=NOW, apply_steps=False)
        self.assertEqual(len(planned.steps), 1)
        self.assertEqual(applied, {})
        self.assertEqual(self.store.get_memory(memory_id)["state"], "active")

    def test_applying_moves_the_memory_and_records_why(self) -> None:
        memory_id = self.memory("moving fact", kind="semantic", quiet_days=200)
        planned, applied = ladder.sweep(self.store, at=NOW, apply_steps=True)
        self.assertEqual(applied, {"active->cold": 1})
        self.assertEqual(self.store.get_memory(memory_id)["state"], "cold")
        events = self.store._conn.execute(
            "SELECT from_state, to_state, reason FROM lifecycle_events WHERE memory_id = ?",
            (memory_id,),
        ).fetchall()
        self.assertTrue(events)
        self.assertEqual((events[-1]["from_state"], events[-1]["to_state"]),
                         ("active", "cold"))
        self.assertIn("forget ladder", events[-1]["reason"])

    def test_a_restore_walks_back_up(self) -> None:
        memory_id = self.memory("restorable", kind="semantic", quiet_days=2000, state="archived")
        ladder.sweep(self.store, at=NOW, apply_steps=True)
        self.assertEqual(self.store.get_memory(memory_id)["state"], "tombstoned")
        self.assertTrue(ladder.restore(self.store, memory_id))
        self.assertEqual(self.store.get_memory(memory_id)["state"], "active")

    def test_a_sweep_is_idempotent_within_the_same_instant(self) -> None:
        self.memory("once", kind="semantic", quiet_days=200)
        ladder.sweep(self.store, at=NOW, apply_steps=True)
        second, _ = ladder.sweep(self.store, at=NOW, apply_steps=False)
        self.assertEqual(second.steps, [])

    def test_counts_are_reported_per_transition(self) -> None:
        for index in range(3):
            self.memory(f"a{index}", kind="episode", quiet_days=10 + index)
        planned, applied = ladder.sweep(self.store, at=NOW, apply_steps=True)
        self.assertEqual(planned.counts(), {"active->cold": 3})
        self.assertEqual(applied, {"active->cold": 3})


if __name__ == "__main__":
    unittest.main()
