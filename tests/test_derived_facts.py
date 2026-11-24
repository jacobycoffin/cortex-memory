"""Derived facts, and what to do when their sources move (#57)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import derived_facts
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, confidence: float = 0.8) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT", confidence=confidence,
        )[0]

    def edit(self, memory_id: str, content: str) -> None:
        """Change what a source says — the only thing that should make an inference dirty."""
        from cortex.store import content_hash

        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET content = ?, content_hash = ?, updated_at = ? WHERE id = ?",
                (content, content_hash(content), NOW.isoformat(), memory_id),
            )

    def touch(self, memory_id: str, *, at: datetime) -> None:
        """Move `updated_at` without changing anything the source says."""
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET updated_at = ? WHERE id = ?", (at.isoformat(), memory_id))

    def fixed(self, memory_id: str, *, at: datetime) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
                (at.isoformat(), at.isoformat(), memory_id),
            )

    def build(self, *, confidence: float | None = None):
        first = self.memory("The dashboard runs on port 8100.")
        second = self.memory("The dashboard is served behind nginx.")
        self.fixed(first, at=NOW - timedelta(days=10))
        self.fixed(second, at=NOW - timedelta(days=10))
        made = derived_facts.derive(
            self.store, "The dashboard is reachable at nginx:8100.",
            [first, second], confidence=confidence, at=NOW,
        )
        return first, second, made

    def test_a_derived_fact_records_its_sources(self) -> None:
        first, second, made = self.build()
        self.assertTrue(made["created"])
        self.assertEqual(derived_facts.sources_of(self.store, made["memory_id"]),
                         sorted([first, second]))

    def test_a_derived_fact_is_marked_as_such(self) -> None:
        _first, _second, made = self.build()
        role = self.store._conn.execute(
            "SELECT record_role FROM memories WHERE id = ?", (made["memory_id"],)
        ).fetchone()[0]
        self.assertEqual(role, "derived")

    def test_the_confidence_is_capped_by_the_weakest_source(self) -> None:
        first = self.memory("Weak source one.", confidence=0.4)
        second = self.memory("Strong source two.", confidence=0.95)
        made = derived_facts.derive(self.store, "An inference.", [first, second],
                                    confidence=0.99, at=NOW)
        self.assertAlmostEqual(made["confidence"], 0.4, places=6)
        self.assertTrue(made["capped"])

    def test_asking_below_the_ceiling_is_allowed(self) -> None:
        first = self.memory("A source.", confidence=0.9)
        made = derived_facts.derive(self.store, "An inference.", [first], confidence=0.5, at=NOW)
        self.assertAlmostEqual(made["confidence"], 0.5, places=6)
        self.assertFalse(made["capped"])

    def test_a_derivation_needs_a_source(self) -> None:
        made = derived_facts.derive(self.store, "Nothing to rest on.", [])
        self.assertFalse(made["created"])
        self.assertIn("at least one source", made["reason"])

    def test_an_unknown_source_is_refused(self) -> None:
        made = derived_facts.derive(self.store, "An inference.", ["no-such-id"])
        self.assertFalse(made["created"])
        self.assertIn("unknown source", made["reason"])

    def test_a_fresh_derivation_is_clean(self) -> None:
        _first, _second, made = self.build()
        self.assertFalse(derived_facts.check(self.store, made["memory_id"]).dirty)

    def test_an_edited_source_makes_it_dirty(self) -> None:
        first, _second, made = self.build()
        self.edit(first, "The dashboard runs on port 8200 now.")
        checked = derived_facts.check(self.store, made["memory_id"])
        self.assertTrue(checked.dirty)
        self.assertIn("changed since the derivation", " ".join(checked.reasons))

    def test_a_timestamp_only_change_does_not_make_it_dirty(self) -> None:
        """The live store stamps `updated_at` in bulk during migrations; that says nothing."""
        first, _second, made = self.build()
        self.touch(first, at=NOW + timedelta(days=30))
        self.assertFalse(derived_facts.check(self.store, made["memory_id"]).dirty)

    def test_a_retired_source_makes_it_dirty(self) -> None:
        first, _second, made = self.build()
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'tombstoned' WHERE id = ?", (first,))
        self.assertTrue(derived_facts.check(self.store, made["memory_id"]).dirty)

    def test_a_replaced_source_makes_it_dirty(self) -> None:
        first, _second, made = self.build()
        replacement = self.memory("The dashboard runs on port 8200.")
        # `supersedes_id` lives on the winner's row: the replacement names the fact it replaces
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET supersedes_id = ? WHERE id = ?",
                         (first, replacement))
        reasons = " ".join(derived_facts.check(self.store, made["memory_id"]).reasons)
        self.assertIn("replaced", reasons)

    def test_revalidating_clears_a_change_that_was_already_there(self) -> None:
        first, _second, made = self.build()
        self.edit(first, "The dashboard runs on port 8200 now.")
        self.assertTrue(derived_facts.check(self.store, made["memory_id"]).dirty)
        derived_facts.revalidate(self.store, made["memory_id"],
                                 at=NOW + timedelta(hours=2))
        self.assertFalse(derived_facts.check(self.store, made["memory_id"]).dirty)

    def test_a_new_change_after_validation_makes_it_dirty_again(self) -> None:
        first, _second, made = self.build()
        derived_facts.revalidate(self.store, made["memory_id"], at=NOW + timedelta(hours=2))
        self.edit(first, "The dashboard runs on port 8300 now.")
        self.assertTrue(derived_facts.check(self.store, made["memory_id"]).dirty)

    def test_revalidating_something_that_is_not_derived_is_refused(self) -> None:
        memory_id = self.memory("An ordinary fact.")
        self.assertFalse(derived_facts.revalidate(self.store, memory_id)["revalidated"])

    def test_retracting_stops_it_being_asserted(self) -> None:
        _first, _second, made = self.build()
        self.assertTrue(derived_facts.retract(self.store, made["memory_id"], reason="wrong")["retracted"])
        state = self.store._conn.execute(
            "SELECT state FROM memories WHERE id = ?", (made["memory_id"],)
        ).fetchone()[0]
        self.assertEqual(state, "tombstoned")

    def test_retracting_keeps_the_dependency_history(self) -> None:
        _first, _second, made = self.build()
        derived_facts.retract(self.store, made["memory_id"], reason="wrong")
        rows = self.store._conn.execute(
            "SELECT active FROM memory_dependencies WHERE memory_id = ?", (made["memory_id"],)
        ).fetchall()
        self.assertTrue(rows)
        self.assertEqual({int(row[0]) for row in rows}, {0})

    def test_dependents_lists_what_rests_on_a_source(self) -> None:
        first, _second, made = self.build()
        self.assertIn(made["memory_id"], derived_facts.dependents(self.store, first))

    def test_dirty_lists_only_the_dirty_ones(self) -> None:
        first, _second, made = self.build()
        self.assertEqual(derived_facts.dirty(self.store), [])
        self.edit(first, "The dashboard runs on port 8200 now.")
        self.assertEqual([item.memory_id for item in derived_facts.dirty(self.store)],
                         [made["memory_id"]])

    def test_the_report_counts_dirty_and_clean(self) -> None:
        first, _second, made = self.build()
        described = derived_facts.report(self.store)
        self.assertEqual(described["derived"], 1)
        self.assertEqual(described["dirty"], 0)
        self.edit(first, "The dashboard runs on port 8200 now.")
        self.assertEqual(derived_facts.report(self.store)["dirty"], 1)
        self.assertTrue(described["note"])

    def test_a_derived_fact_describes_itself(self) -> None:
        first, _second, made = self.build()
        self.assertIn("clean", derived_facts.check(self.store, made["memory_id"]).describe())
        self.edit(first, "The dashboard runs on port 8200 now.")
        self.assertIn("dirty", derived_facts.check(self.store, made["memory_id"]).describe())

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(derived_facts.dirty(self.store), [])
        self.assertEqual(derived_facts.report(self.store)["derived"], 0)


if __name__ == "__main__":
    unittest.main()
