"""The as-of slider surface on top of `as_of.py` (#164)."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import asof_view
from cortex.store import CortexStore

BOUNDARY = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
BEFORE = BOUNDARY - timedelta(days=30)
AFTER = BOUNDARY + timedelta(days=30)


class BrokenStore:
    """A store whose table cannot be read - the shape a half-migrated or locked database has."""

    class _Conn:
        def execute(self, *_args, **_kwargs):
            raise sqlite3.OperationalError("unable to open database file")

    _conn = _Conn()


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self._serial = 0

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, *, created_at: datetime | str, valid_to: datetime | str | None = None,
               state: str = "active", content: str = "The dashboard runs on port 8100.",
               ) -> str:
        self._serial += 1
        memory_id = f"mem-{self._serial:04d}"
        def text(value):
            if value is None:
                return None
            return value.isoformat() if isinstance(value, datetime) else str(value)
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO memories
                   (id, kind, content, content_hash, source_type, source_category, created_at,
                    updated_at, observed_at, state, valid_to)
                   VALUES (?, 'semantic', ?, ?, 'user_turn', 'USER_EXPLICIT', ?, ?, ?, ?, ?)""",
                (memory_id, content, f"hash-{memory_id}", text(created_at), text(created_at),
                 text(created_at), state, text(valid_to)),
            )
        return memory_id

    # -- the window ---------------------------------------------------------------

    def test_a_belief_inside_its_window_is_held(self) -> None:
        memory_id = self.memory(created_at=BEFORE, valid_to=AFTER)
        view = asof_view.at(self.store, BOUNDARY)
        self.assertEqual([belief.memory_id for belief in view.beliefs], [memory_id])
        self.assertEqual(view.total, 1)

    def test_a_belief_retired_exactly_at_the_boundary_is_not_held_there(self) -> None:
        memory_id = self.memory(created_at=BEFORE, valid_to=BOUNDARY)
        self.assertEqual(asof_view.at(self.store, BOUNDARY).total, 0)
        self.assertEqual(asof_view.at(self.store, BOUNDARY - timedelta(seconds=1)).total, 1)
        self.assertEqual(memory_id, asof_view.at(
            self.store, BOUNDARY - timedelta(seconds=1)).beliefs[0].memory_id)

    def test_the_boundary_belongs_to_exactly_one_of_two_adjacent_views(self) -> None:
        """Closed at both ends would count it twice; open at both ends would lose it."""
        self.memory(created_at=BEFORE, valid_to=BOUNDARY)
        earlier = asof_view.at(self.store, BOUNDARY - timedelta(microseconds=1))
        at_boundary = asof_view.at(self.store, BOUNDARY)
        self.assertEqual(earlier.total + at_boundary.total, 1)

    def test_a_memory_created_exactly_at_the_date_is_held(self) -> None:
        self.memory(created_at=BOUNDARY)
        self.assertEqual(asof_view.at(self.store, BOUNDARY).total, 1)

    def test_a_memory_created_after_the_date_is_absent(self) -> None:
        self.memory(created_at=AFTER)
        view = asof_view.at(self.store, BOUNDARY)
        self.assertEqual(view.total, 0)
        self.assertEqual(view.unknown, [])

    def test_an_open_ended_belief_appears_on_every_later_date(self) -> None:
        """No `valid_to` means "not retired", so it is still believed now."""
        memory_id = self.memory(created_at=BEFORE)
        for when in (BOUNDARY, AFTER, AFTER + timedelta(days=365)):
            self.assertIn(memory_id, asof_view.at(self.store, when).ids)

    def test_the_belief_carries_its_own_window(self) -> None:
        self.memory(created_at=BEFORE, valid_to=AFTER)
        belief = asof_view.at(self.store, BOUNDARY).beliefs[0]
        self.assertEqual(belief.since, BEFORE.isoformat())
        self.assertEqual(belief.until, AFTER.isoformat())

    def test_a_zulu_timestamp_is_understood(self) -> None:
        self.memory(created_at="2026-03-01T00:00:00Z")
        self.assertEqual(asof_view.at(self.store, BOUNDARY).total, 1)

    # -- what the view could not decide -------------------------------------------

    def test_an_unparseable_created_at_is_unknown_not_absent(self) -> None:
        memory_id = self.memory(created_at="sometime last spring")
        view = asof_view.at(self.store, BOUNDARY)
        self.assertEqual(view.total, 0)
        self.assertEqual(view.unknown[0]["memory_id"], memory_id)
        self.assertEqual(view.unknown[0]["reason"], "created_at unparseable")
        self.assertIn("undecidable", view.sentence)

    def test_a_missing_created_at_is_unknown(self) -> None:
        self.memory(created_at="")
        view = asof_view.at(self.store, BOUNDARY)
        self.assertEqual(view.unknown[0]["reason"], "created_at missing")

    def test_an_unparseable_valid_to_is_unknown(self) -> None:
        memory_id = self.memory(created_at=BEFORE, valid_to="until further notice")
        view = asof_view.at(self.store, BOUNDARY)
        self.assertEqual(view.total, 0)
        self.assertEqual(view.unknown, [{"memory_id": memory_id, "reason": "valid_to unparseable"}])

    def test_an_empty_store_reports_zero_and_says_so(self) -> None:
        view = asof_view.at(self.store, BOUNDARY)
        self.assertTrue(view.readable)
        self.assertEqual(view.total, 0)
        self.assertEqual(view.unknown, [])
        self.assertIn("held nothing", view.note)

    def test_a_broken_store_reports_unknown_not_zero(self) -> None:
        view = asof_view.at(BrokenStore(), BOUNDARY)
        self.assertFalse(view.readable)
        self.assertEqual(view.beliefs, [])
        self.assertTrue(view.unknown)
        self.assertIn("could not be read", view.unknown[0]["reason"])
        self.assertIn("cannot say", view.note)
        self.assertIn("unknown", view.sentence)

    def test_a_broken_store_does_not_leak_a_path(self) -> None:
        view = asof_view.at(BrokenStore(), BOUNDARY)
        self.assertNotIn(".db", view.sentence)

    # -- the count travels with the window ---------------------------------------

    def test_the_count_and_the_window_are_in_the_answer(self) -> None:
        self.memory(created_at=BEFORE)
        view = asof_view.at(self.store, BOUNDARY)
        self.assertEqual(view.count, view.total)
        self.assertIn("1 belief(s)", view.sentence)
        self.assertIn(BOUNDARY.isoformat(), view.sentence)
        self.assertIn("valid_to", view.sentence)
        self.assertEqual(view.as_dict()["rule"], asof_view.RULE)

    def test_a_limit_truncates_without_changing_the_total(self) -> None:
        for _ in range(5):
            self.memory(created_at=BEFORE)
        view = asof_view.at(self.store, BOUNDARY, limit=2)
        self.assertEqual(view.count, 2)
        self.assertEqual(view.total, 5)
        self.assertIn("showing 2 of 5", view.note)

    def test_a_bad_instant_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            asof_view.at(self.store, "the other day")

    # -- erasure versus demotion --------------------------------------------------

    def test_a_tombstone_never_appears_and_is_counted_as_erased(self) -> None:
        self.memory(created_at=BEFORE, state="tombstoned")
        view = asof_view.at(self.store, BOUNDARY)
        self.assertEqual(view.beliefs, [])
        self.assertEqual(view.forgotten, 1)
        self.assertIn("1 erased", view.sentence)

    def test_archived_memories_were_still_believed(self) -> None:
        memory_id = self.memory(created_at=BEFORE, state="archived")
        self.assertIn(memory_id, asof_view.at(self.store, BOUNDARY).ids)

    # -- the difference between two dates ----------------------------------------

    def test_the_diff_reports_added_and_retired(self) -> None:
        retired = self.memory(created_at=BEFORE, valid_to=BOUNDARY, content="port 8100")
        added = self.memory(created_at=BOUNDARY + timedelta(days=1), content="port 8200")
        diff = asof_view.between(self.store, BEFORE, AFTER)
        self.assertEqual([belief.memory_id for belief in diff.added], [added])
        self.assertEqual([belief.memory_id for belief in diff.retired], [retired])
        self.assertIn("1 belief(s) added and 1 retired", diff.sentence)

    def test_the_diff_counts_what_did_not_change(self) -> None:
        self.memory(created_at=BEFORE)
        diff = asof_view.between(self.store, BEFORE, AFTER)
        self.assertEqual(diff.still, 1)
        self.assertEqual(diff.added, [])
        self.assertEqual(diff.retired, [])

    def test_a_boundary_belief_is_retired_once_across_a_diff(self) -> None:
        """The memory retired at the boundary must not read as both retired and added."""
        self.memory(created_at=BEFORE, valid_to=BOUNDARY)
        diff = asof_view.between(self.store, BOUNDARY - timedelta(days=1), BOUNDARY)
        self.assertEqual(diff.retired_total, 1)
        self.assertEqual(diff.added_total, 0)

    def test_the_diff_reports_a_broken_store_as_unknown(self) -> None:
        diff = asof_view.between(BrokenStore(), BEFORE, AFTER)
        self.assertFalse(diff.readable)
        self.assertEqual(diff.added, [])
        self.assertIn("no difference is claimed", diff.note)

    def test_a_reversed_window_is_flagged(self) -> None:
        self.memory(created_at=BEFORE)
        diff = asof_view.between(self.store, AFTER, BEFORE)
        self.assertIn("backwards", diff.note)

    def test_an_unparseable_row_is_reported_once_across_a_diff(self) -> None:
        self.memory(created_at="sometime last spring")
        diff = asof_view.between(self.store, BEFORE, AFTER)
        self.assertEqual(len(diff.unknown), 1)
        # Unknown in both views, so it is neither added nor retired - and the reason is still counted.
        self.assertEqual(diff.added_total, 0)
        self.assertEqual(diff.retired_total, 0)

    # -- rendering ----------------------------------------------------------------

    def test_render_prints_the_sentence_and_the_beliefs(self) -> None:
        self.memory(created_at=BEFORE)
        lines = asof_view.render(asof_view.at(self.store, BOUNDARY))
        self.assertIn("belief(s)", lines[0])
        self.assertTrue(lines[1].startswith("  "))

    def test_render_diff_marks_additions_and_retirements(self) -> None:
        self.memory(created_at=BEFORE, valid_to=BOUNDARY)
        self.memory(created_at=BOUNDARY + timedelta(days=1))
        lines = asof_view.render_diff(asof_view.between(self.store, BEFORE, AFTER))
        self.assertTrue(any(line.startswith("  + ") for line in lines))
        self.assertTrue(any(line.startswith("  - ") for line in lines))


if __name__ == "__main__":
    unittest.main()
