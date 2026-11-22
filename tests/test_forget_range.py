"""Forget a period, parsed strictly and previewed first (#96)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import forget_range as fr
from cortex import legal_hold as lh
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class ParseTests(unittest.TestCase):
    def test_a_month_is_the_calendar_month(self) -> None:
        start, end = fr.parse_range("2026-03")
        self.assertEqual(start.isoformat(), "2026-03-01T00:00:00+00:00")
        self.assertEqual(end.isoformat(), "2026-04-01T00:00:00+00:00")

    def test_december_rolls_into_the_next_year(self) -> None:
        start, end = fr.parse_range("2026-12")
        self.assertEqual(start.year, 2026)
        self.assertEqual(end.isoformat(), "2027-01-01T00:00:00+00:00")

    def test_an_explicit_range_includes_both_ends(self) -> None:
        start, end = fr.parse_range("2026-03-01..2026-03-31")
        self.assertEqual(start.isoformat(), "2026-03-01T00:00:00+00:00")
        self.assertEqual(end.isoformat(), "2026-04-01T00:00:00+00:00")

    def test_a_single_day_is_that_day(self) -> None:
        start, end = fr.parse_range("2026-03-15")
        self.assertEqual(end - start, __import__("datetime").timedelta(days=1))

    def test_nonsense_is_refused_rather_than_guessed(self) -> None:
        for bad in ("", "March", "last month", "2026-13", "2026-03-32", "2026-3-1",
                    "2026-03-31..2026-03-01"):
            with self.assertRaises(fr.RangeError, msg=bad):
                fr.parse_range(bad)

    def test_a_reversed_range_is_refused(self) -> None:
        with self.assertRaises(fr.RangeError):
            fr.parse_range("2026-05-01..2026-04-01")


class RangeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, observed: str, kind: str = "semantic") -> str:
        memory_id = self.store.add_memory(
            content=content,
            kind=kind,
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET observed_at = ?, created_at = ? WHERE id = ?",
                (observed, observed, memory_id),
            )
        return memory_id


class MatchTests(RangeTestCase):
    def test_memories_observed_in_the_period_are_matched(self) -> None:
        inside = self.memory("learned in March", observed="2026-03-14T10:00:00+00:00")
        self.memory("learned in April", observed="2026-04-02T10:00:00+00:00")
        start, end = fr.parse_range("2026-03")
        self.assertEqual([m["memory_id"] for m in fr.matches(self.store, start, end)], [inside])

    def test_the_upper_bound_is_exclusive_and_the_lower_inclusive(self) -> None:
        first = self.memory("first instant", observed="2026-03-01T00:00:00+00:00")
        last = self.memory("last instant", observed="2026-03-31T23:59:59+00:00")
        april = self.memory("april first", observed="2026-04-01T00:00:00+00:00")
        start, end = fr.parse_range("2026-03")
        ids = [m["memory_id"] for m in fr.matches(self.store, start, end)]
        self.assertIn(first, ids)
        self.assertIn(last, ids)
        self.assertNotIn(april, ids)

    def test_archived_memories_are_not_matched(self) -> None:
        memory_id = self.memory("archived", observed="2026-03-14T10:00:00+00:00")
        self.store.set_state(memory_id, "archived", reason="test")
        start, end = fr.parse_range("2026-03")
        self.assertEqual(fr.matches(self.store, start, end), [])

    def test_content_is_not_consulted(self) -> None:
        """'What you learned in March' is about time, not about words."""
        march = self.memory("mentions nothing in particular", observed="2026-03-05T00:00:00+00:00")
        april = self.memory("mentions March explicitly", observed="2026-04-05T00:00:00+00:00")
        start, end = fr.parse_range("2026-03")
        ids = [m["memory_id"] for m in fr.matches(self.store, start, end)]
        self.assertIn(march, ids)
        self.assertNotIn(april, ids)


class PreviewTests(RangeTestCase):
    def test_a_preview_changes_nothing_and_returns_an_envelope(self) -> None:
        memory_id = self.memory("march note", observed="2026-03-14T10:00:00+00:00")
        preview = fr.preview(self.store, "2026-03", at=NOW)
        self.assertEqual(preview.ids, [memory_id])
        self.assertEqual(len(preview.envelope), 64)
        self.assertEqual(self.store.get_memory(memory_id)["state"], "active")

    def test_the_description_names_the_period_and_the_memories(self) -> None:
        memory_id = self.memory("march note", observed="2026-03-14T10:00:00+00:00")
        described = fr.preview(self.store, "2026-03", at=NOW).describe()
        self.assertIn("2026-03-01", described)
        self.assertIn(memory_id[:8], described)

    def test_the_kind_summary_tells_you_if_you_meant_that_period(self) -> None:
        self.memory("a", observed="2026-03-02T00:00:00+00:00", kind="episode")
        self.memory("b", observed="2026-03-03T00:00:00+00:00", kind="episode")
        self.memory("c", observed="2026-03-04T00:00:00+00:00", kind="semantic")
        found = fr.preview(self.store, "2026-03", at=NOW).matches
        self.assertEqual(fr.summary(self.store, found), {"episode": 2, "semantic": 1})


class ApplyTests(RangeTestCase):
    def test_applying_tombstones_the_period(self) -> None:
        memory_id = self.memory("march note", observed="2026-03-14T10:00:00+00:00")
        preview = fr.preview(self.store, "2026-03", at=NOW)
        applied = fr.apply(self.store, preview.envelope, preview.ids, at=NOW)
        self.assertTrue(applied.ok, applied.summary)
        self.assertEqual(self.store.get_memory(memory_id)["state"], "tombstoned")

    def test_an_id_outside_the_period_is_refused(self) -> None:
        self.memory("march note", observed="2026-03-14T10:00:00+00:00")
        april = self.memory("april note", observed="2026-04-14T10:00:00+00:00")
        preview = fr.preview(self.store, "2026-03", at=NOW)
        applied = fr.apply(self.store, preview.envelope, preview.ids + [april], at=NOW)
        self.assertFalse(applied.ok)
        self.assertEqual(self.store.get_memory(april)["state"], "active")

    def test_a_held_record_blocks_the_period_forget(self) -> None:
        first = self.memory("march note", observed="2026-03-14T10:00:00+00:00")
        held = self.memory("march contract", observed="2026-03-15T10:00:00+00:00")
        lh.place(self.store, [held], reason="contract")
        preview = fr.preview(self.store, "2026-03", at=NOW)
        applied = fr.apply(self.store, preview.envelope, preview.ids, at=NOW, allow_protected=True)
        self.assertFalse(applied.ok)
        self.assertIn("legal hold", applied.refused)
        self.assertEqual(self.store.get_memory(first)["state"], "active")

    def test_a_period_forget_is_receipted(self) -> None:
        from cortex import prune_receipts as rc

        self.memory("march note", observed="2026-03-14T10:00:00+00:00")
        preview = fr.preview(self.store, "2026-03", at=NOW)
        fr.apply(self.store, preview.envelope, preview.ids, at=NOW)
        receipts = rc.recent(self.store, kind="semantic_forget")
        self.assertEqual(len(receipts), 1)
        self.assertIn("range:2026-03", receipts[0].summary)

    def test_an_empty_period_previews_and_applies_harmlessly(self) -> None:
        preview = fr.preview(self.store, "2019-01", at=NOW)
        self.assertEqual(preview.matches, [])
        applied = fr.apply(self.store, preview.envelope, [], at=NOW)
        self.assertFalse(applied.ok)
        self.assertIn("no ids", applied.refused)


if __name__ == "__main__":
    unittest.main()
