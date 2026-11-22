"""An entity's history, in the order the store learned it (#50)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import aliases, entity_timeline
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class TimelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def add(self, content: str, *, entities=("CT116",), when: datetime | None = None,
            supersedes_id: str | None = None, kind: str = "semantic") -> str:
        memory_id = self.store.add_memory(
            content=content,
            kind=kind,
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            entities=list(entities),
            supersedes_id=supersedes_id,
        )[0]
        if when is not None:
            with self.store.transaction() as conn:
                conn.execute(
                    "UPDATE memories SET observed_at = ?, created_at = ? WHERE id = ?",
                    (when.isoformat(), when.isoformat(), memory_id),
                )
        return memory_id


class OrderingTests(TimelineTestCase):
    def test_events_are_oldest_first(self) -> None:
        self.add("The dashboard runs on port 8100.", when=NOW - timedelta(days=30))
        self.add("The dashboard is behind nginx.", when=NOW - timedelta(days=10))
        events = entity_timeline.timeline(self.store, "CT116")
        self.assertEqual(len(events), 2)
        self.assertIn("8100", events[0].content)
        self.assertIn("nginx", events[1].content)

    def test_an_unknown_entity_has_no_timeline(self) -> None:
        self.assertEqual(entity_timeline.timeline(self.store, "Nothing Here"), [])

    def test_the_summary_names_the_span(self) -> None:
        self.add("First fact about the box.", when=NOW - timedelta(days=40))
        self.add("Latest fact about the box.", when=NOW - timedelta(days=2))
        described = entity_timeline.summarise(self.store, "CT116")
        self.assertEqual(described["events"], 2)
        self.assertEqual(described["first"], (NOW - timedelta(days=40)).date().isoformat())
        self.assertEqual(described["last"], (NOW - timedelta(days=2)).date().isoformat())

    def test_everything_about_another_entity_is_excluded(self) -> None:
        self.add("About the box.", entities=("CT116",), when=NOW)
        self.add("About the laptop.", entities=("MacBook",), when=NOW)
        self.assertEqual(len(entity_timeline.timeline(self.store, "CT116")), 1)


class ChangeTests(TimelineTestCase):
    def test_a_supersession_is_marked_as_a_change(self) -> None:
        old = self.add("The dashboard runs on port 8100.", when=NOW - timedelta(days=60))
        self.add("The dashboard runs on port 8200.", when=NOW - timedelta(days=1),
                 supersedes_id=old, kind="correction")
        pairs = entity_timeline.changes(self.store, "CT116")
        self.assertEqual(len(pairs), 1)
        self.assertIn("8100", pairs[0]["was"])
        self.assertIn("8200", pairs[0]["now"])

    def test_the_timeline_marks_both_sides_of_the_change(self) -> None:
        old = self.add("The dashboard runs on port 8100.", when=NOW - timedelta(days=60))
        self.add("The dashboard runs on port 8200.", when=NOW - timedelta(days=1),
                 supersedes_id=old, kind="correction")
        events = entity_timeline.timeline(self.store, "CT116")
        marks = " ".join(event.change for event in events)
        self.assertIn("replaced by", marks)
        self.assertIn("replaces", marks)

    def test_two_facts_without_a_supersession_are_not_a_change(self) -> None:
        """A dispute is not a change: the timeline must not guess which is newer (#48)."""
        self.add("The maintenance window is Sunday.", when=NOW - timedelta(days=30))
        self.add("The maintenance window is Tuesday.", when=NOW - timedelta(days=1))
        self.assertEqual(entity_timeline.changes(self.store, "CT116"), [])

    def test_the_summary_counts_changes(self) -> None:
        old = self.add("Runs on port 8100.", when=NOW - timedelta(days=30))
        self.add("Runs on port 8200.", when=NOW, supersedes_id=old, kind="correction")
        self.assertEqual(entity_timeline.summarise(self.store, "CT116")["changes"], 1)


class AliasTests(TimelineTestCase):
    def test_a_learned_alias_widens_the_timeline(self) -> None:
        self.add("Tagged as CT116.", entities=("CT116",), when=NOW - timedelta(days=5))
        self.add("Tagged as the t3 box.", entities=("the t3 box",), when=NOW)
        self.assertEqual(len(entity_timeline.timeline(self.store, "CT116")), 1)
        aliases.add(self.store, "CT116", "the t3 box", at=NOW)
        self.assertEqual(len(entity_timeline.timeline(self.store, "CT116")), 2)

    def test_expansion_does_not_merge_unrelated_entities(self) -> None:
        self.add("Tagged as CT116.", entities=("CT116",), when=NOW)
        self.add("Tagged as MacBook.", entities=("MacBook",), when=NOW)
        aliases.add(self.store, "CT116", "the box", at=NOW)
        self.assertEqual(len(entity_timeline.timeline(self.store, "CT116")), 1)


class StateTests(TimelineTestCase):
    def test_a_suppressed_memory_still_appears_with_its_state(self) -> None:
        from cortex import suppression

        memory_id = self.add("The dashboard runs on port 8100.", when=NOW)
        suppression.suppress(self.store, memory_id, reason="user asked")
        events = entity_timeline.timeline(self.store, "CT116")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].state, "suppressed")

    def test_a_tombstoned_memory_is_excluded(self) -> None:
        memory_id = self.add("The dashboard runs on port 8100.", when=NOW)
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'tombstoned' WHERE id = ?", (memory_id,))
        self.assertEqual(entity_timeline.timeline(self.store, "CT116"), [])


class RenderTests(TimelineTestCase):
    def test_the_render_has_a_header_and_rows(self) -> None:
        self.add("The dashboard runs on port 8100.", when=NOW - timedelta(days=3))
        rendered = entity_timeline.render(self.store, "CT116")
        self.assertIn("CT116: 1 memories", rendered)
        self.assertIn("port 8100", rendered)

    def test_an_empty_entity_says_so(self) -> None:
        self.assertIn("no memories", entity_timeline.render(self.store, "CT116"))

    def test_undated_events_sort_last(self) -> None:
        dated = self.add("Dated fact about the box.", when=NOW)
        undated = self.add("Undated fact about the box.")
        with self.store.transaction() as conn:
            # created_at is NOT NULL by schema; an empty string is "no date recorded", which is
            # exactly the case the timeline has to place without inventing 1970.
            conn.execute(
                "UPDATE memories SET observed_at = '', created_at = '' WHERE id = ?",
                (undated,),
            )
        events = entity_timeline.timeline(self.store, "CT116")
        self.assertEqual(events[0].memory_id, dated)
        self.assertEqual(events[-1].memory_id, undated)


if __name__ == "__main__":
    unittest.main()
