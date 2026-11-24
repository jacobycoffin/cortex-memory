"""The audit trail, written for a person rather than for a query (#185)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import audit_digest as ad
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, days_ago: int = 0, category: str = "USER_STATED") -> str:
        memory_id = self.store.add_memory(
            content=text, kind="semantic", source_type="user_turn", source_category=category,
        )[0]
        stamp = (NOW - timedelta(days=days_ago)).isoformat()
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET created_at = ? WHERE id = ?", (stamp, memory_id))
        return memory_id

    def section(self, described, name):
        for section in described.sections:
            if section.name == name:
                return section
        raise AssertionError(f"no section {name}")

    def test_the_window_is_stated(self) -> None:
        described = ad.digest(self.store, days=7, now=NOW)
        self.assertIn("2026-11-17", described.window)
        self.assertIn("2026-11-24", described.window)

    def test_written_memories_are_counted_by_source(self) -> None:
        self.memory("a fact")
        self.memory("another fact", category="DOCUMENT_EXTRACTED")
        section = self.section(ad.digest(self.store, now=NOW), "memories written")
        self.assertEqual(section.count, 2)
        self.assertEqual(section.detail.get("DOCUMENT_EXTRACTED"), 1)

    def test_a_memory_outside_the_window_is_not_counted(self) -> None:
        self.memory("an old fact", days_ago=40)
        self.assertEqual(self.section(ad.digest(self.store, days=7, now=NOW),
                                      "memories written").count, 0)

    def test_every_section_names_its_table(self) -> None:
        for section in ad.digest(self.store, now=NOW).sections:
            self.assertTrue(section.table)

    def test_an_unreadable_table_says_unavailable_not_zero(self) -> None:
        """The same `unknown != 0` rule that runs through this codebase."""
        class Broken:
            _conn = None

        described = ad.digest(Broken(), now=NOW)
        self.assertTrue(all(section.unavailable for section in described.sections))
        self.assertIn("unavailable", described.sections[0].sentence())

    def test_a_missing_consent_table_says_unavailable(self) -> None:
        section = self.section(ad.digest(self.store, now=NOW), "consent decisions")
        self.assertTrue(section.unavailable)

    def test_consent_refusals_are_counted_by_class_not_content(self) -> None:
        from cortex import consent_gate as cg

        cg.gate(self.store, "the token is abc123", now=NOW)
        described = ad.digest(self.store, now=NOW)
        section = self.section(described, "consent decisions")
        self.assertEqual(section.count, 1)
        self.assertEqual(section.detail.get("refused"), 1)
        blob = json.dumps(described.as_dict())
        self.assertNotIn("abc123", blob)

    def test_a_quiet_week_is_reported_as_silence(self) -> None:
        described = ad.digest(self.store, days=7, now=NOW)
        self.assertTrue(described.quiet)
        self.assertIn("nothing happened in this window", described.notes)

    def test_a_busy_week_is_not_silent(self) -> None:
        self.memory("a fact")
        self.assertFalse(ad.digest(self.store, now=NOW).quiet)

    def test_zero_corrections_is_a_finding(self) -> None:
        described = ad.digest(self.store, now=NOW)
        self.assertEqual(self.section(described, "corrections in force").count, 0)
        self.assertTrue(any("no correction has ever been recorded" in note
                            for note in described.notes))

    def test_a_correction_is_counted(self) -> None:
        from cortex import correction_memory as cm

        old = self.memory("the port is 8100")
        cm.record(self.store, supersedes=old, content="the port is 8101", now=NOW)
        described = ad.digest(self.store, now=NOW)
        self.assertEqual(self.section(described, "corrections in force").count, 1)
        self.assertEqual(self.section(described, "beliefs retired").count, 1)

    def test_the_notes_state_that_counts_carry_their_table(self) -> None:
        described = ad.digest(self.store, now=NOW)
        self.assertTrue(any("tables named beside them" in note for note in described.notes))

    def test_lifecycle_changes_are_grouped_by_state(self) -> None:
        memory_id = self.memory("a fact")
        with self.store.transaction() as conn:
            conn.execute("INSERT INTO lifecycle_events(memory_id, from_state, to_state, reason, "
                         "created_at) VALUES(?,?,?,?,?)",
                         (memory_id, "active", "archived", "test", NOW.isoformat()))
        section = self.section(ad.digest(self.store, now=NOW), "lifecycle changes")
        self.assertEqual(section.count, 1)
        self.assertEqual(section.detail.get("archived"), 1)

    def test_maintenance_dry_runs_are_distinguished(self) -> None:
        with self.store.transaction() as conn:
            conn.execute("INSERT INTO maintenance_log(action, details, dry_run, created_at) "
                         "VALUES(?,?,?,?)", ("decay", "{}", 1, NOW.isoformat()))
            conn.execute("INSERT INTO maintenance_log(action, details, dry_run, created_at) "
                         "VALUES(?,?,?,?)", ("decay", "{}", 0, NOW.isoformat()))
        section = self.section(ad.digest(self.store, now=NOW), "maintenance actions")
        self.assertEqual(section.count, 2)
        self.assertEqual(section.detail.get("dry runs"), 1)

    def test_all_dry_runs_are_named_as_rehearsal(self) -> None:
        """604 of 604 live maintenance actions were dry runs; the digest must say so unprompted."""
        with self.store.transaction() as conn:
            for _ in range(3):
                conn.execute("INSERT INTO maintenance_log(action, details, dry_run, created_at) "
                             "VALUES(?,?,?,?)", ("decay", "{}", 1, NOW.isoformat()))
        described = ad.digest(self.store, now=NOW)
        self.assertTrue(any("rehearsing, not acting" in note for note in described.notes))

    def test_a_real_maintenance_action_is_not_called_rehearsal(self) -> None:
        with self.store.transaction() as conn:
            conn.execute("INSERT INTO maintenance_log(action, details, dry_run, created_at) "
                         "VALUES(?,?,?,?)", ("decay", "{}", 1, NOW.isoformat()))
            conn.execute("INSERT INTO maintenance_log(action, details, dry_run, created_at) "
                         "VALUES(?,?,?,?)", ("decay", "{}", 0, NOW.isoformat()))
        described = ad.digest(self.store, now=NOW)
        self.assertFalse(any("rehearsing" in note for note in described.notes))

    def test_the_render_reads_as_sentences(self) -> None:
        self.memory("a fact")
        rendered = "\n".join(ad.render(self.store, now=NOW))
        self.assertTrue(rendered.startswith("what happened to my memory"))
        self.assertIn("memories written: 1", rendered)

    def test_the_render_of_a_broken_store_says_unavailable(self) -> None:
        class Broken:
            _conn = None

        rendered = "\n".join(ad.render(Broken(), now=NOW))
        self.assertIn("unavailable", rendered)


if __name__ == "__main__":
    unittest.main()
