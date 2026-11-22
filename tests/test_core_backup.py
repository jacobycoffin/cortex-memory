"""So the profile can be read without Cortex, and rebuilt (#131)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.core_backup import FRONT, collect, export, parse, restore
from cortex.promotion import members, promote
from cortex.store import CortexStore

FACTS = (
    ("Jacoby prefers short answers with no preamble.", "preference", True),
    ("The maintenance window is Sunday 02:00 on CT116.", "semantic", True),
    ("Never change the T3 release channel without asking.", "semantic", False),
)


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        for content, kind, pinned in FACTS:
            created = self.store.add_memory(
                content=content,
                kind=kind,
                source_type="user_turn",
                source_category="USER_EXPLICIT",
                pinned=pinned,
                importance=0.8,
            )
            if not pinned:
                promote(self.store, created[0] if isinstance(created, tuple) else created, by="pin")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_the_file_is_readable_by_a_person(self) -> None:
        backup = export(self.store)
        self.assertTrue(backup.text.startswith(FRONT))
        for content, _kind, _pinned in FACTS:
            self.assertIn(content, backup.text)

    def test_it_contains_a_machine_readable_block(self) -> None:
        payload = json.loads(export(self.store).text.split("```json")[1].split("```")[0])
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(len(payload["entries"]), len(FACTS))

    def test_it_includes_everything_in_the_profile(self) -> None:
        entries = collect(self.store)
        self.assertEqual(len(entries), len(FACTS))
        self.assertTrue(all(entry["content"] for entry in entries))

    def test_the_export_is_deterministic(self) -> None:
        self.assertEqual(export(self.store).text, export(self.store).text)

    def test_the_size_cap_is_respected(self) -> None:
        backup = export(self.store, max_chars=120)
        self.assertGreaterEqual(len(backup.entries), 1)
        self.assertLess(len(backup.text), 2000)

    def test_parsing_the_markdown_works_without_the_json(self) -> None:
        text = "# Core memory backup\n\n- [abcd1234] A fact worth keeping. [pin]\n"
        entries = parse(text)
        self.assertEqual(entries[0]["content"], "A fact worth keeping.")
        self.assertEqual(entries[0]["id"], "abcd1234")


class RestoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.source = CortexStore(Path(self.tmp.name) / "source.db")
        for content, kind, pinned in FACTS:
            created = self.source.add_memory(
                content=content,
                kind=kind,
                source_type="user_turn",
                source_category="USER_EXPLICIT",
                pinned=pinned,
                importance=0.8,
            )
            if not pinned:
                promote(self.source, created[0] if isinstance(created, tuple) else created)
        self.backup = export(self.source).text

    def tearDown(self) -> None:
        self.source.close()
        self.tmp.cleanup()

    def test_a_restore_rebuilds_the_profile_in_a_different_database(self) -> None:
        """Different UUIDs, same facts: matching on content is the whole point."""
        other = CortexStore(Path(self.tmp.name) / "other.db")
        try:
            for content, kind, _pinned in FACTS:
                other.add_memory(
                    content=content,
                    kind=kind,
                    source_type="user_turn",
                    source_category="USER_EXPLICIT",
                )
            result = restore(other, self.backup)
            self.assertEqual(len(result.restored), len(FACTS), result.summary)
            self.assertEqual(len(members(other)), len(FACTS))
        finally:
            other.close()

    def test_a_restore_restores_pins(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "pins.db")
        try:
            for content, kind, _pinned in FACTS:
                other.add_memory(
                    content=content, kind=kind, source_type="user_turn",
                    source_category="USER_EXPLICIT",
                )
            restore(other, self.backup)
            pinned = other._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE pinned = 1"
            ).fetchone()[0]
            self.assertEqual(pinned, 2)
        finally:
            other.close()

    def test_missing_facts_are_reported_not_invented(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            result = restore(other, self.backup)
            self.assertEqual(result.restored, [])
            self.assertEqual(len(result.missing), len(FACTS))
            self.assertEqual(members(other), [])
        finally:
            other.close()

    def test_restoring_twice_is_harmless(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "twice.db")
        try:
            for content, kind, _pinned in FACTS:
                other.add_memory(
                    content=content, kind=kind, source_type="user_turn",
                    source_category="USER_EXPLICIT",
                )
            restore(other, self.backup)
            restore(other, self.backup)
            self.assertEqual(len(members(other)), len(FACTS))
        finally:
            other.close()

    def test_restore_survives_a_corrupt_json_block(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "corrupt.db")
        try:
            for content, kind, _pinned in FACTS:
                other.add_memory(
                    content=content, kind=kind, source_type="user_turn",
                    source_category="USER_EXPLICIT",
                )
            broken = "# Core memory backup\n\n```json\n{not json}\n```\n\n- [ab12] Jacoby prefers short answers with no preamble. [pin]\n"
            result = restore(other, broken)
            self.assertEqual(len(result.restored), 1, result.summary)
        finally:
            other.close()


if __name__ == "__main__":
    unittest.main()
