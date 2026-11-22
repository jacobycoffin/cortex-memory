"""A semantic forget is a proposal until an envelope is applied (#89)."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import forget_preview as fp
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
CONTENTS = (
    "The old NAS was replaced by a Synology DS923 in the basement rack.",
    "Old NAS backups ran nightly to the USB drive on the shelf.",
    "Temple uses a crimson accent colour for the primary action.",
)


class PreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        for content in CONTENTS:
            self.store.add_memory(
                content=content,
                kind="semantic",
                source_type="user_turn",
                source_category="USER_EXPLICIT",
                importance=0.6,
            )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_preview_returns_matches_and_an_envelope(self) -> None:
        preview = fp.preview(self.store, "old NAS replacement", limit=3)
        self.assertTrue(preview.matches)
        self.assertEqual(len(preview.envelope), 64)
        self.assertTrue(preview.ids)

    def test_a_preview_writes_no_state_change(self) -> None:
        before = self.store._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE state = 'tombstoned'"
        ).fetchone()[0]
        fp.preview(self.store, "old NAS replacement", limit=3)
        after = self.store._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE state = 'tombstoned'"
        ).fetchone()[0]
        self.assertEqual(before, after)

    def test_the_envelope_is_addressable_and_repeatable(self) -> None:
        first = fp.preview(self.store, "old NAS replacement", limit=3, at=NOW)
        second = fp.preview(self.store, "old NAS replacement", limit=3, at=NOW)
        self.assertEqual(first.envelope, second.envelope)

    def test_a_different_preview_gets_a_different_envelope(self) -> None:
        first = fp.preview(self.store, "old NAS replacement", limit=3, at=NOW)
        second = fp.preview(self.store, "crimson accent", limit=3, at=NOW)
        self.assertNotEqual(first.envelope, second.envelope)

    def test_the_description_shows_what_would_be_forgotten(self) -> None:
        preview = fp.preview(self.store, "old NAS replacement", limit=3)
        self.assertIn("envelope", preview.describe())
        self.assertIn(preview.ids[0][:8], preview.describe())


class ApplyTests(unittest.TestCase):
    store: CortexStore
    tmp: tempfile.TemporaryDirectory[str]
    setUp = PreviewTests.setUp
    tearDown = PreviewTests.tearDown

    def test_applying_the_previewed_ids_tombstones_them(self) -> None:
        preview = fp.preview(self.store, "old NAS replacement", limit=2)
        applied = fp.apply(self.store, preview.envelope, preview.ids[0:1])
        self.assertTrue(applied.ok, applied.summary)
        self.assertEqual(len(applied.forgotten), 1)
        state = self.store.get_memory(applied.forgotten[0])["state"]
        self.assertEqual(state, "tombstoned")

    def test_an_id_outside_the_preview_is_refused_and_nothing_is_forgotten(self) -> None:
        preview = fp.preview(self.store, "old NAS replacement", limit=2)
        stranger = self.store.add_memory(
            content="Something else entirely about the greenhouse.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        applied = fp.apply(self.store, preview.envelope, preview.ids + [stranger])
        self.assertFalse(applied.ok)
        self.assertIn("not in the previewed set", applied.refused)
        self.assertEqual(self.store.get_memory(stranger)["state"], "active")

    def test_an_unknown_envelope_is_refused(self) -> None:
        applied = fp.apply(self.store, "0" * 64, ["anything"])
        self.assertFalse(applied.ok)
        self.assertIn("unknown envelope", applied.refused)

    def test_an_envelope_cannot_be_applied_twice(self) -> None:
        preview = fp.preview(self.store, "old NAS replacement", limit=2, at=NOW)
        first = fp.apply(self.store, preview.envelope, preview.ids, at=NOW)
        self.assertTrue(first.ok, first.summary)
        second = fp.apply(self.store, preview.envelope, preview.ids, at=NOW)
        self.assertFalse(second.ok)
        self.assertIn("already applied", second.refused)

    def test_an_expired_envelope_is_refused(self) -> None:
        preview = fp.preview(self.store, "old NAS replacement", limit=2, at=NOW)
        late = NOW + timedelta(minutes=fp.PREVIEW_MINUTES + 1)
        applied = fp.apply(self.store, preview.envelope, preview.ids, at=late)
        self.assertFalse(applied.ok)
        self.assertIn("expired", applied.refused)

    def test_the_expiry_window_is_configurable(self) -> None:
        preview = fp.preview(self.store, "old NAS replacement", limit=2, at=NOW)
        with mock.patch.dict(os.environ, {"CORTEX_FORGET_PREVIEW_MINUTES": "5"}):
            late = NOW + timedelta(minutes=6)
            applied = fp.apply(self.store, preview.envelope, preview.ids, at=late)
            self.assertFalse(applied.ok)
            self.assertIn("expired", applied.refused)

    def test_protected_memories_are_refused_unless_forced(self) -> None:
        preview = fp.preview(self.store, "old NAS replacement", limit=3)
        self.store.set_pinned(preview.ids[0], True)
        refused = fp.apply(self.store, preview.envelope, [preview.ids[0]])
        self.assertFalse(refused.ok)
        self.assertIn("protected", refused.refused)
        forced = fp.apply(self.store, preview.envelope, [preview.ids[0]], allow_protected=True)
        self.assertTrue(forced.ok, forced.summary)

    def test_an_empty_request_is_refused(self) -> None:
        preview = fp.preview(self.store, "old NAS replacement", limit=2)
        applied = fp.apply(self.store, preview.envelope, [])
        self.assertFalse(applied.ok)
        self.assertIn("no ids", applied.refused)

    def test_the_audit_trail_records_preview_and_apply(self) -> None:
        preview = fp.preview(self.store, "old NAS replacement", limit=2, at=NOW)
        fp.apply(self.store, preview.envelope, preview.ids[0:1], at=NOW)
        trail = fp.envelopes(self.store)
        self.assertEqual(len(trail), 1)
        self.assertEqual(trail[0]["previewed"], len(preview.ids))
        self.assertEqual(trail[0]["applied"], 1)
        self.assertFalse(trail[0]["expired"])
        self.assertIsNotNone(trail[0]["applied_at"])

    def test_a_forgotten_memory_can_be_restored(self) -> None:
        from cortex.forget_ladder import restore

        preview = fp.preview(self.store, "old NAS replacement", limit=1)
        applied = fp.apply(self.store, preview.envelope, preview.ids)
        self.assertTrue(applied.ok, applied.summary)
        self.assertTrue(restore(self.store, applied.forgotten[0]))
        self.assertEqual(self.store.get_memory(applied.forgotten[0])["state"], "active")


if __name__ == "__main__":
    unittest.main()
