"""A neighbour's metadata may be regenerated; its content may not (#69)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import derived_metadata as dm
from cortex.store import CortexStore


class ChangeTests(unittest.TestCase):
    def test_identical_rows_report_no_change(self) -> None:
        row = {"content": "The dashboard runs on port 8100.", "kind": "semantic"}
        self.assertEqual(dm.changed(row, dict(row)), [])

    def test_a_content_change_is_reported(self) -> None:
        before = {"content": "The dashboard runs on port 8100."}
        after = {"content": "The dashboard runs on port 8200."}
        self.assertEqual(dm.changed(before, after), ["content"])

    def test_a_confidence_change_is_reported(self) -> None:
        self.assertEqual(dm.changed({"confidence": 0.5}, {"confidence": 0.9}), ["confidence"])

    def test_derived_fields_are_not_authoritative(self) -> None:
        before = {"content": "x", "keywords": "one"}
        after = {"content": "x", "keywords": "one,two"}
        self.assertEqual(dm.changed(before, after), [])

    def test_assert_immutable_raises_on_a_content_change(self) -> None:
        with self.assertRaises(ValueError) as caught:
            dm.assert_immutable({"content": "a"}, {"content": "b"})
        self.assertIn("content", str(caught.exception))

    def test_assert_immutable_passes_for_metadata_only_changes(self) -> None:
        dm.assert_immutable({"content": "a", "tags": ""}, {"content": "a", "tags": "new"})

    def test_every_authoritative_field_is_covered(self) -> None:
        before = {name: "a" for name in dm.AUTHORITATIVE_FIELDS}
        after = dict(before)
        after["supersedes_id"] = "b"
        self.assertEqual(dm.changed(before, after), ["supersedes_id"])

    def test_the_fingerprint_is_stable_and_distinguishing(self) -> None:
        self.assertEqual(dm.fingerprint("abc"), dm.fingerprint("abc"))
        self.assertNotEqual(dm.fingerprint("abc"), dm.fingerprint("abd"))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def add(self, content: str = "The dashboard runs on port 8100.") -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def test_regeneration_leaves_content_untouched(self) -> None:
        memory_id = self.add()
        before = self.store.get_memory(memory_id)["content"]
        result = dm.regenerate(self.store, memory_id)
        self.assertTrue(result.regenerated)
        self.assertEqual(self.store.get_memory(memory_id)["content"], before)

    def test_regeneration_reports_which_derived_rows_exist(self) -> None:
        self.add()
        described = dm.report(self.store)
        self.assertGreaterEqual(described["derived_rows"], 0)
        self.assertIn("features", described["derived_fields"])

    def test_an_unknown_memory_is_refused(self) -> None:
        result = dm.regenerate(self.store, "no-such-id")
        self.assertFalse(result.regenerated)
        self.assertIn("no such memory", result.refused)

    def test_many_at_once(self) -> None:
        ids = [self.add(f"Fact number {index} about the box.") for index in range(3)]
        described = dm.regenerate_many(self.store, ids)
        self.assertEqual(described["regenerated"], 3)
        self.assertEqual(described["refused"], [])

    def test_content_stays_byte_identical_across_a_regeneration(self) -> None:
        content = "The dashboard runs on port 8100 and the vault is local."
        memory_id = self.add(content)
        self.store._backfill_memory_features()
        dm.regenerate(self.store, memory_id)
        self.assertEqual(self.store.get_memory(memory_id)["content"], content)

    def test_the_result_describes_itself(self) -> None:
        memory_id = self.add()
        self.assertIn("regenerated", dm.regenerate(self.store, memory_id).describe())
        self.assertIn("no such memory",
                      dm.regenerate(self.store, "missing").describe())


class RestoreIntegrationTests(unittest.TestCase):
    """The restore ladder re-indexes through the guard, so a restore cannot rewrite content."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_restore_leaves_the_record_intact(self) -> None:
        from cortex import restore_ladder

        content = "The dashboard runs on port 8100 and the vault is local."
        memory_id = self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'archived' WHERE id = ?", (memory_id,))
        restore_ladder.restore(self.store, memory_id)
        row = self.store.get_memory(memory_id)
        self.assertEqual(row["content"], content)
        self.assertNotEqual(row["state"], "archived")


if __name__ == "__main__":
    unittest.main()
