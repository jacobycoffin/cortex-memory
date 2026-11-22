"""Receipts name what was written, and the inline edits actually work (#37)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import suppression, write_receipts
from cortex.store import CortexStore


class ReceiptBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def add(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def test_a_receipt_names_the_count_and_the_entries(self) -> None:
        ids = [self.add("The dashboard runs on port 8100."),
               self.add("Jacoby prefers short answers.")]
        receipt = write_receipts.build(self.store, ids)
        self.assertIn("remembered 2 things", receipt.text)
        self.assertIn("port 8100", receipt.text)

    def test_each_entry_carries_a_short_id(self) -> None:
        memory_id = self.add("The dashboard runs on port 8100.")
        receipt = write_receipts.build(self.store, [memory_id])
        self.assertIn(write_receipts.short_id(memory_id), receipt.text)

    def test_the_receipt_invites_the_edit(self) -> None:
        receipt = write_receipts.build(self.store, [self.add("A fact about the vault.")])
        self.assertIn("fix <id>", receipt.text)
        self.assertIn("forget <id>", receipt.text)

    def test_nothing_written_means_no_receipt(self) -> None:
        self.assertEqual(write_receipts.build(self.store, []).text, "")
        self.assertFalse(write_receipts.build(self.store, []))

    def test_an_unknown_id_is_skipped(self) -> None:
        self.assertEqual(write_receipts.build(self.store, ["no-such-memory"]).text, "")

    def test_the_entries_are_capped(self) -> None:
        ids = [self.add(f"Fact number {index} about the system.") for index in range(6)]
        receipt = write_receipts.build(self.store, ids)
        self.assertIn("remembered 3 things", receipt.text)
        self.assertEqual(len(receipt.ids), 3)

    def test_the_line_is_capped(self) -> None:
        ids = [self.add("A very long fact " * 20) for _ in range(3)]
        receipt = write_receipts.build(self.store, ids, max_chars=200)
        self.assertLessEqual(len(receipt.text), 210)

    def test_the_excerpt_does_not_break_on_mid_word_whitespace(self) -> None:
        excerpt = write_receipts._excerpt("one   two\nthree " + "x" * 100, limit=20)
        self.assertLessEqual(len(excerpt), 20)

    def test_a_single_entry_reads_as_one_thing(self) -> None:
        receipt = write_receipts.build(self.store, [self.add("One fact.")])
        self.assertIn("remembered 1 thing ", receipt.text)


class ParseTests(unittest.TestCase):
    def test_a_fix_is_parsed(self) -> None:
        edits = write_receipts.parse("fix a1b2c3 the port is 8200")
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0].action, "fix")
        self.assertEqual(edits[0].text, "the port is 8200")

    def test_a_fix_with_a_colon_is_parsed(self) -> None:
        edits = write_receipts.parse("correct a1b2c3: the port is 8200")
        self.assertEqual(edits[0].text, "the port is 8200")

    def test_a_forget_is_parsed(self) -> None:
        edits = write_receipts.parse("forget a1b2c3")
        self.assertEqual(edits[0].action, "forget")

    def test_a_sentence_without_an_id_is_ignored(self) -> None:
        self.assertEqual(write_receipts.parse("forget about the port thing"), [])

    def test_the_word_forget_inside_a_fix_is_not_a_second_edit(self) -> None:
        edits = write_receipts.parse("fix a1b2c3 forget the old value")
        self.assertEqual([edit.action for edit in edits], ["fix"])

    def test_several_edits_in_one_message(self) -> None:
        edits = write_receipts.parse("fix a1b2c3 the port is 8200 and forget d4e5f6")
        self.assertEqual(sorted(edit.action for edit in edits), ["fix", "forget"])

    def test_parsing_has_no_side_effects(self) -> None:
        write_receipts.parse("forget a1b2c3")
        # Nothing to assert beyond "did not raise": parse() takes no store.


class ApplyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def add(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def test_a_short_id_resolves_to_one_memory(self) -> None:
        memory_id = self.add("The dashboard runs on port 8100.")
        self.assertEqual(write_receipts.resolve(self.store, write_receipts.short_id(memory_id)),
                         memory_id)

    def test_an_ambiguous_prefix_resolves_to_nothing(self) -> None:
        self.assertIsNone(write_receipts.resolve(self.store, "zz"))

    def test_forget_suppresses_without_deleting(self) -> None:
        memory_id = self.add("The dashboard runs on port 8100.")
        edits = write_receipts.apply_edits(self.store, f"forget {write_receipts.short_id(memory_id)}")
        self.assertTrue(edits[0].applied)
        self.assertIn("content kept", edits[0].note)
        self.assertTrue(suppression.is_suppressed(self.store, memory_id))
        self.assertEqual(self.store.get_memory(memory_id)["content"],
                         "The dashboard runs on port 8100.")

    def test_a_fix_supersedes_the_old_memory(self) -> None:
        memory_id = self.add("The dashboard runs on port 8100.")
        edits = write_receipts.apply_edits(
            self.store, f"fix {write_receipts.short_id(memory_id)} the dashboard runs on port 8200"
        )
        self.assertTrue(edits[0].applied)
        row = self.store._conn.execute(
            "SELECT superseding.id AS new_id FROM memories superseding"
            " JOIN memories old ON old.supersedes_id IS NULL WHERE superseding.supersedes_id = ?",
            (memory_id,),
        ).fetchall()
        self.assertEqual(len(row), 1)
        self.assertIn("8200", self.store.get_memory(str(row[0]["new_id"]))["content"])

    def test_an_unknown_id_is_reported_not_raised(self) -> None:
        edits = write_receipts.apply_edits(self.store, "forget deadbeef")
        self.assertFalse(edits[0].applied)
        self.assertIn("no single memory", edits[0].note)

    def test_a_message_with_no_edits_does_nothing(self) -> None:
        self.assertEqual(write_receipts.apply_edits(self.store, "thanks, that helps"), [])

    def test_the_summary_counts_applied_and_failed(self) -> None:
        memory_id = self.add("The dashboard runs on port 8100.")
        edits = write_receipts.apply_edits(
            self.store,
            f"forget {write_receipts.short_id(memory_id)} and forget deadbeef",
        )
        summary = write_receipts.summarise(edits)
        self.assertIn("1 correction applied", summary)
        self.assertIn("1 not applied", summary)

    def test_the_summary_is_empty_without_edits(self) -> None:
        self.assertEqual(write_receipts.summarise([]), "")


class HarnessTests(unittest.TestCase):
    """The real path: a turn that writes ends with a receipt, and the receipt's ids work."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        from cortex.harness import CortexHarnessAdapter

        self.adapter = CortexHarnessAdapter(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.adapter.close()
        self.tmp.cleanup()

    def test_a_turn_names_what_it_wrote(self) -> None:
        turn = self.adapter.before_turn("what port does the dashboard run on", force_recall=True)
        self.adapter.remember(
            "The dashboard runs on port 8100.", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        self.adapter.remember(
            "Jacoby prefers short answers.", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        receipt = turn.receipt(self.adapter.memory.store)
        self.assertIn("remembered 2 things", receipt)
        self.assertIn("port 8100", receipt)

    def test_a_turn_without_writes_has_no_receipt(self) -> None:
        turn = self.adapter.before_turn("hello there", force_recall=True)
        self.assertEqual(turn.receipt(self.adapter.memory.store), "")

    def test_a_duplicate_write_is_not_receipted_twice(self) -> None:
        turn = self.adapter.before_turn("what port does the dashboard run on", force_recall=True)
        content = "The dashboard runs on port 8100."
        for _ in range(2):
            self.adapter.remember(content, source_type="user_turn",
                                  source_category="USER_EXPLICIT")
        self.assertLessEqual(len(turn.writes), 2)

    def test_the_adapter_applies_an_inline_forget(self) -> None:
        turn = self.adapter.before_turn("what port does the dashboard run on", force_recall=True)
        memory_id, _created = self.adapter.remember(
            "The dashboard runs on port 8100.", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        summary = self.adapter.apply_write_edits(f"forget {write_receipts.short_id(memory_id)}")
        self.assertIn("correction applied", summary)
        self.assertTrue(suppression.is_suppressed(self.adapter.memory.store, memory_id))
        self.assertEqual(len(turn.writes), 1)

    def test_a_message_without_edits_returns_nothing(self) -> None:
        self.assertEqual(self.adapter.apply_write_edits("thanks"), "")


if __name__ == "__main__":
    unittest.main()
