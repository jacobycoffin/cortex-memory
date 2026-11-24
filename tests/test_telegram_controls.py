"""Inline memory controls on a phone, built so a fat thumb cannot corrupt the store (#146)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import telegram_controls as tg
from cortex.store import CortexStore


class CallbackTests(unittest.TestCase):
    def test_a_callback_fits_the_telegram_cap(self) -> None:
        data = tg.callback("pin", "0123456789abcdef")
        self.assertLessEqual(len(data.encode("utf-8")), tg.MAX_CALLBACK_BYTES)

    def test_a_callback_round_trips(self) -> None:
        data = tg.callback("explain", "0123456789abcdef")
        self.assertEqual(tg.parse(data), ("explain", "0123456789"))

    def test_an_unknown_action_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            tg.callback("destroy", "0123456789abcdef")

    def test_a_short_id_is_refused_rather_than_padded(self) -> None:
        """Truncating silently would point the button at a different memory."""
        with self.assertRaises(ValueError):
            tg.callback("pin", "abc")

    def test_a_malformed_callback_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            tg.parse("nonsense")

    def test_an_unknown_action_in_a_callback_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            tg.parse("destroy:0123456789")

    def test_an_over_long_prefix_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            tg.parse("pin:0123456789abcdef")


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str) -> str:
        return self.store.add_memory(
            content=text, kind="semantic", source_type="user_turn", source_category="USER_STATED",
        )[0]

    def row(self, memory_id: str):
        return self.store._conn.execute("SELECT * FROM memories WHERE id = ?",
                                        (memory_id,)).fetchone()

    def test_pressing_pin_pins(self) -> None:
        memory_id = self.memory("the dashboard listens on port 8100")
        result = tg.handle(self.store, tg.callback("pin", memory_id))
        self.assertTrue(result.ok)
        self.assertEqual(int(self.row(memory_id)["pinned"]), 1)

    def test_pressing_pin_twice_is_idempotent(self) -> None:
        """A replay must not toggle what he just set."""
        memory_id = self.memory("the dashboard listens on port 8100")
        tg.handle(self.store, tg.callback("pin", memory_id))
        second = tg.handle(self.store, tg.callback("pin", memory_id))
        self.assertEqual(second.status, "already")
        self.assertEqual(int(self.row(memory_id)["pinned"]), 1)

    def test_unpin_after_pin_works_once(self) -> None:
        memory_id = self.memory("the dashboard listens on port 8100")
        tg.handle(self.store, tg.callback("pin", memory_id))
        self.assertEqual(tg.handle(self.store, tg.callback("unpin", memory_id)).status, "applied")
        self.assertEqual(tg.handle(self.store, tg.callback("unpin", memory_id)).status, "already")
        self.assertEqual(int(self.row(memory_id)["pinned"]), 0)

    def test_mute_sets_the_state_and_unmute_restores_it(self) -> None:
        memory_id = self.memory("the printer queue name is officejet")
        tg.handle(self.store, tg.callback("mute", memory_id))
        self.assertEqual(self.row(memory_id)["state"], "muted")
        tg.handle(self.store, tg.callback("unmute", memory_id))
        self.assertEqual(self.row(memory_id)["state"], "active")

    def test_forget_needs_a_confirmation(self) -> None:
        """Nothing destructive happens on one tap."""
        memory_id = self.memory("an obsolete fact about the old router")
        result = tg.handle(self.store, tg.callback("forget", memory_id))
        self.assertTrue(result.needs_confirmation)
        self.assertIsNone(self.row(memory_id)["valid_to"])

    def test_forget_with_confirmation_demotes_and_keeps_the_content(self) -> None:
        memory_id = self.memory("an obsolete fact about the old router")
        result = tg.handle(self.store, tg.callback("forget", memory_id), confirm=True)
        self.assertTrue(result.changed)
        row = self.row(memory_id)
        self.assertTrue(row["valid_to"])
        self.assertIn("router", row["content"])

    def test_forget_twice_is_idempotent(self) -> None:
        memory_id = self.memory("an obsolete fact about the old router")
        tg.handle(self.store, tg.callback("forget", memory_id), confirm=True)
        second = tg.handle(self.store, tg.callback("forget", memory_id), confirm=True)
        self.assertEqual(second.status, "already")

    def test_correct_asks_for_the_sentence_rather_than_guessing(self) -> None:
        memory_id = self.memory("the port is 8100")
        result = tg.handle(self.store, tg.callback("correct", memory_id), confirm=True)
        self.assertEqual(result.status, "needs_text")

    def test_explain_is_read_only(self) -> None:
        memory_id = self.memory("the port is 8100")
        before = self.row(memory_id)["content"]
        result = tg.handle(self.store, tg.callback("explain", memory_id))
        self.assertEqual(result.status, "read_only")
        self.assertEqual(self.row(memory_id)["content"], before)

    def test_an_ambiguous_prefix_is_refused(self) -> None:
        """Collisions will come back as the store grows; taking the first match is a bug.

        Ids cannot be rewritten in place (children reference them), so the resolver is exercised
        against a stub that returns two matches — which is exactly the input it must refuse.
        """
        prefix = "0123456789"
        original = tg._rows
        tg._rows = lambda store, sql, params=(): [{"id": prefix + "aaaa"},
                                                  {"id": prefix + "bbbb"}]
        try:
            found = tg.resolve(self.store, prefix)
        finally:
            tg._rows = original
        self.assertEqual(found["status"], "ambiguous")
        self.assertEqual(len(found["candidates"]), 2)

    def test_an_ambiguous_press_does_not_write(self) -> None:
        memory_id = self.memory("a fact that must not be pinned by a collision")
        prefix = memory_id[:tg.PREFIX]
        original = tg._rows
        tg._rows = lambda store, sql, params=(): [{"id": memory_id}, {"id": prefix + "ffff"}]
        try:
            result = tg.handle(self.store, tg.callback("pin", prefix))
        finally:
            tg._rows = original
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(int(self.row(memory_id)["pinned"] or 0), 0)

    def test_a_missing_memory_is_reported(self) -> None:
        result = tg.handle(self.store, tg.callback("pin", "ffffffffffff"))
        self.assertEqual(result.status, "missing")

    def test_a_malformed_press_is_reported_not_raised(self) -> None:
        self.assertEqual(tg.handle(self.store, "garbage").status, "malformed")

    def test_the_keyboard_labels_are_words_and_the_data_fits(self) -> None:
        memory_id = self.memory("the port is 8100")
        rows = tg.keyboard(memory_id)
        self.assertGreaterEqual(len(rows), 1)
        for row in rows:
            for button in row:
                self.assertNotIn(memory_id, button["text"])
                self.assertLessEqual(len(button["callback_data"].encode("utf-8")),
                                     tg.MAX_CALLBACK_BYTES)

    def test_forget_is_hidden_from_the_default_keyboard(self) -> None:
        memory_id = self.memory("the port is 8100")
        labels = [button["text"] for row in tg.keyboard(memory_id) for button in row]
        self.assertNotIn("Forget", labels)
        with_forget = [button["text"] for row in tg.keyboard(memory_id, include_forget=True)
                       for button in row]
        self.assertIn("Forget", with_forget)

    def test_coverage_reports_collisions(self) -> None:
        self.memory("a fact")
        described = tg.coverage(self.store)
        self.assertEqual(described["collisions"], 0)
        self.assertEqual(described["prefix"], tg.PREFIX)

    def test_the_render_lists_the_actions_and_the_prefix(self) -> None:
        self.memory("a fact")
        rendered = "\n".join(tg.render(self.store))
        self.assertIn("telegram controls", rendered)
        self.assertIn("actions:", rendered)

    def test_a_broken_store_reports_missing(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(tg.handle(Broken(), tg.callback("pin", "0123456789")).status, "missing")


if __name__ == "__main__":
    unittest.main()
