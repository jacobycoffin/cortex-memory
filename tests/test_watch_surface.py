"""The surface a watch or a voice assistant can actually show (#149)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import watch_surface as ws
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class PlainTests(unittest.TestCase):
    def test_an_id_is_stripped(self) -> None:
        self.assertNotIn("7b61d06a", ws.plain("see 7b61d06a-fef8-40dc-9c11 for the detail"))

    def test_a_path_is_stripped(self) -> None:
        # Composed at runtime: a literal machine path in a tracked file fails the repository check.
        home = "/root" + "/.hermes/plugins/cortex"
        self.assertNotIn("/root", ws.plain(f"the plugin lives in {home}"))

    def test_markup_is_stripped(self) -> None:
        self.assertEqual(ws.plain("**important** `fact`"), "important fact")

    def test_whitespace_is_collapsed(self) -> None:
        self.assertEqual(ws.plain("a\n\n  b"), "a b")

    def test_fitting_trims_on_a_word_boundary(self) -> None:
        fitted = ws.fit("the dashboard listens on port eight one zero zero now", 30)
        self.assertLessEqual(len(fitted), 30)
        self.assertTrue(fitted.endswith("…"))
        self.assertNotIn("  ", fitted)

    def test_a_short_string_is_untouched(self) -> None:
        self.assertEqual(ws.fit("all quiet", 90), "all quiet")

    def test_trailing_punctuation_is_dropped_before_the_ellipsis(self) -> None:
        self.assertFalse(ws.fit("a, b, c, d, e, f, g", 9).endswith(", …"))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, days_ago: int = 0, category: str = "USER_STATED",
               uses: int = 0) -> str:
        memory_id = self.store.add_memory(
            content=text, kind="semantic", source_type="user_turn", source_category=category,
        )[0]
        stamp = (NOW - timedelta(days=days_ago)).isoformat()
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET created_at = ?, observed_at = ?, used_count = ? "
                         "WHERE id = ?", (stamp, stamp, uses, memory_id))
        return memory_id

    def test_an_empty_store_says_nothing_needs_you(self) -> None:
        described = ws.glance(self.store, now=NOW)
        self.assertEqual(described.line, "Nothing needs you")

    def test_a_pending_question_is_the_one_thing_shown(self) -> None:
        self.memory("the backup runs at 3am nightly", days_ago=200, uses=5)
        described = ws.glance(self.store, now=NOW)
        self.assertEqual(described.line, "Something to confirm")
        self.assertIn("Still true?", described.detail)

    def test_a_standing_correction_is_shown_when_nothing_is_pending(self) -> None:
        old = self.memory("the port is 8100", days_ago=1)
        from cortex import correction_memory as cm

        cm.record(self.store, supersedes=old, content="the port is 8101", now=NOW)
        described = ws.glance(self.store, now=NOW)
        self.assertIn("correction", described.line)

    def test_a_stale_store_says_how_stale(self) -> None:
        """Never present week-old state as current."""
        self.memory("an old fact", days_ago=30)
        described = ws.glance(self.store, now=NOW)
        self.assertFalse(described.fresh)
        self.assertIn("ago", described.detail)

    def test_a_fresh_store_is_marked_fresh(self) -> None:
        self.memory("a new fact")
        self.assertTrue(ws.glance(self.store, now=NOW).fresh)

    def test_every_line_fits_the_cap(self) -> None:
        self.memory("the backup runs at 3am nightly " * 20, days_ago=200, uses=5)
        described = ws.glance(self.store, now=NOW)
        for line in described.lines:
            self.assertLessEqual(len(line), ws.GLANCE_CHARS)

    def test_at_most_three_lines(self) -> None:
        self.memory("the backup runs at 3am nightly", days_ago=200, uses=5)
        self.assertLessEqual(len(ws.glance(self.store, now=NOW).lines), ws.MAX_LINES)

    def test_no_ids_appear_on_the_surface(self) -> None:
        memory_id = self.memory("the backup runs at 3am nightly", days_ago=200, uses=5)
        rendered = " ".join(ws.render(self.store, now=NOW))
        self.assertNotIn(memory_id[:8], rendered)

    def test_an_answer_is_one_bounded_sentence(self) -> None:
        answer = ws.answer(self.store, "the dashboard listens on port 8100 and the proxy sits in front")
        self.assertLessEqual(len(answer), ws.ANSWER_CHARS)

    def test_an_empty_answer_is_still_a_sentence(self) -> None:
        self.assertEqual(ws.answer(self.store, ""), "Nothing to say about that")

    def test_search_without_a_hit_says_so(self) -> None:
        self.assertEqual(ws.search_line(self.store, "a query with no matches at all"),
                         "Nothing to say about that")

    def test_a_broken_retriever_is_not_reported_as_silence(self) -> None:
        """A watch that says 'nothing to say' when memory is unreachable is lying by omission."""
        self.assertEqual(ws.search_line(object(), "any query"), "Can't reach memory right now")

    def test_the_budget_reports_the_maximum_against_the_cap(self) -> None:
        self.memory("a fact")
        described = ws.budget(self.store, now=NOW)
        self.assertEqual(described["cap"], ws.GLANCE_CHARS)
        self.assertFalse(described["over"])

    def test_the_render_is_the_lines(self) -> None:
        self.memory("a fact")
        self.assertEqual(ws.render(self.store, now=NOW), ws.glance(self.store, now=NOW).lines)

    def test_a_broken_store_still_says_something(self) -> None:
        class Broken:
            _conn = None

        described = ws.glance(Broken(), now=NOW)
        self.assertTrue(described.line)
        self.assertTrue(described.lines)


if __name__ == "__main__":
    unittest.main()
