"""Rewriting may only add signal to underspecified queries (item #16)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.rewrite import (
    MAX_ADDED_TERMS,
    content_terms,
    is_underspecified,
    rewrite,
)
from cortex.store import CortexStore

SESSION = [
    "the seerr telegram notifications stopped after the container update",
    "seerr notifications need the webhook secret rotated",
    "also check the telegram bot token",
]


class TermTests(unittest.TestCase):
    def test_stopwords_are_dropped(self) -> None:
        self.assertEqual(content_terms("do that for the other one"), [])

    def test_real_terms_survive(self) -> None:
        self.assertIn("seerr", content_terms("the seerr notifications"))

    def test_short_words_are_not_terms(self) -> None:
        self.assertNotIn("box", content_terms("the box"))


class UnderspecifiedTests(unittest.TestCase):
    def test_follow_ups_are_underspecified(self) -> None:
        for query in ("do that", "and the other one", "what about that", "yeah"):
            self.assertTrue(is_underspecified(query), query)

    def test_real_questions_are_not(self) -> None:
        for query in ("why did seerr notifications stop", "cortex roadmap next version"):
            self.assertFalse(is_underspecified(query), query)

    def test_empty_query_is_not_rewritten_into_something(self) -> None:
        self.assertEqual(rewrite("", SESSION), ("", []))


class RewriteTests(unittest.TestCase):
    def test_a_follow_up_picks_up_the_session_topic(self) -> None:
        rewritten, added = rewrite("do that", SESSION)
        self.assertTrue(rewritten.startswith("do that"))
        self.assertIn("seerr", added)
        self.assertIn("seerr", rewritten)

    def test_the_original_words_are_always_kept(self) -> None:
        rewritten, _ = rewrite("what about the other one", SESSION)
        self.assertTrue(rewritten.startswith("what about the other one"))

    def test_a_self_contained_query_is_untouched(self) -> None:
        query = "why did the seerr notifications stop after the update"
        self.assertEqual(rewrite(query, SESSION), (query, []))

    def test_recurring_terms_beat_one_off_mentions(self) -> None:
        turns = ["seerr seerr seerr", "unrelated crusade mention"]
        _, added = rewrite("do that", turns)
        self.assertIn("seerr", added)
        self.assertNotIn("crusade", added)

    def test_additions_are_capped(self) -> None:
        turns = [" ".join(f"term{index}x" for index in range(30))] * 2
        _, added = rewrite("do that", turns)
        self.assertLessEqual(len(added), MAX_ADDED_TERMS)

    def test_no_recent_turns_means_no_rewrite(self) -> None:
        self.assertEqual(rewrite("do that", []), ("do that", []))

    def test_recent_turns_without_content_terms_do_not_add_noise(self) -> None:
        self.assertEqual(rewrite("do that", ["ok", "sure", "yes"]), ("do that", []))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_session_queries_can_supply_the_context(self) -> None:
        created = self.store.add_memory(
            content="Seerr notifications stopped after the container update.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        memory_id = created[0] if isinstance(created, tuple) else created
        with self.store.transaction() as conn:
            for offset, text in enumerate(("seerr notifications broke", "seerr webhook rotate")):
                conn.execute(
                    "INSERT INTO access_log (memory_id, event, query, session_id, score, created_at) "
                    "VALUES (?, 'retrieved', ?, 's1', 0.5, datetime('now', ?))",
                    (memory_id, text, f"+{offset} seconds"),
                )
        rows = self.store._conn.execute(
            "SELECT query FROM access_log WHERE session_id='s1' ORDER BY created_at DESC"
        ).fetchall()
        rewritten, added = rewrite("do that", [row[0] for row in rows])
        self.assertIn("seerr", added)
        self.assertIn("seerr", rewritten)


if __name__ == "__main__":
    unittest.main()
