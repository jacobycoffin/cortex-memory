"""Ask about the memories that matter before stating them as current (#177)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import staleness_prompts as sp
from cortex.confidence_phrasing import STALE_AFTER_DAYS
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, days_ago: int = 0, kind: str = "semantic",
               uses: int = 0, state: str = "active", category: str = "USER_STATED") -> str:
        memory_id = self.store.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category=category,
        )[0]
        stamp = (NOW - timedelta(days=days_ago)).isoformat()
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET created_at = ?, observed_at = ?, used_count = ?, state = ? "
                "WHERE id = ?", (stamp, stamp, uses, state, memory_id),
            )
        return memory_id

    def test_a_fresh_memory_is_not_stale(self) -> None:
        self.memory("the port is 8100", days_ago=5)
        self.assertEqual(sp.due(self.store, now=NOW).items, [])

    def test_a_memory_past_its_window_is_due(self) -> None:
        self.memory("the port is 8100", days_ago=STALE_AFTER_DAYS["semantic"] + 10, uses=3)
        described = sp.due(self.store, now=NOW)
        self.assertEqual(len(described.items), 1)
        self.assertIn("Still true?", described.items[0]["prompt"])

    def test_the_window_comes_from_one_place(self) -> None:
        """Two definitions of 'stale' would drift apart, so there is one."""
        self.assertEqual(sp.window_days("semantic"), STALE_AFTER_DAYS["semantic"])

    def test_a_kind_without_a_window_is_never_due(self) -> None:
        self.memory("something odd", days_ago=5000, kind="mystery", uses=2)
        self.assertEqual(sp.due(self.store, now=NOW).items, [])

    def test_a_pinned_memory_is_never_asked_about(self) -> None:
        """A standing instruction does not expire."""
        # A semantic fact 400 days old is well past its 90-day window, so it would be asked about
        # if it were not pinned. (An identity fact has a 10-year window and never reaches the check.)
        memory_id = self.memory("always answer in metric", days_ago=400, kind="semantic", uses=50)
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET pinned = 1 WHERE id = ?", (memory_id,))
        described = sp.due(self.store, now=NOW)
        self.assertEqual(described.items, [])
        self.assertEqual(described.skipped.get("pinned"), 1)

    def test_a_stub_without_an_assertion_is_not_asked_about(self) -> None:
        """'Still true?' about a vault-note stub is noise, not a question (#223's template text)."""
        self.memory("Vault note: 2026-07-17 Path: daily/2026-07-17.md Section: 2026-07-17",
                    days_ago=200, uses=9)
        described = sp.due(self.store, now=NOW)
        self.assertEqual(described.items, [])
        self.assertEqual(described.skipped.get("no assertion"), 1)

    def test_a_real_sentence_is_asked_about(self) -> None:
        self.memory("the backup runs at 3am on the r630", days_ago=200, uses=9)
        self.assertEqual(len(sp.due(self.store, now=NOW).items), 1)

    def test_a_correction_is_never_re_litigated(self) -> None:
        self.memory("no, the port is 8101", days_ago=400, uses=20, category="USER_CORRECTION")
        described = sp.due(self.store, now=NOW)
        self.assertEqual(described.items, [])
        self.assertEqual(described.skipped.get("correction"), 1)

    def test_use_raises_the_priority(self) -> None:
        self.memory("the printer is on the shelf", days_ago=200, uses=0)
        self.memory("the backup runs at 3am nightly", days_ago=200, uses=40)
        items = sp.due(self.store, now=NOW).items
        self.assertIn("backup runs at 3am", items[0]["prompt"])
        self.assertGreater(items[0]["priority"], items[1]["priority"])

    def test_age_raises_the_priority(self) -> None:
        self.memory("the switch lives in the basement", days_ago=400, uses=5)
        self.memory("the nas lives in the attic rack", days_ago=800, uses=5)
        items = sp.due(self.store, now=NOW).items
        self.assertIn("attic rack", items[0]["prompt"])

    def test_the_prompt_names_the_age_and_the_use(self) -> None:
        self.memory("the backup runs at 3am", days_ago=120, uses=14)
        spoken = sp.due(self.store, now=NOW).items[0]["prompt"]
        self.assertIn("120 days ago", spoken)
        self.assertIn("used 14 times", spoken)

    def test_the_prompt_is_bounded(self) -> None:
        self.memory("the very long fact " * 40, days_ago=200, uses=1)
        spoken = sp.due(self.store, now=NOW).items[0]["prompt"]
        self.assertLessEqual(len(spoken), sp.MAX_PROMPT_CHARS)

    def test_a_cooldown_stops_the_same_question_twice(self) -> None:
        memory_id = self.memory("the port is 8100", days_ago=200, uses=3)
        self.assertTrue(sp.record(self.store, memory_id, now=NOW))
        described = sp.due(self.store, now=NOW)
        self.assertEqual(described.items, [])
        self.assertEqual(described.skipped.get("cooldown"), 1)

    def test_the_cooldown_expires(self) -> None:
        memory_id = self.memory("the port is 8100", days_ago=200, uses=3)
        sp.record(self.store, memory_id, now=NOW - timedelta(days=sp.COOLDOWN_DAYS + 5))
        self.assertEqual(len(sp.due(self.store, now=NOW).items), 1)

    def test_a_prompt_is_asked_once_per_memory(self) -> None:
        self.memory("the port is 8100", days_ago=200, uses=3)
        items = sp.due(self.store, now=NOW).items
        self.assertEqual(len({item["memory_id"] for item in items}), len(items))

    def test_the_limit_is_respected(self) -> None:
        for index in range(5):
            self.memory(f"the spare port number {index} is unused", days_ago=200, uses=index)
        self.assertEqual(len(sp.due(self.store, limit=2, now=NOW).items), 2)

    def test_a_stale_memory_is_not_demoted_or_hidden(self) -> None:
        """The prompt asks; it never edits state."""
        memory_id = self.memory("the port is 8100", days_ago=200, uses=3)
        sp.due(self.store, now=NOW)
        row = self.store._conn.execute("SELECT state, content FROM memories WHERE id = ?",
                                       (memory_id,)).fetchone()
        self.assertEqual(row["state"], "active")
        self.assertIn("8100", row["content"])

    def test_archived_memories_are_not_examined(self) -> None:
        self.memory("gone", days_ago=200, uses=3, state="archived")
        self.assertEqual(sp.due(self.store, now=NOW).examined, 0)

    def test_a_kind_filter_is_respected(self) -> None:
        self.memory("the tunnel restarts every night at four", days_ago=200, kind="operational",
                    uses=3)
        self.memory("he prefers metric units everywhere", days_ago=900, kind="preference", uses=3)
        items = sp.due(self.store, kinds=["operational"], now=NOW).items
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["kind"], "operational")

    def test_nothing_due_says_so(self) -> None:
        self.assertIn("nothing is both stale", sp.due(self.store, now=NOW).note)

    def test_the_render_lists_the_questions(self) -> None:
        self.memory("the backup runs at 3am", days_ago=120, uses=4)
        rendered = "\n".join(sp.render(self.store, now=NOW))
        self.assertIn("question(s) due", rendered)
        self.assertIn("Still true?", rendered)

    def test_a_broken_store_examines_nothing(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(sp.due(Broken(), now=NOW).examined, 0)


class PromptTests(unittest.TestCase):
    def row(self, **overrides):
        base = {"kind": "semantic", "content": "the port is 8100", "used_count": 0,
                "created_at": (NOW - timedelta(days=200)).isoformat(), "observed_at": None}
        base.update(overrides)
        return base

    def test_a_non_stale_row_produces_no_prompt(self) -> None:
        self.assertEqual(sp.prompt(self.row(created_at=NOW.isoformat()), now=NOW), "")

    def test_the_snippet_is_shortened_with_an_ellipsis(self) -> None:
        self.assertTrue(sp.snippet("word " * 60).endswith("…"))

    def test_whitespace_is_collapsed(self) -> None:
        self.assertEqual(sp.snippet("a\n\n  b"), "a b")

    def test_singular_use_is_grammatical(self) -> None:
        self.assertIn("used 1 time", sp.prompt(self.row(used_count=1), now=NOW))

    def test_a_scaffolded_stub_has_no_claim(self) -> None:
        self.assertFalse(sp.has_claim("Vault note: 2026-07-17 Path: a.md Section: b"))

    def test_a_sentence_has_a_claim(self) -> None:
        self.assertTrue(sp.has_claim("the dashboard listens on port 8100"))

    def test_an_observed_date_beats_the_created_date(self) -> None:
        row = self.row(observed_at=(NOW - timedelta(days=100)).isoformat())
        self.assertIn("100 days ago", sp.prompt(row, now=NOW))


if __name__ == "__main__":
    unittest.main()
