"""What was injected this turn, in the terminal (#148)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import turn_panel
from cortex.store import CortexStore


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, confidence: float = 0.9, kind: str = "semantic") -> str:
        return self.store.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category="USER_STATED",
            confidence=confidence,
        )[0]

    def trace(self, task_id: str, selected: list[str], *, query: str = "which port") -> None:
        # Stamp an explicit timestamp: two traces written in the same second tie on created_at, and the
        # "latest turn" assertion would then be measuring SQLite's tie-break rather than the panel.
        self.stamp = getattr(self, "stamp", 0) + 1
        self.store.record_memory_trace_decision(
            task_id=task_id,
            session_id="s1",
            goal="answer",
            context_summary="",
            task_type="coding",
            recall_mode="auto",
            retrieval_used=True,
            retrieval_reason="explicit durable-context cue",
            queries=[query],
            candidate_memories=[
                {"memory_id": memory_id, "rank": index + 1, "score": 0.5, "selected": True}
                for index, memory_id in enumerate(selected)
            ],
            retrieval_context={},
        )
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memory_traces SET created_at = ? WHERE task_id = ?",
                (f"2026-11-24T12:00:{self.stamp:02d}+00:00", task_id),
            )

    def test_no_turn_at_all_says_so(self) -> None:
        panel = turn_panel.build(self.store)
        self.assertFalse(panel.recorded)
        self.assertIn("no turn recorded", turn_panel.render(panel)[0])

    def test_the_latest_turn_is_shown_when_no_id_is_given(self) -> None:
        first = self.memory("the dashboard listens on 8100")
        second = self.memory("nginx proxies the dashboard")
        self.trace("t1", [first])
        self.trace("t2", [second])
        panel = turn_panel.build(self.store)
        self.assertEqual([row["memory_id"] for row in panel.rows], [second])

    def test_one_line_per_memory(self) -> None:
        ids = [self.memory(f"fact number {index} about the box") for index in range(3)]
        self.trace("t3", ids)
        lines = turn_panel.render(turn_panel.build(self.store, "t3"))
        self.assertEqual(len([line for line in lines if "semantic" in line]), 3)

    def test_the_clause_is_the_first_phrase(self) -> None:
        memory_id = self.memory(
            "The dashboard listens on port 8100. It sits behind nginx, which adds tls.")
        self.trace("t4", [memory_id])
        row = turn_panel.build(self.store, "t4").rows[0]
        self.assertEqual(row["clause"], "The dashboard listens on port 8100")

    def test_a_long_clause_is_trimmed(self) -> None:
        self.memory("word " * 100)
        self.trace("t5", [self.memory("a")])
        self.assertLessEqual(len(turn_panel.clause("word " * 100)), turn_panel.CLAUSE + 1)

    def test_the_kind_and_confidence_are_shown(self) -> None:
        memory_id = self.memory("a preference about tabs", confidence=0.72, kind="preference")
        self.trace("t6", [memory_id])
        line = [line for line in turn_panel.render(turn_panel.build(self.store, "t6"))
                if "preference" in line][0]
        self.assertIn("0.72", line)

    def test_a_turn_that_injected_nothing_says_so(self) -> None:
        self.trace("t7", [])
        panel = turn_panel.build(self.store, "t7")
        self.assertTrue(panel.recorded)
        self.assertIn("injected no memories", panel.note)

    def test_an_unknown_turn_is_reported_not_empty(self) -> None:
        self.memory("anything")
        panel = turn_panel.build(self.store, "missing")
        self.assertFalse(panel.recorded)
        self.assertIn("not recorded", panel.note)

    def test_used_and_unused_memories_are_marked(self) -> None:
        used = self.memory("the dashboard listens on 8100")
        unused = self.memory("an unrelated note about the vault timer")
        self.trace("t8", [used, unused])
        with self.store.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS recall_receipts (turn_id TEXT PRIMARY KEY, "
                "offered INTEGER, used INTEGER, cited_ids TEXT, unused_ids TEXT, created_at TEXT)")
            conn.execute(
                "INSERT OR REPLACE INTO recall_receipts VALUES ('t8', 2, 1, ?, ?, 'x')", (used, unused))
        panel = turn_panel.build(self.store, "t8")
        marks = {row["memory_id"]: row["used"] for row in panel.rows}
        self.assertTrue(marks[used])
        self.assertFalse(marks[unused])

    def test_without_a_receipt_usage_is_unknown_not_false(self) -> None:
        memory_id = self.memory("the dashboard listens on 8100")
        self.trace("t9", [memory_id])
        self.assertIsNone(turn_panel.build(self.store, "t9").rows[0]["used"])

    def test_the_limit_is_respected_with_a_count_of_the_rest(self) -> None:
        ids = [self.memory(f"fact {index}") for index in range(6)]
        self.trace("t10", ids)
        lines = turn_panel.render(turn_panel.build(self.store, "t10"), limit=2)
        self.assertTrue(any("4 more injected" in line for line in lines))

    def test_plain_text_by_default(self) -> None:
        memory_id = self.memory("the dashboard listens on 8100")
        self.trace("t11", [memory_id])
        rendered = turn_panel.text(turn_panel.build(self.store, "t11"))
        self.assertNotIn("\x1b[", rendered)

    def test_colour_when_asked_for(self) -> None:
        memory_id = self.memory("the dashboard listens on 8100")
        self.trace("t12", [memory_id])
        rendered = turn_panel.text(turn_panel.build(self.store, "t12"), color=True)
        self.assertIn("\x1b[", rendered)

    def test_the_width_is_respected(self) -> None:
        memory_id = self.memory("a long memory " * 20)
        self.trace("t13", [memory_id])
        for line in turn_panel.render(turn_panel.build(self.store, "t13"), width=50):
            self.assertLessEqual(len(line), 60)

    def test_the_terminal_width_is_read_from_the_environment(self) -> None:
        previous = os.environ.get("COLUMNS")
        os.environ["COLUMNS"] = "80"
        try:
            self.assertEqual(turn_panel.terminal_width(), 80)
        finally:
            if previous is None:
                os.environ.pop("COLUMNS", None)
            else:
                os.environ["COLUMNS"] = previous

    def test_an_absurd_width_is_clamped(self) -> None:
        previous = os.environ.get("COLUMNS")
        os.environ["COLUMNS"] = "3"
        try:
            self.assertEqual(turn_panel.terminal_width(), turn_panel.MIN_WIDTH)
        finally:
            if previous is None:
                os.environ.pop("COLUMNS", None)
            else:
                os.environ["COLUMNS"] = previous

    def test_the_query_is_shown(self) -> None:
        memory_id = self.memory("the dashboard listens on 8100")
        self.trace("t14", [memory_id], query="why is the dashboard slow")
        self.assertIn("why is the dashboard slow",
                      "\n".join(turn_panel.render(turn_panel.build(self.store, "t14"))))

    def test_a_broken_store_returns_an_empty_panel(self) -> None:
        class Broken:
            _conn = None

            def memory_traces(self, **_kwargs):
                raise RuntimeError("no store")

        self.assertFalse(turn_panel.build(Broken()).recorded)


if __name__ == "__main__":
    unittest.main()
