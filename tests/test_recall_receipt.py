"""A receipt per answer: what this reply actually used (#174)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import recall_receipt
from cortex.store import CortexStore

QUOTE = "the dashboard listens on port 8100 behind nginx and the proxy adds tls"


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, kind: str = "semantic") -> str:
        return self.store.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category="USER_STATED",
        )[0]

    def trace(self, task_id: str, selected: list[str]) -> None:
        self.store.record_memory_trace_decision(
            task_id=task_id,
            session_id="s1",
            goal="answer",
            context_summary="",
            task_type="coding",
            recall_mode="auto",
            retrieval_used=True,
            retrieval_reason="recalled by score",
            queries=["which port"],
            candidate_memories=[
                {"memory_id": memory_id, "rank": index + 1, "score": 0.5, "selected": True}
                for index, memory_id in enumerate(selected)
            ],
            retrieval_context={},
        )

    def test_a_turn_that_was_not_recorded_says_so(self) -> None:
        found = recall_receipt.build(self.store, "missing", "an answer")
        self.assertFalse(found.recorded)
        self.assertIn("not recorded", found.describe())

    def test_a_repeated_memory_is_used(self) -> None:
        memory_id = self.memory(QUOTE)
        self.trace("t1", [memory_id])
        found = recall_receipt.build(
            self.store, "t1", "Yes — the dashboard listens on port 8100 behind nginx and the proxy")
        self.assertEqual([item["memory_id"] for item in found.used], [memory_id])

    def test_an_injected_memory_that_the_answer_ignores_is_not_used(self) -> None:
        """Being offered is not being used; this is the whole point of the receipt."""
        memory_id = self.memory(QUOTE)
        self.trace("t2", [memory_id])
        found = recall_receipt.build(self.store, "t2", "Unrelated answer about something else.")
        self.assertEqual(found.used, [])
        self.assertEqual(found.offered, 1)
        self.assertIn("offered is not used", found.note)

    def test_a_declared_marker_counts_as_a_citation(self) -> None:
        memory_id = self.memory("a memory nobody repeats verbatim in the reply")
        self.trace("t3", [memory_id])
        found = recall_receipt.build(self.store, "t3", f"See [{memory_id[:8]}] for the detail.")
        self.assertEqual([item["memory_id"] for item in found.used], [memory_id])
        self.assertEqual(found.used[0]["method"], "marker")

    def test_the_receipt_counts_offerings_and_uses(self) -> None:
        first, second = self.memory(QUOTE), self.memory("an unrelated note about the vault timer")
        self.trace("t4", [first, second])
        found = recall_receipt.build(
            self.store, "t4", "The dashboard listens on port 8100 behind nginx.")
        self.assertEqual(found.offered, 2)
        self.assertEqual(found.used_count, 1)

    def test_a_turn_offered_nothing_is_reported_not_zero_cited(self) -> None:
        self.trace("t5", [])
        found = recall_receipt.build(self.store, "t5", "an answer")
        self.assertTrue(found.recorded)
        self.assertIn("nothing was offered", found.describe())

    def test_the_sentence_reads_as_a_receipt(self) -> None:
        memory_id = self.memory(QUOTE)
        self.trace("t6", [memory_id])
        found = recall_receipt.build(
            self.store, "t6", "the dashboard listens on port 8100 behind nginx and the proxy")
        self.assertIn("used 1 of 1 offered memory", found.describe())

    def test_the_render_is_short(self) -> None:
        ids = [self.memory(QUOTE) for _ in range(3)]
        self.trace("t7", ids)
        lines = recall_receipt.build(
            self.store, "t7", "the dashboard listens on port 8100 behind nginx and the proxy"
        ).render(limit=1)
        self.assertLessEqual(len(lines), 5)
        self.assertTrue(any("more used" in line for line in lines))

    def test_recording_is_idempotent_per_turn(self) -> None:
        memory_id = self.memory(QUOTE)
        self.trace("t8", [memory_id])
        answer = "the dashboard listens on port 8100 behind nginx and the proxy"
        recall_receipt.record(self.store, "t8", answer)
        recall_receipt.record(self.store, "t8", answer)
        rows = self.store._conn.execute(
            "SELECT COUNT(*) FROM recall_receipts WHERE turn_id = 't8'").fetchone()[0]
        self.assertEqual(rows, 1)

    def test_a_stored_receipt_can_be_read_without_the_answer(self) -> None:
        memory_id = self.memory(QUOTE)
        self.trace("t9", [memory_id])
        recall_receipt.record(
            self.store, "t9", "the dashboard listens on port 8100 behind nginx and the proxy")
        found = recall_receipt.show(self.store, "t9")
        self.assertTrue(found["recorded"])
        self.assertEqual([item["memory_id"] for item in found["used"]], [memory_id])
        self.assertIn("used 1 of 1", found["sentence"])

    def test_a_missing_receipt_says_so(self) -> None:
        found = recall_receipt.show(self.store, "never")
        self.assertFalse(found["recorded"])

    def test_recording_a_turn_with_nothing_offered_stores_nothing(self) -> None:
        self.trace("t10", [])
        recall_receipt.record(self.store, "t10", "an answer")
        self.assertFalse(recall_receipt.show(self.store, "t10")["recorded"])

    def test_the_stats_report_the_used_share(self) -> None:
        first, second = self.memory(QUOTE), self.memory("an unrelated note about the vault timer")
        self.trace("t11", [first, second])
        recall_receipt.record(
            self.store, "t11", "the dashboard listens on port 8100 behind nginx and the proxy")
        described = recall_receipt.stats(self.store)
        self.assertEqual(described["turns"], 1)
        self.assertEqual((described["offered"], described["used"]), (2, 1))
        self.assertEqual(described["used_share"], 0.5)

    def test_the_stats_name_the_turns_that_used_nothing(self) -> None:
        memory_id = self.memory(QUOTE)
        self.trace("t12", [memory_id])
        recall_receipt.record(self.store, "t12", "nothing relevant here at all")
        self.assertEqual(recall_receipt.stats(self.store)["turns_with_no_use"], 1)

    def test_unused_memories_are_ranked_by_how_often_they_were_ignored(self) -> None:
        memory_id = self.memory("a memory offered twice and never used")
        for turn in ("t13", "t14"):
            self.trace(turn, [memory_id])
            recall_receipt.record(self.store, turn, "an answer that ignores it")
        ranked = recall_receipt.unused_memories(self.store)
        self.assertEqual(ranked[0]["memory_id"], memory_id)
        self.assertEqual(ranked[0]["offered_unused"], 2)

    def test_no_answer_text_is_reported_rather_than_blamed(self) -> None:
        memory_id = self.memory(QUOTE)
        self.trace("t15", [memory_id])
        found = recall_receipt.build(self.store, "t15", "")
        self.assertIn("no answer text", found.note)

    def test_a_broken_store_does_not_raise(self) -> None:
        class Broken:
            _conn = None

            def memory_traces(self, **_kwargs):
                raise RuntimeError("no store")

        found = recall_receipt.build(Broken(), "t", "an answer")
        self.assertFalse(found.recorded)


if __name__ == "__main__":
    unittest.main()
