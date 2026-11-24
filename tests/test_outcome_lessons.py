"""Failures become questions, not rewritten memories (#208)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import outcome_lessons as ol, outcomes
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str = "Run the deploy from CT116 with the wrapper script.",
               kind: str = "procedure") -> str:
        return self.store.add_memory(
            content=content, kind=kind, source_type="user_turn", source_category="USER_STATED",
        )[0]

    def record(self, memory_id: str, task: str, verdict: str) -> None:
        outcomes.record(self.store, memory_id, task, verdict, at=NOW)

    def test_two_failures_raise_a_lesson(self) -> None:
        memory_id = self.memory()
        self.record(memory_id, "deploy-1", "failure")
        self.record(memory_id, "deploy-2", "failure")
        found = ol.as_lessons(self.store)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].failures, 2)
        self.assertEqual(found[0].memory_id, memory_id)

    def test_one_failure_is_not_a_lesson(self) -> None:
        memory_id = self.memory()
        self.record(memory_id, "deploy-1", "failure")
        self.assertEqual(ol.as_lessons(self.store), [])

    def test_a_mixed_record_is_not_a_lesson(self) -> None:
        """Three successes and two failures needs more data, not a rewrite."""
        memory_id = self.memory()
        for task in ("a", "b", "c"):
            self.record(memory_id, task, "success")
        for task in ("d", "e"):
            self.record(memory_id, task, "failure")
        self.assertEqual(ol.as_lessons(self.store), [])

    def test_a_bad_record_is_a_lesson(self) -> None:
        memory_id = self.memory()
        self.record(memory_id, "a", "success")
        for task in ("b", "c"):
            self.record(memory_id, task, "failure")
        self.assertEqual(len(ol.as_lessons(self.store)), 1)

    def test_the_lesson_is_a_question(self) -> None:
        memory_id = self.memory()
        self.record(memory_id, "deploy-1", "failure")
        self.record(memory_id, "deploy-2", "failure")
        question = ol.as_lessons(self.store)[0].question()
        self.assertIn("is the memory wrong", question)
        self.assertIn("wrong situation", question)

    def test_the_question_names_the_failing_tasks(self) -> None:
        memory_id = self.memory()
        self.record(memory_id, "deploy-1", "failure")
        self.record(memory_id, "deploy-2", "failure")
        self.assertIn("deploy-1", ol.as_lessons(self.store)[0].question())

    def test_the_lesson_carries_the_memory_text(self) -> None:
        memory_id = self.memory("Deploy with the wrapper.")
        self.record(memory_id, "a", "failure")
        self.record(memory_id, "b", "failure")
        self.assertIn("wrapper", ol.as_lessons(self.store)[0].content)

    def test_persisting_is_idempotent(self) -> None:
        memory_id = self.memory()
        self.record(memory_id, "a", "failure")
        self.record(memory_id, "b", "failure")
        self.assertEqual(ol.persist(self.store, at=NOW), 1)
        self.assertEqual(ol.persist(self.store, at=NOW), 0)
        self.assertEqual(len(ol.lessons(self.store)), 1)

    def test_the_round_trip(self) -> None:
        memory_id = self.memory()
        self.record(memory_id, "a", "failure")
        self.record(memory_id, "b", "failure")
        payload = ol.as_lessons(self.store)[0].as_dict()
        for key in ("lesson_id", "memory_id", "failures", "share", "tasks", "question", "state"):
            self.assertIn(key, payload)

    def test_a_correction_needs_a_person(self) -> None:
        memory_id = self.memory()
        with self.assertRaises(ol.ConsentRequired):
            ol.propose_correction(self.store, memory_id, "Deploy with the other script.", by="")

    def test_a_correction_needs_a_statement(self) -> None:
        memory_id = self.memory()
        with self.assertRaises(ValueError):
            ol.propose_correction(self.store, memory_id, "  ", by="jacoby")

    def test_a_correction_supersedes_rather_than_overwrites(self) -> None:
        memory_id = self.memory("Deploy with the wrapper.")
        self.record(memory_id, "a", "failure")
        self.record(memory_id, "b", "failure")
        made = ol.propose_correction(self.store, memory_id, "Deploy with the systemd unit.",
                                     by="jacoby", at=NOW)
        self.assertTrue(made["corrected"])
        row = self.store._conn.execute(
            "SELECT state, content FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        self.assertEqual(row["state"], "superseded")
        self.assertEqual(row["content"], "Deploy with the wrapper.")  # text is preserved
        winner = self.store._conn.execute(
            "SELECT supersedes_id FROM memories WHERE id = ?", (made["correction_id"],)
        ).fetchone()
        self.assertEqual(winner["supersedes_id"], memory_id)

    def test_a_correction_closes_the_lesson(self) -> None:
        memory_id = self.memory()
        self.record(memory_id, "a", "failure")
        self.record(memory_id, "b", "failure")
        ol.persist(self.store, at=NOW)
        ol.propose_correction(self.store, memory_id, "Use the systemd unit.", by="jacoby", at=NOW)
        self.assertEqual(ol.report(self.store)["open"], 0)
        self.assertEqual(ol.report(self.store)["by_state"].get("corrected"), 1)

    def test_correcting_a_missing_memory_is_not_a_crash(self) -> None:
        made = ol.propose_correction(self.store, "nope", "anything", by="jacoby", at=NOW)
        self.assertFalse(made["corrected"])

    def test_a_lesson_can_close_as_situational(self) -> None:
        memory_id = self.memory()
        self.record(memory_id, "a", "failure")
        self.record(memory_id, "b", "failure")
        ol.persist(self.store, at=NOW)
        closed = ol.close(self.store, memory_id, by="jacoby", note="wrong box that day", at=NOW)
        self.assertEqual(closed["closed"], 1)
        self.assertEqual(ol.lessons(self.store, state="situational")[0]["note"],
                         "wrong box that day (jacoby)")

    def test_closing_needs_a_person(self) -> None:
        with self.assertRaises(ol.ConsentRequired):
            ol.close(self.store, "nope", by="")

    def test_the_report_states_its_thresholds(self) -> None:
        described = ol.report(self.store)
        self.assertEqual(described["open"], 0)
        self.assertEqual(described["thresholds"]["min_failures"], 2)
        self.assertIn("judgement", described["note"])

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(ol.as_lessons(self.store), [])
        self.assertEqual(ol.persist(self.store), 0)
        self.assertEqual(ol.report(self.store)["lessons"], 0)


if __name__ == "__main__":
    unittest.main()
