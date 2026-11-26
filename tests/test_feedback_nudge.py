"""The harness asks for a grade, and one word is enough to give it.

Jacoby stated the failure directly: he is willing to grade and forgets, because he is "always working on a bunch of
things through the day". So the ask must come from the receipt, and it must name a token the plugin actually reads —
a prompt suggesting a word nothing recognises is worse than no prompt, because it teaches him the gesture does nothing.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import CortexMemoryProvider
from cortex.hermes_receipts import format_grade_hint, parse_explicit_grade


class TokenTests(unittest.TestCase):
    def test_plus_validates_and_minus_corrects(self) -> None:
        self.assertEqual(parse_explicit_grade("cortex +"), "validated")
        self.assertEqual(parse_explicit_grade("cortex -"), "corrected")

    def test_the_token_survives_being_typed_into_a_sentence(self) -> None:
        self.assertEqual(parse_explicit_grade("that one was right, cortex +"), "validated")
        self.assertEqual(parse_explicit_grade("nope cortex - redo it"), "corrected")
        self.assertEqual(parse_explicit_grade("cortex+"), "validated")

    def test_bare_words_are_not_grades(self) -> None:
        # "no" and "good" appear in ordinary sentences; grading the wrong turn is worse than not grading.
        for text in ("no", "good", "yes please", "cortex", "cortex is fine", "the cortex project", ""):
            self.assertIsNone(parse_explicit_grade(text), f"{text!r} must not be read as a grade")

    def test_the_hint_names_the_token_that_works(self) -> None:
        hint = format_grade_hint()
        self.assertIn("cortex +", hint)
        self.assertIn("cortex -", hint)
        self.assertLess(len(hint), 160, "the ask has to stay one short line")


class ProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = CortexMemoryProvider({"db_path": "$HERMES_HOME/cortex/test.db"})
        self.provider.initialize("session-1", hermes_home=self.tmp.name, agent_context="primary")

    def tearDown(self) -> None:
        self.provider.shutdown()
        self.tmp.cleanup()

    def recall_happened(self, *memory_ids: str, task_id: str = "task-1") -> None:
        self.provider._pending_prefetches["session-1"] = [(list(memory_ids), task_id)]

    # -- the ask ----------------------------------------------------------------------

    def test_a_turn_that_recalled_asks_for_a_grade(self) -> None:
        self.recall_happened("m1", "m2")
        out = self.provider.transform_llm_output("Here is the answer.", session_id="session-1")
        self.assertIsNotNone(out)
        self.assertIn("Cortex recall:", str(out))
        self.assertIn("cortex +", str(out))

    def test_a_turn_that_recalled_nothing_does_not_ask(self) -> None:
        self.assertIsNone(self.provider.transform_llm_output("Just chatting.", session_id="session-1"))

    def test_the_ask_can_be_switched_off(self) -> None:
        self.provider._config["memory_feedback_hint"] = False
        self.recall_happened("m1")
        out = str(self.provider.transform_llm_output("Answer.", session_id="session-1"))
        self.assertNotIn("cortex +", out)

    # -- and one word grades the turn -------------------------------------------------

    def spent_task(self) -> str:
        memory_id = self.provider._store.add_memory(
            content="Run the suite on the Mac with -n 4.", kind="procedure",
            source_type="user_turn", source_category="USER_EXPLICIT",
        )[0]
        task_id = self.provider._store.create_usage_batch(
            [(memory_id, 0.9)], query="how do i run the suite", session_id="session-1",
            task_type="code", recall_mode="focused", requested_budget=600, estimated_tokens=540,
        )
        self.provider._store.resolve_usage(task_id, {memory_id: 0.9})
        self.provider._completed_task_by_session["session-1"] = [task_id]
        return task_id

    def outcome_of(self, task_id: str) -> str:
        row = self.provider._store._conn.execute(
            "SELECT outcome FROM usage_records WHERE task_id=? AND used=1", (task_id,)
        ).fetchone()
        return str(row["outcome"])

    def test_minus_grades_the_previous_turn_as_corrected(self) -> None:
        task_id = self.spent_task()
        self.provider.sync_turn("cortex -", "answer", session_id="session-1")
        self.assertEqual(self.outcome_of(task_id), "corrected")

    def test_plus_grades_the_previous_turn_as_validated(self) -> None:
        task_id = self.spent_task()
        self.provider.sync_turn("cortex +", "answer", session_id="session-1")
        self.assertEqual(self.outcome_of(task_id), "validated")

    def test_ordinary_text_grades_nothing(self) -> None:
        task_id = self.spent_task()
        self.provider.sync_turn("what about the other repo", "answer", session_id="session-1")
        # Attribution happened ("used"); no grade was applied, which is exactly the state a reminder exists to fix.
        self.assertEqual(self.outcome_of(task_id), "used")

    def test_an_explicit_grade_also_grades_a_used_procedure(self) -> None:
        """The whole point of the loop: one word has to reach procedure_confidence."""
        task_id = self.spent_task()
        self.provider.sync_turn("cortex +", "answer", session_id="session-1")
        self.assertEqual(self.outcome_of(task_id), "validated")
        runs = self.provider._store._conn.execute(
            "SELECT COUNT(*) FROM procedure_outcomes"
        ).fetchone()[0]
        self.assertEqual(runs, 1)


if __name__ == "__main__":
    unittest.main()
