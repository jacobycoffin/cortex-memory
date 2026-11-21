"""Scored task classification and its labelled set (#218)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.store import CortexStore
from cortex.task_class import GENERAL, MIN_SCORE, classify
from cortex.tooling import ToolExecution, classify_task


def execution(index: int, task_type: str, context: str, tool: str, success: bool) -> ToolExecution:
    return ToolExecution(
        execution_id=f"exec-{index}",
        session_id="scope-session",
        task_type=task_type,
        task_context=context,
        tool_name=tool,
        argument_keys=("command",),
        success=success,
        error_type=None if success else "permission",
        result_summary="ok" if success else "permission denied",
        step_index=index,
    )


class ScoredClassificationTests(unittest.TestCase):
    def test_the_reported_misfile_is_fixed(self) -> None:
        """The case from the backlog item: 'dashboard' used to decide the class."""
        self.assertEqual(classify_task("which systemd unit runs the cortex dashboard"), "shell")

    def test_a_strong_phrase_beats_a_weak_word(self) -> None:
        self.assertEqual(classify("check the dashboard csv").task_type, "data_analysis")

    def test_split_evidence_abstains(self) -> None:
        result = classify("research current agent memory papers")
        self.assertEqual(result.task_type, GENERAL)
        self.assertIn("tie", result.reason)

    def test_a_single_weak_word_is_not_a_class(self) -> None:
        result = classify("what should we build next")
        self.assertEqual(result.task_type, GENERAL)
        self.assertIn(str(MIN_SCORE), result.reason)

    def test_no_vocabulary_is_general(self) -> None:
        self.assertEqual(classify("tell me a joke about homelabs").task_type, GENERAL)

    def test_plurals_count(self) -> None:
        self.assertEqual(
            classify("open the container log").scores, classify("open the containers logs").scores
        )

    def test_tool_name_participates(self) -> None:
        self.assertEqual(classify_task("do it", "shell_exec"), "shell")

    def test_scores_are_returned_for_every_matching_class(self) -> None:
        result = classify("which systemd unit runs the cortex dashboard")
        self.assertGreaterEqual(set(result.scores), {"shell", "data_analysis"})
        self.assertEqual(result.runner_up, "data_analysis")


class LabelledSetTests(unittest.TestCase):
    """The labelled set is synthetic, so it is allowed in the repo."""

    @classmethod
    def setUpClass(cls) -> None:
        path = Path(ROOT) / "evalsets" / "task-class-v1.jsonl"
        cls.cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_the_set_is_not_trivially_small(self) -> None:
        self.assertGreaterEqual(len(self.cases), 50)

    def test_accuracy_holds_and_beats_the_first_match_baseline(self) -> None:
        from cortex.tooling import _TASK_RULES

        def legacy(query: str) -> str:
            for name, pattern in _TASK_RULES:
                if pattern.search(query):
                    return name
            return GENERAL

        scored_right = sum(
            1 for case in self.cases if classify(case["query"]).task_type == case["task_type"]
        )
        legacy_right = sum(1 for case in self.cases if legacy(case["query"]) == case["task_type"])
        scored = scored_right / len(self.cases)
        first_match = legacy_right / len(self.cases)
        self.assertGreaterEqual(scored, 0.80, f"scored accuracy regressed to {scored:.1%}")
        self.assertGreater(scored, first_match, "scored must beat first-match-wins")

    def test_misses_abstain_more_often_than_they_misfile(self) -> None:
        """A wrong class spends recall and tokens silently; `general` spends neither."""
        wrong = [
            case
            for case in self.cases
            if classify(case["query"]).task_type not in (case["task_type"], GENERAL)
        ]
        self.assertLessEqual(len(wrong), len(self.cases) * 0.2)


class ToolLedgerScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        for index, context in enumerate(("first shell task", "second shell task"), start=1):
            self.store.record_tool_execution(
                execution(index, "shell", context, "shell_exec", success=True)
            )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_an_unknown_class_sees_guidance_from_any_class(self) -> None:
        """`general` turns used to lose every tool warning (#218)."""
        self.assertEqual(self.store.tool_guidance("general"), [])
        scoped = self.store.tool_guidance(None)
        self.assertTrue(scoped)
        self.assertEqual(scoped[0]["task_type"], "shell")


if __name__ == "__main__":
    unittest.main()
