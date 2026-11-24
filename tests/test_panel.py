from __future__ import annotations

import unittest

from tests._bootstrap import ROOT  # noqa: F401

from cortex.panel import render_memory_panel


def _candidate(memory_id: str, *, rank: int, score: float, selected: bool, **extra):
    base = {
        "memory_id": memory_id,
        "kind": "semantic",
        "rank": rank,
        "score": score,
        "selected": selected,
        "estimated_tokens": 120,
        "content_preview": f"preview for {memory_id}",
        "reason": "selected by score, context budget, and diversity constraints"
        if selected
        else "would exceed the task context budget",
        "components": {"context_gate": 1.0},
    }
    base.update(extra)
    return base


def _trace(**extra):
    trace = {
        "trace_id": "2e98b6b5-680c-4385-8e1c-cb71fe3df44c",
        "task_id": "task-1",
        "goal": "where did I put the spare key",
        "recall_mode": "focused",
        "outcome": "completed_unlabeled",
        "retrieval_used": True,
        "retrieval_reason": "selected 2 memories",
        "created_at": "2026-11-24T12:00:00+00:00",
        "candidate_memories": [
            _candidate("aaaaaaaa-1111", rank=1, score=0.71, selected=True),
            _candidate("bbbbbbbb-2222", rank=2, score=0.55, selected=True),
            _candidate("cccccccc-3333", rank=3, score=0.40, selected=False),
            _candidate("dddddddd-4444", rank=4, score=0.39, selected=False),
            _candidate("eeeeeeee-5555", rank=5, score=0.38, selected=False),
            _candidate("ffffffff-6666", rank=6, score=0.37, selected=False),
        ],
        "influence": [
            {"memory_id": "aaaaaaaa-1111", "influenced": True, "attribution_score": 0.42},
            {"memory_id": "bbbbbbbb-2222", "influenced": False, "attribution_score": 0.0},
        ],
    }
    trace.update(extra)
    return trace


class MemoryPanelTests(unittest.TestCase):
    def test_lists_injected_and_excluded_with_counts(self) -> None:
        text = render_memory_panel([_trace()], width=100, color=False)
        self.assertIn("injected 2 of 6 candidates", text)
        self.assertIn("~240 tok", text)
        self.assertIn("aaaaaaaa", text)
        self.assertIn("preview for aaaaaaaa-1111", text)
        self.assertIn("EXCLUDED (4)", text)
        self.assertIn("would exceed the task context budget", text)
        self.assertNotIn("more excluded", text)

    def test_distinguishes_cited_from_merely_retrieved(self) -> None:
        text = render_memory_panel([_trace()], width=100)
        self.assertIn("cited in the answer (0.42)", text)
        self.assertIn("retrieved, no answer-use evidence", text)

    def test_caps_excluded_and_says_how_many_are_hidden(self) -> None:
        text = render_memory_panel([_trace()], width=100, max_excluded=2)
        self.assertIn("EXCLUDED (4)", text)
        self.assertIn("… 2 more excluded (--all to show)", text)
        self.assertNotIn("ffffffff", text)

    def test_abstention_is_stated_not_implied(self) -> None:
        trace = _trace(
            candidate_memories=[
                _candidate("cccccccc-3333", rank=1, score=0.2, selected=False)
            ],
            influence=[],
            retrieval_used=False,
            retrieval_reason="no candidate cleared the relevance floor",
        )
        text = render_memory_panel([trace], width=100)
        self.assertIn("NOTHING INJECTED", text)
        self.assertIn("no candidate cleared the relevance floor", text)
        self.assertNotIn("\nINJECTED\n", text)

    def test_empty_database_is_explicit(self) -> None:
        text = render_memory_panel([], width=100)
        self.assertIn("no turns recorded", text)

    def test_no_rendered_line_exceeds_the_requested_width(self) -> None:
        trace = _trace(
            goal="a very long goal " * 40,
            candidate_memories=[
                _candidate(
                    "aaaaaaaa-1111",
                    rank=1,
                    score=0.71,
                    selected=True,
                    content_preview="long preview " * 60,
                    reason="reason " * 60,
                    metacognition={
                        "applied": False,
                        "decision": "use",
                        "calibrated_probability": 0.8,
                        "reason": "long judge reason " * 20,
                    },
                ),
                _candidate("cccccccc-3333", rank=2, score=0.4, selected=False),
            ],
        )
        text = render_memory_panel([trace], width=60, show_all_excluded=True)
        for line in text.splitlines():
            self.assertLessEqual(len(line), 60, line)

    def test_color_can_be_disabled(self) -> None:
        text = render_memory_panel([_trace()], width=100, color=False)
        self.assertNotIn("\x1b[", text)


if __name__ == "__main__":
    unittest.main()
