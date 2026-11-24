"""“Why did you say that?” — the audit view (#158)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import audit_view
from cortex.store import CortexStore


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

    def trace(self, task_id: str, *, selected: list[dict], rejected: list[str] | None = None,
              queries: list[str] | None = None, reason: str = "recalled by score") -> str:
        return self.store.record_memory_trace_decision(
            task_id=task_id,
            session_id="s1",
            goal="answer the question",
            context_summary="",
            task_type="coding",
            recall_mode="auto",
            retrieval_used=True,
            retrieval_reason=reason,
            queries=queries or ["which port does the dashboard use"],
            candidate_memories=selected,
            retrieval_context={"rejected_memory_ids": rejected or []},
        )

    def candidate(self, memory_id: str, *, rank: int, score: float, chosen: bool,
                  reason: str = "", components: dict | None = None) -> dict:
        return {
            "memory_id": memory_id,
            "rank": rank,
            "score": score,
            "selected": chosen,
            "reason": reason,
            "components": components or {},
        }

    def test_a_turn_with_no_trace_says_so(self) -> None:
        found = audit_view.audit_turn(self.store, "missing")
        self.assertFalse(found.recorded)
        self.assertIn("no trace recorded", found.render()[0])

    def test_the_selected_candidates_come_back_in_rank_order(self) -> None:
        first, second = self.memory("The dashboard listens on 8100."), self.memory("Nginx proxies it.")
        self.trace("t1", selected=[
            self.candidate(second, rank=2, score=0.4, chosen=True),
            self.candidate(first, rank=1, score=0.9, chosen=True),
        ])
        found = audit_view.audit_turn(self.store, "t1")
        self.assertEqual([row["memory_id"] for row in found.selected], [first, second])

    def test_the_excluded_candidates_are_counted_not_dropped(self) -> None:
        chosen, other = self.memory("kept"), self.memory("dropped")
        self.trace("t2", selected=[
            self.candidate(chosen, rank=1, score=0.9, chosen=True),
            self.candidate(other, rank=4, score=0.2, chosen=False,
                           reason="lower diversified rank than the selected result limit"),
        ])
        found = audit_view.audit_turn(self.store, "t2")
        self.assertEqual(len(found.excluded), 1)
        self.assertIn("lower diversified rank", found.excluded[0]["reason"])

    def test_reasons_are_grouped_for_the_summary(self) -> None:
        chosen = self.memory("kept")
        rows = [self.candidate(chosen, rank=1, score=0.9, chosen=True)]
        for index in range(3):
            rows.append(self.candidate(self.memory(f"no {index}"), rank=5 + index, score=0.1,
                                       chosen=False, reason="below the relevance floor"))
        self.trace("t3", selected=rows)
        found = audit_view.audit_turn(self.store, "t3")
        self.assertEqual(found.by_reason["below the relevance floor"], 3)

    def test_a_candidate_without_a_reason_is_reported_as_unrecorded(self) -> None:
        chosen = self.memory("kept")
        self.trace("t4", selected=[
            self.candidate(chosen, rank=1, score=0.9, chosen=True),
            self.candidate(self.memory("orphan"), rank=2, score=0.5, chosen=False),
        ])
        found = audit_view.audit_turn(self.store, "t4")
        self.assertEqual(found.unrecorded_reasons, 1)
        self.assertTrue(any("no recorded reason" in note for note in found.notes))

    def test_the_score_composition_is_shown_not_just_the_number(self) -> None:
        memory_id = self.memory("popular but thin")
        self.trace("t5", selected=[self.candidate(
            memory_id, rank=1, score=0.61, chosen=True,
            components={"relevance": 0.55, "use_boost": 0.06},
        )])
        found = audit_view.audit_turn(self.store, "t5")
        keys = [pair[0] for pair in found.selected[0]["top_components"]]
        self.assertIn("relevance", keys)

    def test_history_is_named_as_a_share_of_the_score(self) -> None:
        memory_id = self.memory("kept because it was used")
        self.trace("t6", selected=[self.candidate(
            memory_id, rank=1, score=0.5, chosen=True,
            components={"relevance": 0.1, "use_boost": 0.4},
        )])
        found = audit_view.audit_turn(self.store, "t6")
        self.assertGreater(found.selected[0]["history_share"], 0.5)
        self.assertIn("history", " ".join(found.render()))

    def test_a_rejected_id_that_was_never_a_candidate_is_a_recall_gap(self) -> None:
        """The recorder cannot produce this shape, so the row is written directly.

        `record_memory_trace_decision` derives the rejected list from the candidates it was given, so
        an id that was rejected before it ever became a candidate never appears. That is exactly the
        case the view has to survive: a row from an older build, or a pipeline that rejected early.
        """
        chosen = self.memory("kept")
        ghost = self.memory("never offered")
        self.trace("t7", selected=[self.candidate(chosen, rank=1, score=0.9, chosen=True)])
        with self.store.transaction() as conn:
            conn.execute("UPDATE memory_traces SET rejected_memory_ids_json = ? WHERE task_id = ?",
                         (json.dumps([ghost]), "t7"))
        found = audit_view.audit_turn(self.store, "t7")
        self.assertTrue(any("recall gap" in note for note in found.notes), found.notes)

    def test_the_query_is_quoted(self) -> None:
        chosen = self.memory("kept")
        self.trace("t8", selected=[self.candidate(chosen, rank=1, score=0.9, chosen=True)],
                   queries=["why is the dashboard slow"])
        self.assertEqual(audit_view.audit_turn(self.store, "t8").query,
                         "why is the dashboard slow")

    def test_the_retrieval_reason_is_quoted(self) -> None:
        chosen = self.memory("kept")
        self.trace("t9", selected=[self.candidate(chosen, rank=1, score=0.9, chosen=True)],
                   reason="needs continuity, not search")
        self.assertEqual(audit_view.audit_turn(self.store, "t9").retrieval_reason,
                         "needs continuity, not search")

    def test_render_is_short_enough_to_read(self) -> None:
        chosen = self.memory("kept")
        rows = [self.candidate(chosen, rank=1, score=0.9, chosen=True)]
        rows += [self.candidate(self.memory(f"x{index}"), rank=2 + index, score=0.1,
                                chosen=False, reason="below the floor") for index in range(6)]
        self.trace("t10", selected=rows)
        lines = audit_view.audit_turn(self.store, "t10").render(limit=3)
        self.assertLessEqual(len(lines), 9)
        self.assertTrue(any("more:" in line for line in lines))

    def test_why_memory_reports_selections_and_rejections(self) -> None:
        memory_id = self.memory("asked about twice")
        other = self.memory("other")
        self.trace("t11", selected=[
            self.candidate(memory_id, rank=1, score=0.9, chosen=True),
            self.candidate(other, rank=2, score=0.2, chosen=False, reason="below the floor"),
        ])
        self.trace("t12", selected=[
            self.candidate(other, rank=1, score=0.9, chosen=True),
            self.candidate(memory_id, rank=3, score=0.2, chosen=False, reason="below the floor"),
        ])
        found = audit_view.why_memory(self.store, memory_id)
        self.assertEqual(found["turns"], 2)
        self.assertEqual(found["selected"], 1)
        self.assertEqual(found["rejected"]["below the floor"], 1)

    def test_why_memory_says_when_a_memory_was_never_considered(self) -> None:
        found = audit_view.why_memory(self.store, "never-seen")
        self.assertEqual(found["turns"], 0)
        self.assertIn("does not appear", found["sentence"])

    def test_history_share_of_an_empty_score_is_zero(self) -> None:
        self.assertEqual(audit_view.history_share({}), 0.0)

    def test_a_trace_with_no_candidates_is_reported_not_guessed(self) -> None:
        self.trace("t13", selected=[])
        found = audit_view.audit_turn(self.store, "t13")
        self.assertTrue(found.recorded)
        self.assertFalse(found.selected)
        self.assertTrue(any("no candidate list" in note for note in found.notes))

    def test_the_audit_survives_a_store_that_cannot_be_read(self) -> None:
        class Broken:
            _conn = None

            def memory_traces(self, **_kwargs):
                raise RuntimeError("no store")

        found = audit_view.audit_turn(Broken(), "t")
        self.assertFalse(found.recorded)


if __name__ == "__main__":
    unittest.main()


class ComponentTests(unittest.TestCase):
    """A gate is not a contribution."""

    def test_a_gate_is_not_a_graded_component(self) -> None:
        graded, flags = audit_view.split_components({"context_gate": 1.0, "lexical": 0.42})
        self.assertEqual([name for name, _ in graded], ["lexical"])
        self.assertEqual([name for name, _ in flags], ["context_gate"])

    def test_a_zero_is_not_a_gate_worth_listing(self) -> None:
        _, flags = audit_view.split_components({"context_gate": 0.0})
        self.assertEqual(flags, [])

    def test_a_negative_gate_is_still_a_gate(self) -> None:
        _, flags = audit_view.split_components({"blocked": -1.0})
        self.assertEqual([name for name, _ in flags], ["blocked"])

    def test_a_gate_only_row_is_marked_unexplainable(self) -> None:
        self.assertFalse(audit_view.split_components({"context_gate": 1.0})[0])
