"""What a memory was chosen over (#77)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import counterfactuals
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class PairTests(unittest.TestCase):
    def trace(self, *, selected: list[str], rejected: list[str],
              candidates: list[tuple[str, int, float]] | None = None) -> dict:
        rows = candidates or [(memory_id, index + 1, 1.0 - index * 0.01)
                              for index, memory_id in enumerate(selected + rejected)]
        return {
            "trace_id": "t1",
            "selected_memory_ids": selected,
            "rejected_memory_ids": rejected,
            "candidate_memories": [
                {"memory_id": memory_id, "rank": rank, "score": score}
                for memory_id, rank, score in rows
            ],
        }

    def test_a_selected_memory_is_paired_with_the_nearest_rejection(self) -> None:
        pairs = counterfactuals.pairs_for(
            self.trace(selected=["a"], rejected=["b", "c", "d", "e"], candidates=[
                ("a", 1, 1.00), ("b", 2, 0.99), ("c", 3, 0.98), ("d", 4, 0.90), ("e", 5, 0.80),
            ])
        )
        self.assertEqual([pair.rejected_id for pair in pairs], ["b", "c", "d"])

    def test_only_the_nearest_few_are_kept(self) -> None:
        pairs = counterfactuals.pairs_for(
            self.trace(selected=["a"], rejected=["b", "c", "d", "e"], candidates=[
                ("a", 1, 1.00), ("b", 2, 0.99), ("c", 3, 0.98), ("d", 4, 0.90), ("e", 5, 0.80),
            ]), alternatives=2)
        self.assertEqual(len(pairs), 2)

    def test_the_gap_is_the_recorded_score_difference(self) -> None:
        pairs = counterfactuals.pairs_for(
            self.trace(selected=["a"], rejected=["b"], candidates=[
                ("a", 1, 0.90), ("b", 2, 0.85),
            ])
        )
        self.assertAlmostEqual(pairs[0].score_gap, 0.05, places=6)

    def test_every_selected_memory_gets_its_own_pairs(self) -> None:
        pairs = counterfactuals.pairs_for(
            self.trace(selected=["a", "b"], rejected=["c"], candidates=[
                ("a", 1, 1.0), ("b", 2, 0.9), ("c", 3, 0.8),
            ])
        )
        self.assertEqual({pair.chosen_id for pair in pairs}, {"a", "b"})

    def test_a_trace_without_rejections_pairs_nothing(self) -> None:
        self.assertEqual(counterfactuals.pairs_for(self.trace(selected=["a"], rejected=[])), [])

    def test_a_trace_without_selections_pairs_nothing(self) -> None:
        self.assertEqual(counterfactuals.pairs_for(self.trace(selected=[], rejected=["b"])), [])

    def test_a_rejection_missing_from_the_candidates_is_ignored(self) -> None:
        pairs = counterfactuals.pairs_for(self.trace(selected=["a"], rejected=["ghost"],
                                                     candidates=[("a", 1, 1.0)]))
        self.assertEqual(pairs, [])

    def test_the_pair_round_trips(self) -> None:
        pairs = counterfactuals.pairs_for(self.trace(selected=["a"], rejected=["b"]))
        payload = pairs[0].as_dict()
        for key in ("chosen_id", "rejected_id", "chosen_rank", "rejected_rank", "score_gap"):
            self.assertIn(key, payload)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def trace(self, *, trace_id: str, selected: list[str], rejected: list[str],
              scores: dict[str, float] | None = None) -> None:
        scores = scores or {}
        rows = []
        for index, memory_id in enumerate(selected + rejected, start=1):
            rows.append({"memory_id": memory_id, "rank": index,
                         "score": scores.get(memory_id, 1.0 - index * 0.01),
                         "selected": memory_id in selected})
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO memory_traces (trace_id, task_id, session_id, goal, task_type,
                       recall_mode, context_summary, retrieval_reason, created_at,
                       candidate_memories_json, selected_memory_ids_json,
                       rejected_memory_ids_json, updated_at)
                   VALUES (?, ?, 's', 'goal', 'general', 'standard', 'summary', '',
                           '2026-11-24T10:00:00+00:00', ?, ?, ?, '2026-11-24T10:00:00+00:00')""",
                (trace_id, f"task-{trace_id}", json.dumps(rows), json.dumps(selected),
                 json.dumps(rejected)),
            )

    def test_recording_keeps_the_near_misses(self) -> None:
        self.trace(trace_id="t1", selected=["a"], rejected=["b", "c"])
        found = counterfactuals.record(self.store, at=NOW)
        self.assertEqual(found.paired, 2)
        self.assertEqual(len(counterfactuals.alternatives(self.store, "a")), 2)

    def test_recording_twice_keeps_one_row_per_pair(self) -> None:
        self.trace(trace_id="t1", selected=["a"], rejected=["b"])
        counterfactuals.record(self.store, at=NOW)
        counterfactuals.record(self.store, at=NOW)
        self.assertEqual(counterfactuals.report(self.store)["pairs"], 1)

    def test_the_mirror_question_works(self) -> None:
        self.trace(trace_id="t1", selected=["a"], rejected=["b"])
        counterfactuals.record(self.store, at=NOW)
        self.assertEqual(len(counterfactuals.beaten_by(self.store, "b")), 1)

    def test_the_narrowest_calls_come_first(self) -> None:
        self.trace(trace_id="t1", selected=["a"], rejected=["b"],
                   scores={"a": 0.90, "b": 0.89})
        self.trace(trace_id="t2", selected=["c"], rejected=["d"],
                   scores={"c": 0.90, "d": 0.50})
        counterfactuals.record(self.store, at=NOW)
        narrowest = counterfactuals.narrowest(self.store)
        self.assertEqual(narrowest[0]["chosen_id"], "a")
        self.assertAlmostEqual(abs(narrowest[0]["score_gap"]), 0.01, places=6)

    def test_a_trace_without_rejections_is_counted_not_hidden(self) -> None:
        self.trace(trace_id="t1", selected=["a"], rejected=[])
        found = counterfactuals.record(self.store, at=NOW)
        self.assertEqual(found.paired, 0)
        self.assertEqual(found.skipped_no_rejections, 1)
        self.assertIn("recorded no rejections", found.describe())

    def test_the_report_counts_distinct_ids(self) -> None:
        self.trace(trace_id="t1", selected=["a"], rejected=["b", "c"])
        self.trace(trace_id="t2", selected=["a"], rejected=["b"])
        counterfactuals.record(self.store, at=NOW)
        described = counterfactuals.report(self.store)
        self.assertEqual(described["pairs"], 3)
        self.assertEqual(described["traces"], 2)
        self.assertEqual(described["chosen"], 1)
        self.assertEqual(described["rejected"], 2)

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(counterfactuals.record(self.store, at=NOW).paired, 0)
        self.assertEqual(counterfactuals.alternatives(self.store, "x"), [])
        self.assertEqual(counterfactuals.report(self.store)["pairs"], 0)

    def test_the_limit_bounds_the_traces_read(self) -> None:
        for index in range(4):
            self.trace(trace_id=f"t{index}", selected=["a"], rejected=["b"])
        found = counterfactuals.record(self.store, limit=2, at=NOW)
        self.assertEqual(found.traces, 2)


if __name__ == "__main__":
    unittest.main()
