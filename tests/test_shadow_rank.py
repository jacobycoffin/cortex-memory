"""What a scoring change would have done, measured on history (#202)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import shadow_rank
from cortex.store import CortexStore


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def trace(self, candidates: list[dict], *, trace_id: str = "t1") -> None:
        """Insert a recorded trace: each candidate carries score, rank, selected and components."""
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO memory_traces (trace_id, task_id, session_id, goal, task_type,
                       recall_mode, context_summary, retrieval_reason, created_at,
                       candidate_memories_json, selected_memory_ids_json, updated_at)
                   VALUES (?, 'task', 'session', 'goal', 'general', 'standard', 'summary', '',
                           '2026-11-24T10:00:00+00:00', ?, '[]', '2026-11-24T10:00:00+00:00')""",
                (trace_id, json.dumps(candidates)),
            )

    def candidate(self, memory_id: str, score: float, rank: int, *, selected: bool,
                  use_boost: float = 0.0) -> dict:
        return {
            "memory_id": memory_id,
            "score": score,
            "rank": rank,
            "selected": selected,
            "components": {"use_boost": use_boost} if use_boost else {},
        }

    def test_a_policy_that_changes_nothing_says_inert(self) -> None:
        """No candidate carries a usage boost, so removing it must not read as 'measured, harmless'."""
        self.trace([
            self.candidate("a", 0.9, 1, selected=True),
            self.candidate("b", 0.8, 2, selected=True),
        ])
        found = shadow_rank.replay(self.store, shadow_rank.Policy("no_usage_boost", ("usage",)))
        self.assertTrue(found.inert)
        self.assertIn("inert", found.describe())

    def test_a_boost_that_flips_the_top_is_reported(self) -> None:
        """b is only first because of a 0.15 usage boost; remove it and a wins."""
        self.trace([
            self.candidate("a", 0.90, 2, selected=True),
            self.candidate("b", 0.95, 1, selected=True, use_boost=0.15),
        ])
        found = shadow_rank.replay(self.store, shadow_rank.Policy("no_usage_boost", ("usage",)))
        self.assertEqual(found.changed, 1)
        self.assertEqual(found.top1_changed, 1)
        self.assertFalse(found.inert)

    def test_the_arithmetic_is_exact(self) -> None:
        """b at 0.95 with a 0.15 boost scores 0.80 without it, which is below a's 0.90."""
        self.trace([
            self.candidate("a", 0.90, 2, selected=True),
            self.candidate("b", 0.95, 1, selected=True, use_boost=0.15),
        ])
        found = shadow_rank.replay(self.store, shadow_rank.Policy("no_usage_boost", ("usage",)))
        self.assertEqual(found.examples[0]["recorded"], ["b", "a"])
        self.assertEqual(found.examples[0]["shadow"], ["a", "b"])

    def test_a_removal_that_does_not_change_the_order_is_counted_as_touched(self) -> None:
        self.trace([
            self.candidate("a", 0.99, 1, selected=True, use_boost=0.05),
            self.candidate("b", 0.80, 2, selected=True),
        ])
        found = shadow_rank.replay(self.store, shadow_rank.Policy("no_usage_boost", ("usage",)))
        self.assertEqual(found.touched, 1)
        self.assertEqual(found.changed, 0)

    def test_the_replay_keeps_the_recorded_selection_size(self) -> None:
        self.trace([
            self.candidate("a", 0.99, 1, selected=True),
            self.candidate("b", 0.98, 2, selected=True),
            self.candidate("c", 0.10, 3, selected=False),
        ])
        found = shadow_rank.replay(self.store, shadow_rank.Policy("no_boost_at_all",
                                                                  tuple(shadow_rank.COMPONENTS)))
        self.assertEqual(found.compared, 1)
        self.assertEqual(found.traces, 1)

    def test_a_trace_with_one_candidate_is_skipped(self) -> None:
        self.trace([self.candidate("a", 0.9, 1, selected=True)])
        found = shadow_rank.replay(self.store, shadow_rank.Policy("no_usage_boost", ("usage",)))
        self.assertEqual(found.compared, 0)
        self.assertEqual(found.changed_share, 0.0)

    def test_ties_break_on_id_not_on_dict_order(self) -> None:
        """Two equal scores must not produce a phantom change from list order."""
        self.trace([
            self.candidate("b", 0.90, 2, selected=True),
            self.candidate("a", 0.90, 1, selected=True),
        ])
        found = shadow_rank.replay(self.store, shadow_rank.Policy("no_usage_boost", ("usage",)))
        self.assertEqual(found.changed, 0)

    def test_the_report_replays_every_built_in_policy(self) -> None:
        self.trace([
            self.candidate("a", 0.90, 2, selected=True),
            self.candidate("b", 0.95, 1, selected=True, use_boost=0.15),
        ])
        described = shadow_rank.report(self.store)
        self.assertEqual(len(described["policies"]), len(shadow_rank.policies()))
        names = {item["policy"] for item in described["policies"]}
        self.assertIn("no_usage_boost", names)
        self.assertIn("no_boost_at_all", names)

    def test_an_empty_store_is_quiet(self) -> None:
        described = shadow_rank.report(self.store)
        self.assertEqual(described["traces_replayed"], 0)
        self.assertTrue(all(item["inert"] for item in described["policies"]))

    def test_the_policy_names_its_components(self) -> None:
        described = shadow_rank.policies()[0].as_dict()
        self.assertEqual(described["remove"], ["usage"])
        self.assertTrue(described["note"])

    def test_an_unknown_family_is_ignored_rather_than_crashing(self) -> None:
        self.trace([
            self.candidate("a", 0.9, 1, selected=True),
            self.candidate("b", 0.8, 2, selected=True),
        ])
        found = shadow_rank.replay(self.store, shadow_rank.Policy("odd", ("nonexistent",)))
        self.assertEqual(found.changed, 0)


if __name__ == "__main__":
    unittest.main()
