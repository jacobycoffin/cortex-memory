"""The shadow ledger must record what abstention *would* have done (#15)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.abstention_ledger import DEFAULT_SHADOW_THRESHOLD, decisions, record, summary
from cortex.store import CortexStore


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_low_score_is_recorded_as_a_shadow_abstention(self) -> None:
        self.assertEqual(
            record(self.store, "t1", top_score=0.20, top_memory_id="m1", session_id="s1"), 1
        )
        stats = summary(self.store)
        self.assertEqual((stats["turns"], stats["would_abstain"]), (1, 1))
        self.assertEqual(stats["abstained"], 0)  # the real policy is still off

    def test_a_high_score_is_not_a_shadow_abstention(self) -> None:
        record(self.store, "t1", top_score=0.80, top_memory_id="m1")
        self.assertEqual(summary(self.store)["would_abstain"], 0)

    def test_re_recording_a_turn_replaces_its_row(self) -> None:
        record(self.store, "t1", top_score=0.20, top_memory_id="m1")
        record(self.store, "t1", top_score=0.90, top_memory_id="m2")
        stats = summary(self.store)
        self.assertEqual((stats["turns"], stats["would_abstain"]), (1, 0))

    def test_precision_counts_withheld_memories_that_were_used(self) -> None:
        self.store.set_pinned  # attribute exists; not used here, keeps linters honest
        memory_id = self.store.add_memory(
            content="A fact that mattered later.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        memory_id = memory_id[0] if isinstance(memory_id, tuple) else memory_id
        record(self.store, "t-hit", top_score=0.20, top_memory_id=memory_id)
        record(self.store, "t-miss", top_score=0.20, top_memory_id="never-used")
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO access_log (memory_id, event, created_at) "
                "VALUES (?, 'used', datetime('now'))",
                (memory_id,),
            )
        stats = summary(self.store)
        self.assertEqual(stats["would_abstain"], 2)
        self.assertEqual(stats["would_abstain_then_used"], 1)
        self.assertAlmostEqual(stats["would_abstain_precision"], 0.5, places=4)

    def test_empty_ledger_reports_zeros(self) -> None:
        stats = summary(self.store)
        self.assertEqual((stats["turns"], stats["would_abstain"]), (0, 0))
        self.assertEqual(stats["would_abstain_precision"], 0.0)

    def test_decisions_lists_only_shadow_abstentions(self) -> None:
        record(self.store, "low", top_score=0.10, top_memory_id="m1")
        record(self.store, "high", top_score=0.90, top_memory_id="m2")
        rows = decisions(self.store)
        self.assertEqual([r[0] for r in rows], ["low"])

    def test_a_missing_connection_is_not_an_error(self) -> None:
        class Broken:
            pass

        self.assertEqual(record(Broken(), "t", top_score=0.1, top_memory_id="m"), 0)
        self.assertEqual(summary(Broken())["turns"], 0)

    def test_default_threshold_matches_the_calibration(self) -> None:
        self.assertAlmostEqual(DEFAULT_SHADOW_THRESHOLD, 0.55, places=2)


if __name__ == "__main__":
    unittest.main()
