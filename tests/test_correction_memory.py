"""A correction is a first-class memory, not an edit (#178)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import correction_memory as cm
from cortex.store import CortexStore


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, kind: str = "semantic",
               category: str = "USER_STATED") -> str:
        return self.store.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category=category,
        )[0]

    def row(self, memory_id: str):
        return self.store._conn.execute("SELECT * FROM memories WHERE id = ?",
                                        (memory_id,)).fetchone()

    def test_a_correction_is_written_linked_and_the_old_belief_retired(self) -> None:
        old = self.memory("the dashboard listens on port 8100")
        result = cm.record(self.store, supersedes=old, content="the dashboard listens on port 8101")
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(self.row(result.correction_id)["source_category"], "USER_CORRECTION")
        self.assertEqual(self.row(result.correction_id)["supersedes_id"], old)
        self.assertEqual(self.row(old)["valid_to"] is not None, True)
        self.assertEqual(self.row(old)["state"], "superseded")

    def test_the_old_belief_is_retired_not_deleted(self) -> None:
        """'What did I think in October' still has an answer."""
        old = self.memory("the dashboard listens on port 8100")
        cm.record(self.store, supersedes=old, content="it listens on 8101")
        self.assertIsNotNone(self.row(old))
        self.assertIn("8100", self.row(old)["content"])

    def test_a_correction_carries_the_ceiling(self) -> None:
        old = self.memory("a fact about the port")
        result = cm.record(self.store, supersedes=old, content="the port is 8101")
        self.assertEqual(self.row(result.correction_id)["confidence"], cm.CORRECTION_CEILING)

    def test_an_explicit_confidence_cannot_raise_the_ceiling(self) -> None:
        old = self.memory("a fact about the port")
        result = cm.record(self.store, supersedes=old, content="the port is 8101", confidence=0.99)
        self.assertEqual(self.row(result.correction_id)["confidence"], cm.CORRECTION_CEILING)

    def test_a_correction_without_a_target_is_refused(self) -> None:
        """Otherwise it is just a new fact, and the store ends up with two contradictory beliefs."""
        result = cm.record(self.store, supersedes="", content="the port is 8101")
        self.assertFalse(result.ok)
        self.assertIn("must name what it replaces", result.reason)

    def test_a_correction_with_empty_content_is_refused(self) -> None:
        old = self.memory("a fact about the port")
        self.assertFalse(cm.record(self.store, supersedes=old, content="   ").ok)

    def test_a_correction_of_a_missing_memory_is_refused(self) -> None:
        result = cm.record(self.store, supersedes="nope", content="the port is 8101")
        self.assertFalse(result.ok)
        self.assertIn("does not exist", result.reason)

    def test_an_inference_may_not_supersede_a_correction(self) -> None:
        """A later consolidation pass must not quietly re-derive the belief he just fixed."""
        old = self.memory("the port is 8100")
        corrected = cm.record(self.store, supersedes=old, content="the port is 8101")
        allowed, why = cm.can_supersede(self.store, corrected.correction_id,
                                        source_category="AGENT_INFERENCE")
        self.assertFalse(allowed)
        self.assertIn("he already fixed this", why)

    def test_a_later_correction_may_supersede_an_earlier_one(self) -> None:
        first = self.memory("the port is 8100")
        second = cm.record(self.store, supersedes=first, content="the port is 8101")
        third = cm.record(self.store, supersedes=second.correction_id, content="the port is 8102")
        self.assertTrue(third.ok, third.reason)

    def test_the_chain_is_walkable(self) -> None:
        first = self.memory("the port is 8100")
        second = cm.record(self.store, supersedes=first, content="the port is 8101")
        third = cm.record(self.store, supersedes=second.correction_id, content="the port is 8102")
        walked = cm.chain(self.store, third.correction_id)
        self.assertEqual(walked["length"], 3)
        self.assertIn("8100", walked["items"][-1]["content"])
        self.assertFalse(walked["cycle"])

    def test_a_cycle_is_reported_not_hung(self) -> None:
        first = self.memory("the port is 8100")
        second = self.memory("the port is 8101")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET supersedes_id = ? WHERE id = ?", (second, first))
            conn.execute("UPDATE memories SET supersedes_id = ? WHERE id = ?", (first, second))
        walked = cm.chain(self.store, first)
        self.assertTrue(walked["cycle"])
        self.assertLessEqual(walked["length"], cm.MAX_CHAIN)

    def test_the_chain_of_an_unknown_memory_is_empty(self) -> None:
        self.assertEqual(cm.chain(self.store, "nope")["length"], 0)

    def test_standing_corrections_exclude_retired_ones(self) -> None:
        old = self.memory("the port is 8100")
        first = cm.record(self.store, supersedes=old, content="the port is 8101")
        cm.record(self.store, supersedes=first.correction_id, content="the port is 8102")
        standing = cm.standing(self.store)
        self.assertEqual(len(standing), 1)
        self.assertIn("8102", standing[0]["content"])

    def test_standing_is_newest_first(self) -> None:
        for index in range(3):
            target = self.memory(f"the port is {9000 + index}")
            cm.record(self.store, supersedes=target, content=f"the port is {9100 + index}")
        standing = cm.standing(self.store)
        self.assertEqual(len(standing), 3)

    def test_coverage_reports_an_unused_path(self) -> None:
        self.memory("the port is 8100")
        described = cm.coverage(self.store)
        self.assertEqual(described.corrections, 0)
        self.assertIn("indistinguishable from a fact", described.note)
        self.assertIn("valid_to", described.note)

    def test_coverage_recognises_a_used_path(self) -> None:
        old = self.memory("the port is 8100")
        cm.record(self.store, supersedes=old, content="the port is 8101")
        described = cm.coverage(self.store)
        self.assertEqual(described.corrections, 1)
        self.assertEqual(described.retired, 1)
        self.assertEqual(described.note, "the correction path is in use")

    def test_the_render_states_the_gaps(self) -> None:
        self.memory("the port is 8100")
        rendered = "\n".join(cm.render(self.store))
        self.assertIn("corrections:", rendered)
        self.assertIn("valid_to", rendered)

    def test_a_broken_store_covers_nothing(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(cm.coverage(Broken()).total, 0)


if __name__ == "__main__":
    unittest.main()
