"""Current, superseded or disputed — and a hedge instead of a coin flip (#48)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import dispute
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class DisputeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content,
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]


class MarkTests(DisputeTestCase):
    def test_marking_creates_one_open_pair(self) -> None:
        left = self.memory("The maintenance window is Sunday.")
        right = self.memory("The maintenance window is Tuesday.")
        dispute.mark(self.store, left, right, reason="two sources disagree", at=NOW)
        listed = dispute.pairs(self.store)
        self.assertEqual(len(listed), 1)
        self.assertEqual({listed[0].left_id, listed[0].right_id}, {left, right})

    def test_the_pair_is_order_independent(self) -> None:
        left = self.memory("window is Sunday")
        right = self.memory("window is Tuesday")
        dispute.mark(self.store, left, right, at=NOW)
        dispute.mark(self.store, right, left, at=NOW)
        self.assertEqual(len(dispute.pairs(self.store)), 1)

    def test_a_memory_cannot_dispute_itself(self) -> None:
        memory_id = self.memory("window is Sunday")
        self.assertIsNone(dispute.mark(self.store, memory_id, memory_id))

    def test_an_unknown_id_is_refused(self) -> None:
        memory_id = self.memory("window is Sunday")
        self.assertIsNone(dispute.mark(self.store, memory_id, "00000000-0000-0000-0000-000000000000"))

    def test_prefixes_resolve(self) -> None:
        left = self.memory("window is Sunday")
        right = self.memory("window is Tuesday")
        dispute.mark(self.store, left[:8], right[:8], at=NOW)
        self.assertEqual(dispute.disputed_ids(self.store), {left, right})


class StatusTests(DisputeTestCase):
    def test_an_ordinary_memory_is_current(self) -> None:
        memory_id = self.memory("the greenhouse runs on an ESP32")
        self.assertEqual(dispute.status(self.store, memory_id), dispute.CURRENT)

    def test_a_disputed_memory_reports_disputed(self) -> None:
        left = self.memory("window is Sunday")
        right = self.memory("window is Tuesday")
        dispute.mark(self.store, left, right)
        self.assertEqual(dispute.status(self.store, left), dispute.DISPUTED)
        self.assertEqual(dispute.status(self.store, right), dispute.DISPUTED)

    def test_a_superseded_memory_reports_superseded(self) -> None:
        old = self.memory("window is Sunday")
        new = self.memory("window is Tuesday")
        with self.store.transaction() as conn:
            # the newer memory supersedes the older one
            conn.execute("UPDATE memories SET supersedes_id = ? WHERE id = ?", (old, new))
        self.assertEqual(dispute.status(self.store, old), dispute.SUPERSEDED)
        self.assertEqual(dispute.status(self.store, new), dispute.CURRENT)

    def test_dispute_outranks_supersession(self) -> None:
        """A memory can be both; the hedge matters more than the demotion."""
        old = self.memory("window is Sunday")
        new = self.memory("window is Tuesday")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET supersedes_id = ? WHERE id = ?", (old, new))
        dispute.mark(self.store, old, new)
        self.assertEqual(dispute.status(self.store, old), dispute.DISPUTED)


class ResolveTests(DisputeTestCase):
    def test_resolving_without_a_winner_closes_the_pair(self) -> None:
        left = self.memory("window is Sunday")
        right = self.memory("window is Tuesday")
        dispute.mark(self.store, left, right)
        self.assertTrue(dispute.resolve(self.store, left, right, reason="vendor confirmed Sunday"))
        self.assertEqual(dispute.pairs(self.store), [])
        self.assertEqual(dispute.status(self.store, left), dispute.CURRENT)

    def test_resolving_with_a_winner_supersedes_the_loser(self) -> None:
        left = self.memory("window is Sunday")
        right = self.memory("window is Tuesday")
        dispute.mark(self.store, left, right)
        dispute.resolve(self.store, left, right, winner=right)
        self.assertEqual(dispute.status(self.store, left), dispute.SUPERSEDED)
        self.assertEqual(dispute.status(self.store, right), dispute.CURRENT)

    def test_resolving_an_unknown_pair_is_a_no_op(self) -> None:
        left = self.memory("window is Sunday")
        right = self.memory("window is Tuesday")
        self.assertFalse(dispute.resolve(self.store, left, right))

    def test_a_resolved_pair_can_be_reopened(self) -> None:
        left = self.memory("window is Sunday")
        right = self.memory("window is Tuesday")
        dispute.mark(self.store, left, right)
        dispute.resolve(self.store, left, right)
        dispute.mark(self.store, left, right, reason="vendor changed it again")
        self.assertEqual(dispute.status(self.store, left), dispute.DISPUTED)


class HedgeTests(DisputeTestCase):
    def test_no_dispute_means_no_hedge(self) -> None:
        memory_id = self.memory("the greenhouse runs on an ESP32")
        self.assertEqual(dispute.hedge(self.store, [memory_id]), "")

    def test_a_disputed_result_produces_a_hedge_naming_the_count(self) -> None:
        left = self.memory("window is Sunday")
        right = self.memory("window is Tuesday")
        dispute.mark(self.store, left, right)
        note = dispute.hedge(self.store, [left, right])
        self.assertIn("2 of the recalled memories are disputed", note)
        self.assertIn("uncertain", note)

    def test_a_hedge_does_not_demote_anything(self) -> None:
        """A dispute is not evidence that one side is wrong, so neither is penalised."""
        left = self.memory("window is Sunday")
        right = self.memory("window is Tuesday")
        before = self.store.get_memory(left)
        dispute.mark(self.store, left, right)
        after = self.store.get_memory(left)
        self.assertEqual(before["state"], after["state"])
        self.assertEqual(before["importance"], after["importance"])

    def test_the_report_counts_open_pairs_and_memories(self) -> None:
        first = self.memory("a1")
        second = self.memory("a2")
        third = self.memory("a3")
        dispute.mark(self.store, first, second)
        dispute.mark(self.store, second, third)
        described = dispute.report(self.store)
        self.assertEqual(described["open"], 2)
        self.assertEqual(described["memories"], 3)

    def test_statuses_are_computed_in_one_pass(self) -> None:
        left = self.memory("window is Sunday")
        right = self.memory("window is Tuesday")
        plain = self.memory("unrelated")
        dispute.mark(self.store, left, right)
        state = dispute.statuses(self.store, [left, right, plain])
        self.assertEqual(state[left], dispute.DISPUTED)
        self.assertEqual(state[plain], dispute.CURRENT)


if __name__ == "__main__":
    unittest.main()
