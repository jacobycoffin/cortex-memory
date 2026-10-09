"""Batched recall-eligibility must agree with the per-id predicate.

``recall_eligible_ids`` exists so read-only snapshots stop probing once per row
(the review inbox issued ~600 such probes per snapshot, 1,000+ statements).
These tests pin the batched result to ``is_memory_recall_eligible`` across every
dimension that predicate reads — memory state, membership eligibility, revoked
memberships, recall-set status — plus the chunking boundary.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401

from cortex.store import CortexStore


class RecallEligibilityBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.ids = [
            self.store.add_memory(f"batched eligibility fixture memory {index}", kind="semantic")[0]
            for index in range(6)
        ]
        self.active_set_id = self.store.recall_set_snapshot()["active"]["recall_set_id"]

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _set_state(self, memory_id: str, state: str) -> None:
        self.store._conn.execute(
            "UPDATE memories SET state=? WHERE id=?", (state, memory_id)
        )
        self.store._conn.commit()

    def _add_membership(self, memory_id: str, eligibility: str, *, revoked: bool = False) -> None:
        self.store._conn.execute(
            """INSERT INTO memory_recall_memberships
                   (memory_id, recall_set_id, eligibility, origin, created_at, revoked_at)
               VALUES (?, ?, ?, 'trusted_write', '2026-09-14T00:00:00Z', ?)""",
            (memory_id, self.active_set_id, eligibility, "2026-09-14T00:00:00Z" if revoked else None),
        )
        self.store._conn.commit()

    def _parity(self, ids: list[str], **kwargs) -> None:
        expected = {
            memory_id
            for memory_id in ids
            if self.store.is_memory_recall_eligible(memory_id, **kwargs)
        }
        self.assertEqual(self.store.recall_eligible_ids(ids, **kwargs), expected)

    def test_batched_matches_per_id_across_states_and_memberships(self) -> None:
        self._set_state(self.ids[1], "cold")
        self._set_state(self.ids[2], "archived")
        self._set_state(self.ids[3], "quarantine")
        # An active memory whose only live membership is evidence-only: the
        # default predicate (primary only) must exclude it, evidence_lookup
        # must include it.
        self.store._conn.execute(
            "DELETE FROM memory_recall_memberships WHERE memory_id=? AND recall_set_id=?",
            (self.ids[4], self.active_set_id),
        )
        self._add_membership(self.ids[4], "evidence_only")
        # A memory whose membership was revoked is not recall-eligible.
        self.store._conn.execute(
            "DELETE FROM memory_recall_memberships WHERE memory_id=? AND recall_set_id=?",
            (self.ids[5], self.active_set_id),
        )
        self._add_membership(self.ids[5], "primary", revoked=True)

        for kwargs in ({}, {"evidence_lookup": True}, {"include_archived": True}):
            with self.subTest(evidence_lookup=kwargs.get("evidence_lookup", False)):
                self._parity(self.ids, **kwargs)

    def test_batched_handles_memories_without_memberships(self) -> None:
        self.store._conn.execute(
            "DELETE FROM memory_recall_memberships WHERE memory_id=?", (self.ids[0],)
        )
        self.store._conn.commit()
        self.assertFalse(self.store.is_memory_recall_eligible(self.ids[0]))
        self._parity(self.ids)

    def test_chunk_boundary_and_unknown_ids(self) -> None:
        """More ids than one parameter chunk, most of them nonexistent."""
        ids = [f"missing-{index:04d}" for index in range(900)] + self.ids
        expected = {
            memory_id
            for memory_id in ids
            if self.store.is_memory_recall_eligible(memory_id)
        }
        self.assertEqual(self.store.recall_eligible_ids(ids), expected)
        self.assertTrue(expected.issubset(set(self.ids)))

    def test_duplicate_and_empty_inputs(self) -> None:
        self.assertEqual(self.store.recall_eligible_ids([]), set())
        self.assertEqual(
            self.store.recall_eligible_ids([self.ids[0], self.ids[0]]),
            self.store.recall_eligible_ids([self.ids[0]]),
        )


if __name__ == "__main__":
    unittest.main()
