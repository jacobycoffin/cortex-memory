from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import sleep_proposals
from cortex.store import CortexStore


class SleepProposalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    # -- fixtures ---------------------------------------------------------------------------

    def _memory(self, text: str, *, state: str = "active") -> str:
        memory_id, _ = self.store.add_memory(
            text,
            kind="semantic",
            source_category="TOOL_VERIFIED",
            state=state,
        )
        return memory_id

    def _observations_table(self, *, pk: bool = True) -> None:
        primary = "PRIMARY KEY (left_id, right_id, session_id)" if pk else ""
        with self.store.transaction() as conn:
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS co_retrieval_observations (
                        left_id TEXT NOT NULL,
                        right_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        turn_id TEXT,
                        observed_at TEXT NOT NULL
                        {"," if primary else ""} {primary}
                    )"""
            )

    def _observe(
        self,
        left: str,
        right: str,
        session: str,
        *,
        turn: str | None = None,
        at: str | None = None,
        pk: bool = True,
    ) -> None:
        self._observations_table(pk=pk)
        stamp = at or datetime.now(timezone.utc).isoformat()
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO co_retrieval_observations
                   (left_id, right_id, session_id, turn_id, observed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (left, right, session, turn, stamp),
            )

    def _observe_pair(self, left: str, right: str, sessions: tuple[str, ...]) -> None:
        for session in sessions:
            self._observe(left, right, session)

    def _veto(self, left: str, right: str) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS edge_feedback (
                       left_id TEXT NOT NULL,
                       right_id TEXT NOT NULL,
                       polarity TEXT NOT NULL,
                       note TEXT NOT NULL DEFAULT '',
                       created_at TEXT NOT NULL,
                       PRIMARY KEY (left_id, right_id)
                   )"""
            )
            conn.execute(
                """INSERT INTO edge_feedback(left_id, right_id, polarity, note, created_at)
                   VALUES (?, ?, 'unrelated', 'operator said so', ?)""",
                (left, right, datetime.now(timezone.utc).isoformat()),
            )
        return None

    def _edge_count(self) -> int:
        with self.store._lock:
            return int(self.store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])

    # -- build ------------------------------------------------------------------------------

    def test_build_proposes_pair_seen_in_two_sessions(self) -> None:
        first, second = self._memory("restic backs up the cortex store"), self._memory(
            "restic snapshots are verified nightly"
        )
        self._observe_pair(first, second, ("session-a", "session-b"))

        built = sleep_proposals.build(self.store)

        self.assertEqual(len(built), 1)
        proposal = built[0]
        self.assertEqual(proposal.pair, tuple(sorted((first, second))))
        self.assertEqual(proposal.relation, "co_retrieved")
        self.assertEqual(proposal.sessions, 2)
        self.assertEqual(proposal.state, "proposed")
        self.assertGreater(proposal.confidence, 0.0)
        self.assertLessEqual(proposal.confidence, sleep_proposals.CONFIDENCE_CAP)
        self.assertEqual(proposal.evidence["session_ids"], ["session-a", "session-b"])
        self.assertFalse(proposal.evidence["meets_edge_evidence_bar"])

    def test_build_is_read_only_and_writes_no_edges(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        self._observe_pair(first, second, ("a", "b"))

        sleep_proposals.build(self.store)

        self.assertEqual(self._edge_count(), 0)
        with self.store._lock:
            rows = self.store._conn.execute("SELECT COUNT(*) FROM edge_proposals").fetchone()[0]
        self.assertEqual(int(rows), 0)

    def test_build_on_missing_observations_table_returns_empty(self) -> None:
        self.assertEqual(sleep_proposals.build(self.store), [])
        report = sleep_proposals.report(self.store)
        self.assertEqual(report["observations"]["pairs"], 0)
        self.assertEqual(report["observations"]["source"], "absent")
        self.assertTrue(report["observations"]["empty"])

    def test_build_refuses_single_session_even_across_many_turns(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        # No primary key on this fixture table: five rows, one conversation.
        for index in range(5):
            self._observe(first, second, "chatty-session", turn=f"turn-{index}", pk=False)

        self.assertEqual(sleep_proposals.build(self.store), [])

    def test_build_respects_min_sessions_parameter(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        self._observe_pair(first, second, ("a", "b"))

        self.assertEqual(len(sleep_proposals.build(self.store, min_sessions=2)), 1)
        self.assertEqual(sleep_proposals.build(self.store, min_sessions=3), [])

        self._observe(first, second, "c")
        built = sleep_proposals.build(self.store, min_sessions=3)
        self.assertEqual(len(built), 1)
        self.assertEqual(built[0].sessions, 3)

    def test_build_skips_pair_already_linked_by_any_edge(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        self._observe_pair(first, second, ("a", "b"))
        self.store.add_edge(first, second, "related", weight=0.4, evidence_type="test", evidence_key="k1")

        self.assertEqual(sleep_proposals.build(self.store), [])

    def test_build_skips_pair_with_existing_proposal_and_keeps_rejection(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        self._observe_pair(first, second, ("a", "b"))
        result = sleep_proposals.propose(self.store)
        self.assertEqual(result["written"], 1)
        sleep_proposals.reject(self.store, result["proposal_ids"][0], by="jacoby", reason="not related")

        self.assertEqual(sleep_proposals.build(self.store), [])
        again = sleep_proposals.propose(self.store)
        self.assertEqual(again["written"], 0)
        self.assertEqual(sleep_proposals.proposals(self.store, "rejected")[0].reason, "not related")

    def test_build_skips_vetoed_pair_in_either_direction(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        self._observe_pair(first, second, ("a", "b"))
        self._veto(second, first)

        self.assertEqual(sleep_proposals.build(self.store), [])

    def test_build_skips_pair_over_the_degree_cap(self) -> None:
        self.assertEqual(sleep_proposals.DEFAULT_MAX_DEGREE, 24)
        hub = self._memory("hub memory")
        partner = self._memory("already linked partner")
        candidate = self._memory("candidate memory")
        self.store.add_edge(hub, partner, "related", weight=0.3, evidence_type="test", evidence_key="k1")
        self._observe_pair(hub, candidate, ("a", "b"))

        with patch.dict(os.environ, {"CORTEX_MAX_DEGREE": "1"}):
            self.assertEqual(sleep_proposals.build(self.store), [])

        with patch.dict(os.environ, {"CORTEX_MAX_DEGREE": "24"}):
            self.assertEqual(len(sleep_proposals.build(self.store)), 1)

    def test_build_skips_memory_that_is_not_active_or_cold(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        self._observe_pair(first, second, ("a", "b"))
        self.store.set_state(second, "tombstoned", reason="test")

        self.assertEqual(sleep_proposals.build(self.store), [])

    def test_build_canonicalises_reversed_observations_to_one_stable_id(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        self._observe(second, first, "a")
        self._observe(second, first, "b")

        built = sleep_proposals.build(self.store)

        self.assertEqual(len(built), 1)
        left, right = sorted((first, second))
        self.assertEqual(built[0].pair, (left, right))
        self.assertEqual(built[0].proposal_id, sleep_proposals.proposal_id(second, first))
        self.assertEqual(
            sleep_proposals.proposal_id(first, second),
            sleep_proposals.proposal_id(second, first),
        )

    def test_build_confidence_grows_with_sessions_and_is_capped(self) -> None:
        small = sleep_proposals.confidence_for(2)
        large = sleep_proposals.confidence_for(40)
        self.assertGreater(large, small)
        self.assertLessEqual(large, sleep_proposals.CONFIDENCE_CAP)
        self.assertEqual(sleep_proposals.confidence_for(1), 0.0)

    # -- propose / decisions ----------------------------------------------------------------

    def test_propose_persists_one_row_with_evidence_json(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        self._observe_pair(first, second, ("a", "b"))

        result = sleep_proposals.propose(self.store)

        self.assertEqual(result["built"], 1)
        self.assertEqual(result["written"], 1)
        self.assertEqual(result["skipped"]["candidates"], 1)
        with self.store._lock:
            row = self.store._conn.execute("SELECT * FROM edge_proposals").fetchone()
        self.assertEqual(row["state"], "proposed")
        self.assertEqual(row["sessions"], 2)
        self.assertIsNone(row["decided_at"])
        self.assertIsNone(row["decided_by"])
        evidence = json.loads(row["evidence_json"])
        self.assertEqual(evidence["signal"], "co_retrieval")
        self.assertEqual(evidence["sessions"], 2)

    def test_propose_is_idempotent_for_the_same_pair(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        self._observe_pair(first, second, ("a", "b"))

        first_run = sleep_proposals.propose(self.store)
        second_run = sleep_proposals.propose(self.store)

        self.assertEqual(first_run["written"], 1)
        self.assertEqual(second_run["written"], 0)
        self.assertEqual(second_run["skipped"]["already_proposed"], 1)
        self.assertEqual(second_run["built"], 0)
        with self.store._lock:
            count = int(self.store._conn.execute("SELECT COUNT(*) FROM edge_proposals").fetchone()[0])
        self.assertEqual(count, 1)

    def test_accept_records_named_decision_without_creating_an_edge(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        self._observe_pair(first, second, ("a", "b"))
        proposal_id = sleep_proposals.propose(self.store)["proposal_ids"][0]

        result = sleep_proposals.accept(self.store, proposal_id, by="jacoby")

        self.assertTrue(result["ok"])
        self.assertTrue(result["changed"])
        self.assertEqual(result["decided_by"], "jacoby")
        self.assertFalse(result["edge_created"])
        self.assertIn("caller must create the edge", result["note"])
        self.assertEqual(self._edge_count(), 0)
        decided = sleep_proposals.proposals(self.store, "accepted")[0]
        self.assertEqual(decided.decided_by, "jacoby")
        self.assertTrue(decided.decided_at)
        self.assertEqual(sleep_proposals.proposals(self.store, "proposed"), [])

    def test_accept_unknown_proposal_is_refused(self) -> None:
        result = sleep_proposals.accept(self.store, "ep_missing", by="jacoby")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "unknown proposal")

        first, second = self._memory("one"), self._memory("two")
        self._observe_pair(first, second, ("a", "b"))
        proposal_id = sleep_proposals.propose(self.store)["proposal_ids"][0]
        with self.assertRaises(ValueError):
            sleep_proposals.accept(self.store, proposal_id, by="   ")
        with self.assertRaises(ValueError):
            sleep_proposals.reject(self.store, proposal_id, by="")

    def test_reject_records_reason_and_keeps_the_row(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        self._observe_pair(first, second, ("a", "b"))
        proposal_id = sleep_proposals.propose(self.store)["proposal_ids"][0]

        result = sleep_proposals.reject(
            self.store, proposal_id, by="jacoby", reason="same backup, different words"
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(result["reason"], "same backup, different words")
        rejected = sleep_proposals.proposals(self.store, "rejected")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].decided_by, "jacoby")
        self.assertEqual(self._edge_count(), 0)

    def test_decisions_are_final_and_idempotent(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        self._observe_pair(first, second, ("a", "b"))
        proposal_id = sleep_proposals.propose(self.store)["proposal_ids"][0]
        sleep_proposals.reject(self.store, proposal_id, by="jacoby")

        denied = sleep_proposals.accept(self.store, proposal_id, by="someone-else")
        repeated = sleep_proposals.reject(self.store, proposal_id, by="jacoby")

        self.assertFalse(denied["ok"])
        self.assertIn("already rejected", denied["error"])
        self.assertTrue(repeated["ok"])
        self.assertFalse(repeated["changed"])
        self.assertEqual(len(sleep_proposals.proposals(self.store, "any")), 1)

    # -- listing and reporting --------------------------------------------------------------

    def test_proposals_filters_by_state_and_rejects_unknown_state(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        self._observe_pair(first, second, ("a", "b"))
        proposal_id = sleep_proposals.propose(self.store)["proposal_ids"][0]
        sleep_proposals.accept(self.store, proposal_id, by="jacoby")

        self.assertEqual(sleep_proposals.proposals(self.store, "proposed"), [])
        self.assertEqual(len(sleep_proposals.proposals(self.store, "accepted")), 1)
        self.assertEqual(len(sleep_proposals.proposals(self.store, "any")), 1)
        with self.assertRaises(ValueError):
            sleep_proposals.proposals(self.store, "maybe")

    def test_report_counts_states_and_states_the_signal_limits(self) -> None:
        first, second = self._memory("one"), self._memory("two")
        third, fourth = self._memory("three"), self._memory("four")
        self._observe_pair(first, second, ("a", "b"))
        self._observe_pair(third, fourth, ("c", "d"))
        ids = sleep_proposals.propose(self.store)["proposal_ids"]
        sleep_proposals.accept(self.store, ids[0], by="jacoby")
        sleep_proposals.reject(self.store, ids[1], by="jacoby", reason="no")

        report = sleep_proposals.report(self.store)

        self.assertEqual(report["proposals"], 2)
        self.assertEqual(report["by_state"]["accepted"], 1)
        self.assertEqual(report["by_state"]["rejected"], 1)
        self.assertEqual(report["by_relation"]["co_retrieved"], 2)
        self.assertEqual(report["pending"], 0)
        self.assertEqual(report["thresholds"]["edge_evidence_bar"], 2)
        self.assertEqual(report["thresholds"]["max_degree"], 24)
        self.assertFalse(report["thresholds"]["min_sessions"] == 0)
        self.assertEqual(report["observations"]["source"], "table")
        self.assertEqual(report["observations"]["pairs"], 2)
        self.assertFalse(report["observations"]["empty"])
        self.assertTrue(any("cannot tell why" in line for line in report["limits"]))


if __name__ == "__main__":
    unittest.main()
