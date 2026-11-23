"""One chatty session must never manufacture an edge on its own (#204)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import co_retrieval, session_dedupe as sd
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.left = self.memory("The dashboard runs on port 8100.")
        self.right = self.memory("Nginx proxies the dashboard.")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def observe(self, session: str) -> None:
        co_retrieval.observe(
            self.store, [self.left, self.right], session_id=session, at=NOW
        )

    def test_no_observations_means_no_sessions(self) -> None:
        self.assertEqual(sd.sessions(self.store, self.left, self.right), set())

    def test_turns_in_one_session_count_once(self) -> None:
        for _ in range(5):
            self.observe("s1")
        self.assertEqual(len(sd.sessions(self.store, self.left, self.right)), 1)

    def test_distinct_sessions_are_counted(self) -> None:
        self.observe("s1")
        self.observe("s2")
        self.assertEqual(sd.sessions(self.store, self.left, self.right), {"s1", "s2"})

    def test_one_session_is_refused(self) -> None:
        for _ in range(9):
            self.observe("s1")
        verdict = sd.admit(self.store, self.left, self.right)
        self.assertFalse(verdict.allowed)
        self.assertIn("one session cannot make a link", verdict.reason)

    def test_two_sessions_are_admitted(self) -> None:
        self.observe("s1")
        self.observe("s2")
        verdict = sd.admit(self.store, self.left, self.right)
        self.assertTrue(verdict.allowed)
        self.assertEqual(verdict.sessions, 2)

    def test_the_order_of_the_pair_does_not_matter(self) -> None:
        self.observe("s1")
        self.assertEqual(
            sd.sessions(self.store, self.right, self.left), sd.sessions(self.store, self.left,
                                                                        self.right)
        )

    def test_a_stricter_bar_can_be_asked_for(self) -> None:
        self.observe("s1")
        self.observe("s2")
        self.assertFalse(sd.admit(self.store, self.left, self.right, minimum=3).allowed)
        self.observe("s3")
        self.assertTrue(sd.admit(self.store, self.left, self.right, minimum=3).allowed)

    def test_the_audit_finds_a_single_session_pair(self) -> None:
        self.observe("s1")
        described = sd.audit(self.store)
        self.assertEqual(described["single_session"], 1)
        self.assertEqual(described["busiest_single"][0]["turns"], 1)

    def test_the_audit_separates_multi_session_pairs(self) -> None:
        self.observe("s1")
        self.observe("s2")
        described = sd.audit(self.store)
        self.assertEqual(described["single_session"], 0)
        self.assertEqual(described["multi_session"], 1)

    def test_the_weight_cap_is_zero_below_the_bar(self) -> None:
        self.assertEqual(sd.weight_cap_for(1, cap=0.45), 0.0)

    def test_the_weight_cap_is_full_at_the_bar(self) -> None:
        self.assertEqual(sd.weight_cap_for(2, cap=0.45), 0.45)
        self.assertEqual(sd.weight_cap_for(50, cap=0.45), 0.45)

    def test_co_retrieval_still_requires_its_own_three_sessions(self) -> None:
        """The stricter bar in the caller is untouched by the general rule."""
        self.observe("s1")
        self.observe("s2")
        self.assertEqual(co_retrieval.weight_for(2), 0.0)
        self.assertGreater(co_retrieval.weight_for(3), 0.0)

    def test_materialise_creates_nothing_from_one_session(self) -> None:
        for _ in range(6):
            self.observe("s1")
        result = co_retrieval.materialize(self.store, at=NOW)
        self.assertEqual(result["created"], 0)
        self.assertEqual(result["skipped"], 1)

    def test_the_report_carries_the_bar_and_the_audit(self) -> None:
        self.observe("s1")
        described = sd.report(self.store)
        self.assertEqual(described["min_sessions"], sd.MIN_SESSIONS)
        self.assertTrue(described["observations_table"])
        self.assertEqual(described["single_session"], 1)

    def test_a_store_without_observations_reports_cleanly(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            self.assertEqual(sd.sessions(other, "a", "b"), set())
            self.assertEqual(sd.audit(other)["pairs"], 0)
            self.assertFalse(sd.admit(other, "a", "b").allowed)
        finally:
            other.close()


if __name__ == "__main__":
    unittest.main()
