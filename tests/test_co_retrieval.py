"""Co-retrieval edges appear only after three independent sessions agree (#67)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import co_retrieval
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class CoRetrievalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.a, self.b, self.c = (
            self.store.add_memory(
                content=content,
                kind="semantic",
                source_type="user_turn",
                source_category="USER_EXPLICIT",
            )[0]
            for content in (
                "The dashboard runs on port 8100.",
                "Nginx proxies the dashboard to 127.0.0.1:8100.",
                "The dashboard has a Cloudflare tunnel.",
            )
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def edges(self) -> int:
        return self.store._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE relation = ?", (co_retrieval.RELATION,)
        ).fetchone()[0]


class ObservationTests(CoRetrievalTestCase):
    def test_observing_records_every_pair(self) -> None:
        written = co_retrieval.observe(self.store, [self.a, self.b, self.c],
                                       session_id="s1", at=NOW)
        self.assertEqual(written, 3)
        self.assertEqual(len(co_retrieval.observations(self.store)), 3)

    def test_a_single_memory_is_not_a_pair(self) -> None:
        self.assertEqual(
            co_retrieval.observe(self.store, [self.a], session_id="s1", at=NOW), 0
        )

    def test_the_same_session_counts_once(self) -> None:
        co_retrieval.observe(self.store, [self.a, self.b], session_id="s1", at=NOW)
        co_retrieval.observe(self.store, [self.a, self.b], session_id="s1", at=NOW)
        rows = [row for row in co_retrieval.observations(self.store)]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sessions"], 1)

    def test_duplicate_ids_inside_one_turn_are_deduplicated(self) -> None:
        co_retrieval.observe(self.store, [self.a, self.b, self.a], session_id="s1", at=NOW)
        self.assertEqual(len(co_retrieval.observations(self.store)), 1)

    def test_observing_does_not_create_edges(self) -> None:
        co_retrieval.observe(self.store, [self.a, self.b, self.c], session_id="s1", at=NOW)
        self.assertEqual(self.edges(), 0)


class GateTests(CoRetrievalTestCase):
    def test_no_edge_before_three_sessions(self) -> None:
        for session in ("s1", "s2"):
            co_retrieval.observe(self.store, [self.a, self.b], session_id=session, at=NOW)
        result = co_retrieval.materialize(self.store, at=NOW)
        self.assertEqual(result["created"], 0)
        self.assertEqual(self.edges(), 0)

    def test_an_edge_appears_at_the_third_session(self) -> None:
        for session in ("s1", "s2", "s3"):
            co_retrieval.observe(self.store, [self.a, self.b], session_id=session, at=NOW)
        result = co_retrieval.materialize(self.store, at=NOW)
        self.assertEqual(result["created"], 1)
        self.assertEqual(self.edges(), 1)

    def test_three_retrievals_in_one_session_do_not_count(self) -> None:
        """The gate is independent sessions, not independent queries."""
        for turn in ("t1", "t2", "t3"):
            co_retrieval.observe(self.store, [self.a, self.b], session_id="s1",
                                 turn_id=turn, at=NOW)
        co_retrieval.materialize(self.store, at=NOW)
        self.assertEqual(self.edges(), 0)

    def test_the_plan_exposes_the_gate(self) -> None:
        for session in ("s1", "s2", "s3"):
            co_retrieval.observe(self.store, [self.a, self.b], session_id=session, at=NOW)
        gates = co_retrieval.plan(self.store, at=NOW)
        self.assertEqual(len(gates), 1)
        self.assertTrue(gates[0].eligible)
        self.assertEqual(gates[0].sessions, 3)


class WeightTests(unittest.TestCase):
    def test_below_the_gate_the_weight_is_zero(self) -> None:
        self.assertEqual(co_retrieval.weight_for(2), 0.0)

    def test_weight_grows_logarithmically(self) -> None:
        third = co_retrieval.weight_for(3)
        tenth = co_retrieval.weight_for(10)
        hundredth = co_retrieval.weight_for(100)
        self.assertGreater(tenth, third)
        self.assertGreater(hundredth, tenth)
        self.assertLess(hundredth - tenth, tenth - third)

    def test_the_cap_holds(self) -> None:
        self.assertLessEqual(co_retrieval.weight_for(10_000), co_retrieval.WEIGHT_CAP)

    def test_age_decays_the_weight(self) -> None:
        fresh = co_retrieval.weight_for(5, age_days=0)
        old = co_retrieval.weight_for(5, age_days=co_retrieval.HALF_LIFE_DAYS)
        self.assertAlmostEqual(old, fresh / 2, places=3)

    def test_decay_reaches_zero_so_a_fossil_pair_stops_qualifying(self) -> None:
        """A century-old association should not keep an edge alive; it decays out of the gate."""
        self.assertEqual(co_retrieval.weight_for(5, age_days=10_000), 0.0)
        gate = co_retrieval.Gate(pair=("a", "b"), sessions=5, weight=0.0)
        self.assertFalse(gate.eligible)


class MaterializeTests(CoRetrievalTestCase):
    def test_materialize_is_idempotent(self) -> None:
        for session in ("s1", "s2", "s3"):
            co_retrieval.observe(self.store, [self.a, self.b], session_id=session, at=NOW)
        co_retrieval.materialize(self.store, at=NOW)
        again = co_retrieval.materialize(self.store, at=NOW)
        self.assertEqual(again["created"], 0)
        self.assertEqual(again["reinforced"], 1)
        self.assertEqual(self.edges(), 1)

    def test_the_edge_is_attributed_and_explained(self) -> None:
        for session in ("s1", "s2", "s3"):
            co_retrieval.observe(self.store, [self.a, self.b], session_id=session, at=NOW)
        co_retrieval.materialize(self.store, at=NOW)
        row = self.store._conn.execute(
            "SELECT weight, relation FROM edges WHERE src_id = ? AND dst_id = ?",
            (min(self.a, self.b), max(self.a, self.b)),
        ).fetchone()
        self.assertEqual(row["relation"], co_retrieval.RELATION)
        self.assertGreater(row["weight"], 0)
        self.assertLessEqual(row["weight"], co_retrieval.WEIGHT_CAP)

    def test_an_old_observation_does_not_create_a_strong_edge(self) -> None:
        long_ago = NOW - timedelta(days=int(co_retrieval.HALF_LIFE_DAYS * 6))
        for session in ("s1", "s2", "s3"):
            co_retrieval.observe(self.store, [self.a, self.b], session_id=session, at=long_ago)
        co_retrieval.materialize(self.store, at=NOW)
        row = self.store._conn.execute(
            "SELECT weight FROM edges WHERE relation = ?", (co_retrieval.RELATION,)
        ).fetchone()
        if row is not None:
            self.assertLess(row["weight"], co_retrieval.weight_for(3))

    def test_the_report_counts_observed_and_eligible(self) -> None:
        for session in ("s1", "s2", "s3"):
            co_retrieval.observe(self.store, [self.a, self.b], session_id=session, at=NOW)
        co_retrieval.observe(self.store, [self.b, self.c], session_id="s1", at=NOW)
        described = co_retrieval.report(self.store, at=NOW)
        self.assertEqual(described["pairs_observed"], 2)
        self.assertEqual(described["eligible"], 1)

    def test_stale_observations_can_be_pruned(self) -> None:
        co_retrieval.observe(self.store, [self.a, self.b], session_id="s1",
                             at=NOW - timedelta(days=400))
        self.assertEqual(co_retrieval.prune_stale(self.store, days=365, at=NOW), 1)
        self.assertEqual(co_retrieval.observations(self.store), [])


if __name__ == "__main__":
    unittest.main()
