"""Where in a session something was used changes how useful it is (#82)."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import serial_position as sp
from cortex.store import CortexStore

START = datetime(2026, 11, 22, 9, 0, tzinfo=timezone.utc)


class FlagTests(unittest.TestCase):
    def test_off_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(sp.ENABLED_ENV, None)
            self.assertFalse(sp.enabled())

    def test_switchable(self) -> None:
        with mock.patch.dict(os.environ, {sp.ENABLED_ENV: "1"}):
            self.assertTrue(sp.enabled())


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.session = "session-a"
        self.memory = self.add("The dashboard runs on port 8100.")
        self.other = self.add("The vault lives under ~/.hermes/obsidian-vault.")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def add(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def add_run(self, *, minutes: int, recall_id: str | None = None) -> str:
        recall = recall_id or f"run-{minutes}"
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO recall_runs
                   (recall_id, session_id, query, mode, reason, requested_limit, token_budget,
                    created_at)
                   VALUES (?, ?, ?, 'adaptive', '', 6, 700, ?)""",
                (recall, self.session, "a query",
                 (START + timedelta(minutes=minutes)).isoformat()),
            )
        return recall

    def event(self, memory_id: str, kind: str, *, minutes: int) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO access_log (memory_id, event, query, session_id, score, created_at)
                   VALUES (?, ?, 'a query', ?, 0.5, ?)""",
                (memory_id, kind, self.session, (START + timedelta(minutes=minutes)).isoformat()),
            )

    def test_turn_index_counts_prior_runs(self) -> None:
        self.add_run(minutes=0)
        self.add_run(minutes=1)
        self.assertEqual(sp.turn_index(self.store, self.session), 2)

    def test_turn_index_is_zero_for_an_unknown_session(self) -> None:
        self.assertEqual(sp.turn_index(self.store, "no such session"), 0)

    def test_recency_reports_distance_in_recalls(self) -> None:
        self.add_run(minutes=0, recall_id="old")
        self.add_run(minutes=10, recall_id="recent")
        self.event(self.memory, "used", minutes=11)
        recency = sp.recency_in_session(self.store, self.session)
        self.assertEqual(recency[self.memory], 0)

    def test_recency_uses_the_most_recent_appearance(self) -> None:
        self.add_run(minutes=0, recall_id="old")
        self.event(self.memory, "selected", minutes=1)
        self.add_run(minutes=10, recall_id="recent")
        self.event(self.memory, "selected", minutes=11)
        self.assertEqual(sp.recency_in_session(self.store, self.session)[self.memory], 0)

    def test_early_in_a_session_nothing_is_weighed(self) -> None:
        weighed = sp.weight_for(self.store, self.memory, session_id=self.session, turns=2)
        self.assertEqual(weighed.multiplier, 1.0)
        self.assertIn("no session position", weighed.reason)

    def test_late_in_a_session_the_working_set_is_boosted(self) -> None:
        with mock.patch.dict(os.environ, {sp.ENABLED_ENV: "1"}):
            for index in range(10):
                self.add_run(minutes=index, recall_id=f"run-{index}")
            self.event(self.memory, "used", minutes=10)
            weighed = sp.weight_for(self.store, self.memory, session_id=self.session, turns=10)
        self.assertEqual(weighed.multiplier, sp.LATE_BOOST)

    def test_late_in_a_session_a_stale_memory_is_penalised(self) -> None:
        with mock.patch.dict(os.environ, {sp.ENABLED_ENV: "1"}):
            for index in range(20):
                self.add_run(minutes=index, recall_id=f"run-{index}")
            self.event(self.memory, "used", minutes=1)
            weighed = sp.weight_for(self.store, self.memory, session_id=self.session, turns=20)
        self.assertEqual(weighed.multiplier, sp.STALE_PENALTY)

    def test_a_memory_not_seen_this_session_is_untouched(self) -> None:
        with mock.patch.dict(os.environ, {sp.ENABLED_ENV: "1"}):
            for index in range(10):
                self.add_run(minutes=index, recall_id=f"run-{index}")
            weighed = sp.weight_for(self.store, self.other, session_id=self.session, turns=10)
        self.assertEqual(weighed.multiplier, 1.0)

    def test_the_flag_off_means_no_weight_ever(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(sp.ENABLED_ENV, None)
            for index in range(10):
                self.add_run(minutes=index, recall_id=f"run-{index}")
            self.event(self.memory, "used", minutes=10)
            weighed = sp.weight_for(self.store, self.memory, session_id=self.session, turns=10)
        self.assertEqual(weighed.multiplier, 1.0)

    def test_apply_reports_counters_and_multiplies(self) -> None:
        from types import SimpleNamespace

        with mock.patch.dict(os.environ, {sp.ENABLED_ENV: "1"}):
            for index in range(10):
                self.add_run(minutes=index, recall_id=f"run-{index}")
            self.event(self.memory, "used", minutes=10)
            results = [SimpleNamespace(memory={"id": self.memory}, score=1.0)]
            counters = sp.apply(self.store, results, session_id=self.session, turns=10)
        self.assertEqual(counters["serial_position_boosted_count"], 1)
        self.assertAlmostEqual(results[0].score, sp.LATE_BOOST, places=6)

    def test_apply_is_inert_with_the_flag_off(self) -> None:
        from types import SimpleNamespace

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(sp.ENABLED_ENV, None)
            results = [SimpleNamespace(memory={"id": self.memory}, score=1.0)]
            counters = sp.apply(self.store, results, session_id=self.session, turns=99)
        self.assertEqual(counters, {})
        self.assertEqual(results[0].score, 1.0)

    def test_the_measurement_reproduces_the_quartile_table(self) -> None:
        """Late beats early on constructed data, so the constants have a runnable justification."""
        for index in range(12):
            self.add_run(minutes=index, recall_id=f"run-{index}")
            self.event(self.memory, "selected", minutes=index)
            # every other early turn is used; every late turn is used
            if index >= 9 or index % 2 == 0:
                self.event(self.memory, "used", minutes=index)
        measured = sp.measure(self.store, buckets=4)
        rates = [bucket["use_rate"] for bucket in measured["buckets"]]
        self.assertEqual(len(rates), 4)
        self.assertGreater(rates[-1], rates[0])

    def test_the_measurement_is_quiet_on_an_empty_store(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            measured = sp.measure(other)
            self.assertEqual(measured["sessions"], 0)
            self.assertEqual(measured["buckets"], [])
        finally:
            other.close()

    def test_the_report_carries_the_measurement_it_reads(self) -> None:
        described = sp.report(self.store)
        self.assertEqual(described["measured_quartile_use_rates"][-1], 0.281)
        self.assertFalse(described["enabled"])

    def test_the_description_names_the_multiplier(self) -> None:
        self.assertIn("x1.08", sp.Weight("abcdef12", 1.08, "working set").describe())


if __name__ == "__main__":
    unittest.main()
