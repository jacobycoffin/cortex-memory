"""Recall is an immutable event log, and the query stays out of the memory (#66)."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import recall_events
from cortex.retrieval import MemoryRetriever
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
QUERY = "what port does the cortex dashboard run on"


class RecallEventTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.ids = [
            self.store.add_memory(
                content=content,
                kind="semantic",
                source_type="user_turn",
                source_category="USER_EXPLICIT",
            )[0]
            for content in (
                "The cortex dashboard runs on port 8100 behind nginx.",
                "The dashboard is published at cortex.example.com.",
                "Nginx proxies the dashboard to 127.0.0.1:8100.",
            )
        ]

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def results(self, query: str = QUERY) -> list:
        found, _diagnostics = MemoryRetriever(self.store).search_detailed(query, limit=3)
        return found


class RecordingTests(RecallEventTestCase):
    def test_one_row_per_recalled_memory(self) -> None:
        results = self.results()
        written = recall_events.record(self.store, "turn-1", results, at=NOW)
        self.assertEqual(len(written), len(results))
        self.assertEqual(len(recall_events.events(self.store, "turn-1")), len(results))

    def test_the_row_carries_rank_score_components_and_retriever(self) -> None:
        results = self.results()
        recall_events.record(self.store, "turn-1", results, retriever="hybrid", at=NOW)
        detail = recall_events.detail(self.store, "turn-1")
        first = detail["results"][0]
        self.assertEqual(first["rank"], 1)
        self.assertIsNotNone(first["score"])
        self.assertEqual(first["retriever"], "hybrid")
        self.assertFalse(first["used_in"])

    def test_components_are_stored_as_numbers(self) -> None:
        results = self.results()
        recall_events.record(self.store, "turn-1", results, at=NOW)
        row = self.store._conn.execute(
            "SELECT components_json FROM recall_events WHERE turn_id = 'turn-1' LIMIT 1"
        ).fetchone()
        self.assertIn("{", row["components_json"])

    def test_recording_a_turn_with_no_results_is_harmless(self) -> None:
        self.assertEqual(recall_events.record(self.store, "turn-empty", [], at=NOW), [])


class CitationTests(RecallEventTestCase):
    def test_a_citation_is_its_own_event(self) -> None:
        results = self.results()
        recall_events.record(self.store, "turn-1", results, at=NOW)
        recall_events.cite(self.store, "turn-1", [self.ids[0]], at=NOW)
        detail = recall_events.detail(self.store, "turn-1")
        self.assertEqual(detail["cited"], 1)
        used = {row["memory_id"]: row["used_in"] for row in detail["results"]}
        self.assertTrue(used[self.ids[0]])

    def test_used_in_distinguishes_injected_from_cited(self) -> None:
        results = self.results()
        recall_events.record(self.store, "turn-1", results, at=NOW)
        self.assertEqual(recall_events.used_in(self.store, "turn-1"), set())
        recall_events.cite(self.store, "turn-1", [self.ids[1]], at=NOW)
        self.assertEqual(recall_events.used_in(self.store, "turn-1"), {self.ids[1]})

    def test_the_ranking_is_preserved_alongside_the_citation(self) -> None:
        """Append-only means both facts survive: it was ranked 2nd AND it was cited."""
        results = self.results()
        recall_events.record(self.store, "turn-1", results, at=NOW)
        recall_events.cite(self.store, "turn-1", [self.ids[1]], at=NOW)
        detail = recall_events.detail(self.store, "turn-1")
        self.assertEqual(detail["ranked"], len(results))
        self.assertIn(self.ids[1], recall_events.used_in(self.store, "turn-1"))


class ImmutabilityTests(RecallEventTestCase):
    def test_an_event_cannot_be_updated(self) -> None:
        recall_events.record(self.store, "turn-1", self.results(), at=NOW)
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store.transaction() as conn:
                conn.execute("UPDATE recall_events SET rank = 99 WHERE turn_id = 'turn-1'")

    def test_an_event_cannot_be_deleted(self) -> None:
        recall_events.record(self.store, "turn-1", self.results(), at=NOW)
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store.transaction() as conn:
                conn.execute("DELETE FROM recall_events WHERE turn_id = 'turn-1'")

    def test_the_events_survive_after_the_attempt(self) -> None:
        written = recall_events.record(self.store, "turn-1", self.results(), at=NOW)
        try:
            with self.store.transaction() as conn:
                conn.execute("DELETE FROM recall_events")
        except sqlite3.IntegrityError:
            pass
        self.assertEqual(len(recall_events.events(self.store, "turn-1")), len(written))


class QueryIsNotContentTests(RecallEventTestCase):
    def test_recording_a_recall_does_not_change_any_memory_text(self) -> None:
        before = {memory_id: self.store.get_memory(memory_id)["content"] for memory_id in self.ids}
        results = self.results()
        recall_events.record(self.store, "turn-1", results, at=NOW)
        recall_events.cite(self.store, "turn-1", [self.ids[0]], at=NOW)
        after = {memory_id: self.store.get_memory(memory_id)["content"] for memory_id in self.ids}
        self.assertEqual(before, after)

    def test_the_query_text_is_stored_nowhere_as_a_memory(self) -> None:
        results = self.results()
        recall_events.record(self.store, "turn-1", results, at=NOW)
        self.assertEqual(recall_events.query_leaked_into_memory(self.store, QUERY), [])

    def test_no_recall_event_column_holds_the_query(self) -> None:
        recall_events.record(self.store, "turn-1", self.results(), at=NOW)
        row = dict(self.store._conn.execute(
            "SELECT * FROM recall_events WHERE turn_id = 'turn-1' LIMIT 1"
        ).fetchone())
        self.assertNotIn(QUERY, " ".join(str(value) for value in row.values()))


class SummaryTests(RecallEventTestCase):
    def test_the_summary_counts_turns_rankings_and_citations(self) -> None:
        recall_events.record(self.store, "turn-1", self.results(), at=NOW)
        recall_events.record(self.store, "turn-2", self.results(), at=NOW)
        recall_events.cite(self.store, "turn-2", [self.ids[0]], at=NOW)
        described = recall_events.summary(self.store)
        self.assertEqual(described["turns"], 2)
        self.assertGreaterEqual(described["ranked"], 2)
        self.assertEqual(described["citations"], 1)

    def test_an_empty_log_reports_zero(self) -> None:
        described = recall_events.summary(self.store)
        self.assertEqual((described["events"], described["turns"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
