"""The sentence a fact came from is kept, and drift is measurable (#42)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import quotes
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
SAID = "yeah the dashboard is on 8100, nginx sits in front of it"


class QuoteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str = "The dashboard runs on port 8100.") -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]


class RecordTests(QuoteTestCase):
    def test_a_quote_is_kept(self) -> None:
        memory_id = self.memory()
        self.assertTrue(quotes.record(self.store, memory_id, SAID, at=NOW))
        self.assertEqual(len(quotes.quotes_for(self.store, memory_id)), 1)

    def test_recording_the_same_quote_twice_is_idempotent(self) -> None:
        memory_id = self.memory()
        quotes.record(self.store, memory_id, SAID, at=NOW)
        self.assertFalse(quotes.record(self.store, memory_id, SAID, at=NOW))

    def test_whitespace_is_normalised(self) -> None:
        memory_id = self.memory()
        quotes.record(self.store, memory_id, "  the   dashboard\n is on 8100  ", at=NOW)
        self.assertEqual(quotes.quotes_for(self.store, memory_id)[0].quote,
                         "the dashboard is on 8100")

    def test_an_empty_quote_is_refused(self) -> None:
        memory_id = self.memory()
        self.assertFalse(quotes.record(self.store, memory_id, "   ", at=NOW))

    def test_an_empty_subject_is_refused(self) -> None:
        self.assertFalse(quotes.record(self.store, "", SAID, at=NOW))

    def test_a_long_quote_is_capped(self) -> None:
        memory_id = self.memory()
        quotes.record(self.store, memory_id, "word " * 2000, at=NOW)
        self.assertLessEqual(len(quotes.quotes_for(self.store, memory_id)[0].quote),
                             quotes.MAX_QUOTE_CHARS)

    def test_the_source_reference_is_kept(self) -> None:
        memory_id = self.memory()
        quotes.record(self.store, memory_id, SAID, source_ref="session 41, turn 7", at=NOW)
        self.assertIn("session 41", quotes.quotes_for(self.store, memory_id)[0].describe())


class AuditTests(QuoteTestCase):
    def test_find_points_back_at_the_source_sentence(self) -> None:
        memory_id = self.memory()
        quotes.record(self.store, memory_id, SAID, at=NOW)
        found = quotes.find(self.store, "nginx sits")
        self.assertEqual([item.subject_id for item in found], [memory_id])

    def test_find_with_no_match_returns_nothing(self) -> None:
        self.assertEqual(quotes.find(self.store, "nothing like this"), [])

    def test_an_empty_search_returns_nothing(self) -> None:
        self.assertEqual(quotes.find(self.store, ""), [])

    def test_a_quote_is_not_a_memory(self) -> None:
        """Quotes exist for audit only: they must not become recallable records."""
        memory_id = self.memory()
        quotes.record(self.store, memory_id, SAID, at=NOW)
        count = self.store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        self.assertEqual(count, 1)


class AdoptionTests(QuoteTestCase):
    def test_a_promotion_carries_the_quote_to_the_memory(self) -> None:
        memory_id = self.memory()
        quotes.record(self.store, "proposal-1", SAID, kind="proposal", at=NOW)
        self.assertEqual(quotes.adopt(self.store, "proposal-1", memory_id), 1)
        self.assertEqual(quotes.quotes_for(self.store, memory_id)[0].quote, SAID)

    def test_adoption_is_idempotent(self) -> None:
        memory_id = self.memory()
        quotes.record(self.store, "proposal-1", SAID, kind="proposal", at=NOW)
        quotes.adopt(self.store, "proposal-1", memory_id)
        self.assertEqual(quotes.adopt(self.store, "proposal-1", memory_id), 0)

    def test_adoption_with_nothing_to_move_is_harmless(self) -> None:
        self.assertEqual(quotes.adopt(self.store, "proposal-x", "memory-x"), 0)


class VerifyTests(QuoteTestCase):
    def test_a_faithful_memory_is_consistent_with_its_quote(self) -> None:
        memory_id = self.memory("The dashboard runs on port 8100.")
        quotes.record(self.store, memory_id, SAID, at=NOW)
        result = quotes.verify(self.store, memory_id)
        self.assertTrue(result.quoted)
        self.assertFalse(result.drifted)
        self.assertIn("consistent", result.describe())

    def test_a_paraphrase_far_from_the_quote_is_flagged(self) -> None:
        memory_id = self.memory("A completely different fact about sourdough bread baking.")
        quotes.record(self.store, memory_id, SAID, at=NOW)
        result = quotes.verify(self.store, memory_id)
        self.assertTrue(result.drifted)
        self.assertIn("drifted", result.describe())

    def test_no_quote_is_reported_as_such(self) -> None:
        memory_id = self.memory()
        result = quotes.verify(self.store, memory_id)
        self.assertFalse(result.quoted)
        self.assertEqual(result.describe(), "no quote recorded")

    def test_the_audit_shortlist_finds_drifted_rows(self) -> None:
        drifted_id = self.memory("Unrelated content about bread.")
        quotes.record(self.store, drifted_id, SAID, at=NOW)
        faithful_id = self.memory("The dashboard runs on port 8100 today.")
        quotes.record(self.store, faithful_id, SAID, at=NOW)
        found = quotes.drifted(self.store)
        self.assertEqual([item["memory_id"] for item in found], [drifted_id])


class ReportTests(QuoteTestCase):
    def test_the_report_counts_quotes_and_coverage(self) -> None:
        first = self.memory("The dashboard runs on port 8100.")
        second = self.memory("The vault lives at home.")
        quotes.record(self.store, first, SAID, at=NOW)
        quotes.record(self.store, "proposal-1", "staged fact", kind="proposal", at=NOW)
        described = quotes.report(self.store)
        self.assertEqual(described["quotes"], 2)
        self.assertEqual(described["memories_with_quotes"], 1)
        self.assertEqual(described["memories"], 2)
        self.assertEqual(described["coverage"], 0.5)
        self.assertIn(second, [first, second])

    def test_an_empty_table_reports_zero_coverage(self) -> None:
        described = quotes.report(self.store)
        self.assertEqual((described["quotes"], described["coverage"]), (0, 0.0))


class CandidateSourceTests(unittest.TestCase):
    """The quote has to be the sentence as written, not the normalised one."""

    def test_a_candidate_keeps_its_unanchored_source_text(self) -> None:
        from cortex.extraction import extract_candidates

        candidate = extract_candidates(
            "The dashboard requires port 8100 and the deploy broke yesterday."
        )[0]
        self.assertIn("yesterday", candidate.source_text)
        self.assertNotIn("yesterday (", candidate.source_text)

    def test_a_candidate_without_relative_time_is_identical(self) -> None:
        from cortex.extraction import extract_candidates

        candidate = extract_candidates("The dashboard requires port 8100.")[0]
        self.assertEqual(candidate.source_text, candidate.content)


if __name__ == "__main__":
    unittest.main()
