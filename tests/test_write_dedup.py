"""Decide insert / merge / reject before the row exists (#28)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import write_dedup
from cortex.store import CortexStore


class SimilarityTests(unittest.TestCase):
    def test_identical_text_is_one(self) -> None:
        self.assertEqual(write_dedup.jaccard("port 8100 dashboard", "port 8100 dashboard"), 1.0)

    def test_unrelated_text_is_zero(self) -> None:
        self.assertEqual(write_dedup.jaccard("port 8100 dashboard", "sourdough starter"), 0.0)

    def test_containment_measures_the_shorter_side(self) -> None:
        self.assertEqual(
            write_dedup.containment("port 8100", "the dashboard runs on port 8100 behind nginx"),
            1.0,
        )

    def test_containment_of_a_disjoint_text_is_zero(self) -> None:
        self.assertEqual(write_dedup.containment("sourdough", "port 8100 dashboard"), 0.0)

    def test_trailing_punctuation_does_not_hide_a_duplicate(self) -> None:
        """The first version kept the full stop and turned a duplicate into an insert."""
        self.assertEqual(write_dedup.tokens("behind nginx."), write_dedup.tokens("behind nginx"))

    def test_stopwords_do_not_create_similarity(self) -> None:
        self.assertEqual(write_dedup.jaccard("the and of", "the and of"), 0.0)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.existing = self.memory("The dashboard runs on port 8100 behind nginx.")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, category: str = "USER_EXPLICIT") -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category=category,
        )[0]

    def count(self) -> int:
        return int(self.store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def test_a_new_fact_is_an_insert(self) -> None:
        verdict = write_dedup.decide(self.store, "The sourdough starter is fed at eight.")
        self.assertEqual(verdict.action, "insert")
        self.assertTrue(verdict.writes)

    def test_the_same_fact_verbatim_is_rejected(self) -> None:
        verdict = write_dedup.decide(self.store, "The dashboard runs on port 8100 behind nginx.")
        self.assertEqual(verdict.action, "reject")
        self.assertEqual(verdict.target_id, self.existing)
        self.assertIn("verbatim", verdict.reason)

    def test_a_restatement_merges(self) -> None:
        verdict = write_dedup.decide(
            self.store, "The dashboard runs on port 8100 behind nginx, via the proxy."
        )
        self.assertEqual(verdict.action, "merge")
        self.assertEqual(verdict.target_id, self.existing)

    def test_a_neighbour_with_different_words_is_not_a_duplicate(self) -> None:
        self.memory("Nginx proxies the dashboard.")
        verdict = write_dedup.decide(self.store, "The sourdough starter is fed at eight.")
        self.assertEqual(verdict.action, "insert")

    def test_the_verdict_carries_its_neighbours(self) -> None:
        verdict = write_dedup.decide(self.store, "The dashboard port is 8100.")
        self.assertTrue(verdict.neighbours)
        self.assertGreaterEqual(verdict.neighbours[0]["similarity"], 0.0)

    def test_an_empty_write_is_rejected(self) -> None:
        verdict = write_dedup.decide(self.store, "   ")
        self.assertEqual(verdict.action, "reject")
        self.assertIn("nothing to write", verdict.reason)

    def test_a_reject_writes_nothing(self) -> None:
        before = self.count()
        verdict = write_dedup.decide(self.store, "The dashboard runs on port 8100 behind nginx.")
        result = write_dedup.apply(self.store, "The dashboard runs on port 8100 behind nginx.",
                                   verdict)
        self.assertFalse(result["written"])
        self.assertEqual(self.count(), before)

    def test_a_merge_updates_the_existing_row(self) -> None:
        before = self.count()
        text = "The dashboard runs on port 8100 behind nginx, via the proxy."
        verdict = write_dedup.decide(self.store, text)
        result = write_dedup.apply(self.store, text, verdict)
        self.assertTrue(result["written"])
        self.assertEqual(result["action"], "merge")
        self.assertEqual(self.count(), before)
        stored = self.store._conn.execute(
            "SELECT content FROM memories WHERE id = ?", (self.existing,)
        ).fetchone()[0]
        self.assertEqual(stored, text)

    def test_an_insert_requires_the_caller_to_write(self) -> None:
        verdict = write_dedup.decide(self.store, "The sourdough starter is fed at eight.")
        result = write_dedup.apply(self.store, "The sourdough starter is fed at eight.", verdict)
        self.assertFalse(result["written"])
        self.assertEqual(result["action"], "insert_required")

    def test_a_neighbour_in_another_state_is_ignored(self) -> None:
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'tombstoned' WHERE id = ?", (self.existing,))
        verdict = write_dedup.decide(self.store, "The dashboard runs on port 8100 behind nginx.")
        self.assertEqual(verdict.action, "insert")

    def test_the_verdict_describes_itself(self) -> None:
        self.assertIn("write it",
                      write_dedup.decide(self.store, "Something entirely new about bicycles.").describe())
        self.assertIn("reject",
                      write_dedup.decide(
                          self.store, "The dashboard runs on port 8100 behind nginx."
                      ).describe())

    def test_the_dict_round_trips(self) -> None:
        payload = write_dedup.decide(self.store, "The dashboard port is 8100.").as_dict()
        self.assertIn("sentence", payload)
        self.assertIn("neighbours", payload)

    def test_a_row_can_be_asked_about_without_finding_itself(self) -> None:
        """A survey over existing rows must not report every one as its own duplicate."""
        content = "The dashboard runs on port 8100 behind nginx."
        self.assertEqual(
            write_dedup.decide(self.store, content, exclude=self.existing).action, "insert"
        )
        self.assertNotIn(
            self.existing,
            [item["memory_id"] for item in
             write_dedup.neighbours(self.store, content, exclude=self.existing)],
        )

    def test_the_report_carries_the_thresholds(self) -> None:
        described = write_dedup.report(
            self.store, samples=["The dashboard runs on port 8100 behind nginx.", "A new fact."]
        )
        self.assertEqual(described["thresholds"]["verbatim_jaccard"],
                         write_dedup.VERBATIM_JACCARD)
        self.assertEqual(described["samples"][0]["action"], "reject")
        self.assertEqual(described["samples"][1]["action"], "insert")

    def test_an_empty_store_is_quiet(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            self.assertEqual(write_dedup.neighbours(other, "anything"), [])
            self.assertEqual(write_dedup.decide(other, "anything at all").action, "insert")
        finally:
            other.close()


if __name__ == "__main__":
    unittest.main()
