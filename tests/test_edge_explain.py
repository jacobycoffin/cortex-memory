"""Every relation renders as a sentence a person can check (#85)."""

from __future__ import annotations

import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import assoc_feedback, edge_explain, near_miss
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.left = self.memory("Route 73 job runs at 06:00.")
        self.right = self.memory("The Vertex account is billed for route 73.")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def witness(self, evidence_type: str, count: int = 1) -> None:
        for _ in range(count):
            with self.store.transaction() as conn:
                conn.execute(
                    """INSERT INTO edge_evidence
                       (evidence_id, src_id, dst_id, relation, evidence_type, evidence_key, summary,
                        created_at)
                       VALUES (?, ?, ?, 'related', ?, ?, ?, ?)""",
                    (str(uuid.uuid4()), self.left, self.right, evidence_type,
                     f"{evidence_type}:{uuid.uuid4()}", f"{evidence_type} witness",
                     NOW.isoformat()),
                )

    def test_a_missing_link_says_so(self) -> None:
        explained = edge_explain.explain(self.store, self.left, self.right)
        self.assertEqual(explained.status, "absent")
        self.assertIn("no link is recorded", explained.sentence)

    def test_a_recorded_link_names_the_evidence_and_the_count(self) -> None:
        self.store.add_edge(self.left, self.right, "related", weight=0.6)
        self.witness("co_retrieval", 3)
        sentence = edge_explain.explain(self.store, self.left, self.right).sentence
        self.assertIn("retrieved together 3×", sentence)

    def test_the_sentence_uses_both_memories(self) -> None:
        self.store.add_edge(self.left, self.right, "related", weight=0.6)
        self.witness("co_retrieval")
        sentence = edge_explain.explain(self.store, self.left, self.right).sentence
        self.assertIn("Route 73 job", sentence)
        self.assertIn("Vertex account", sentence)

    def test_strength_is_a_word_not_a_decimal(self) -> None:
        self.store.add_edge(self.left, self.right, "related", weight=0.85)
        self.witness("co_retrieval")
        sentence = edge_explain.explain(self.store, self.left, self.right).sentence
        self.assertIn("strongly", sentence.lower())

    def test_the_relation_is_rendered_as_a_phrase(self) -> None:
        self.store.add_edge(self.left, self.right, "supports", weight=0.4)
        self.witness("explicit_wikilink")
        sentence = edge_explain.explain(self.store, self.left, self.right).sentence
        self.assertIn("supports", sentence)  # named in the tail relation phrase

    def test_a_link_with_no_evidence_says_so_plainly(self) -> None:
        self.store.add_edge(self.left, self.right, "related", weight=0.3)
        # add_edge writes only the unattributed row, which must be named as such
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM edge_evidence WHERE evidence_type = 'legacy_unattributed'")
        sentence = edge_explain.explain(self.store, self.left, self.right).sentence
        self.assertIn("no preserved evidence", sentence)

    def test_a_vetoed_pair_says_a_person_said_so(self) -> None:
        self.store.add_edge(self.left, self.right, "related", weight=0.6)
        self.witness("co_retrieval")
        assoc_feedback.apply(self.store, self.left, self.right, "unrelated", at=NOW)
        explained = edge_explain.explain(self.store, self.left, self.right)
        self.assertEqual(explained.status, "vetoed")
        self.assertIn("unrelated", explained.sentence)

    def test_a_near_miss_pair_says_only_the_stronger_surfaces(self) -> None:
        self.store.add_edge(self.left, self.right, "related", weight=0.6)
        self.witness("co_retrieval")
        near_miss.record(self.store, self.left, self.right, at=NOW)
        explained = edge_explain.explain(self.store, self.left, self.right)
        self.assertEqual(explained.status, "near-miss")
        self.assertIn("only the stronger", explained.sentence)

    def test_a_quarantined_link_says_it_does_not_score_yet(self) -> None:
        from cortex import edge_quarantine

        edge_quarantine.adopt(self.store, at=NOW)
        self.store.add_edge(self.left, self.right, "related", weight=0.6)
        self.witness("co_retrieval")
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE edges SET created_at = ? WHERE src_id = ? AND dst_id = ?",
                ((NOW.isoformat()), self.left, self.right),
            )
        explained = edge_explain.explain(self.store, self.left, self.right)
        self.assertIn("not scoring yet", explained.sentence)

    def test_resolution_accepts_phrases(self) -> None:
        self.store.add_edge(self.left, self.right, "related", weight=0.6)
        self.witness("co_retrieval")
        explained = edge_explain.explain(self.store, "Route 73 job", "Vertex account")
        self.assertNotEqual(explained.status, "absent")

    def test_for_memory_lists_links_strongest_first(self) -> None:
        third = self.memory("A third fact about route 73.")
        self.store.add_edge(self.left, self.right, "related", weight=0.9)
        self.store.add_edge(self.left, third, "related", weight=0.2)
        self.witness("co_retrieval")
        explained = edge_explain.for_memory(self.store, self.left)
        self.assertGreaterEqual(len(explained), 2)
        self.assertGreaterEqual(explained[0].weight, explained[1].weight)

    def test_the_dict_round_trips(self) -> None:
        self.store.add_edge(self.left, self.right, "related", weight=0.6)
        self.witness("co_retrieval")
        payload = edge_explain.explain(self.store, self.left, self.right).as_dict()
        self.assertIn("sentence", payload)
        self.assertEqual(payload["status"], "counting")

    def test_a_specific_relation_can_be_asked_for(self) -> None:
        self.store.add_edge(self.left, self.right, "supports", weight=0.5)
        self.witness("explicit_wikilink")
        explained = edge_explain.explain(self.store, self.left, self.right, "supports")
        self.assertEqual(explained.relation, "supports")
        self.assertEqual(
            edge_explain.explain(self.store, self.left, self.right, "extends").status, "absent"
        )

    def test_strength_words_cover_the_range(self) -> None:
        self.assertEqual(edge_explain.strength_word(0.9), "strongly")
        self.assertEqual(edge_explain.strength_word(0.6), "moderately")
        self.assertEqual(edge_explain.strength_word(0.3), "weakly")
        self.assertEqual(edge_explain.strength_word(0.05), "very weakly")

    def test_the_report_counts_what_is_explainable(self) -> None:
        self.store.add_edge(self.left, self.right, "related", weight=0.5)
        described = edge_explain.report(self.store)
        self.assertEqual(described["edges"], 1)
        self.assertGreater(described["phrases"], 5)

    def test_an_empty_store_is_quiet(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            self.assertEqual(edge_explain.for_memory(other, "nope"), [])
            self.assertEqual(edge_explain.report(other)["edges"], 0)
        finally:
            other.close()


if __name__ == "__main__":
    unittest.main()
