"""An edge the retriever made is not something the user said (#206)."""

from __future__ import annotations

import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import edge_explain, edge_provenance as ep
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class OriginTests(unittest.TestCase):
    def test_a_user_statement_is_human(self) -> None:
        self.assertEqual(ep.origin_of("user_feedback"), ep.HUMAN)

    def test_a_wikilink_is_a_document(self) -> None:
        self.assertEqual(ep.origin_of("explicit_wikilink"), ep.DOCUMENT)

    def test_co_retrieval_is_machine(self) -> None:
        self.assertEqual(ep.origin_of("co_retrieval"), ep.MACHINE)

    def test_an_unknown_type_defaults_to_machine(self) -> None:
        """Unknown evidence must never be more citable than known evidence."""
        self.assertEqual(ep.origin_of("something_new"), ep.MACHINE)

    def test_an_unattributed_legacy_row_is_machine(self) -> None:
        self.assertEqual(ep.origin_of("legacy_unattributed"), ep.MACHINE)

    def test_only_human_and_document_are_citable(self) -> None:
        self.assertEqual(ep.CITABLE, frozenset({ep.HUMAN, ep.DOCUMENT}))

    def test_human_outranks_document_and_machine(self) -> None:
        self.assertGreater(ep.RANK[ep.HUMAN], ep.RANK[ep.DOCUMENT])
        self.assertGreater(ep.RANK[ep.DOCUMENT], ep.RANK[ep.MACHINE])


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

    def witness(self, evidence_type: str) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO edge_evidence
                   (evidence_id, src_id, dst_id, relation, evidence_type, evidence_key, summary,
                    created_at)
                   VALUES (?, ?, ?, 'related', ?, ?, ?, ?)""",
                (str(uuid.uuid4()), self.left, self.right, evidence_type,
                 f"{evidence_type}:{uuid.uuid4()}", "witness", NOW.isoformat()),
            )

    def link(self) -> None:
        self.store.add_edge(self.left, self.right, "related", weight=0.5)
        with self.store.transaction() as conn:
            conn.execute(
                "DELETE FROM edge_evidence WHERE evidence_type = 'legacy_unattributed'"
            )

    def test_no_evidence_is_not_citable(self) -> None:
        verdict = ep.provenance(self.store, self.left, self.right)
        self.assertFalse(verdict.citable)
        self.assertEqual(verdict.origin, ep.MACHINE)

    def test_a_machine_only_edge_is_not_citable(self) -> None:
        self.link()
        self.witness("co_retrieval")
        self.witness("auto_judge")
        verdict = ep.provenance(self.store, self.left, self.right)
        self.assertFalse(verdict.citable)
        self.assertIn("not citable", verdict.describe())

    def test_a_document_edge_is_citable(self) -> None:
        self.link()
        self.witness("explicit_wikilink")
        self.assertTrue(ep.citable(self.store, self.left, self.right))

    def test_a_human_edge_is_citable(self) -> None:
        self.link()
        self.witness("user_feedback")
        verdict = ep.provenance(self.store, self.left, self.right)
        self.assertTrue(verdict.citable)
        self.assertEqual(verdict.origin, ep.HUMAN)

    def test_one_human_word_outranks_many_machine_rows(self) -> None:
        self.link()
        for _ in range(5):
            self.witness("co_retrieval")
        self.witness("user_feedback")
        verdict = ep.provenance(self.store, self.left, self.right)
        self.assertEqual(verdict.origin, ep.HUMAN)
        self.assertTrue(verdict.citable)
        self.assertTrue(verdict.mixed)

    def test_mixed_witnesses_are_reported_as_mixed(self) -> None:
        self.link()
        self.witness("explicit_wikilink")
        self.witness("co_retrieval")
        verdict = ep.provenance(self.store, self.left, self.right)
        self.assertTrue(verdict.mixed)
        self.assertEqual(verdict.origin, ep.DOCUMENT)

    def test_the_caveat_names_the_origin(self) -> None:
        self.link()
        self.witness("co_retrieval")
        self.assertIn("not stated by anyone",
                      ep.caveat_for(self.store, self.left, self.right))

    def test_the_relation_can_be_narrowed(self) -> None:
        self.link()
        self.witness("explicit_wikilink")
        self.assertTrue(ep.citable(self.store, self.left, self.right, "related"))
        self.assertFalse(ep.citable(self.store, self.left, self.right, "extends"))

    def test_the_explanation_carries_the_provenance_caveat(self) -> None:
        """#85 renders edges to people, so it must say when the link is only inferred."""
        self.store.add_edge(self.left, self.right, "related", weight=0.5)
        self.witness("co_retrieval")
        sentence = edge_explain.explain(self.store, self.left, self.right).sentence
        self.assertIn("inferred from recall history", sentence)

    def test_a_document_link_needs_no_inference_warning(self) -> None:
        self.store.add_edge(self.left, self.right, "related", weight=0.5)
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM edge_evidence")
        self.witness("explicit_wikilink")
        sentence = edge_explain.explain(self.store, self.left, self.right).sentence
        self.assertNotIn("inferred from recall history", sentence)
        self.assertIn("supported by a note", sentence)

    def test_the_report_counts_rows_by_origin(self) -> None:
        self.link()
        self.witness("co_retrieval")
        self.witness("explicit_wikilink")
        described = ep.report(self.store)
        self.assertEqual(described["by_origin"][ep.MACHINE], 1)
        self.assertEqual(described["by_origin"][ep.DOCUMENT], 1)
        self.assertIn("co_retrieval", described["machine_types"])

    def test_an_empty_store_is_quiet(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            self.assertEqual(ep.report(other)["evidence_rows"], 0)
            self.assertFalse(ep.citable(other, "a", "b"))
        finally:
            other.close()


if __name__ == "__main__":
    unittest.main()
