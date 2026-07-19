from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT

from cortex.retrieval import MemoryRetriever
from cortex.store import CortexStore, creation_proposal_revision


class RecallSetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_baseline_migration_preserves_ordinary_recall(self) -> None:
        memory_id, _ = self.store.add_memory(
            "Project Cedar uses a blue deployment switch.",
            kind="decision",
        )
        snapshot = self.store.recall_set_snapshot()
        self.assertEqual(snapshot["active"]["kind"], "legacy")
        self.assertIn(memory_id, {row["id"] for row in self.store.fts_search("Cedar deployment")})

    def test_trained_set_is_empty_for_personal_recall_and_reversible(self) -> None:
        personal_id, _ = self.store.add_memory(
            "Project Cedar uses a blue deployment switch.",
            kind="decision",
            record_role="canonical",
        )
        reference_id, _ = self.store.add_memory(
            "The Cedar manual documents the blue switch on page eight.",
            kind="semantic",
            source_type="vault_markdown",
            record_role="reference",
        )
        baseline_id = self.store.recall_set_snapshot()["active"]["recall_set_id"]

        preview = self.store.preview_trained_recall_set()
        self.assertEqual(preview["legacy_primary_excluded"], 1)
        self.assertEqual(preview["references_kept_as_evidence"], 1)
        activated = self.store.start_trained_recall_set(actor="test-operator")
        trained_id = activated["active"]["recall_set_id"]
        self.assertNotEqual(trained_id, baseline_id)

        ordinary = self.store.fts_search("Cedar blue switch")
        self.assertNotIn(personal_id, {row["id"] for row in ordinary})
        self.assertNotIn(reference_id, {row["id"] for row in ordinary})
        evidence = self.store.fts_search("Cedar manual page eight", evidence_lookup=True)
        self.assertIn(reference_id, {row["id"] for row in evidence})

        switched = self.store.activate_recall_set(baseline_id, actor="test-operator")
        self.assertEqual(switched["active"]["recall_set_id"], baseline_id)
        restored = self.store.fts_search("Cedar blue switch")
        self.assertIn(personal_id, {row["id"] for row in restored})

    def test_promoting_legacy_memory_adds_membership_without_rewriting_it(self) -> None:
        memory_id, _ = self.store.add_memory("Jacoby prefers the compact review layout.", kind="preference")
        original = self.store.get_memory(memory_id)
        self.store.start_trained_recall_set(actor="test-operator")

        promoted = self.store.promote_memory_to_active_recall_set(
            memory_id,
            eligibility="primary",
            actor="test-operator",
        )
        self.assertTrue(promoted["changed"])
        recalled = MemoryRetriever(self.store, threshold=0.0).search("compact review layout", limit=3)
        self.assertIn(memory_id, {result.memory["id"] for result in recalled})
        current = self.store.get_memory(memory_id)
        self.assertEqual(current["content"], original["content"])
        self.assertEqual(current["state"], original["state"])
        self.assertEqual(current["record_role"], original["record_role"])

    def test_trained_set_carries_trusted_operator_imports(self) -> None:
        trusted_id, _ = self.store.add_memory(
            "Hermes deploy credentials are stored in the password manager under Hermes VPS.",
            kind="semantic",
            source_type="operator_import",
            source_category="OPERATOR_APPROVED",
            storage_policy="operator_approved",
        )
        preview = self.store.preview_trained_recall_set()
        self.assertEqual(preview["approved_memories_carried"], 1)

        self.store.start_trained_recall_set(actor="test-operator")
        self.assertTrue(self.store.is_memory_recall_eligible(trusted_id))

    def test_trained_set_preserves_automatic_membership_provenance(self) -> None:
        proposals = [
            self.store.propose_memory_creation(
                "Project Cedar requires the verified deployment checklist.",
                kind="procedure",
                source_type="user_turn",
                source_category="USER_STATED",
            ),
            self.store.propose_memory_creation(
                "The Cedar deployment manual documents the checklist on page eight.",
                kind="semantic",
                source_type="document",
                source_category="DOCUMENT_EXTRACTED",
            ),
        ]
        decisions = [
            self.store.review_memory_creation(
                proposals[0]["proposal_id"],
                "remember",
                actor="cortex-auto-judge:synthetic",
                approval_authority="automatic",
                expected_revision=creation_proposal_revision(proposals[0]),
            ),
            self.store.review_memory_creation(
                proposals[1]["proposal_id"],
                "evidence_only",
                actor="cortex-auto-judge:synthetic",
                approval_authority="automatic",
                expected_revision=creation_proposal_revision(proposals[1]),
            ),
        ]
        memory_ids = {decision["memory_id"] for decision in decisions}
        with self.store._lock:
            before = {
                row["memory_id"]: dict(row)
                for row in self.store._conn.execute(
                    """SELECT memory_id,eligibility,origin,review_id,actor
                       FROM memory_recall_memberships WHERE revoked_at IS NULL"""
                ).fetchall()
                if row["memory_id"] in memory_ids
            }

        self.store.start_trained_recall_set(actor="test-operator")

        with self.store._lock:
            after = {
                row["memory_id"]: dict(row)
                for row in self.store._conn.execute(
                    """SELECT memory_id,eligibility,origin,review_id,actor
                       FROM memory_recall_memberships
                       WHERE recall_set_id=(
                         SELECT recall_set_id FROM memory_recall_sets WHERE status='active'
                       ) AND revoked_at IS NULL"""
                ).fetchall()
                if row["memory_id"] in memory_ids
            }
        self.assertEqual(after, before)

    def test_dashboard_map_marks_active_recall_eligibility(self) -> None:
        legacy_id, _ = self.store.add_memory("Legacy-only map detail for Project Cedar.")
        trusted_id, _ = self.store.add_memory(
            "Approved map detail for Project Cedar.",
            source_category="OPERATOR_APPROVED",
            storage_policy="operator_approved",
        )
        self.store.start_trained_recall_set(actor="test-operator")

        by_id = {row["id"]: row for row in self.store.dashboard_snapshot()["memories"]}
        self.assertIsNone(by_id[legacy_id]["recall_eligibility"])
        self.assertEqual(by_id[trusted_id]["recall_eligibility"], "primary")

    def test_graph_expansion_cannot_cross_the_active_recall_set(self) -> None:
        seed_id, _ = self.store.add_memory("Project Cedar deploys through Hermes.", kind="decision")
        legacy_neighbor_id, _ = self.store.add_memory(
            "A stale Cedar note says deployment uses an unrelated legacy host.", kind="operational"
        )
        self.store.add_edge(
            seed_id,
            legacy_neighbor_id,
            "supports",
            evidence_type="test",
            explanation="Test-only relationship.",
            evidence_key="recall-set-test",
        )
        self.store.start_trained_recall_set(actor="test-operator")
        self.store.promote_memory_to_active_recall_set(seed_id, actor="test-operator")

        results = MemoryRetriever(self.store, threshold=0.0).search(
            "How does Project Cedar deploy?", limit=5, graph_depth=2
        )
        ids = {result.memory["id"] for result in results}
        self.assertIn(seed_id, ids)
        self.assertNotIn(legacy_neighbor_id, ids)

    def test_reopening_database_does_not_backfill_legacy_into_trained_set(self) -> None:
        legacy_id, _ = self.store.add_memory("Legacy-only detail for Project Cedar.")
        self.store.start_trained_recall_set(actor="test-operator")
        path = self.store.path
        self.store.close()
        self.store = CortexStore(path)

        self.assertEqual(self.store.recall_set_snapshot()["active"]["kind"], "trained")
        self.assertFalse(self.store.is_memory_recall_eligible(legacy_id))

    def test_exact_legacy_duplicate_requires_review_and_promotes_without_copying(self) -> None:
        content = "Project Cedar deploys through the verified blue gateway."
        legacy_id, _ = self.store.add_memory(content, kind="decision")
        before = self.store.get_memory(legacy_id)
        self.store.start_trained_recall_set(actor="test-operator")
        proposal = self.store.propose_memory_creation(
            content,
            kind="decision",
            source_type="user_turn",
            source_category="USER_STATED",
        )
        self.assertEqual(proposal["assessment"]["duplicate_memory_id"], legacy_id)
        self.assertFalse(self.store.is_memory_recall_eligible(legacy_id))

        result = self.store.review_memory_creation(
            proposal["proposal_id"], "remember", actor="test-operator"
        )
        self.assertEqual(result["memory_id"], legacy_id)
        self.assertFalse(result["memory_created"])
        self.assertTrue(result["memory_promoted"])
        self.assertEqual(self.store.stats()["memories"], 1)
        self.assertTrue(self.store.is_memory_recall_eligible(legacy_id))

        self.assertTrue(self.store.undo_review_decision(result["review_id"]))
        self.assertFalse(self.store.is_memory_recall_eligible(legacy_id))
        restored = self.store.get_memory(legacy_id)
        self.assertEqual(restored["approval_state"], before["approval_state"])
        self.assertEqual(
            restored["origin_source_category"], before["origin_source_category"]
        )
        self.assertEqual(restored["updated_at"], before["updated_at"])

    def test_exact_duplicate_primary_promotion_undo_restores_evidence_membership(self) -> None:
        content = "The Cedar deployment manual documents the blue gateway."
        memory_id, _ = self.store.add_memory(
            content,
            source_type="document",
            record_role="reference",
            recall_eligibility="evidence_only",
        )
        with self.store._lock:
            prior = dict(
                self.store._conn.execute(
                    """SELECT * FROM memory_recall_memberships
                       WHERE memory_id=? AND revoked_at IS NULL""",
                    (memory_id,),
                ).fetchone()
            )
        proposal = self.store.propose_memory_creation(
            content,
            source_type="user_turn",
            source_category="USER_STATED",
        )

        result = self.store.review_memory_creation(
            proposal["proposal_id"], "remember", actor="test-operator"
        )

        self.assertTrue(result["memory_promoted"])
        self.assertTrue(self.store.is_memory_recall_eligible(memory_id))
        self.assertTrue(self.store.undo_review_decision(result["review_id"]))
        with self.store._lock:
            restored = dict(
                self.store._conn.execute(
                    """SELECT * FROM memory_recall_memberships
                       WHERE recall_set_id=? AND memory_id=?""",
                    (prior["recall_set_id"], memory_id),
                ).fetchone()
            )
        self.assertEqual(restored, prior)
        self.assertFalse(self.store.is_memory_recall_eligible(memory_id))
        self.assertTrue(self.store.is_memory_recall_eligible(memory_id, evidence_lookup=True))


if __name__ == "__main__":
    unittest.main()
