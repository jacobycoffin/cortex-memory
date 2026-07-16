from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT

from cortex.store import CortexStore


class MemoryNeighborhoodTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_one_credential_reference_uses_broad_groups_not_named_service(self) -> None:
        memory_id, _ = self.store.add_memory(
            "The Hermes deploy credential is stored in 1Password under Hermes VPS.",
            kind="operational",
            applicable_systems=["Hermes"],
            source_category="OPERATOR_APPROVED",
            storage_policy="operator_approved",
        )
        snapshot = self.store.dashboard_snapshot(memory_limit=20)
        memory = next(row for row in snapshot["memories"] if row["id"] == memory_id)
        labels = {item["label"] for item in memory["neighborhoods"]}
        self.assertIn("Credential references", labels)
        self.assertIn("Tool use", labels)
        self.assertIn("Services", labels)
        self.assertNotIn("Service: Hermes", labels)
        self.assertEqual(memory["primary_neighborhood"], "Credential references")

        results = self.store.neighborhood_search("Which Hermes credential should the tool use?")
        self.assertIn(memory_id, {row["id"] for row in results})

    def test_plaintext_password_is_redacted_before_proposal_storage(self) -> None:
        proposal = self.store.propose_memory_creation(
            "The experimental service password is orange-door-4829.",
            source_type="user_turn",
            source_category="USER_STATED",
        )
        self.assertNotIn("orange-door-4829", proposal["content"])
        self.assertIn("[REDACTED]", proposal["content"])
        self.assertTrue(proposal["redacted"])
        self.assertIn("secret value removed", proposal["quarantine_reason"])
        with self.store._lock:
            raw = self.store._conn.execute(
                "SELECT content,candidate_json FROM memory_creation_proposals WHERE proposal_id=?",
                (proposal["proposal_id"],),
            ).fetchone()
        self.assertNotIn("orange-door-4829", str(dict(raw)))

    def test_password_policy_is_not_a_credential_reference(self) -> None:
        memory_id, _ = self.store.add_memory(
            "Passwords must not be stored in Cortex.",
            source_category="OPERATOR_APPROVED",
            storage_policy="operator_approved",
        )
        memory = next(
            row
            for row in self.store.dashboard_snapshot(memory_limit=20)["memories"]
            if row["id"] == memory_id
        )
        labels = {item["label"] for item in memory["neighborhoods"]}
        self.assertNotIn("Credential references", labels)

    def test_credential_text_does_not_invent_atlas_or_promote_one_off_service(self) -> None:
        proposal = self.store.propose_memory_creation(
            "The Atlas deploy credential is stored in 1Password under Atlas Production.",
            source_type="user_turn",
            source_category="USER_STATED",
            applicable_systems=["Hermes"],
        )
        review_item = next(
            item
            for item in self.store.dashboard_snapshot(memory_limit=20)["review_inbox"]["items"]
            if item.get("proposal_id") == proposal["proposal_id"]
        )
        preview = set(review_item["creation"]["proposed_neighborhoods"])
        self.assertIn("Credential references", preview)
        self.assertIn("Tool use", preview)
        self.assertIn("Services", preview)
        self.assertNotIn("Service: Hermes", preview)
        self.assertNotIn("Service: Atlas", preview)

        result = self.store.review_memory_creation(proposal["proposal_id"], "remember")
        snapshot = self.store.dashboard_snapshot(memory_limit=20)
        memory = next(row for row in snapshot["memories"] if row["id"] == result["memory_id"])
        labels = {item["label"] for item in memory["neighborhoods"]}
        self.assertIn("Services", labels)
        self.assertNotIn("Service: Hermes", labels)
        self.assertNotIn("Service: Atlas", labels)

    def test_named_service_requires_repeated_approved_cross_context_evidence(self) -> None:
        ids = []
        for index in range(5):
            memory_id, _ = self.store.add_memory(
                f"Verified Hermes operating fact number {index}.",
                applicable_systems=["Hermes"],
                session_id=f"session-{index % 2}",
                source_category="OPERATOR_APPROVED",
                storage_policy="operator_approved",
            )
            ids.append(memory_id)

        snapshot = self.store.dashboard_snapshot(memory_limit=20)
        by_id = {row["id"]: row for row in snapshot["memories"]}
        for memory_id in ids:
            labels = {item["label"] for item in by_id[memory_id]["neighborhoods"]}
            self.assertIn("Services", labels)
            self.assertIn("Service: Hermes", labels)
        training = snapshot["review_inbox"]["neighborhood_training"]
        evaluation = next(
            item for item in training["evaluations"] if item["name_key"] == "hermes"
        )
        self.assertEqual(evaluation["decision"], "admitted")
        self.assertEqual(evaluation["memory_count"], 5)
        self.assertEqual(evaluation["context_count"], 2)
        self.assertTrue(training["recent_events"])

    def test_paused_project_stays_broad_and_is_logged_inactive(self) -> None:
        memory_id, _ = self.store.add_memory(
            "Atlas is paused while priorities are reconsidered.",
            scope={"project": "Atlas", "project_status": "paused"},
            session_id="atlas-session",
            source_category="OPERATOR_APPROVED",
            storage_policy="operator_approved",
        )
        snapshot = self.store.dashboard_snapshot(memory_limit=20)
        memory = next(row for row in snapshot["memories"] if row["id"] == memory_id)
        labels = {item["label"] for item in memory["neighborhoods"]}
        self.assertIn("Projects", labels)
        self.assertNotIn("Project: Atlas", labels)
        evaluation = next(
            item
            for item in snapshot["review_inbox"]["neighborhood_training"]["evaluations"]
            if item["name_key"] == "atlas"
        )
        self.assertEqual(evaluation["decision"], "inactive")

    def test_reopen_removes_legacy_one_mention_dynamic_neighborhood(self) -> None:
        memory_id, _ = self.store.add_memory("A normal durable fact.")
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO memory_neighborhoods(
                     neighborhood_id,slug,label,category,parent_neighborhood_id,
                     description,safety_class,created_at
                   ) VALUES('neighborhood:service-atlas','service-atlas','Service: Atlas',
                            'service','neighborhood:services','legacy heuristic','normal',
                            '2026-07-16T00:00:00+00:00')"""
            )
            conn.execute(
                """INSERT INTO memory_neighborhood_memberships(
                     memory_id,neighborhood_id,confidence,origin,explanation,created_at
                   ) VALUES(?,'neighborhood:service-atlas',0.9,'deterministic',
                            'legacy one-mention heuristic','2026-07-16T00:00:00+00:00')""",
                (memory_id,),
            )
        path = self.store.path
        self.store.close()
        self.store = CortexStore(path)
        labels = {
            row["label"]
            for row in self.store.dashboard_snapshot(memory_limit=20)["memories"][0][
                "neighborhoods"
            ]
        }
        self.assertNotIn("Service: Atlas", labels)


if __name__ == "__main__":
    unittest.main()
