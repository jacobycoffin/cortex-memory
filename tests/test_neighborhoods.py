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

    def test_credential_reference_groups_with_tool_and_service_without_secret(self) -> None:
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
        self.assertIn("Service: Hermes", labels)
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

    def test_credential_text_adds_named_service_beside_capture_system(self) -> None:
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
        self.assertIn("Service: Hermes", preview)
        self.assertIn("Service: Atlas", preview)

        result = self.store.review_memory_creation(proposal["proposal_id"], "remember")
        snapshot = self.store.dashboard_snapshot(memory_limit=20)
        memory = next(row for row in snapshot["memories"] if row["id"] == result["memory_id"])
        labels = {item["label"] for item in memory["neighborhoods"]}
        self.assertIn("Service: Hermes", labels)
        self.assertIn("Service: Atlas", labels)


if __name__ == "__main__":
    unittest.main()
