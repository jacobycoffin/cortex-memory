from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from tests._bootstrap import ROOT

from cortex.store import CortexStore, _retention_score
from cortex.retrieval import MemoryRetriever


class LearningBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_review_preserves_origin_separately_from_approval(self) -> None:
        proposal = self.store.propose_memory_creation(
            "Jacoby prefers an amber release theme.",
            kind="preference",
            source_type="user_turn",
            source_category="USER_STATED",
            session_id="origin-test",
        )
        decision = self.store.review_memory_creation(
            proposal["proposal_id"], "remember", actor="test-operator"
        )

        memory = self.store.get_memory(decision["memory_id"])

        self.assertEqual(memory["source_category"], "OPERATOR_APPROVED")
        self.assertEqual(memory["origin_source_category"], "USER_STATED")
        self.assertEqual(memory["approval_state"], "operator_approved")

    def test_candidate_occurrences_and_dataset_are_text_free_by_default(self) -> None:
        for session in ("first-session", "second-session"):
            self.store.propose_memory_creation(
                "The verified release checklist is reusable.",
                source_category="USER_STATED",
                session_id=session,
            )

        dataset = self.store.learning_experience_dataset()
        observed = [
            item for item in dataset["experiences"] if item["event_type"] == "candidate_observed"
        ]

        self.assertEqual(len(observed), 2)
        self.assertEqual({item["session_id"] for item in observed}, {"first-session", "second-session"})
        self.assertEqual(dataset["privacy_mode"], "no_memory_text")
        self.assertTrue(all("memory_text" not in item for item in observed))
        self.assertTrue(all(item["split"] in {"train", "validation", "test"} for item in observed))

    def test_spaced_helpful_use_outweighs_one_burst(self) -> None:
        burst, _ = self.store.add_memory("Burst-only reusable fact.")
        spaced, _ = self.store.add_memory("Spaced reusable fact.")
        with self.store.transaction() as conn:
            for index in range(10):
                task_id = f"burst-{index}"
                conn.execute(
                    """INSERT INTO memory_experience_events(
                         event_id,event_type,memory_id,task_id,session_id,evidence_key,event_day,
                         outcome,metadata_json,created_at
                       ) VALUES(?,'memory_used',?,?,?,?,?,'used','{}',?)""",
                    (
                        str(uuid.uuid4()), burst, task_id, "same-session",
                        f"burst:{index}", "2026-07-16", f"2026-07-16T10:00:{index:02d}+00:00",
                    ),
                )
            for index, day in enumerate(("2026-07-14", "2026-07-16")):
                task_id = f"spaced-{index}"
                conn.execute(
                    """INSERT INTO memory_experience_events(
                         event_id,event_type,memory_id,task_id,session_id,evidence_key,event_day,
                         outcome,metadata_json,created_at
                       ) VALUES(?,'memory_used',?,?,?,?,?,'used','{}',?)""",
                    (
                        str(uuid.uuid4()), spaced, task_id, f"session-{index}",
                        f"spaced-use:{index}", day, f"{day}T10:00:00+00:00",
                    ),
                )
                conn.execute(
                    """INSERT INTO memory_experience_events(
                         event_id,event_type,memory_id,task_id,session_id,evidence_key,event_day,
                         outcome,metadata_json,created_at
                       ) VALUES(?,'outcome_labeled',?,?,?,?,?,'helpful','{}',?)""",
                    (
                        str(uuid.uuid4()), spaced, task_id, f"session-{index}",
                        f"spaced-outcome:{index}", day, f"{day}T10:05:00+00:00",
                    ),
                )
        strengths = self.store.memory_experience_strengths([burst, spaced])

        self.assertGreater(
            strengths[spaced]["spaced_reinforcement"],
            strengths[burst]["spaced_reinforcement"],
        )
        self.assertGreater(
            _retention_score({"used_count": 2, **strengths[spaced]}),
            _retention_score({"used_count": 10, **strengths[burst]}),
        )

    def test_stale_clean_start_preview_is_rejected(self) -> None:
        preview = self.store.preview_trained_recall_set()
        self.store.add_memory(
            "A newly approved memory changes the clean-start membership.",
            source_category="OPERATOR_APPROVED",
            origin_source_category="USER_STATED",
            approval_state="operator_approved",
        )

        with self.assertRaisesRegex(ValueError, "preview changed"):
            self.store.start_trained_recall_set(
                actor="test-operator", expected_preview_id=preview["preview_id"]
            )

        current = self.store.preview_trained_recall_set()
        result = self.store.start_trained_recall_set(
            actor="test-operator", expected_preview_id=current["preview_id"]
        )
        self.assertEqual(result["active"]["kind"], "trained")

    def test_due_prospective_memory_is_cued_and_completion_suppresses_it(self) -> None:
        memory_id, _ = self.store.add_memory(
            "Send the verified release note.",
            kind="prospective",
            valid_from="2026-01-01T09:00:00+00:00",
        )

        recalled = MemoryRetriever(self.store, threshold=0.0).search(
            "Hello", limit=2, graph_depth=0
        )
        self.assertIn(memory_id, {item.memory["id"] for item in recalled})
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE prospective_items SET status='completed' WHERE memory_id=?", (memory_id,)
            )

        recalled = MemoryRetriever(self.store, threshold=0.0).search(
            "Hello", limit=2, graph_depth=0
        )
        self.assertNotIn(memory_id, {item.memory["id"] for item in recalled})

    def test_unified_decision_log_contains_neighborhood_rule_events(self) -> None:
        for index in range(5):
            self.store.add_memory(
                f"Hermes approved operational detail {index}.",
                source_category="OPERATOR_APPROVED",
                origin_source_category="USER_STATED",
                approval_state="operator_approved",
                session_id=f"schema-{index % 2}",
                applicable_systems=["Hermes"],
            )

        entries = self.store.decision_log(limit=100)

        neighborhood = next(item for item in entries if item["category"] == "neighborhood")
        self.assertEqual(neighborhood["action"], "admitted")
        self.assertIn("Hermes", neighborhood["summary"])
        self.assertTrue(neighborhood["reversible"])


if __name__ == "__main__":
    unittest.main()
