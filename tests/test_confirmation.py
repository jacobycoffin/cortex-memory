"""Confirmation capture: restatements reinforce, corrections do not (#216)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.confirmation import (
    corroboration_stats,
    find_confirmation,
    is_correction,
    record_confirmation,
)
from cortex.store import CortexStore

FACT = "The maintenance window is Sunday 02:00 on CT116."
RESTATED = "ct116 maintenance window runs sunday 02:00"  # 0.667 Jaccard, 0.857 containment
CORRECTED = "The maintenance window changed to Tuesday 03:00 on CT116."
UNRELATED = "The Temple app uses a crimson accent on the workout screen."


class CorrectionDetectionTests(unittest.TestCase):
    def test_change_markers_are_corrections(self) -> None:
        for text in (
            "the window changed to tuesday",
            "we no longer use sqlcipher",
            "it moved to ct117 instead of ct116",
            "this supersedes the old plan",
        ):
            self.assertTrue(is_correction(text), text)

    def test_ordinary_negation_is_not_a_correction(self) -> None:
        """A factual sentence with 'not' in it is still a statement of fact.

        The first version matched bare `not`/`never`/`don't` and rejected 55% of the store
        as corrections, which would have made the whole feature inert.
        """
        for text in (
            "do not run the migration before the backup finishes",
            "the container never exposes port 8100 externally",
            "jacoby doesn't want production deploys without approval",
        ):
            self.assertFalse(is_correction(text), text)


class ConfirmationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        created = self.store.add_memory(
            content=FACT,
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        self.memory_id = created[0] if isinstance(created, tuple) else created

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_restatement_is_found(self) -> None:
        found = find_confirmation(self.store, RESTATED)
        self.assertIsNotNone(found)
        self.assertEqual(found.memory_id, self.memory_id)

    def test_an_unrelated_fact_is_not_a_confirmation(self) -> None:
        self.assertIsNone(find_confirmation(self.store, UNRELATED))

    def test_a_correction_is_never_a_confirmation(self) -> None:
        """Corrections share vocabulary with what they replace — that is the trap."""
        self.assertIsNone(find_confirmation(self.store, CORRECTED))

    def test_recording_increments_the_counters(self) -> None:
        before = corroboration_stats(self.store)
        self.assertIsNotNone(record_confirmation(self.store, RESTATED, session_id="s1"))
        after = corroboration_stats(self.store)
        self.assertEqual(after["confirmations"], before["confirmations"] + 1)
        self.assertEqual(after["validations"], before["validations"] + 1)
        self.assertEqual(after["corroborated"], before["corroborated"] + 1)

    def test_the_event_is_attributed_to_the_original_memory(self) -> None:
        record_confirmation(self.store, RESTATED, session_id="session-42")
        row = self.store._conn.execute(
            "SELECT memory_id, session_id FROM access_log WHERE event='confirmed'"
        ).fetchone()
        self.assertEqual(row["memory_id"], self.memory_id)
        self.assertEqual(row["session_id"], "session-42")

    def test_no_match_records_nothing(self) -> None:
        before = corroboration_stats(self.store)
        self.assertIsNone(record_confirmation(self.store, UNRELATED))
        self.assertEqual(corroboration_stats(self.store), before)

    def test_a_memory_does_not_confirm_itself(self) -> None:
        found = find_confirmation(self.store, FACT, exclude_id=self.memory_id)
        self.assertIsNone(found)

    def test_an_exact_copy_is_near_verbatim(self) -> None:
        found = find_confirmation(self.store, FACT + " ", exclude_id="other")
        self.assertIsNotNone(found)
        self.assertGreaterEqual(found.similarity, 0.85)
        self.assertTrue(found.verbatim)


class ProviderWiringTests(unittest.TestCase):
    def test_the_remember_action_confirms_instead_of_duplicating(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            provider = __import__(
                "cortex.hermes_provider", fromlist=["CortexMemoryProvider"]
            ).CortexMemoryProvider(
                {"token_budget": 700, "top_k": 6, "database_path": str(Path(tmp.name) / "p.db")}
            )
            provider.initialize("confirm-session", hermes_home=tmp.name, agent_context="primary")
            store = provider._store
            created = store.add_memory(
                content=FACT,
                kind="semantic",
                source_type="user_turn",
                source_category="USER_EXPLICIT",
            )
            original = created[0] if isinstance(created, tuple) else created

            import json

            reply = json.loads(
                provider.handle_tool_call(
                    "cortex_memory", {"action": "remember", "content": RESTATED}
                )
            )
            self.assertTrue(reply.get("confirmed"), reply)
            self.assertEqual(reply.get("memory_id"), original)
            self.assertFalse(reply.get("created"))
            self.assertGreater(
                store._conn.execute(
                    "SELECT confirmed_count FROM memories WHERE id=?", (original,)
                ).fetchone()[0],
                0,
            )
            store.close()
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
