"""Core memory never holds two contradictory lines (#130)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.contradiction import all_conflicts, check, conflicts_with
from cortex.promotion import ContradictionError, promote
from cortex.store import CortexStore


class ContradictionDetectionTests(unittest.TestCase):
    def test_the_same_claim_with_a_different_value(self) -> None:
        reason = conflicts_with(
            {"id": "a", "content": "window is Tuesday", "subject": "maintenance window",
             "predicate": "day", "object_value": "tuesday"},
            {"id": "b", "content": "window is Sunday", "subject": "maintenance window",
             "predicate": "day", "object_value": "sunday"},
        )
        self.assertIsNotNone(reason)
        self.assertIn("different value", str(reason))

    def test_the_same_claim_with_the_same_value_is_not_a_conflict(self) -> None:
        self.assertIsNone(
            conflicts_with(
                {"id": "a", "content": "x", "subject": "s", "predicate": "p", "object_value": "v"},
                {"id": "b", "content": "x", "subject": "s", "predicate": "p", "object_value": "v"},
            )
        )

    def test_opposite_polarity_on_the_same_wording(self) -> None:
        reason = conflicts_with(
            {"id": "a", "content": "The channel is the V2 nightly build for the phone."},
            {"id": "b", "content": "The channel is not the V2 nightly build for the phone."},
        )
        self.assertIsNotNone(reason)
        self.assertIn("polarity", str(reason))

    def test_a_restatement_is_not_a_conflict(self) -> None:
        """Restatements are #216's business; treating them as conflicts would block writes."""
        self.assertIsNone(
            conflicts_with(
                {"id": "a", "content": "Jacoby prefers short answers with no preamble."},
                {"id": "b", "content": "Jacoby prefers short answers without a preamble."},
            )
        )

    def test_unrelated_memories_never_conflict(self) -> None:
        self.assertIsNone(
            conflicts_with(
                {"id": "a", "content": "The Temple app uses a crimson accent."},
                {"id": "b", "content": "The maintenance window is Sunday."},
            )
        )

    def test_a_memory_never_conflicts_with_itself(self) -> None:
        row = {"id": "a", "content": "the channel is not nightly", "subject": "c",
               "predicate": "p", "object_value": "v"}
        self.assertIsNone(conflicts_with(row, row))


class WriteBlockingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _memory(self, content: str, **kwargs) -> str:
        created = self.store.add_memory(
            content=content,
            kind=kwargs.pop("kind", "semantic"),
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            **kwargs,
        )
        return created[0] if isinstance(created, tuple) else created

    def test_promoting_a_contradiction_is_refused(self) -> None:
        first = self._memory("The release channel is the V2 nightly build for the phone.")
        promote(self.store, first, by="pin")
        second = self._memory("The release channel is not the V2 nightly build for the phone.")
        with self.assertRaises(ContradictionError) as caught:
            promote(self.store, second, by="pin")
        self.assertIn("contradictory", str(caught.exception))

    def test_the_refused_promotion_leaves_core_untouched(self) -> None:
        first = self._memory("The release channel is the V2 nightly build for the phone.")
        promote(self.store, first, by="pin")
        second = self._memory("The release channel is not the V2 nightly build for the phone.")
        with self.assertRaises(ContradictionError):
            promote(self.store, second, by="pin")
        from cortex.promotion import members

        self.assertEqual(members(self.store), [first])

    def test_a_structured_conflict_is_caught_without_negation(self) -> None:
        first = self._memory("Window is Sunday.", subject="window", predicate="day",
                             object_value="sunday")
        promote(self.store, first, by="pin")
        second = self._memory("Window is Tuesday.", subject="window", predicate="day",
                              object_value="tuesday")
        with self.assertRaises(ContradictionError):
            promote(self.store, second, by="pin")

    def test_a_restatement_still_promotes(self) -> None:
        first = self._memory("Jacoby prefers short answers with no preamble.")
        promote(self.store, first, by="pin")
        second = self._memory("Jacoby prefers short answers without a preamble.")
        self.assertIsNotNone(promote(self.store, second, by="use"))

    def test_the_check_can_be_skipped_explicitly(self) -> None:
        """Ingestion of a known-bad import should be able to say so on purpose."""
        first = self._memory("The channel is nightly.")
        promote(self.store, first, by="pin")
        second = self._memory("The channel is not nightly.")
        self.assertIsNotNone(promote(self.store, second, by="pin", check_conflicts=False))

    def test_check_reports_before_the_write(self) -> None:
        first = self._memory("The channel is nightly.")
        promote(self.store, first, by="pin")
        second = self._memory("The channel is not nightly.")
        found = check(self.store, second)
        self.assertTrue(found)
        self.assertEqual(found[0].memory_id, first)

    def test_an_empty_core_has_no_conflicts(self) -> None:
        self.assertEqual(all_conflicts(self.store), [])


class ToolSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        provider_module = __import__(
            "cortex.hermes_provider", fromlist=["CortexMemoryProvider"]
        )
        self.provider = provider_module.CortexMemoryProvider(
            {"token_budget": 700, "top_k": 6, "database_path": str(Path(self.tmp.name) / "p.db")}
        )
        self.provider.initialize("conflict", hermes_home=self.tmp.name, agent_context="primary")
        self.store: CortexStore = self.provider._store

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_core_reports_conflicts(self) -> None:
        reply = json.loads(self.provider.handle_tool_call("cortex_memory", {"action": "core"}))
        self.assertIn("conflicts", reply)
        self.assertEqual(reply["conflicts"], [])

    def test_editing_into_a_contradiction_reports_the_failure(self) -> None:
        first = self.store.add_memory(
            content="The channel is nightly.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            pinned=True,
        )[0]
        second = self.store.add_memory(
            content="The channel is not nightly.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            pinned=True,
        )[0]
        reply = json.loads(
            self.provider.handle_tool_call(
                "cortex_memory",
                {"action": "core_edit", "memory_id": second, "content": "The channel is not nightly at all."},
            )
        )
        self.assertFalse(reply["success"], reply)
        self.assertIn("contradictory", reply.get("error", ""))
        self.assertIsNotNone(self.store.get_memory(first))


if __name__ == "__main__":
    unittest.main()
