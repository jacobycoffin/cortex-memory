"""Hard test: a finance or work memory must never reach a personal channel (item #20).

Every path that can put a memory in front of a user is exercised, not just the happy one:
retrieval, the always-on floor, and the floor's own protections. A leak on any one of them
is a leak.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.floor_context import build_floor
from cortex.namespace import admissible, channel_namespaces, filter_results, memory_namespaces
from cortex.retrieval import MemoryRetriever, RetrievalContext
from cortex.store import CortexStore

FINANCE = "The advisory fee is 1.2% and the wire hits the account on the 3rd."
WORK = "The county contract renews in March and the SOW needs a signature."
PERSONAL = "The truck needs new tires before winter."


def memory(scope: dict | None = None, **metadata) -> dict:
    record = {"id": "m", "content": "x"}
    if scope is not None:
        record["scope_json"] = scope
    if metadata:
        record["metadata_json"] = metadata
    return record


class ClassificationTests(unittest.TestCase):
    def test_scope_namespace_is_read(self) -> None:
        self.assertEqual(memory_namespaces(memory({"namespace": "finance"})), {"finance"})

    def test_metadata_namespace_is_read(self) -> None:
        self.assertEqual(memory_namespaces(memory(namespace="WORK")), {"work"})

    def test_domain_and_lists_are_accepted(self) -> None:
        self.assertEqual(memory_namespaces(memory({"domain": ["finance", "tax"]})), {"finance", "tax"})

    def test_unscoped_memory_is_shared(self) -> None:
        self.assertEqual(memory_namespaces({"id": "m", "content": "x"}), set())

    def test_malformed_scope_is_ignored_not_fatal(self) -> None:
        self.assertEqual(memory_namespaces({"scope_json": "{not json"}), set())

    def test_channel_reads_scope_and_system_state(self) -> None:
        context = RetrievalContext(scope={"namespace": "personal"}, system_state={"domain": "work"})
        self.assertEqual(channel_namespaces(context), {"personal", "work"})


class AdmissibilityTests(unittest.TestCase):
    def test_a_finance_memory_is_blocked_from_a_personal_channel(self) -> None:
        self.assertFalse(admissible(memory({"namespace": "finance"}), {"personal"}))

    def test_a_finance_memory_is_allowed_in_its_own_channel(self) -> None:
        self.assertTrue(admissible(memory({"namespace": "finance"}), {"finance"}))

    def test_a_shared_memory_reaches_everyone(self) -> None:
        self.assertTrue(admissible({"id": "m", "content": "x"}, {"personal"}))

    def test_an_undeclared_channel_sees_everything(self) -> None:
        """No namespace declared on the turn means no isolation was requested."""
        self.assertTrue(admissible(memory({"namespace": "finance"}), set()))

    def test_multi_namespace_memory_matches_any(self) -> None:
        record = memory({"namespace": ["work", "finance"]})
        self.assertTrue(admissible(record, {"finance"}))
        self.assertTrue(admissible(record, {"work"}))
        self.assertFalse(admissible(record, {"personal"}))

    def test_filter_results_counts_what_it_dropped(self) -> None:
        class R:
            def __init__(self, mid: str, scope: dict | None) -> None:
                self.memory = {"id": mid, "content": "x", "scope_json": scope}

        kept, dropped = filter_results(
            [R("a", {"namespace": "finance"}), R("b", None), R("c", {"namespace": "work"})],
            {"personal"},
        )
        self.assertEqual([r.memory["id"] for r in kept], ["b"])
        self.assertEqual(dropped, 2)


class StorePathTests(unittest.TestCase):
    """The same fact, through the real store and the real retriever."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.finance = self._add(FINANCE, {"namespace": "finance"})
        self.work = self._add(WORK, {"namespace": "work"})
        self.personal = self._add(PERSONAL, None)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _add(self, content: str, scope: dict | None) -> str:
        created = self.store.add_memory(
            content=content,
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            importance=0.9,
            scope=scope,
        )
        return created[0] if isinstance(created, tuple) else created

    def test_retrieval_never_returns_a_finance_memory_to_a_personal_turn(self) -> None:
        retriever = MemoryRetriever(self.store)
        results, _diag = retriever.search_detailed(
            "advisory fee wire transfer", context=RetrievalContext(scope={"namespace": "personal"})
        )
        ids = [r.memory["id"] for r in results]
        self.assertNotIn(self.finance, ids)

    def test_retrieval_still_returns_it_to_its_own_channel(self) -> None:
        retriever = MemoryRetriever(self.store)
        results, _diag = retriever.search_detailed(
            "advisory fee wire transfer", context=RetrievalContext(scope={"namespace": "finance"})
        )
        self.assertIn(self.finance, [r.memory["id"] for r in results])

    def test_a_work_memory_is_blocked_from_a_personal_turn(self) -> None:
        retriever = MemoryRetriever(self.store)
        results, _diag = retriever.search_detailed(
            "county contract renewal SOW", context=RetrievalContext(scope={"namespace": "personal"})
        )
        self.assertNotIn(self.work, [r.memory["id"] for r in results])

    def test_the_floor_never_carries_a_finance_memory_into_a_personal_channel(self) -> None:
        block = build_floor(self.store, limit=5, channel={"personal"})
        self.assertNotIn("advisory fee", block)
        self.assertNotIn("county contract", block)

    def test_the_floor_still_works_for_a_shared_memory(self) -> None:
        block = build_floor(self.store, limit=5, channel={"personal"})
        self.assertIn("new tires", block)

    def test_a_finance_channel_still_gets_the_finance_fact(self) -> None:
        block = build_floor(self.store, limit=5, channel={"finance"})
        self.assertIn("advisory fee", block)


if __name__ == "__main__":
    unittest.main()
