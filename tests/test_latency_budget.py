"""A latency budget must degrade, not fail (item #22)."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.latency_budget import ENV, Budget, budget_from
from cortex.retrieval import MemoryRetriever
from cortex.store import CortexStore


class BudgetTests(unittest.TestCase):
    def test_a_disabled_budget_never_expires(self) -> None:
        budget = Budget(0)
        self.assertFalse(budget.enabled)
        self.assertFalse(budget.expired())
        self.assertEqual(budget.remaining_ms(), float("inf"))

    def test_a_spent_budget_expires(self) -> None:
        budget = Budget(1)
        time.sleep(0.01)
        self.assertTrue(budget.expired())
        self.assertLessEqual(budget.remaining_ms(), 0.0)

    def test_a_generous_budget_does_not_expire(self) -> None:
        self.assertFalse(Budget(10_000).expired())

    def test_explicit_value_wins_over_the_environment(self) -> None:
        os.environ[ENV] = "500"
        try:
            self.assertEqual(budget_from(50).ms, 50.0)
        finally:
            os.environ.pop(ENV, None)

    def test_the_environment_can_enable_it(self) -> None:
        os.environ[ENV] = "123"
        try:
            self.assertEqual(budget_from(None).ms, 123.0)
        finally:
            os.environ.pop(ENV, None)

    def test_garbage_in_the_environment_disables_rather_than_crashes(self) -> None:
        os.environ[ENV] = "not-a-number"
        try:
            self.assertFalse(budget_from(None).enabled)
        finally:
            os.environ.pop(ENV, None)


class RetrievalBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        for index in range(30):
            self.store.add_memory(
                content=f"Fact number {index} about the maintenance window and the store.",
                kind="semantic",
                source_type="user_turn",
                source_category="USER_EXPLICIT",
            )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_an_exhausted_budget_still_returns_without_raising(self) -> None:
        retriever = MemoryRetriever(self.store, latency_budget_ms=0.0001)
        time.sleep(0.002)
        results, diagnostics = retriever.search_detailed("maintenance window", limit=5)
        self.assertIsInstance(results, list)  # best-so-far, possibly short
        self.assertIn("budget_exhausted", diagnostics.stage_ms)

    def test_a_generous_budget_is_invisible(self) -> None:
        retriever = MemoryRetriever(self.store, latency_budget_ms=30_000)
        results, diagnostics = retriever.search_detailed("maintenance window", limit=5)
        self.assertGreater(len(results), 0)
        self.assertNotIn("budget_exhausted", diagnostics.stage_ms)

    def test_no_budget_means_no_time_dependence(self) -> None:
        retriever = MemoryRetriever(self.store)
        first, _ = retriever.search_detailed("maintenance window", limit=5)
        second, _ = retriever.search_detailed("maintenance window", limit=5)
        self.assertEqual(
            [r.memory["id"] for r in first], [r.memory["id"] for r in second]
        )


if __name__ == "__main__":
    unittest.main()
