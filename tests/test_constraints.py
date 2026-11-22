"""Constraints are always in view, whatever the question (#60)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import constraints, preference_strength
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class DetectionTests(unittest.TestCase):
    def test_allergies_are_constraints(self) -> None:
        self.assertEqual(constraints.detect("Allergic to shellfish."), constraints.ALLERGY)
        self.assertEqual(constraints.detect("He can't eat gluten."), constraints.ALLERGY)

    def test_budgets_are_constraints(self) -> None:
        for text in (
            "The hosting budget is 40 dollars a month.",
            "Spend at most 200 on the upgrade.",
            "$25 per month for software.",
        ):
            self.assertEqual(constraints.detect(text), constraints.BUDGET, text)

    def test_hard_nos_are_constraints(self) -> None:
        for text in (
            "Never touch the production database.",
            "Do not ever deploy on Fridays.",
            "No exceptions on this one.",
        ):
            self.assertEqual(constraints.detect(text), constraints.HARD_NO, text)

    def test_limits_are_constraints(self) -> None:
        self.assertEqual(constraints.detect("Only text messages, no calls."),
                         constraints.LIMIT)

    def test_ordinary_statements_are_not_constraints(self) -> None:
        for text in ("The dashboard runs on port 8100.", "Jacoby prefers short answers.", ""):
            self.assertIsNone(constraints.detect(text), text)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def add(self, content: str, *, importance: float = 0.8, kind: str = "preference") -> str:
        return self.store.add_memory(
            content=content, kind=kind, source_type="user_turn",
            source_category="USER_EXPLICIT", importance=importance,
        )[0]

    def test_a_detected_constraint_is_promoted(self) -> None:
        memory_id = self.add("Allergic to shellfish.")
        self.assertEqual(constraints.promote(self.store, memory_id, "Allergic to shellfish."),
                         constraints.ALLERGY)
        self.assertEqual(len(constraints.all(self.store)), 1)

    def test_a_plain_fact_is_not_promoted(self) -> None:
        memory_id = self.add("The dashboard runs on port 8100.")
        self.assertIsNone(
            constraints.promote(self.store, memory_id, "The dashboard runs on port 8100.")
        )

    def test_a_hard_preference_is_a_constraint_without_a_keyword(self) -> None:
        """#59's hard class fills the gap the keyword list leaves."""
        content = "Ship the release checklist every time."
        memory_id = self.add(content)
        preference_strength.record(self.store, memory_id, "Always " + content, at=NOW)
        self.assertEqual(constraints.promote_from_strength(self.store, memory_id, content),
                         constraints.HARD_NO)
        # The record is against the memory id, so the lookup works with the stored strength.
        self.assertEqual(len(constraints.all(self.store)), 1)

    def test_promotion_is_idempotent(self) -> None:
        memory_id = self.add("Never deploy on Fridays.")
        constraints.promote(self.store, memory_id, "Never deploy on Fridays.", at=NOW)
        constraints.promote(self.store, memory_id, "Never deploy on Fridays.", at=NOW)
        self.assertEqual(constraints.report(self.store)["constraints"], 1)

    def test_a_constraint_can_be_removed(self) -> None:
        memory_id = self.add("Never deploy on Fridays.")
        constraints.promote(self.store, memory_id, "Never deploy on Fridays.", at=NOW)
        self.assertTrue(constraints.remove(self.store, memory_id))
        self.assertEqual(constraints.all(self.store), [])


class BlockTests(StoreTests):
    def test_the_block_lists_the_constraints(self) -> None:
        memory_id = self.add("Allergic to shellfish.")
        constraints.promote(self.store, memory_id, "Allergic to shellfish.", at=NOW)
        block = constraints.context_block(self.store)
        self.assertIn("Constraints", block)
        self.assertIn("shellfish", block)

    def test_no_constraints_means_no_block(self) -> None:
        self.assertEqual(constraints.context_block(self.store), "")

    def test_the_block_is_capped(self) -> None:
        for index in range(20):
            content = f"Never do dangerous thing number {index} because it hurts."
            memory_id = self.add(content)
            constraints.promote(self.store, memory_id, content, at=NOW)
        block = constraints.context_block(self.store)
        self.assertLessEqual(len(block), constraints.MAX_BLOCK_CHARS + 80)
        self.assertLessEqual(block.count("- "), constraints.MAX_ITEMS)

    def test_important_constraints_come_first(self) -> None:
        low = self.add("Never use the blue theme, it looks wrong.", importance=0.2)
        high = self.add("Allergic to shellfish.", importance=0.95)
        constraints.promote(self.store, low, "Never use the blue theme, it looks wrong.", at=NOW)
        constraints.promote(self.store, high, "Allergic to shellfish.", at=NOW)
        lines = constraints.context_block(self.store).splitlines()
        self.assertIn("shellfish", lines[1])

    def test_suppressed_constraints_are_not_shown(self) -> None:
        from cortex import suppression

        memory_id = self.add("Never deploy on Fridays.")
        constraints.promote(self.store, memory_id, "Never deploy on Fridays.", at=NOW)
        suppression.suppress(self.store, memory_id, reason="user lifted it")
        self.assertEqual(constraints.context_block(self.store), "")


class RelevantTests(StoreTests):
    def test_a_matching_constraint_is_flagged(self) -> None:
        memory_id = self.add("Allergic to shellfish.")
        constraints.promote(self.store, memory_id, "Allergic to shellfish.", at=NOW)
        self.assertEqual(len(constraints.relevant(self.store, "book a shellfish restaurant")), 1)

    def test_an_unrelated_action_is_not_flagged(self) -> None:
        memory_id = self.add("Allergic to shellfish.")
        constraints.promote(self.store, memory_id, "Allergic to shellfish.", at=NOW)
        self.assertEqual(constraints.relevant(self.store, "restart the nginx service"), [])


class HarnessTests(unittest.TestCase):
    """The constraint block has to reach the actual turn context."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        from cortex.harness import CortexHarnessAdapter

        self.adapter = CortexHarnessAdapter(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.adapter.close()
        self.tmp.cleanup()

    def test_a_constraint_appears_in_a_turn_regardless_of_the_query(self) -> None:
        memory_id, _created = self.adapter.remember(
            "Allergic to shellfish.", source_type="user_turn", source_category="USER_EXPLICIT",
            kind="preference",
        )
        constraints.promote(self.adapter.memory.store, memory_id, "Allergic to shellfish.")
        turn = self.adapter.before_turn("how do I restart nginx", force_recall=True)
        self.assertIn("shellfish", turn.context)

    def test_without_constraints_the_context_is_unchanged(self) -> None:
        self.adapter.remember("The dashboard runs on port 8100.", source_type="user_turn",
                             source_category="USER_EXPLICIT")
        turn = self.adapter.before_turn("what port does the dashboard use", force_recall=True)
        self.assertNotIn("Constraints (", turn.context)


if __name__ == "__main__":
    unittest.main()
