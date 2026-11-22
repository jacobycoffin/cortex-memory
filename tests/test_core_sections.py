"""Section budgets: one topic cannot crowd out another (#127)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.core_block import (
    SECTIONS,
    SECTION_SHARES,
    build,
    classify_section,
)
from cortex.store import CortexStore


class ClassificationTests(unittest.TestCase):
    def test_kind_wins_over_keywords(self) -> None:
        """A memory captured as a preference is one, whatever words it uses."""
        self.assertEqual(
            classify_section({"kind": "preference", "content": "never do the thing"}),
            "preferences",
        )
        self.assertEqual(
            classify_section({"kind": "prospective", "content": "a project build"}),
            "commitments",
        )

    def test_a_project_scope_is_decisive(self) -> None:
        self.assertEqual(
            classify_section(
                {"kind": "semantic", "content": "a rule", "scope_json": '{"active_project":"website"}'}
            ),
            "projects",
        )

    def test_constraints_beat_preferences_in_the_fallback(self) -> None:
        self.assertEqual(
            classify_section({"kind": "semantic", "content": "never change the channel"}),
            "constraints",
        )

    def test_an_unclassifiable_memory_lands_in_general(self) -> None:
        self.assertEqual(
            classify_section({"kind": "semantic", "content": "the kettle boils in four minutes"}),
            "general",
        )


class BudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _memory(self, content: str, kind: str = "semantic", **kwargs) -> str:
        created = self.store.add_memory(
            content=content,
            kind=kind,
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            **kwargs,
        )
        return created[0] if isinstance(created, tuple) else created

    def test_a_flood_of_one_section_cannot_evict_another(self) -> None:
        """The whole point: forty project facts must not push out the one preference."""
        for index in range(40):
            self._memory(
                f"Project fact number {index} about repositories and release builds.",
                kind="semantic",
                pinned=True,
                importance=0.9,
            )
        preference = self._memory(
            "Jacoby prefers short answers with no preamble.",
            kind="preference",
            pinned=True,
            importance=0.1,
        )
        block = build(self.store, max_chars=2000)
        self.assertTrue(
            any(memory_id.startswith(preference[:8]) for memory_id in block.memory_ids),
            block.text,
        )
        self.assertIn("Preferences:", block.text)

    def test_sections_appear_in_a_fixed_order(self) -> None:
        from cortex.core_block import SECTION_TITLES

        self._memory("Jacoby prefers tea.", kind="preference", pinned=True)
        self._memory("Never deploy on Fridays.", kind="semantic", pinned=True)
        self._memory("Open decision: pick a database.", kind="prospective", pinned=True)
        block = build(self.store)
        from cortex.core_block import HEADER

        titles = [
            line[:-1]
            for line in block.text.splitlines()
            if line.endswith(":") and line != HEADER
        ]
        order = [name for name in SECTIONS if SECTION_TITLES[name] in titles]
        self.assertEqual(titles, [SECTION_TITLES[name] for name in order], titles)

    def test_the_block_stays_inside_its_band(self) -> None:
        for index in range(40):
            self._memory(f"Fact {index} " + "detail " * 30, pinned=True, importance=0.9)
        block = build(self.store, max_chars=2000, min_chars=800)
        self.assertLessEqual(block.chars, 2000)
        self.assertGreaterEqual(block.chars, 800)

    def test_an_empty_section_does_not_waste_its_slot(self) -> None:
        """People is empty here, so the space goes to the sections that have content."""
        for index in range(20):
            self._memory(f"Constraint fact {index}: never use port 22 for deploys.", pinned=True, importance=0.9)
        block = build(self.store, max_chars=2000)
        self.assertNotIn("People:", block.text)
        # The constraint section therefore carries more than its nominal share would allow.
        constraint_lines = block.text.split("Constraints:")[1].count("\n- ")
        self.assertGreater(constraint_lines, 2)

    def test_the_shares_do_not_oversubscribe_the_block(self) -> None:
        self.assertLessEqual(sum(SECTION_SHARES.values()), 1.0)

    def test_rendering_is_deterministic(self) -> None:
        self._memory("Jacoby prefers tea.", kind="preference", pinned=True)
        self._memory("Never deploy on Fridays.", kind="semantic", pinned=True)
        self.assertEqual(build(self.store).text, build(self.store).text)


if __name__ == "__main__":
    unittest.main()
