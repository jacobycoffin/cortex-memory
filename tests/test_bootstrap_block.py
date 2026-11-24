"""The fixed preamble: who, what, and what is not allowed (#150)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import bootstrap_block
from cortex.store import CortexStore


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, kind: str = "semantic", pinned: bool = False,
               scope: dict | None = None, confidence: float = 0.9) -> str:
        memory_id = self.store.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category="USER_STATED",
            pinned=pinned, confidence=confidence,
        )[0]
        if scope:
            with self.store.transaction() as conn:
                conn.execute("UPDATE memories SET scope_json = ? WHERE id = ?",
                             (json.dumps(scope), memory_id))
        return memory_id

    def headings(self, block) -> list[str]:
        return [line for line in block.lines if line.endswith(":")]

    def test_the_three_sections_are_always_present_in_order(self) -> None:
        block = bootstrap_block.build(self.store)
        self.assertEqual(self.headings(block),
                         ["Pinned:", "Working on:", "Constraints:"])

    def test_an_empty_section_says_so_rather_than_disappearing(self) -> None:
        """A heading that vanishes with its section silently reshapes the prompt."""
        block = bootstrap_block.build(self.store)
        self.assertEqual(block.lines.count("- (nothing recorded)"), 3)

    def test_identity_comes_from_pinned_memories(self) -> None:
        pinned = self.memory("Jacoby is a network tech specialist in Elmira Heights.", pinned=True)
        self.memory("an unpinned fact that should not lead")
        block = bootstrap_block.build(self.store)
        self.assertTrue(any(pinned[:8] in line for line in block.lines))

    def test_an_unreliable_kind_label_is_not_trusted_for_identity(self) -> None:
        """The live store's `identity` kind contains a gateway-restart log."""
        self.memory("`hermes gateway restart` (2748739 -> 2947533)", kind="identity")
        block = bootstrap_block.build(self.store)
        self.assertFalse(any("gateway restart" in line for line in block.lines))

    def test_projects_come_from_scope_tags_and_are_counted(self) -> None:
        self.memory("a duelist-road fact", scope={"project": "duelist-road"})
        self.memory("another duelist-road fact", scope={"project": "duelist-road"})
        self.memory("a cortex fact", scope={"project": "cortex-memory"})
        block = bootstrap_block.build(self.store)
        line = [line for line in block.lines if "duelist-road" in line][0]
        self.assertIn("(2 memories)", line)

    def test_projects_are_ranked_by_how_much_is_known(self) -> None:
        self.memory("one", scope={"project": "small"})
        for index in range(3):
            self.memory(f"big {index}", scope={"project": "big"})
        block = bootstrap_block.build(self.store)
        big = [line for line in block.lines if "big" in line][0]
        small = [line for line in block.lines if "small" in line][0]
        self.assertLess(block.lines.index(big), block.lines.index(small))

    def test_a_scope_project_list_is_flattened(self) -> None:
        self.memory("x", scope={"project": ["one", "two"]})
        block = bootstrap_block.build(self.store)
        self.assertTrue(any("one" in line for line in block.lines))
        self.assertTrue(any("two" in line for line in block.lines))

    def test_every_line_is_traceable_to_a_memory(self) -> None:
        pinned = self.memory("Jacoby keeps the vault in Obsidian.", pinned=True)
        block = bootstrap_block.build(self.store)
        self.assertIn(pinned[:8], block.ids)

    def test_the_same_store_gives_the_same_block(self) -> None:
        for index in range(4):
            self.memory(f"pinned fact {index}", pinned=True)
        first = bootstrap_block.build(self.store)
        second = bootstrap_block.build(self.store)
        self.assertEqual(first.lines, second.lines)

    def test_the_budget_is_respected(self) -> None:
        for index in range(6):
            self.memory(f"a long pinned fact number {index} " * 4, pinned=True)
        block = bootstrap_block.build(self.store, budget=200)
        self.assertLessEqual(block.chars, 240)

    def test_truncation_is_counted_not_hidden(self) -> None:
        for index in range(6):
            self.memory(f"pinned fact {index} " * 6, pinned=True)
        block = bootstrap_block.build(self.store, budget=160)
        self.assertGreater(block.truncated, 0)

    def test_the_line_cap_is_applied(self) -> None:
        for index in range(8):
            self.memory(f"pinned fact {index}", pinned=True)
        block = bootstrap_block.build(self.store, who_items=2)
        self.assertEqual(len([line for line in block.lines if line.startswith("- ") and "[" in line
                              and block.lines.index(line) < block.lines.index("Working on:")]), 2)

    def test_a_pinned_memory_is_read_as_a_clause_not_a_paragraph(self) -> None:
        self.memory("Jacoby runs a homelab. " + "And a great deal more text follows here." * 6,
                    pinned=True)
        block = bootstrap_block.build(self.store)
        who_line = block.lines[1]
        self.assertLessEqual(len(who_line), bootstrap_block.WHO_CHARS + 20)

    def test_the_constraints_section_uses_the_constraints_module(self) -> None:
        # Constraints are *recorded* by the constraints module, never inferred from a memory's text:
        # a preamble that promoted sentence keywords would invent limits nobody stated.
        from cortex import constraints

        memory_id = self.memory("Never restart the cortex dashboard without asking.")
        self.assertIsNotNone(constraints.promote(
            self.store, memory_id, "Never restart the cortex dashboard without asking."))
        block = bootstrap_block.build(self.store)
        self.assertTrue(any("Constraints" in line for line in block.lines))
        self.assertTrue(any("never" in line.lower() for line in block.lines))

    def test_the_dict_carries_the_rendered_text(self) -> None:
        described = bootstrap_block.build(self.store).as_dict()
        self.assertIn("Pinned:", described["text"])

    def test_a_broken_store_still_returns_three_sections(self) -> None:
        class Broken:
            _conn = None

        block = bootstrap_block.build(Broken())
        self.assertEqual(len(self.headings(block)), 3)


if __name__ == "__main__":
    unittest.main()
