"""Compile the system prompt from memory, deterministically (#151)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import prompt_compiler as pc
from cortex.store import CortexStore


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, kind: str = "semantic", category: str = "USER_STATED",
               pinned: bool = False) -> str:
        memory_id = self.store.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category=category,
        )[0]
        if pinned:
            with self.store.transaction() as conn:
                conn.execute("UPDATE memories SET pinned = 1 WHERE id = ?", (memory_id,))
        return memory_id

    def test_an_empty_store_compiles_to_nothing(self) -> None:
        compiled = pc.compile_prompt(self.store)
        self.assertEqual(compiled.text, "")
        self.assertEqual(compiled.digest, pc.hashlib.sha256(b"").hexdigest()[:12])

    def test_a_pinned_memory_reaches_the_prompt(self) -> None:
        self.memory("answer in metric units", pinned=True)
        compiled = pc.compile_prompt(self.store)
        self.assertIn("Pinned:", compiled.text)
        self.assertIn("metric units", compiled.text)

    def test_every_line_carries_a_short_tag(self) -> None:
        memory_id = self.memory("answer in metric units", pinned=True)
        compiled = pc.compile_prompt(self.store)
        self.assertIn(f"[{memory_id[:8]}]", compiled.text)

    def test_the_same_state_gives_the_same_bytes(self) -> None:
        """A prompt that changes every turn destroys prefix caching."""
        self.memory("answer in metric units", pinned=True)
        self.memory("the dashboard listens on port 8100", kind="preference")
        described = pc.stable(self.store)
        self.assertTrue(described["stable"])
        self.assertEqual(len(described["digest"]), 12)

    def test_no_timestamp_is_embedded(self) -> None:
        self.memory("a fact", pinned=True)
        compiled = pc.compile_prompt(self.store)
        for marker in ("2026-", "T12:", "generated at"):
            self.assertNotIn(marker, compiled.text)

    def test_a_correction_gets_its_own_section(self) -> None:
        old = self.memory("the port is 8100")
        from cortex import correction_memory as cm

        cm.record(self.store, supersedes=old, content="the port is 8101")
        compiled = pc.compile_prompt(self.store)
        self.assertIn("Corrections:", compiled.text)
        self.assertIn("8101", compiled.text)

    def test_a_correction_survives_a_tiny_budget(self) -> None:
        """He said it in so many words; a prompt that drops it re-litigates a settled question."""
        old = self.memory("the port is 8100")
        from cortex import correction_memory as cm

        cm.record(self.store, supersedes=old, content="the port is 8101")
        for index in range(8):
            self.memory(f"a long decision note number {index} " * 4, kind="decision")
        compiled = pc.compile_prompt(self.store, budget=200)
        self.assertIn("8101", compiled.text)
        self.assertTrue(compiled.dropped > 0)

    def test_an_empty_section_is_omitted_not_printed(self) -> None:
        self.memory("a fact", pinned=True)
        compiled = pc.compile_prompt(self.store)
        self.assertNotIn("Commitments:", compiled.text)
        self.assertNotIn("Limits:", compiled.text)

    def test_truncation_is_counted_and_reported(self) -> None:
        for index in range(6):
            self.memory(f"a decision note number {index} with plenty of words in it", kind="decision")
        compiled = pc.compile_prompt(self.store, budget=120)
        self.assertLessEqual(compiled.chars, 120)
        self.assertGreater(compiled.dropped, 0)
        self.assertIn("dropped", "\n".join(pc.render(self.store, budget=120)))

    def test_an_over_budget_prompt_says_so(self) -> None:
        """Only reachable when protected sections alone exceed the budget."""
        for index in range(6):
            self.memory(f"a pinned instruction number {index} with quite a few words", pinned=True)
        compiled = pc.compile_prompt(self.store, budget=80)
        self.assertTrue(compiled.over_budget)

    def test_the_budget_is_stated_in_the_result(self) -> None:
        self.memory("a fact", pinned=True)
        self.assertEqual(pc.compile_prompt(self.store, budget=500).budget, 500)

    def test_sections_are_in_a_fixed_order(self) -> None:
        self.memory("a pinned instruction", pinned=True)
        self.memory("a preference about metric units", kind="preference")
        self.memory("a decision about the roadmap", kind="decision")
        compiled = pc.compile_prompt(self.store)
        names = [section["name"] for section in compiled.sections]
        self.assertEqual(names, [name for name, _title, _keep in pc.SECTIONS])

    def test_protected_sections_are_flagged_in_the_table(self) -> None:
        protected = [name for name, _title, keep in pc.SECTIONS if keep]
        self.assertEqual(protected, ["pinned", "corrections"])

    def test_a_tag_is_eight_characters(self) -> None:
        self.assertEqual(len(pc.tag("0123456789abcdef")), 8)

    def test_a_long_line_is_shortened_with_an_ellipsis(self) -> None:
        self.assertTrue(pc.one_line("word " * 80).endswith("…"))

    def test_whitespace_is_collapsed_in_a_line(self) -> None:
        self.assertEqual(pc.one_line("a\n\n  b"), "a b")

    def test_the_render_states_chars_and_digest(self) -> None:
        self.memory("a fact", pinned=True)
        rendered = "\n".join(pc.render(self.store))
        self.assertIn("digest", rendered)
        self.assertIn("pinned:", rendered)

    def test_a_broken_store_compiles_to_nothing(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(pc.compile_prompt(Broken()).text, "")


if __name__ == "__main__":
    unittest.main()
