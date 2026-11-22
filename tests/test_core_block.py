"""Core memory block: always present, plain SQL, fixed size (#120)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.core_block import (
    HEADER,
    MAX_LINE_CHARS,
    build,
    dedupe_against,
    enabled,
    stats,
)
from cortex.store import CortexStore


class CoreBlockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.pinned = self.store.add_memory(
            content="Jacoby reads T3 Code on his phone; never change the release channel.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            importance=0.99,
            pinned=True,
        )[0]
        for index in range(12):
            self.store.add_memory(
                content=f"Ordinary fact number {index} about the homelab and its services.",
                kind="semantic",
                source_type="user_turn",
                source_category="USER_EXPLICIT",
                importance=0.2 + index * 0.01,
            )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_the_block_is_inside_the_specified_band(self) -> None:
        block = build(self.store, min_chars=400, max_chars=800)
        self.assertLessEqual(block.chars, 800)
        self.assertGreaterEqual(block.chars, 400)

    def test_it_never_exceeds_the_ceiling(self) -> None:
        block = build(self.store, max_chars=500)
        self.assertLessEqual(block.chars, 500)

    def test_pinned_memories_come_first(self) -> None:
        block = build(self.store)
        first_line = block.text.splitlines()[1]
        self.assertIn(self.pinned[:8], first_line)
        self.assertIn("[pinned]", first_line)

    def test_the_header_says_it_may_be_incomplete(self) -> None:
        """An always-present block that reads as authoritative becomes a source of error."""
        self.assertEqual(build(self.store).text.splitlines()[0], HEADER)

    def test_lines_are_capped(self) -> None:
        self.store.add_memory(
            content="word " * 200,
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            importance=1.0,
        )
        for line in build(self.store, max_chars=2000).text.splitlines()[1:]:
            self.assertLessEqual(len(line), MAX_LINE_CHARS + 40, line[:60])

    def test_it_is_deterministic(self) -> None:
        self.assertEqual(build(self.store).text, build(self.store).text)

    def test_it_uses_plain_sql_only(self) -> None:
        """The block is the memory of last resort: no FTS, no embeddings, no ranking model."""
        import cortex.core_block as module

        source = Path(module.__file__).read_text()
        self.assertNotIn("fts_search", source)
        self.assertNotIn("semantic", source.replace("semantic_features", ""))

    def test_commitments_are_rendered_from_the_join(self) -> None:
        commitment = self.store.add_memory(
            content="Publish the Cortex release notes.",
            kind="commitment",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO prospective_items(memory_id,status,due_at,created_at,updated_at) "
                "VALUES(?, 'open', ?, ?, ?)",
                (commitment, "2026-12-01T00:00:00+00:00", "2026-11-01T00:00:00+00:00",
                 "2026-11-01T00:00:00+00:00"),
            )
        block = build(self.store, max_chars=4000)
        self.assertIn("commitment", block.text)
        self.assertIn("Publish the Cortex release notes.", block.text)

    def test_stats_report_the_cost(self) -> None:
        numbers = stats(self.store)
        self.assertGreater(numbers["chars"], 0)
        self.assertEqual(numbers["pinned"], 1)
        self.assertEqual(numbers["estimated_tokens"], (numbers["chars"] + 3) // 4)

    def test_off_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORTEX_CORE_MEMORY", None)
            self.assertFalse(enabled())

    def test_switched_on_by_the_environment(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_CORE_MEMORY": "1"}):
            self.assertTrue(enabled())


class VerifiedDateTests(unittest.TestCase):
    """Every core line says when it was last verified (#125)."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _memory(self, content: str, **kwargs) -> str:
        created = self.store.add_memory(
            content=content,
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            **kwargs,
        )
        return created[0] if isinstance(created, tuple) else created

    def test_every_line_carries_a_label(self) -> None:
        self._memory("Pinned fact with no history.", pinned=True, importance=0.9)
        block = build(self.store)
        for line in block.text.splitlines()[1:]:
            self.assertTrue(
                "[verified " in line or "[unverified]" in line, f"no label: {line[:80]}"
            )

    def test_a_fact_with_no_history_is_marked_unverified(self) -> None:
        """Silence would invite the reader to assume freshness."""
        self._memory("Never used, never confirmed.", pinned=True, importance=0.9)
        self.assertIn("[unverified]", build(self.store).text)

    def test_use_stamps_the_date(self) -> None:
        memory_id = self._memory("Used yesterday.", pinned=True, importance=0.9)
        self.store.log_access(memory_id, "used")
        self.assertNotIn("[unverified]", build(self.store).text)

    def test_a_confirmation_stamps_the_date(self) -> None:
        from cortex.promotion import promote, touch_confirmations

        memory_id = self._memory("Confirmed this month.", importance=0.9)
        promote(self.store, memory_id, at="2026-01-01T00:00:00+00:00")
        touch_confirmations(self.store, [memory_id], at="2026-11-01T00:00:00+00:00")
        self.assertIn("2026-11-01", build(self.store).text)

    def test_promotion_alone_is_not_a_verification(self) -> None:
        """A rule firing is not someone vouching for the fact."""
        from cortex.promotion import promote

        memory_id = self._memory("Promoted by a rule, never used since.", importance=0.9)
        promote(self.store, memory_id, at="2026-11-01T00:00:00+00:00")
        self.assertIn("[unverified]", build(self.store).text)

    def test_the_later_of_use_and_confirmation_wins(self) -> None:
        from cortex.promotion import promote, touch_confirmations

        memory_id = self._memory("Used long ago, confirmed recently.", importance=0.9)
        promote(self.store, memory_id, at="2024-01-01T00:00:00+00:00")
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET last_used_at = ? WHERE id = ?",
                ("2025-01-01T00:00:00+00:00", memory_id),
            )
        touch_confirmations(self.store, [memory_id], at="2026-11-01T00:00:00+00:00")
        text = build(self.store).text
        self.assertIn("2026-11-01", text)
        self.assertNotIn("2025-01-01", text)

    def test_the_label_respects_the_line_cap(self) -> None:
        self._memory("word " * 300, pinned=True, importance=0.9)
        for line in build(self.store, max_chars=2000).text.splitlines()[1:]:
            self.assertLessEqual(len(line), MAX_LINE_CHARS + 40, line[:60])


class DedupeTests(unittest.TestCase):
    def test_a_recalled_memory_is_not_repeated(self) -> None:
        text = "Core memory:\n- [M:aaaa1111 semantic] one\n- [M:bbbb2222 semantic] two"
        self.assertEqual(
            dedupe_against(text, {"aaaa1111-0000-0000-0000-000000000000"}),
            "Core memory:\n- [M:bbbb2222 semantic] two",
        )

    def test_nothing_recalled_leaves_it_alone(self) -> None:
        text = "Core memory:\n- [M:aaaa1111 semantic] one"
        self.assertEqual(dedupe_against(text, set()), text)

    def test_the_header_survives(self) -> None:
        text = "Core memory:\n- [M:aaaa1111 semantic] one"
        self.assertIn("Core memory:", dedupe_against(text, {"aaaa1111-x"}))


class ProviderInjectionTests(unittest.TestCase):
    def _provider(self, tmp: str):
        provider = __import__(
            "cortex.hermes_provider", fromlist=["CortexMemoryProvider"]
        ).CortexMemoryProvider(
            {"token_budget": 700, "top_k": 6, "database_path": str(Path(tmp) / "p.db")}
        )
        provider.initialize("core-session", hermes_home=tmp, agent_context="primary")
        return provider

    def test_the_block_is_absent_by_default(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            provider = self._provider(tmp.name)
            provider._store.add_memory(
                content="Pinned release-channel rule: never change it.",
                kind="semantic",
                source_type="user_turn",
                source_category="USER_EXPLICIT",
                pinned=True,
            )
            context = provider.prefetch("what is the release channel rule", session_id="core-session")
            self.assertNotIn(HEADER, context)
            provider._store.close()
        finally:
            tmp.cleanup()

    def test_the_block_is_prepended_when_enabled(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            with mock.patch.dict(os.environ, {"CORTEX_CORE_MEMORY": "1"}):
                provider = self._provider(tmp.name)
                provider._store.add_memory(
                    content="Pinned release-channel rule: never change it.",
                    kind="semantic",
                    source_type="user_turn",
                    source_category="USER_EXPLICIT",
                    pinned=True,
                )
                context = provider.prefetch(
                    "what is the release channel rule", session_id="core-session"
                )
                self.assertIn(HEADER, context)
                provider._store.close()
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
