"""Core memory in model voice: imperative, scoped, stripped (#128)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.core_block import build
from cortex.core_voice import enabled, measure, to_model_voice
from cortex.store import CortexStore


class VoiceTests(unittest.TestCase):
    def test_a_narrated_preference_becomes_an_instruction(self) -> None:
        self.assertEqual(
            to_model_voice("Jacoby prefers short answers with no preamble."),
            "Prefer short answers with no preamble.",
        )

    def test_an_exception_is_lifted_out(self) -> None:
        rewritten = to_model_voice(
            "Jacoby prefers the V2 nightly channel, except when a TestFlight build is pending."
        )
        self.assertTrue(rewritten.startswith("Prefer the V2 nightly channel"), rewritten)
        self.assertIn("Except:", rewritten)
        self.assertIn("TestFlight build is pending", rewritten)

    def test_vault_scaffolding_is_removed(self) -> None:
        rewritten = to_model_voice(
            "Vault note: Duo Tabletop Path: Temple/Temple/Duo/View.swift Section: Layout — the "
            "tabletop view keeps the rest timer visible."
        )
        self.assertNotIn("Vault note", rewritten)
        self.assertNotIn("Path:", rewritten)
        self.assertNotIn("Section:", rewritten)
        self.assertIn("rest timer", rewritten)

    def test_scoped_rules_keep_their_condition(self) -> None:
        rewritten = to_model_voice(
            "Jacoby should never change the release channel while a build is in review."
        )
        self.assertIn("release channel", rewritten)
        self.assertNotIn("Jacoby should", rewritten)

    def test_both_halves_survive_a_dash(self) -> None:
        """Measured data loss: the first version kept only the tail and dropped the rule."""
        text = (
            "Also: check the Obsidian vault + Cortex FIRST before manual SSH/probing. "
            "wiki.js is now insignificant (legacy mirror of the vault) — do NOT treat it as a source."
        )
        rewritten = to_model_voice(text)
        self.assertIn("check the Obsidian vault", rewritten)
        self.assertIn("do NOT treat it as a source", rewritten)

    def test_a_short_label_before_a_dash_is_kept_as_a_label(self) -> None:
        rewritten = to_model_voice("Delegation — prefer bounded background work.")
        self.assertIn("Delegation", rewritten)
        self.assertIn("bounded background work", rewritten)

    def test_an_ordinary_fact_is_kept_intact(self) -> None:
        fact = "The maintenance window is Sunday 02:00 on CT116."
        self.assertEqual(to_model_voice(fact), fact)

    def test_an_empty_memory_stays_empty(self) -> None:
        self.assertEqual(to_model_voice("   "), "")

    def test_long_text_is_capped(self) -> None:
        rewritten = to_model_voice("word " * 200, max_chars=120)
        self.assertLessEqual(len(rewritten), 120)

    def test_the_transform_is_deterministic(self) -> None:
        text = "Jacoby prefers X, except Y. Jacoby should also Z."
        self.assertEqual(to_model_voice(text), to_model_voice(text))

    def test_it_never_invents_content(self) -> None:
        """A verbose true memory beats a crisp invented one."""
        text = "Sonarr imports stall on RAR archives; the fix is to disable unpacking."
        rewritten = to_model_voice(text)
        for token in ("Sonarr", "RAR", "unpacking"):
            self.assertIn(token, rewritten)

    def test_switched_off_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORTEX_CORE_VOICE", None)
            self.assertFalse(enabled())


class BlockIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.store.add_memory(
            content=(
                "Vault note: Delegation Path: preferences/Delegation.md Section: Rules — "
                "Jacoby prefers Jarvis to delegate bounded independent background work to "
                "subagents so the main conversation stays available."
            ),
            kind="preference",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            pinned=True,
            importance=0.9,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_the_block_keeps_the_original_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORTEX_CORE_VOICE", None)
            self.assertIn("Vault note", build(self.store).text)

    def test_the_block_uses_model_voice_when_enabled(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_CORE_VOICE": "1"}):
            text = build(self.store).text
            self.assertNotIn("Vault note", text)
            self.assertIn("Prefer", text)

    def test_the_rewrite_fits_more_into_the_same_budget(self) -> None:
        rows = [{"content": "Jacoby prefers X " * 20}]
        stats = measure(rows)
        self.assertGreater(stats["chars_saved"], 0)
        self.assertGreater(stats["share_saved"], 0)


if __name__ == "__main__":
    unittest.main()
