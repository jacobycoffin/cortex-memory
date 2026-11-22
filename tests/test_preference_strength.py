"""A preference carries how hard it is (#59)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import preference_strength as ps
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class ClassifyTests(unittest.TestCase):
    def test_hard_markers(self) -> None:
        for text in (
            "Never touch the production database.",
            "Always use the release job for deploys.",
            "No exceptions on this one.",
            "He is allergic to shellfish.",
            "The budget is 40 dollars a month.",
        ):
            self.assertEqual(ps.classify(text).strength, ps.HARD, text)

    def test_firm_markers(self) -> None:
        for text in (
            "Prefers short answers.",
            "Please don't add preamble.",
            "He'd rather get a table than a bullet list.",
        ):
            self.assertEqual(ps.classify(text).strength, ps.FIRM, text)

    def test_soft_markers(self) -> None:
        for text in (
            "Usually replies in the evening.",
            "If convenient, send a summary.",
            "Mildly prefers the dark theme.",
            "Tends to skim long documents.",
        ):
            self.assertEqual(ps.classify(text).strength, ps.SOFT, text)

    def test_a_hard_marker_beats_a_soft_one(self) -> None:
        """'always, when convenient' is still always: the safe reading wins."""
        result = ps.classify("Always run the tests, when convenient.")
        self.assertEqual(result.strength, ps.HARD)

    def test_a_contrast_clause_carries_the_real_instruction(self) -> None:
        result = ps.classify("I don't usually care about formatting, but this one matters.")
        self.assertEqual(result.strength, ps.FIRM)

    def test_an_unmarked_preference_is_firm(self) -> None:
        """Silence about intensity is not evidence of weakness."""
        self.assertEqual(ps.classify("Short answers.").strength, ps.FIRM)

    def test_an_empty_preference_is_firm(self) -> None:
        self.assertEqual(ps.classify("").strength, ps.FIRM)

    def test_only_a_hard_rule_forbids_deviation(self) -> None:
        self.assertFalse(ps.classify("Never deploy on Fridays.").may_deviate)
        self.assertTrue(ps.classify("Prefers short answers.").may_deviate)
        self.assertTrue(ps.classify("Usually replies in the evening.").may_deviate)

    def test_the_guidance_names_the_permitted_behaviour(self) -> None:
        self.assertIn("do not deviate", ps.classify("Never deploy.").describe())
        self.assertIn("stated reason", ps.classify("Prefers short answers.").describe())
        self.assertIn("deviate freely", ps.classify("Usually brief.").describe())

    def test_signals_are_reported(self) -> None:
        result = ps.classify("Never deploy on Fridays.")
        self.assertIn("never", [signal.lower() for signal in result.signals])


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def add(self, content: str, kind: str = "preference") -> str:
        return self.store.add_memory(
            content=content, kind=kind, source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def test_strength_is_recorded_and_read_back(self) -> None:
        memory_id = self.add("Never deploy on Fridays.")
        ps.record(self.store, memory_id, "Never deploy on Fridays.", at=NOW)
        self.assertEqual(ps.of(self.store, memory_id), ps.HARD)

    def test_recording_is_idempotent(self) -> None:
        memory_id = self.add("Prefers short answers.")
        ps.record(self.store, memory_id, "Prefers short answers.", at=NOW)
        ps.record(self.store, memory_id, "Prefers short answers.", at=NOW)
        self.assertEqual(ps.report(self.store)["recorded"], 1)

    def test_an_unrecorded_memory_reports_nothing(self) -> None:
        self.assertIsNone(ps.of(self.store, "no-such-id"))

    def test_the_report_counts_each_strength(self) -> None:
        for content in ("Never deploy.", "Prefers short answers.", "Usually brief."):
            memory_id = self.add(content)
            ps.record(self.store, memory_id, content, at=NOW)
        described = ps.report(self.store)
        self.assertEqual(described[ps.HARD], 1)
        self.assertEqual(described[ps.FIRM], 1)
        self.assertEqual(described[ps.SOFT], 1)

    def test_hard_rules_lists_only_hard_ones(self) -> None:
        hard = self.add("Never deploy on Fridays.")
        soft = self.add("Usually brief.")
        ps.record(self.store, hard, "Never deploy on Fridays.", at=NOW)
        ps.record(self.store, soft, "Usually brief.", at=NOW)
        listed = ps.hard_rules(self.store)
        self.assertEqual([item["memory_id"] for item in listed], [hard])

    def test_an_archived_hard_rule_is_not_listed(self) -> None:
        memory_id = self.add("Never deploy on Fridays.")
        ps.record(self.store, memory_id, "Never deploy on Fridays.", at=NOW)
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'archived' WHERE id = ?", (memory_id,))
        self.assertEqual(ps.hard_rules(self.store), [])


if __name__ == "__main__":
    unittest.main()
