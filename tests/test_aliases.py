"""Aliases are remembered from usage and can be corrected (#39)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import aliases
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class KeyTests(unittest.TestCase):
    def test_case_and_spacing_fold(self) -> None:
        self.assertEqual(aliases.key("CT-116"), aliases.key("ct 116"))

    def test_a_leading_article_folds(self) -> None:
        self.assertEqual(aliases.key("the Proxmox box"), aliases.key("Proxmox box"))

    def test_surrounding_punctuation_folds(self) -> None:
        self.assertEqual(aliases.key("(CT116)"), aliases.key("CT116"))

    def test_similar_names_stay_distinct(self) -> None:
        self.assertNotEqual(aliases.key("ct1"), aliases.key("ct11"))

    def test_blank_input_is_blank(self) -> None:
        self.assertEqual(aliases.key("   "), "")


class AliasTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()


class AddTests(AliasTestCase):
    def test_an_alias_resolves_to_its_entity(self) -> None:
        aliases.add(self.store, "CT116", "the t3 box", at=NOW)
        self.assertEqual(aliases.resolve(self.store, "t3 box"), "CT116")

    def test_a_repeat_use_increments_hits(self) -> None:
        for _ in range(3):
            aliases.add(self.store, "CT116", "t3code", at=NOW)
        entries = aliases.aliases_for(self.store, "CT116")
        self.assertEqual(entries[0].hits, 3)

    def test_a_short_alias_is_refused(self) -> None:
        self.assertIsNone(aliases.add(self.store, "CT116", "t", at=NOW))

    def test_an_empty_entity_is_refused(self) -> None:
        self.assertIsNone(aliases.add(self.store, "", "t3code", at=NOW))

    def test_a_document_title_is_not_an_entity_name(self) -> None:
        """`entities_json` holds imported document headings too; they are not things you name."""
        long_title = "Cortex Brain Mechanics — Design Document With A Long Heading"
        self.assertIsNone(aliases.add(self.store, long_title, long_title, at=NOW))
        self.assertLessEqual(len(long_title), 200)

    def test_an_unknown_name_does_not_resolve(self) -> None:
        self.assertIsNone(aliases.resolve(self.store, "nothing here"))

    def test_resolution_prefers_the_habit(self) -> None:
        aliases.add(self.store, "CT116", "the box", at=NOW)
        for _ in range(3):
            aliases.add(self.store, "r630", "the box", at=NOW)
        self.assertEqual(aliases.resolve(self.store, "the box"), "r630")


class LearnTests(AliasTestCase):
    def test_learn_records_several_names(self) -> None:
        learned = aliases.learn(self.store, "CT116", ["t3code", "the t3 box", "ct116"])
        self.assertEqual(learned, 3)
        self.assertEqual(len(aliases.aliases_for(self.store, "CT116")), 3)

    def test_learn_skips_names_too_short(self) -> None:
        self.assertEqual(aliases.learn(self.store, "CT116", ["x"]), 0)

    def test_expand_returns_canonical_names(self) -> None:
        aliases.add(self.store, "CT116", "t3code", at=NOW)
        self.assertEqual(aliases.expand(self.store, ["t3code", "r630"]), ["CT116", "r630"])

    def test_expand_deduplicates(self) -> None:
        aliases.add(self.store, "CT116", "t3code", at=NOW)
        aliases.add(self.store, "CT116", "the t3 box", at=NOW)
        self.assertEqual(aliases.expand(self.store, ["t3code", "t3 box"]), ["CT116"])

    def test_from_memories_refuses_values_that_are_not_names(self) -> None:
        """The live entity lists carry vault paths, dates and section headings (#38)."""
        self.store.add_memory(
            content="A memory whose entity list is polluted.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            entities=["/tank", "2026-07-15", "2026-07-17 › HomeBox asset ID structure", "CT116"],
        )
        aliases.from_memories(self.store)
        self.assertIsNone(aliases.resolve(self.store, "/tank"))
        self.assertIsNone(aliases.resolve(self.store, "2026-07-15"))
        self.assertEqual(aliases.resolve(self.store, "CT116"), "CT116")

    def test_looks_like_a_name_judges_the_shapes(self) -> None:
        self.assertFalse(aliases.looks_like_a_name("/tank"))
        self.assertFalse(aliases.looks_like_a_name("2026-07-15"))
        self.assertFalse(aliases.looks_like_a_name("2026-07-17 › HomeBox asset ID structure"))
        self.assertTrue(aliases.looks_like_a_name("CT116"))
        self.assertTrue(aliases.looks_like_a_name("229 Scottwood Ave"))

    def test_from_memories_teaches_the_names_in_use(self) -> None:
        self.store.add_memory(
            content="The t3 box hosts the nightly build.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            entities=["CT116", "Proxmox"],
        )
        result = aliases.from_memories(self.store)
        self.assertEqual(result["memories"], 1)
        self.assertEqual(aliases.resolve(self.store, "CT116"), "CT116")
        self.assertEqual(aliases.resolve(self.store, "proxmox"), "Proxmox")


class EditTests(AliasTestCase):
    def test_remove_by_name(self) -> None:
        aliases.add(self.store, "CT116", "t3code", at=NOW)
        self.assertEqual(aliases.remove(self.store, "t3code"), 1)
        self.assertIsNone(aliases.resolve(self.store, "t3code"))

    def test_remove_only_for_one_entity(self) -> None:
        aliases.add(self.store, "CT116", "the box", at=NOW)
        aliases.add(self.store, "r630", "the box", at=NOW)
        self.assertEqual(aliases.remove(self.store, "the box", entity_id="r630"), 1)
        self.assertEqual(aliases.resolve(self.store, "the box"), "CT116")

    def test_merge_moves_aliases(self) -> None:
        aliases.learn(self.store, "t3", ["t3code", "t3 box"])
        result = aliases.merge(self.store, "t3", "CT116")
        self.assertEqual(result["moved"], 2)
        self.assertEqual(aliases.resolve(self.store, "t3code"), "CT116")
        self.assertEqual(aliases.entities(self.store)[0]["entity_id"], "CT116")

    def test_merge_adds_hits_when_the_alias_already_exists(self) -> None:
        aliases.add(self.store, "t3", "t3code", at=NOW)
        aliases.add(self.store, "CT116", "t3code", at=NOW)
        result = aliases.merge(self.store, "t3", "CT116")
        self.assertEqual(result["dropped"], 1)
        self.assertEqual(aliases.aliases_for(self.store, "CT116")[0].hits, 2)

    def test_merging_an_entity_into_itself_is_a_no_op(self) -> None:
        aliases.add(self.store, "CT116", "t3code", at=NOW)
        self.assertEqual(aliases.merge(self.store, "CT116", "ct116"),
                         {"moved": 0, "dropped": 0})


class ReportTests(AliasTestCase):
    def test_the_report_counts_entities_and_aliases(self) -> None:
        aliases.learn(self.store, "CT116", ["t3code", "t3 box"])
        aliases.add(self.store, "r630", "the host", at=NOW)
        described = aliases.report(self.store)
        self.assertEqual(described["entities"], 2)
        self.assertEqual(described["aliases"], 3)

    def test_a_shared_name_is_reported_as_ambiguous(self) -> None:
        aliases.add(self.store, "CT116", "the box", at=NOW)
        aliases.add(self.store, "r630", "the box", at=NOW)
        # The report speaks in folded keys, which is what the lookup actually collides on.
        self.assertIn("box", aliases.report(self.store)["ambiguous"])

    def test_an_empty_table_reports_zero(self) -> None:
        described = aliases.report(self.store)
        self.assertEqual((described["entities"], described["aliases"]), (0, 0))


class IntegrationTests(AliasTestCase):
    def test_a_learned_alias_changes_the_entity_match_set(self) -> None:
        """The point of the table: two spellings of one box now find the same memories."""
        memory_id = self.store.add_memory(
            content="The t3 box runs the nightly build.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            entities=["CT116"],
        )[0]
        from cortex import entity_anchor

        self.assertEqual(
            entity_anchor.entity_memory_ids(self.store, ["t3 box"]), []
        )
        aliases.add(self.store, "CT116", "t3 box", at=NOW)
        expanded = aliases.expand(self.store, ["t3 box"])
        self.assertIn(memory_id, entity_anchor.entity_memory_ids(self.store, expanded))


if __name__ == "__main__":
    unittest.main()
