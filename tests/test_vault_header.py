"""Take the vault header out of stored content (#223)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import vault_header
from cortex.store import CortexStore

DELIMITED = ("Vault note: AdGuard Home Path: homelab/AdGuard Home.md "
             "Section: AdGuard Home › Notes - Router DHCP DNS points clients to AdGuard Home.")
AMBIGUOUS = ("Vault note: Bazarr Path: homelab/Bazarr.md Section: Bazarr "
             "**Container**: [[homelab/Infrastructure|CT100]] (192.168.1.18)")


class ParseTests(unittest.TestCase):
    def test_a_delimited_header_is_split_exactly(self) -> None:
        parsed = vault_header.parse(DELIMITED)
        self.assertTrue(parsed.matched)
        self.assertEqual(parsed.title, "AdGuard Home")
        self.assertEqual(parsed.path, "homelab/AdGuard Home.md")
        self.assertEqual(parsed.section, "AdGuard Home › Notes")
        self.assertTrue(parsed.body.startswith("Router DHCP DNS"))
        self.assertEqual(parsed.mode, vault_header.DELIMITED)

    def test_a_header_without_a_delimiter_does_not_invent_a_body(self) -> None:
        parsed = vault_header.parse(AMBIGUOUS)
        self.assertTrue(parsed.matched)
        self.assertEqual(parsed.body, "")
        self.assertEqual(parsed.mode, "")
        self.assertFalse(parsed.changed)

    def test_a_plain_memory_does_not_match(self) -> None:
        self.assertFalse(vault_header.parse("The dashboard listens on 8100.").matched)

    def test_a_memory_that_merely_mentions_a_vault_note_is_untouched(self) -> None:
        """Only a leading header is scaffolding; the phrase in the middle is content."""
        text = "I wrote a Vault note: about the box Path: somewhere"
        self.assertFalse(vault_header.parse(text).matched)

    def test_extra_whitespace_in_the_title_is_collapsed(self) -> None:
        parsed = vault_header.parse(
            "Vault note:   Bazarr   Path: a.md Section: Bazarr - Subtitle automation.")
        self.assertEqual(parsed.title, "Bazarr")

    def test_the_prefix_mode_strips_only_the_deterministic_part(self) -> None:
        parsed = vault_header.parse(AMBIGUOUS, mode=vault_header.PREFIX)
        self.assertEqual(parsed.mode, vault_header.PREFIX)
        self.assertTrue(parsed.body.startswith("Bazarr **Container**"))

    def test_a_body_containing_a_dash_separator_keeps_the_rest_intact(self) -> None:
        parsed = vault_header.parse(
            "Vault note: X Path: x.md Section: X - first part - second part")
        self.assertEqual(parsed.body, "first part - second part")


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str) -> str:
        return self.store.add_memory(
            content=text, kind="semantic", source_type="vault", source_category="AGENT_INFERENCE",
        )[0]

    def content(self, memory_id: str) -> str:
        row = self.store._conn.execute(
            "SELECT content FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return str(row["content"])

    def test_a_dry_run_changes_nothing(self) -> None:
        memory_id = self.memory(DELIMITED)
        found = vault_header.strip(self.store)
        self.assertEqual(found.rewritten, 1)
        self.assertFalse(found.applied)
        self.assertEqual(self.content(memory_id), DELIMITED)

    def test_an_applied_strip_rewrites_the_content(self) -> None:
        memory_id = self.memory(DELIMITED)
        vault_header.strip(self.store, apply=True)
        self.assertTrue(self.content(memory_id).startswith("Router DHCP DNS"))

    def test_the_header_is_kept_as_metadata(self) -> None:
        memory_id = self.memory(DELIMITED)
        vault_header.strip(self.store, apply=True)
        stored = vault_header.header_of(self.store, memory_id)
        self.assertEqual(stored["title"], "AdGuard Home")
        self.assertEqual(stored["path"], "homelab/AdGuard Home.md")

    def test_the_original_is_kept_verbatim(self) -> None:
        memory_id = self.memory(DELIMITED)
        vault_header.strip(self.store, apply=True)
        self.assertEqual(vault_header.header_of(self.store, memory_id)["original"], DELIMITED)

    def test_a_strip_can_be_undone(self) -> None:
        memory_id = self.memory(DELIMITED)
        vault_header.strip(self.store, apply=True)
        self.assertTrue(vault_header.restore(self.store, memory_id))
        self.assertEqual(self.content(memory_id), DELIMITED)

    def test_restoring_something_never_stripped_is_refused(self) -> None:
        memory_id = self.memory("a plain memory")
        self.assertFalse(vault_header.restore(self.store, memory_id))

    def test_the_second_run_is_a_no_op(self) -> None:
        self.memory(DELIMITED)
        vault_header.strip(self.store, apply=True)
        found = vault_header.strip(self.store, apply=True)
        self.assertEqual(found.rewritten, 0)
        self.assertEqual(found.examined, 0)  # the row no longer starts with the header

    def test_an_ambiguous_row_is_left_alone_by_default(self) -> None:
        memory_id = self.memory(AMBIGUOUS)
        found = vault_header.strip(self.store, apply=True)
        self.assertEqual(found.ambiguous, 1)
        self.assertEqual(found.rewritten, 0)
        self.assertEqual(self.content(memory_id), AMBIGUOUS)

    def test_the_prefix_mode_also_rewrites_ambiguous_rows(self) -> None:
        memory_id = self.memory(AMBIGUOUS)
        found = vault_header.strip(self.store, mode=vault_header.PREFIX, apply=True)
        self.assertEqual(found.rewritten, 1)
        self.assertTrue(self.content(memory_id).startswith("Bazarr **Container**"))

    def test_a_header_whose_body_is_empty_falls_back_to_the_title(self) -> None:
        memory_id = self.memory("Vault note: Empty Note Path: a/Empty Note.md Section: Empty Note - ")
        vault_header.strip(self.store, apply=True)
        self.assertEqual(self.content(memory_id), "Empty Note")

    def test_nothing_is_ever_blanked(self) -> None:
        memory_id = self.memory("Vault note: Bare Path: a/Bare.md Section: Bare - ")
        vault_header.strip(self.store, apply=True)
        self.assertTrue(self.content(memory_id).strip())

    def test_a_plain_memory_is_not_touched(self) -> None:
        memory_id = self.memory("The dashboard listens on 8100.")
        found = vault_header.strip(self.store, apply=True)
        self.assertEqual(found.examined, 0)
        self.assertEqual(self.content(memory_id), "The dashboard listens on 8100.")

    def test_the_report_states_the_shape_of_the_problem(self) -> None:
        self.memory(DELIMITED)
        self.memory(AMBIGUOUS)
        self.memory("a plain memory")
        described = vault_header.report(self.store)
        self.assertEqual(described["active"], 3)
        self.assertEqual(described["with_header"], 2)
        self.assertEqual(described["delimited"], 1)
        self.assertEqual(described["ambiguous"], 1)

    def test_the_sweep_says_what_it_would_do(self) -> None:
        self.memory(DELIMITED)
        self.assertIn("would rewrite 1", vault_header.strip(self.store).describe())
        self.assertIn("saving", vault_header.strip(self.store).as_dict()["sentence"])


if __name__ == "__main__":
    unittest.main()
