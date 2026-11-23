"""Entities as records with a page, not just strings on memories (#49)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import aliases, entity_nodes
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class ClassifyTests(unittest.TestCase):
    def test_a_known_person_is_a_person(self) -> None:
        self.assertEqual(entity_nodes.classify("Jacoby"), "person")

    def test_a_project_is_a_project(self) -> None:
        self.assertEqual(entity_nodes.classify("duelist-road"), "project")
        self.assertEqual(entity_nodes.classify("cortex-memory"), "project")

    def test_a_service_is_a_service(self) -> None:
        self.assertEqual(entity_nodes.classify("nginx"), "service")

    def test_a_host_shaped_name_is_a_device(self) -> None:
        self.assertEqual(entity_nodes.classify("CT116"), "device")
        self.assertEqual(entity_nodes.classify("r630"), "device")

    def test_an_email_is_an_account(self) -> None:
        self.assertEqual(entity_nodes.classify("jacoby@example.com"), "account")

    def test_a_street_address_is_a_place(self) -> None:
        self.assertEqual(entity_nodes.classify("229 Scottwood Ave"), "place")

    def test_an_unclear_name_stays_unknown(self) -> None:
        self.assertEqual(entity_nodes.classify("HomeBox asset ID structure"), "unknown")
        self.assertEqual(entity_nodes.classify(""), "unknown")

    def test_every_kind_is_in_the_vocabulary(self) -> None:
        for name in ("Jacoby", "cortex", "nginx", "CT116", "a@b.co", "1 Main St", "???"):
            self.assertIn(entity_nodes.classify(name), entity_nodes.KINDS)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, entities=None) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT", entities=entities,
        )[0]

    def test_create_returns_a_node_with_a_classified_kind(self) -> None:
        node = entity_nodes.create(self.store, "CT116", "CT116", at=NOW)
        self.assertIsNotNone(node)
        self.assertEqual(node.kind, "device")

    def test_create_is_idempotent(self) -> None:
        first = entity_nodes.create(self.store, "CT116", "CT116", at=NOW)
        second = entity_nodes.create(self.store, "CT116", "something else", at=NOW)
        self.assertEqual(first.entity_id, second.entity_id)
        self.assertEqual(second.canonical_name, "CT116")

    def test_an_explicit_kind_wins_over_the_lexicon(self) -> None:
        node = entity_nodes.create(self.store, "box-7", "box-7", kind="service", at=NOW)
        self.assertEqual(node.kind, "service")

    def test_an_invalid_kind_falls_back_to_classification(self) -> None:
        node = entity_nodes.create(self.store, "CT116", "CT116", kind="spaceship", at=NOW)
        self.assertEqual(node.kind, "device")

    def test_a_missing_identifier_creates_nothing(self) -> None:
        self.assertIsNone(entity_nodes.create(self.store, "", "x"))

    def test_a_person_can_correct_the_kind(self) -> None:
        entity_nodes.create(self.store, "65VZZ72", "65VZZ72", at=NOW)
        corrected = entity_nodes.set_kind(self.store, "65VZZ72", "account", at=NOW)
        self.assertEqual(corrected.kind, "account")

    def test_an_unknown_kind_is_refused(self) -> None:
        entity_nodes.create(self.store, "ct116", "ct116", at=NOW)
        self.assertIsNone(entity_nodes.set_kind(self.store, "ct116", "nonsense"))
        self.assertEqual(entity_nodes.get(self.store, "ct116").kind, "device")

    def test_a_digit_leading_code_is_not_assumed_to_be_a_device(self) -> None:
        """A serial number is not a hostname; guessing a device here would be a wrong kind."""
        self.assertEqual(entity_nodes.classify("65VZZ72"), "unknown")

    def test_a_summary_is_stored_once_and_kept(self) -> None:
        entity_nodes.create(self.store, "CT116", "CT116", at=NOW)
        entity_nodes.set_summary(self.store, "CT116", "the T3 container", at=NOW)
        node = entity_nodes.get(self.store, "CT116")
        self.assertEqual(node.summary, "the T3 container")

    def test_sync_gives_known_alias_names_nodes(self) -> None:
        aliases.add(self.store, "CT116", "CT116", source="manual")
        aliases.add(self.store, "cortex", "cortex", source="manual")
        described = entity_nodes.sync(self.store, at=NOW)
        self.assertEqual(described["created"], 2)
        self.assertEqual(entity_nodes.get(self.store, "CT116").kind, "device")

    def test_sync_is_idempotent(self) -> None:
        aliases.add(self.store, "CT116", "CT116", source="manual")
        entity_nodes.sync(self.store, at=NOW)
        self.assertEqual(entity_nodes.sync(self.store, at=NOW)["created"], 0)

    def test_sync_also_reads_entity_lists_on_memories(self) -> None:
        self.memory("The box hosts the service.", entities=["Proxmox"])
        described = entity_nodes.sync(self.store, at=NOW)
        self.assertGreaterEqual(described["created"], 1)
        self.assertEqual(entity_nodes.get(self.store, "Proxmox").kind, "service")

    def test_a_page_carries_the_aliases_and_the_facts(self) -> None:
        aliases.add(self.store, "CT116", "CT116", source="manual")
        aliases.add(self.store, "CT116", "the t3 box", source="usage")
        entity_nodes.sync(self.store, at=NOW)
        self.memory("CT116 runs the nightly build.", entities=["CT116"])
        page = entity_nodes.page(self.store, "CT116")
        self.assertIn("the t3 box", page["aliases"])
        self.assertGreaterEqual(page["fact_count"], 1)
        self.assertTrue(page["facts"])

    def test_a_page_reports_when_the_entity_was_seen(self) -> None:
        aliases.add(self.store, "CT116", "CT116", source="manual")
        entity_nodes.sync(self.store, at=NOW)
        self.memory("CT116 runs the nightly build.", entities=["CT116"])
        page = entity_nodes.page(self.store, "CT116")
        self.assertTrue(page["first_seen"])
        self.assertTrue(page["last_seen"])

    def test_a_page_lists_neighbours_it_shares_a_memory_with(self) -> None:
        aliases.add(self.store, "CT116", "CT116", source="manual")
        entity_nodes.sync(self.store, at=NOW)
        self.memory("CT116 serves cortex for Jacoby.", entities=["CT116", "cortex"])
        page = entity_nodes.page(self.store, "CT116")
        self.assertIn("cortex", page["neighbours"])

    def test_a_page_for_an_unknown_entity_is_none(self) -> None:
        self.assertIsNone(entity_nodes.page(self.store, "no-such-entity"))

    def test_the_report_counts_nodes_by_kind(self) -> None:
        aliases.add(self.store, "CT116", "CT116", source="manual")
        aliases.add(self.store, "Jacoby", "Jacoby", source="manual")
        entity_nodes.sync(self.store, at=NOW)
        described = entity_nodes.report(self.store)
        self.assertEqual(described["nodes"], 2)
        self.assertEqual(described["by_kind"], {"device": 1, "person": 1})

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(entity_nodes.sync(self.store)["created"], 0)
        self.assertEqual(entity_nodes.report(self.store)["nodes"], 0)

    def test_the_node_round_trips(self) -> None:
        node = entity_nodes.create(self.store, "CT116", "CT116", summary="s", at=NOW)
        payload = node.as_dict()
        for key in ("entity_id", "canonical_name", "kind", "summary", "created_at", "updated_at"):
            self.assertIn(key, payload)
        self.assertIsInstance(json.dumps(payload), str)


if __name__ == "__main__":
    unittest.main()
