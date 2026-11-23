"""Resolve an entity's names to one identity before the fact is stored (#38)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import aliases, entity_resolve
from cortex.store import CortexStore


class MentionTests(unittest.TestCase):
    def test_a_host_shaped_token_is_a_mention(self) -> None:
        names = [name for name, _ in entity_resolve.mentions("CT116 runs the service.")]
        self.assertIn("CT116", names)

    def test_a_descriptor_is_a_mention(self) -> None:
        pairs = entity_resolve.mentions("The Proxmox box holds the containers.")
        self.assertIn(("Proxmox", "descriptor"), pairs)

    def test_a_bare_number_is_not_a_mention(self) -> None:
        names = [name for name, _ in entity_resolve.mentions("It uses 8100 for the port.")]
        self.assertNotIn("8100", names)

    def test_a_weekday_is_not_a_mention(self) -> None:
        names = [name for name, _ in entity_resolve.mentions("Monday is when it runs.")]
        self.assertNotIn("Monday", names)

    def test_duplicates_collapse(self) -> None:
        names = [name for name, _ in entity_resolve.mentions("CT116 and CT116 again.")]
        self.assertEqual(names.count("CT116"), 1)

    def test_a_name_is_recorded_once_with_its_kind(self) -> None:
        pairs = entity_resolve.mentions("r630 runs CT116.")
        kinds = dict((name, kind) for name, kind in pairs)
        self.assertIn("r630", kinds)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.entity = "ent-proxmox"

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def entities(self, memory_id: str) -> list:
        raw = self.store._conn.execute(
            "SELECT entities_json FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        return json.loads(raw or "[]")

    def test_an_unknown_name_is_unresolved(self) -> None:
        result = entity_resolve.resolve(self.store, "CT116 runs the service.")
        self.assertTrue(result.unresolved)

    def test_a_known_name_resolves(self) -> None:
        aliases.add(self.store, self.entity, "CT116", source="manual")
        result = entity_resolve.resolve(self.store, "CT116 runs the service.")
        self.assertEqual(result.canonical, [self.entity])

    def test_the_three_names_of_one_box_resolve_together(self) -> None:
        """The item's own example: "the box", "CT116", "the Proxmox box"."""
        aliases.add(self.store, self.entity, "proxmox", source="manual")
        aliases.add(self.store, self.entity, "CT116", source="manual")
        text = "CT116 is the Proxmox box."
        result = entity_resolve.resolve(self.store, text)
        self.assertEqual(result.canonical, [self.entity])

    def test_a_descriptor_resolves_through_its_qualifier(self) -> None:
        aliases.add(self.store, self.entity, "proxmox", source="manual")
        result = entity_resolve.resolve(self.store, "The Proxmox box holds them.")
        self.assertEqual(result.canonical, [self.entity])

    def test_a_bare_descriptor_resolves_to_nothing(self) -> None:
        """“the box” alone names no machine, so it is not resolved by guesswork."""
        aliases.add(self.store, self.entity, "proxmox", source="manual")
        result = entity_resolve.resolve(self.store, "The box holds them.")
        self.assertEqual(result.canonical, [])

    def test_an_ambiguous_name_is_left_alone(self) -> None:
        aliases.add(self.store, "ent-a", "CT116", source="manual")
        aliases.add(self.store, "ent-b", "CT116", source="manual")
        result = entity_resolve.resolve(self.store, "CT116 runs the service.")
        self.assertEqual(result.canonical, [])
        self.assertTrue(result.mentions[0].ambiguous)

    def test_a_resolved_name_is_learned_as_an_alias(self) -> None:
        aliases.add(self.store, self.entity, "CT116", source="manual")
        result = entity_resolve.resolve(self.store, "CT116.lan answers.", learn=True)
        learned = [mention for mention in result.mentions if mention.learned]
        self.assertTrue(learned)

    def test_attach_writes_the_canonical_entity(self) -> None:
        aliases.add(self.store, self.entity, "CT116", source="manual")
        memory_id = self.memory("CT116 runs the service.")
        entity_resolve.attach(self.store, memory_id)
        self.assertEqual(self.entities(memory_id), [self.entity])

    def test_attach_unions_with_what_the_memory_already_had(self) -> None:
        aliases.add(self.store, self.entity, "CT116", source="manual")
        memory_id = self.store.add_memory(
            content="CT116 runs the service.", kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT", entities=["cortex"],
        )[0]
        entity_resolve.attach(self.store, memory_id)
        self.assertIn("cortex", self.entities(memory_id))
        self.assertIn(self.entity, self.entities(memory_id))

    def test_attach_never_removes_an_entity(self) -> None:
        memory_id = self.store.add_memory(
            content="Nothing resolvable here.", kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT", entities=["cortex"],
        )[0]
        entity_resolve.attach(self.store, memory_id)
        self.assertEqual(self.entities(memory_id), ["cortex"])

    def test_attach_of_a_missing_memory_is_quiet(self) -> None:
        result = entity_resolve.attach(self.store, "no-such-id")
        self.assertIn("canonical", result)

    def test_the_resolution_round_trips(self) -> None:
        aliases.add(self.store, self.entity, "CT116", source="manual")
        payload = entity_resolve.resolve(self.store, "CT116 answers.").as_dict()
        for key in ("canonical", "unresolved", "mentions"):
            self.assertIn(key, payload)

    def test_coverage_counts_memories_with_an_entity(self) -> None:
        aliases.add(self.store, self.entity, "CT116", source="manual")
        one = self.memory("CT116 runs the service.")
        self.memory("Something else entirely.")
        entity_resolve.attach(self.store, one)
        counts = entity_resolve.coverage(self.store)
        self.assertEqual(counts["memories"], 2)
        self.assertEqual(counts["with_entities"], 1)

    def test_the_report_includes_the_alias_table(self) -> None:
        aliases.add(self.store, self.entity, "CT116", source="manual")
        described = entity_resolve.report(self.store)
        self.assertIn("aliases", described)

    def test_the_audit_reports_the_resolved_share(self) -> None:
        aliases.add(self.store, self.entity, "CT116", source="manual")
        measured = entity_resolve.audit(
            self.store, ["CT116 runs it.", "an unrelated sentence here."]
        )
        self.assertGreaterEqual(measured["resolved"], 1)
        self.assertIn("resolved_share", measured)

    def test_an_audit_with_no_mentions_has_no_share(self) -> None:
        measured = entity_resolve.audit(self.store, ["nothing to see"])
        self.assertNotIn("resolved_share", measured)


if __name__ == "__main__":
    unittest.main()
