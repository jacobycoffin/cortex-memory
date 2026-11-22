"""Forgetting everything about one entity, safely and auditably (#95)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import forget_entity as fe
from cortex import legal_hold as lh
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class EntityTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, kind: str = "semantic", entities: list[str] | None = None) -> str:
        created = self.store.add_memory(
            content=content,
            kind=kind,
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        memory_id = created[0] if isinstance(created, tuple) else created
        if entities:
            import json

            with self.store.transaction() as conn:
                conn.execute(
                    "UPDATE memories SET entities_json = ? WHERE id = ?",
                    (json.dumps(entities), memory_id),
                )
        return memory_id


class MatchTests(EntityTestCase):
    def test_a_content_mention_matches(self) -> None:
        memory_id = self.memory("Acme Corp signed the MSA in March.")
        self.assertEqual([m["memory_id"] for m in fe.matches(self.store, "Acme Corp")], [memory_id])

    def test_a_structured_entity_matches_even_without_a_text_mention(self) -> None:
        memory_id = self.memory("Renewal terms discussed.", entities=["Acme Corp"])
        self.assertIn(memory_id, [m["memory_id"] for m in fe.matches(self.store, "Acme Corp")])

    def test_a_graph_neighbour_is_not_matched_by_default(self) -> None:
        """A neighbour of a memory about the client is not itself about the client."""
        seed = self.memory("Acme Corp is the client for the retainer.")
        neighbour = self.memory("Invoicing cadence is monthly, net 30.")
        self.store.add_edge(seed, neighbour, "relates_to", evidence_type="fixture")
        ids = [m["memory_id"] for m in fe.matches(self.store, "Acme Corp")]
        self.assertIn(seed, ids)
        self.assertNotIn(neighbour, ids)

    def test_a_graph_neighbour_can_be_included_deliberately(self) -> None:
        seed = self.memory("Acme Corp is the client for the retainer.")
        neighbour = self.memory("Invoicing cadence is monthly, net 30.")
        self.store.add_edge(seed, neighbour, "relates_to", evidence_type="fixture")
        rows = fe.matches(self.store, "Acme Corp", include_neighbours=True)
        by_id = {row["memory_id"]: row for row in rows}
        self.assertEqual(by_id[neighbour]["evidence"], "connected only")
        self.assertEqual(by_id[seed]["evidence"], "matched")

    def test_boundaries_are_respected(self) -> None:
        """`Acme` must not match `AcmeCorp` or `Acmeson`."""
        wanted = self.memory("Acme signed the contract.")
        self.memory("AcmeCorp signed a different contract.")
        self.memory("Acmeson is a different vendor entirely.")
        self.assertEqual([m["memory_id"] for m in fe.matches(self.store, "Acme")], [wanted])

    def test_matching_is_case_insensitive(self) -> None:
        memory_id = self.memory("ACME signed the contract.")
        self.assertIn(memory_id, [m["memory_id"] for m in fe.matches(self.store, "acme")])

    def test_archived_memories_are_not_matched(self) -> None:
        memory_id = self.memory("Acme signed the contract.")
        self.store.set_state(memory_id, "archived", reason="test")
        self.assertEqual(fe.matches(self.store, "Acme"), [])

    def test_an_empty_entity_matches_nothing(self) -> None:
        self.memory("Acme signed the contract.")
        self.assertEqual(fe.matches(self.store, "   "), [])

    def test_an_unknown_entity_matches_nothing(self) -> None:
        self.memory("Acme signed the contract.")
        self.assertEqual(fe.matches(self.store, "Northwind Traders"), [])


class PreviewTests(EntityTestCase):
    def test_a_preview_returns_an_envelope_and_changes_nothing(self) -> None:
        first = self.memory("Acme Corp is the client.")
        second = self.memory("Acme Corp renewal is in March.")
        preview = fe.preview(self.store, "Acme Corp", at=NOW)
        self.assertEqual(set(preview.ids), {first, second})
        self.assertEqual(len(preview.envelope), 64)
        self.assertEqual(self.store.get_memory(first)["state"], "active")

    def test_the_description_lists_what_would_go(self) -> None:
        memory_id = self.memory("Acme Corp is the client.")
        described = fe.preview(self.store, "Acme Corp", at=NOW).describe()
        self.assertIn(memory_id[:8], described)
        self.assertIn("Acme Corp", described)


class ApplyTests(EntityTestCase):
    def test_applying_the_envelope_tombstones_the_matches(self) -> None:
        first = self.memory("Acme Corp is the client.")
        preview = fe.preview(self.store, "Acme Corp", at=NOW)
        applied = fe.apply(self.store, preview.envelope, preview.ids, at=NOW)
        self.assertTrue(applied.ok, applied.summary)
        self.assertEqual(self.store.get_memory(first)["state"], "tombstoned")

    def test_an_id_outside_the_entity_preview_is_refused(self) -> None:
        self.memory("Acme Corp is the client.")
        other = self.memory("Northwind Traders is a different client.")
        preview = fe.preview(self.store, "Acme Corp", at=NOW)
        applied = fe.apply(self.store, preview.envelope, preview.ids + [other], at=NOW)
        self.assertFalse(applied.ok)
        self.assertIn("not in the previewed set", applied.refused)

    def test_a_held_memory_blocks_the_whole_entity_forget(self) -> None:
        """Broad requests are exactly where a contract clause gets swept up by accident."""
        self.memory("Acme Corp is the client.")
        contract = self.memory("Acme Corp contract clause 7.2 governs renewal.")
        lh.place(self.store, [contract], reason="contract")
        preview = fe.preview(self.store, "Acme Corp", at=NOW)
        applied = fe.apply(self.store, preview.envelope, preview.ids, at=NOW, allow_protected=True)
        self.assertFalse(applied.ok)
        self.assertIn("legal hold", applied.refused)
        self.assertEqual(self.store.get_memory(contract)["state"], "active")

    def test_an_entity_forget_is_receipted(self) -> None:
        from cortex import prune_receipts as rc

        self.memory("Acme Corp is the client.")
        preview = fe.preview(self.store, "Acme Corp", at=NOW)
        fe.apply(self.store, preview.envelope, preview.ids, at=NOW)
        receipts = rc.recent(self.store, kind="semantic_forget")
        self.assertEqual(len(receipts), 1)
        self.assertIn("entity:Acme Corp", receipts[0].summary)

    def test_a_forgotten_entity_memory_can_be_restored(self) -> None:
        from cortex.restore_ladder import restore

        memory_id = self.memory("Acme Corp is the client.")
        preview = fe.preview(self.store, "Acme Corp", at=NOW)
        fe.apply(self.store, preview.envelope, preview.ids, at=NOW)
        self.assertTrue(restore(self.store, memory_id).restored)
        self.assertEqual(self.store.get_memory(memory_id)["state"], "active")


if __name__ == "__main__":
    unittest.main()
