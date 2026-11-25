"""The receipt for anything destructive (#180): computed before, written with the change, verifiable after."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import delete_proof
from cortex.store import CortexStore


class Broken:
    _conn = None


class ProofTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str = "the proxmox box runs 8.4") -> str:
        return self.store.add_memory(
            content=text, kind="semantic", source_type="user_turn", source_category="USER_STATED",
        )[0]

    def state(self, memory_id: str) -> str:
        return str(self.store._conn.execute(
            "SELECT state FROM memories WHERE id = ?", (memory_id,)).fetchone()[0])

    def content(self, memory_id: str) -> str:
        return str(self.store._conn.execute(
            "SELECT content FROM memories WHERE id = ?", (memory_id,)).fetchone()[0])

    def snapshot(self) -> tuple:
        tables = tuple(sorted(r["name"] for r in self.store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")))
        counts = tuple(int(self.store._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                       for t in tables)
        return tables, counts

    # -- the proof comes first, and refuses without a reason -------------------------

    def test_a_destructive_action_without_a_reason_is_refused(self) -> None:
        proof = delete_proof.plan(self.store, self.memory(), action=delete_proof.ARCHIVE)
        self.assertFalse(proof.ok)
        self.assertIn("reason", proof.refusal)

    def test_an_unknown_action_lists_the_choices(self) -> None:
        proof = delete_proof.plan(self.store, self.memory(), action="obliterate", reason="why not")
        self.assertFalse(proof.ok)
        for action in delete_proof.ACTIONS:
            self.assertIn(action, proof.refusal)

    def test_an_unknown_memory_is_named(self) -> None:
        proof = delete_proof.plan(self.store, "no-such-id", reason="test")
        self.assertFalse(proof.ok)
        self.assertIn("unknown memory", proof.refusal)

    def test_planning_writes_nothing(self) -> None:
        memory_id = self.memory()
        before = self.snapshot()
        delete_proof.plan(self.store, memory_id, action=delete_proof.PURGE, reason="leaked value")
        self.assertEqual(before, self.snapshot())

    def test_the_proof_counts_every_table_that_points_at_the_memory(self) -> None:
        memory_id = self.memory()
        proof = delete_proof.plan(self.store, memory_id, reason="test")
        self.assertTrue(proof.ok)
        self.assertIn("memories", proof.affects)
        self.assertEqual(proof.affects["memories"], 1)
        self.assertGreater(proof.keeps["tables_with_memory_id"], 5,
                           "the reference list must be discovered, not assumed")

    def test_a_purge_says_it_is_not_reversible_and_an_archive_says_it_is(self) -> None:
        memory_id = self.memory()
        archive = delete_proof.plan(self.store, memory_id, action=delete_proof.ARCHIVE, reason="tidying")
        purge = delete_proof.plan(self.store, memory_id, action=delete_proof.PURGE, reason="leaked")
        self.assertTrue(archive.reversible)
        self.assertFalse(purge.reversible)
        self.assertTrue(purge.render()[-1].endswith("NOT reversible"))

    def test_the_proof_states_what_is_lost(self) -> None:
        memory_id = self.memory()
        purge = delete_proof.plan(self.store, memory_id, action=delete_proof.PURGE, reason="leaked")
        self.assertIn("content", purge.lost)
        self.assertIn("edge", purge.survives)

    def test_the_receipt_names_tables_and_counts_but_never_content(self) -> None:
        memory_id = self.memory("the deploy token is hunter2")
        proof = delete_proof.plan(self.store, memory_id, action=delete_proof.PURGE, reason="leaked secret")
        rendered = "\n".join(proof.render())
        self.assertIn("memories", rendered)
        self.assertNotIn("hunter2", rendered, "a receipt that quotes the row is a second copy of it")
        self.assertIn("leaked secret", rendered)

    # -- applying --------------------------------------------------------------------

    def test_apply_without_consent_changes_nothing(self) -> None:
        memory_id = self.memory()
        proof = delete_proof.plan(self.store, memory_id, action=delete_proof.PURGE, reason="leaked")
        result = delete_proof.apply(self.store, proof)
        self.assertFalse(result["ok"])
        self.assertEqual(self.content(memory_id), "the proxmox box runs 8.4")
        self.assertEqual(delete_proof.receipts(self.store), [], "no receipt, no change")

    def test_an_archive_marks_the_state_and_writes_one_receipt(self) -> None:
        memory_id = self.memory()
        proof = delete_proof.plan(self.store, memory_id, action=delete_proof.ARCHIVE, reason="tidying")
        result = delete_proof.apply(self.store, proof, consent=True)
        self.assertTrue(result["ok"], result.get("reason"))
        self.assertEqual(self.state(memory_id), "archived")
        self.assertEqual(self.content(memory_id), "the proxmox box runs 8.4", "an archive loses nothing")
        self.assertEqual(len(delete_proof.receipts(self.store)), 1)

    def test_a_purge_removes_the_content_and_keeps_the_row(self) -> None:
        memory_id = self.memory("the deploy token is hunter2")
        proof = delete_proof.plan(self.store, memory_id, action=delete_proof.PURGE, reason="leaked secret")
        result = delete_proof.apply(self.store, proof, consent=True)
        self.assertTrue(result["ok"], result.get("reason"))
        self.assertEqual(self.content(memory_id), delete_proof.PURGE_MARKER)
        self.assertEqual(self.state(memory_id), "tombstoned")
        self.assertIsNotNone(self.store._conn.execute(
            "SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone(), "the row survives")

    def test_a_failed_change_writes_no_receipt(self) -> None:
        memory_id = self.memory()
        proof = delete_proof.plan(self.store, memory_id, action=delete_proof.ARCHIVE, reason="tidying")
        proof.memory_id = "gone-from-under-us"
        result = delete_proof.apply(self.store, proof, consent=True)
        self.assertFalse(result["ok"])
        self.assertEqual(delete_proof.receipts(self.store), [])

    # -- checking a receipt later ----------------------------------------------------

    def test_verify_confirms_the_change_happened(self) -> None:
        memory_id = self.memory()
        proof = delete_proof.plan(self.store, memory_id, action=delete_proof.ARCHIVE, reason="tidying")
        receipt_id = delete_proof.apply(self.store, proof, consent=True)["receipt_id"]
        checked = delete_proof.verify(self.store, receipt_id)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(checked["memory_id"], memory_id)
        self.assertTrue(checked["reversible"])

    def test_verify_notices_when_the_store_no_longer_matches(self) -> None:
        memory_id = self.memory()
        proof = delete_proof.plan(self.store, memory_id, action=delete_proof.ARCHIVE, reason="tidying")
        receipt_id = delete_proof.apply(self.store, proof, consent=True)["receipt_id"]
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'active' WHERE id = ?", (memory_id,))
        checked = delete_proof.verify(self.store, receipt_id)
        self.assertFalse(checked["ok"])
        self.assertIn("does not match", checked["reason"])

    def test_verify_of_an_unknown_receipt_is_named(self) -> None:
        checked = delete_proof.verify(self.store, "not-a-receipt")
        self.assertFalse(checked["ok"])
        self.assertIn("no receipt", checked["reason"])

    def test_a_receipt_records_the_reason_it_was_given(self) -> None:
        memory_id = self.memory()
        proof = delete_proof.plan(self.store, memory_id, action=delete_proof.ARCHIVE, reason="GDPR request")
        receipt_id = delete_proof.apply(self.store, proof, consent=True)["receipt_id"]
        self.assertEqual(delete_proof.verify(self.store, receipt_id)["reason_given"], "GDPR request")

    def test_a_broken_store_is_refused_not_raised(self) -> None:
        proof = delete_proof.plan(Broken(), "abc", action=delete_proof.ARCHIVE, reason="test")
        self.assertFalse(proof.ok)
        self.assertIn("could not be read", proof.refusal)

    def test_the_receipt_json_is_checkable(self) -> None:
        memory_id = self.memory()
        proof = delete_proof.plan(self.store, memory_id, action=delete_proof.ARCHIVE, reason="tidying")
        delete_proof.apply(self.store, proof, consent=True)
        receipt = delete_proof.receipts(self.store)[0]
        affects = json.loads(receipt["affects_json"])
        self.assertEqual(affects["memories"], 1)
        self.assertEqual(receipt["action"], delete_proof.ARCHIVE)


if __name__ == "__main__":
    unittest.main()
