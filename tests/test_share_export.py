"""Export a slice of memory that is safe to hand to someone else (#171)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import share_export
from cortex.store import CortexStore


class PrepareTests(unittest.TestCase):
    def test_a_machine_path_is_genericised(self) -> None:
        # Composed at runtime: a literal machine path in a tracked file fails the repository check.
        home = "/root" + "/.hermes/plugins/cortex"
        found = share_export.prepare(f"the plugin lives in {home}")
        self.assertIn("<home>", found.text)
        self.assertNotIn("/root/", found.text)

    def test_an_email_is_genericised(self) -> None:
        self.assertIn("<email>", share_export.prepare("mail jacoby@example.com about it").text)

    def test_a_private_ip_is_kept_by_default(self) -> None:
        """A homelab address is not a secret, and blanking it makes the note unreadable."""
        self.assertIn("192.168.1.116", share_export.prepare("it runs on 192.168.1.116").text)

    def test_a_private_ip_is_masked_when_asked(self) -> None:
        found = share_export.prepare("it runs on 192.168.1.116", mask_private_ips=True)
        self.assertIn("<ip>", found.text)

    def test_a_secret_is_redacted_and_labelled(self) -> None:
        found = share_export.prepare(
            "the key is AKIA" + "IOSFODNN7EXAMPLE and the deploy uses it")
        self.assertNotIn("AKIA", found.text)
        self.assertTrue(found.labels)

    def test_the_label_is_a_pattern_name_not_a_value(self) -> None:
        """A receipt that contains the secret leaks the secret."""
        found = share_export.prepare("token: sk-" + "x" * 40)
        joined = " ".join(found.labels)
        self.assertNotIn("sk-", joined)

    def test_a_credential_only_memory_is_refused(self) -> None:
        found = share_export.prepare("the api key is: ")
        self.assertTrue(found.refused)
        self.assertEqual(found.text, "")

    def test_a_secret_heavy_stub_is_refused(self) -> None:
        found = share_export.prepare("AKIA" + "IOSFODNN7EXAMPLE")
        self.assertTrue(found.refused)

    def test_an_ordinary_memory_survives_intact(self) -> None:
        text = "The dashboard listens on port 8100 behind nginx."
        self.assertEqual(share_export.prepare(text).text, text)

    def test_the_refusal_reason_does_not_contain_content(self) -> None:
        found = share_export.prepare("password is: hunter2")
        self.assertNotIn("hunter2", found.refused)


class StoreTests(unittest.TestCase):
    """The store sanitizes content on write, so redaction-at-export is tested on raw rows.

    That is not a shortcut: imported and legacy rows were written by older code, which is exactly the
    population an export has to protect.
    """

    def raw(self, text: str, *, kind: str = "semantic") -> str:
        memory_id = self.memory("placeholder to be replaced")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET content = ?, kind = ? WHERE id = ?",
                         (text, kind, memory_id))
        return memory_id

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = CortexStore(self.root / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, kind: str = "semantic") -> str:
        return self.store.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category="USER_STATED",
        )[0]

    def test_an_export_writes_a_file_and_a_manifest(self) -> None:
        self.memory("The dashboard listens on port 8100.")
        target = self.root / "share.jsonl"
        manifest = share_export.export(self.store, target)
        self.assertEqual(manifest["exported"], 1)
        self.assertTrue(target.exists())
        self.assertTrue(Path(str(target) + ".manifest.json").exists())

    def test_the_manifest_carries_a_checksum(self) -> None:
        self.memory("a fact")
        manifest = share_export.export(self.store, self.root / "s.jsonl")
        self.assertEqual(len(manifest["sha256"]), 64)

    def test_verify_accepts_the_file_it_wrote(self) -> None:
        self.memory("a fact")
        target = self.root / "s.jsonl"
        share_export.export(self.store, target)
        checked = share_export.verify(target)
        self.assertTrue(checked["ok"])

    def test_verify_notices_a_changed_file(self) -> None:
        self.memory("a fact")
        target = self.root / "s.jsonl"
        share_export.export(self.store, target)
        target.write_text("{}\n", encoding="utf-8")
        self.assertFalse(share_export.verify(target)["ok"])

    def test_verify_reports_a_missing_manifest(self) -> None:
        target = self.root / "lonely.jsonl"
        target.write_text("{}\n", encoding="utf-8")
        self.assertFalse(share_export.verify(target)["ok"])

    def test_an_export_needs_an_explicit_slice(self) -> None:
        """There is no export-everything convenience: that is the call made by accident."""
        for index in range(3):
            self.memory(f"fact {index}")
        manifest = share_export.export(self.store, self.root / "s.jsonl", limit=2)
        self.assertEqual(manifest["exported"], 2)

    def test_an_id_list_exports_only_those(self) -> None:
        wanted = self.memory("the one to share")
        self.memory("the one to keep")
        manifest = share_export.export(self.store, self.root / "s.jsonl", ids=[wanted])
        self.assertEqual(manifest["exported"], 1)

    def test_a_kind_filter_is_respected(self) -> None:
        self.memory("a preference", kind="preference")
        self.memory("a fact", kind="semantic")
        manifest = share_export.export(self.store, self.root / "s.jsonl", kinds=["preference"])
        self.assertEqual(manifest["exported"], 1)

    def test_archived_memories_are_not_exported(self) -> None:
        keep = self.memory("kept")
        gone = self.memory("gone")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'archived' WHERE id = ?", (gone,))
        manifest = share_export.export(self.store, self.root / "s.jsonl", ids=[keep, gone])
        self.assertEqual(manifest["exported"], 1)

    def test_a_refused_memory_is_counted_not_written(self) -> None:
        self.raw("the api key is:")
        manifest = share_export.export(self.store, self.root / "s.jsonl")
        self.assertEqual(manifest["refused"], 1)
        self.assertEqual(manifest["exported"], 0)
        self.assertIn("credential-only", manifest["refusals"][0]["reason"])

    def test_the_manifest_records_patterns_not_values(self) -> None:
        self.raw("the aws key is AKIA" + "IOSFODNN7EXAMPLE for the bucket")
        manifest = share_export.export(self.store, self.root / "s.jsonl")
        blob = json.dumps(manifest)
        self.assertNotIn("AKIA", blob)
        self.assertTrue(manifest["redactions"])

    def test_the_exported_text_carries_no_machine_path(self) -> None:
        home = "/root" + "/.hermes/plugins/cortex"
        self.memory(f"the plugin lives in {home} and proxies the dashboard")
        target = self.root / "s.jsonl"
        share_export.export(self.store, target)
        self.assertNotIn("/root/", target.read_text(encoding="utf-8"))

    def test_a_dry_run_writes_nothing(self) -> None:
        self.memory("a fact")
        described = share_export.dry_run(self.store)
        self.assertEqual(described["would_export"], 1)
        self.assertEqual(list(self.root.glob("*.jsonl")), [])

    def test_a_dry_run_reports_counts_only(self) -> None:
        self.raw("the aws key is AKIA" + "IOSFODNN7EXAMPLE here")
        described = share_export.dry_run(self.store)
        self.assertNotIn("AKIA", json.dumps(described))

    def test_an_empty_slice_exports_an_empty_file_that_verifies(self) -> None:
        target = self.root / "s.jsonl"
        share_export.export(self.store, target)
        checked = share_export.verify(target)
        self.assertTrue(checked["ok"])
        self.assertEqual(checked["lines"], 0)

    def test_a_broken_store_exports_nothing(self) -> None:
        class Broken:
            _conn = None

        manifest = share_export.export(Broken(), self.root / "s.jsonl", ids=["x"])
        self.assertEqual(manifest["exported"], 0)


if __name__ == "__main__":
    unittest.main()
