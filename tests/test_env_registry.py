"""Environment facts registry (#135): typed, dated, and never a secret."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import env_registry
from cortex.store import CortexStore

NOW = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)


class DetectTests(unittest.TestCase):
    def kinds(self, text: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for fact in env_registry.detect(text):
            out.setdefault(fact.kind, []).append(fact.value)
        return out

    def test_labelled_host_keeps_the_name(self) -> None:
        facts = env_registry.detect("T3 lives on CT116 (192.168.1.116) and nothing else does.")
        hosts = [(f.subject, f.value) for f in facts if f.kind == "host"]
        self.assertIn(("ct116", "192.168.1.116"), hosts)

    def test_bare_ip_is_its_own_subject(self) -> None:
        facts = [f for f in env_registry.detect("The VPS answers at 76.13.98.26 over ssh.") if f.kind == "host"]
        self.assertTrue(facts)
        self.assertEqual(facts[0].subject, "76.13.98.26")

    def test_port_takes_the_host_it_is_attached_to(self) -> None:
        facts = [f for f in env_registry.detect("`t3 serve` listens on 127.0.0.1:3773.") if f.kind == "port"]
        self.assertEqual([(f.subject, f.value) for f in facts], [("127.0.0.1", "3773")])

    def test_unattributed_port_is_not_a_fact(self) -> None:
        self.assertEqual([f for f in env_registry.detect("It listens on port 8100.") if f.kind == "port"], [])

    def test_docker_mapping_is_not_a_port(self) -> None:
        self.assertEqual([f for f in env_registry.detect("Compose sets user: root:1001 for the job.") if f.kind == "port"], [])

    def test_stack_trace_line_is_not_a_port(self) -> None:
        self.assertEqual([f for f in env_registry.detect("Traceback at store.py:13177 in the worker.") if f.kind == "port"], [])

    def test_path_and_credential_pointer(self) -> None:
        facts = env_registry.detect("The Cloudflare token is in /root/.hermes/secrets/cloudflare-api-token.")
        kinds = {f.value: f.kind for f in facts}
        self.assertEqual(kinds.get("/root/.hermes/secrets/cloudflare-api-token"), "credential")
        facts = env_registry.detect("Cortex installs to /root/.hermes/plugins/cortex on the VPS.")
        kinds = {f.value: f.kind for f in facts}
        self.assertEqual(kinds.get("/root/.hermes/plugins/cortex"), "path")

    def test_version_pin_keeps_its_owner(self) -> None:
        facts = env_registry.detect("T3 runs build 0.1.4-nightly.20261028.3824 on the box.")
        pins = [(f.subject, f.value) for f in facts if f.kind == "version"]
        self.assertIn(("t3", "0.1.4-nightly.20261028.3824"), pins)

    def test_version_without_owner_is_dropped(self) -> None:
        self.assertEqual([f for f in env_registry.detect("It was 2.4.1 before the change.") if f.kind == "version"], [])

    def test_mod_version_is_not_an_address(self) -> None:
        text = "FTB set (`26.1.2.8` + FTB Library `26.1.2.8` + FTB Teams `26.1.2.4`) is installed."
        self.assertEqual([f for f in env_registry.detect(text) if f.kind == "host"], [])

    def test_markdown_table_row_is_not_a_version(self) -> None:
        self.assertEqual([f for f in env_registry.detect("| store | 8.160.3 | yes |") if f.kind == "version"], [])

    def test_secrets_are_never_values(self) -> None:
        text = (
            "GitHub token ghp_abcdefghijklmnopqrstuvwxyz0123456789 lives in /root/.hermes/secrets/gh. "
            "password: hunter2 and api_key=sk-live-abcdefghijklmnop. Private key "
            "-----BEGIN OPENSSH PRIVATE KEY----- also on disk."
        )
        values = [f.value for f in env_registry.detect(text)]
        self.assertFalse([v for v in values if "ghp_" in v or "hunter2" in v or "sk-live" in v])
        self.assertIn("/root/.hermes/secrets/gh", values)

    def test_hex_token_is_not_a_name(self) -> None:
        self.assertNotIn("deadbeefcafe", [f.subject for f in env_registry.detect("Set 1.2.3 on deadbeefcafe.")])


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, when: datetime = NOW) -> str:
        memory_id = self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT", confidence=0.8,
        )[0]
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET created_at = ?, observed_at = ? WHERE id = ?",
                         (when.isoformat(), when.isoformat(), memory_id))
        return memory_id

    def test_build_persists_dated_facts(self) -> None:
        self.memory("r630 (192.168.1.208) runs Proxmox; t3 serve listens on 127.0.0.1:3773.")
        facts = env_registry.build(self.store)
        self.assertTrue(facts)
        report = env_registry.report(self.store)
        self.assertGreater(report["total"], 0)
        self.assertEqual(report["undated"], 0)

    def test_same_fact_from_two_memories_confirms_does_not_duplicate(self) -> None:
        self.memory("The VPS is at 76.13.98.26.", when=datetime(2026, 10, 1, tzinfo=timezone.utc))
        self.memory("The VPS is at 76.13.98.26, still.", when=datetime(2026, 11, 1, tzinfo=timezone.utc))
        env_registry.build(self.store)
        rows = env_registry.lookup(self.store, "76.13.98.26", kind="host")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["confirmed_at"][:10], "2026-11-01")
        self.assertEqual(rows[0]["confirmations"], 2)

    def test_conflict_keeps_both_sides(self) -> None:
        self.memory("CT110 is at 192.168.1.41.", when=datetime(2026, 7, 1, tzinfo=timezone.utc))
        self.memory("CT110 is at 192.168.1.42.", when=datetime(2026, 7, 2, tzinfo=timezone.utc))
        env_registry.build(self.store)
        found = env_registry.conflicts(self.store)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].subject, "ct110")
        self.assertEqual({value for value, _ in found[0].values}, {"192.168.1.41", "192.168.1.42"})
        self.assertEqual(len(env_registry.lookup(self.store, "192.168.1.41")), 1)

    def test_unknown_host_does_not_conflict_with_known_host(self) -> None:
        self.memory("CT110 is at 192.168.1.41.")
        self.memory("Something else sits at 192.168.1.42.")
        env_registry.build(self.store)
        self.assertEqual(env_registry.conflicts(self.store), [])
        self.assertEqual(env_registry.report(self.store)["conflicts"], 0)

    def test_ports_on_one_host_are_not_a_conflict(self) -> None:
        self.memory("CT107 runs 25575 and 13177.")
        self.memory("CT107 runs 25575 and 13177.")
        env_registry.build(self.store)
        self.assertEqual(env_registry.report(self.store)["conflicts"], 0)

    def test_drift_lists_older_pins_first_as_history(self) -> None:
        self.memory("T3 is on 0.1.3.", when=datetime(2026, 9, 1, tzinfo=timezone.utc))
        self.memory("T3 is on 0.1.4.", when=datetime(2026, 11, 1, tzinfo=timezone.utc))
        env_registry.build(self.store)
        drift = env_registry.drift(self.store)
        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0].subject, "t3")

    def test_lookup_and_render(self) -> None:
        self.memory("CT116 (192.168.1.116) runs t3 serve on 127.0.0.1:3773, build 0.1.4-nightly.20261028.3824.")
        env_registry.build(self.store)
        self.assertTrue(env_registry.lookup(self.store, "3773"))
        text = env_registry.render(self.store)
        self.assertIn("192.168.1.116", text)
        self.assertIn("3773", text)
        self.assertLessEqual(len(text), 2000)

    def test_render_respects_a_small_budget(self) -> None:
        for index in range(30):
            self.memory(f"CT1{index:02d} is at 192.168.1.{index + 10}.")
        env_registry.build(self.store)
        self.assertLessEqual(len(env_registry.render(self.store, budget=400)), 400)

    def test_rebuild_is_idempotent(self) -> None:
        self.memory("The VPS is at 76.13.98.26.")
        env_registry.build(self.store)
        first = env_registry.report(self.store)["total"]
        env_registry.build(self.store)
        self.assertEqual(env_registry.report(self.store)["total"], first)

    def test_report_shapes(self) -> None:
        self.memory("CT116 (192.168.1.116) runs t3 on 127.0.0.1:3773.")
        env_registry.build(self.store)
        report = env_registry.report(self.store)
        for key in ("total", "by_kind", "undated", "conflicts", "drift", "subjects"):
            self.assertIn(key, report)
        self.assertEqual(set(report["by_kind"]), set(env_registry.KINDS))


if __name__ == "__main__":
    unittest.main()
