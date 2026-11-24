"""One store, several devices, no cross-talk (#156)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import device_namespaces as dn
from cortex.store import CortexStore


class NormaliseTests(unittest.TestCase):
    def test_a_name_is_lowercased(self) -> None:
        self.assertEqual(dn.normalise("Phone"), "phone")

    def test_spaces_become_hyphens(self) -> None:
        self.assertEqual(dn.normalise("Jacoby's iPhone"), "jacoby-s-iphone")

    def test_an_empty_name_is_empty_not_a_wildcard(self) -> None:
        """A blank device must not mean 'show me everything'."""
        self.assertEqual(dn.normalise("   "), "")

    def test_a_name_is_bounded(self) -> None:
        self.assertLessEqual(len(dn.normalise("x" * 100)), dn._MAX_NAME)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, scope: dict | None = None) -> str:
        memory_id = self.store.add_memory(
            content=text, kind="semantic", source_type="user_turn", source_category="USER_STATED",
        )[0]
        if scope is not None:
            with self.store.transaction() as conn:
                conn.execute("UPDATE memories SET scope_json = ? WHERE id = ?",
                             (json.dumps(scope), memory_id))
        return memory_id

    def scope(self, memory_id: str) -> dict:
        row = self.store._conn.execute("SELECT scope_json FROM memories WHERE id = ?",
                                       (memory_id,)).fetchone()
        return json.loads(row["scope_json"])

    def test_setting_a_device_merges_rather_than_replaces(self) -> None:
        """Scoping a memory must not silently de-project it."""
        memory_id = self.memory("a homelab fact", scope={"project": "homelab"})
        result = dn.set_device(self.store, memory_id, "phone")
        self.assertTrue(result["ok"])
        self.assertEqual(self.scope(memory_id), {"device": "phone", "project": "homelab"})

    def test_an_empty_device_is_refused(self) -> None:
        memory_id = self.memory("a fact")
        result = dn.set_device(self.store, memory_id, "  ")
        self.assertFalse(result["ok"])
        self.assertIn("not a wildcard", result["reason"])

    def test_scoping_a_missing_memory_is_refused(self) -> None:
        self.assertFalse(dn.set_device(self.store, "nope", "phone")["ok"])

    def test_an_unscoped_memory_is_shared_not_hidden(self) -> None:
        """If an empty scope meant 'no device', a phone would open to an empty store."""
        self.memory("a preference about metric units")
        visible = dn.visible(self.store, "phone")
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0]["scope"], "shared")

    def test_a_scoped_memory_is_only_visible_to_its_device(self) -> None:
        self.memory("this phone is on 192.168.1.44", scope={"device": "phone"})
        self.assertEqual(len(dn.visible(self.store, "phone")), 1)
        self.assertEqual(len(dn.visible(self.store, "desktop")), 0)

    def test_include_shared_can_be_turned_off(self) -> None:
        self.memory("a shared preference")
        self.memory("a phone-only fact", scope={"device": "phone"})
        self.assertEqual(len(dn.visible(self.store, "phone", include_shared=False)), 1)

    def test_an_unknown_device_sees_nothing_rather_than_everything(self) -> None:
        self.memory("a shared preference")
        self.assertEqual(dn.visible(self.store, ""), [])

    def test_deny_blocks_another_devices_memory(self) -> None:
        memory_id = self.memory("a phone-only fact", scope={"device": "phone"})
        self.assertFalse(dn.deny(self.store, "phone", memory_id))
        self.assertTrue(dn.deny(self.store, "desktop", memory_id))

    def test_deny_allows_shared_memory(self) -> None:
        memory_id = self.memory("a shared preference")
        self.assertFalse(dn.deny(self.store, "desktop", memory_id))

    def test_deny_refuses_a_missing_memory(self) -> None:
        self.assertTrue(dn.deny(self.store, "phone", "nope"))

    def test_clearing_a_device_returns_it_to_shared(self) -> None:
        memory_id = self.memory("a phone-only fact", scope={"device": "phone"})
        result = dn.clear_device(self.store, memory_id)
        self.assertTrue(result["changed"])
        self.assertEqual(self.scope(memory_id), {})

    def test_clearing_twice_is_a_no_op(self) -> None:
        memory_id = self.memory("a fact")
        self.assertFalse(dn.clear_device(self.store, memory_id)["changed"])

    def test_the_split_counts_shared_and_per_device(self) -> None:
        self.memory("a shared preference")
        self.memory("a phone fact", scope={"device": "phone"})
        self.memory("a watch fact", scope={"device": "watch"})
        described = dn.split(self.store)
        self.assertEqual(described.memories, 3)
        self.assertEqual(described.shared, 1)
        self.assertEqual(described.by_device, {"phone": 1, "watch": 1})

    def test_a_messy_stored_name_still_matches_its_device(self) -> None:
        """Five live memories hold names like `COBY-CLX gaming-pc`; normalising only on write would
        hide them from the device that owns them."""
        self.memory("a gaming pc fact", scope={"device": "COBY-CLX gaming-pc"})
        self.assertEqual(len(dn.visible(self.store, "coby-clx-gaming-pc")), 1)
        self.assertEqual(len(dn.visible(self.store, "COBY-CLX Gaming-PC")), 1)

    def test_the_split_normalises_the_names_it_counts(self) -> None:
        self.memory("a fact", scope={"device": "iPad 10th Gen"})
        self.assertIn("ipad-10th-gen", dn.split(self.store).by_device)

    def test_an_unscoped_store_says_so(self) -> None:
        self.memory("a shared preference")
        self.assertIn("not a namespace", dn.split(self.store).note)

    def test_the_render_states_the_split(self) -> None:
        self.memory("a phone fact", scope={"device": "phone"})
        rendered = "\n".join(dn.render(self.store))
        self.assertIn("device namespaces", rendered)
        self.assertIn("phone 1", rendered)

    def test_a_broken_store_splits_to_zero(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(dn.split(Broken()).memories, 0)
        self.assertEqual(dn.visible(Broken(), "phone"), [])


if __name__ == "__main__":
    unittest.main()
