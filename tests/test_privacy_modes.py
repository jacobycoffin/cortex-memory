"""How much of memory a caller may see, and why a blank page is not a broken store (#183)."""

from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import privacy_modes as pm
from cortex.store import CortexStore


def _insert(store: CortexStore, content: str, *, pinned: int = 0, state: str = "active") -> str:
    memory_id = str(uuid.uuid4())
    stamp = "2026-11-01T00:00:00+00:00"
    store._conn.execute(
        "INSERT INTO memories(id, content, content_hash, created_at, updated_at, observed_at, state, "
        "pinned) VALUES(?,?,?,?,?,?,?,?)",
        (memory_id, content, str(uuid.uuid4()), stamp, stamp, stamp, state, pinned),
    )
    store._conn.commit()
    return memory_id


class BaseCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def visible_ids(self, mode: str) -> set[str]:
        return {row["id"] for row in pm.view(self.store, {"privacy_mode": mode}).memories}


class NarrowingTests(BaseCase):
    def test_normal_shows_every_active_memory(self) -> None:
        ids = {_insert(self.store, "the dashboard listens on port 8100"),
               _insert(self.store, "the mortgage statement arrives on the third"),
               _insert(self.store, "the token is in the vault")}
        self.assertEqual(self.visible_ids(pm.NORMAL), ids)

    def test_private_excludes_a_credential_tag(self) -> None:
        secret = _insert(self.store, "a note about rotating keys")
        pm.tag(self.store, secret, "credential")
        self.assertNotIn(secret, self.visible_ids(pm.PRIVATE))

    def test_private_excludes_finance_and_health_tags(self) -> None:
        money = _insert(self.store, "a note about the budget")
        health = _insert(self.store, "a note about the pharmacy")
        pm.tag(self.store, money, "finance")
        pm.tag(self.store, health, "health")
        self.assertEqual(self.visible_ids(pm.PRIVATE), set())

    def test_private_keeps_ordinary_technical_content(self) -> None:
        ordinary = _insert(self.store, "the reverse proxy sits in front of the dashboard")
        pm.tag(self.store, ordinary, "technical")
        self.assertEqual(self.visible_ids(pm.PRIVATE), {ordinary})

    def test_private_hides_an_untagged_memory_the_keyword_screen_flags(self) -> None:
        """An untagged store would otherwise show everything in private mode, which is the leak."""
        money = _insert(self.store, "the mortgage is due in October")
        visible = self.visible_ids(pm.PRIVATE)
        self.assertNotIn(money, visible)

    def test_private_does_not_widen_to_classes_it_was_not_given(self) -> None:
        """Only the three named classes are hidden; a relationship note is not private by this rule."""
        relation = _insert(self.store, "my brother called about the boat")
        self.assertIn(relation, self.visible_ids(pm.PRIVATE))

    def test_untagging_restores_visibility(self) -> None:
        """A tag is a record, so it can be taken back - which a guess could not."""
        memory_id = _insert(self.store, "a note about rotating keys")
        pm.tag(self.store, memory_id, "credential")
        self.assertNotIn(memory_id, self.visible_ids(pm.PRIVATE))
        pm.untag(self.store, memory_id, "credential")
        self.assertIn(memory_id, self.visible_ids(pm.PRIVATE))

    def test_sealed_hides_everything_not_pinned_safe(self) -> None:
        _insert(self.store, "the dashboard listens on port 8100")
        self.assertEqual(self.visible_ids(pm.SEALED), set())

    def test_sealed_shows_a_memory_explicitly_pinned_safe(self) -> None:
        memory_id = _insert(self.store, "the dashboard listens on port 8100")
        pm.pin_safe(self.store, memory_id, why="already public in the README")
        self.assertEqual(self.visible_ids(pm.SEALED), {memory_id})

    def test_sealed_ignores_the_scratchpad_pinned_column(self) -> None:
        """`pinned` is already used as a scratchpad (#151); safe-to-show is a different sentence."""
        memory_id = _insert(self.store, "remember to buy milk", pinned=1)
        self.assertNotIn(memory_id, self.visible_ids(pm.SEALED))

    def test_off_record_retrieves_nothing(self) -> None:
        _insert(self.store, "the dashboard listens on port 8100")
        view = pm.view(self.store, {"privacy_mode": pm.OFF_RECORD})
        self.assertEqual(view.memories, [])
        self.assertEqual(view.excluded, 1)

    def test_archived_memories_are_invisible_in_every_mode(self) -> None:
        _insert(self.store, "an old homelab note", state="archived")
        self.assertEqual(pm.view(self.store, "normal").total, 0)


class ResolutionTests(BaseCase):
    def test_an_unstated_mode_is_normal(self) -> None:
        self.assertEqual(pm.resolve(None).name, pm.NORMAL)

    def test_the_mode_is_read_from_the_caller_not_from_the_content(self) -> None:
        """Content that looks financial does not switch the caller into private mode on its own."""
        _insert(self.store, "the mortgage statement arrives on the third")
        self.assertEqual(pm.view(self.store, None).mode, pm.NORMAL)

    def test_an_unknown_mode_falls_back_to_the_most_restrictive_known_mode(self) -> None:
        resolution = pm.resolve({"privacy_mode": "secret"})
        self.assertEqual(resolution.name, pm.OFF_RECORD)
        self.assertTrue(resolution.fell_back)

    def test_an_unknown_mode_hides_the_store_rather_than_showing_it(self) -> None:
        _insert(self.store, "the dashboard listens on port 8100")
        view = pm.view(self.store, {"privacy_mode": "paranoid"})
        self.assertEqual(view.memories, [])
        self.assertEqual(view.excluded, 1)

    def test_mode_names_are_case_and_space_insensitive(self) -> None:
        self.assertEqual(pm.resolve({"privacy_mode": "  PRIVATE "}).name, pm.PRIVATE)

    def test_an_object_context_works_like_a_mapping(self) -> None:
        class Caller:
            privacy_mode = "sealed"

        self.assertEqual(pm.resolve(Caller()).name, pm.SEALED)

    def test_the_unknown_mode_is_reported_not_swallowed(self) -> None:
        line = pm.view(self.store, {"privacy_mode": "secret"}).statement
        self.assertIn("secret", line)
        self.assertIn(pm.OFF_RECORD, line)


class StatedModeTests(BaseCase):
    def test_every_mode_states_itself_in_the_output(self) -> None:
        _insert(self.store, "the dashboard listens on port 8100")
        for name in pm.MODES:
            with self.subTest(mode=name):
                self.assertIn(f"privacy mode: {name}", pm.view(self.store, {"privacy_mode": name}).statement)

    def test_the_statement_counts_what_was_excluded(self) -> None:
        for index in range(3):
            _insert(self.store, f"the mortgage statement number {index} arrived")
        _insert(self.store, "the dashboard listens on port 8100")
        line = pm.view(self.store, {"privacy_mode": pm.PRIVATE}).statement
        self.assertIn("excluded 3 of 4 memories", line)

    def test_a_clean_view_says_nothing_was_excluded(self) -> None:
        _insert(self.store, "the dashboard listens on port 8100")
        self.assertIn("nothing excluded", pm.view(self.store, "private").statement)

    def test_the_output_carries_the_statement_as_data(self) -> None:
        _insert(self.store, "the dashboard listens on port 8100")
        payload = pm.view(self.store, "private").as_dict()
        self.assertEqual(payload["mode"], "private")
        self.assertIn("privacy mode: private", payload["statement"])

    def test_render_leads_with_the_mode(self) -> None:
        _insert(self.store, "the dashboard listens on port 8100")
        lines = pm.render(pm.view(self.store, "sealed"))
        self.assertTrue(lines[0].startswith("privacy mode: sealed"))
        self.assertIn("nothing to show in this mode", "\n".join(lines))

    def test_census_counts_without_returning_content(self) -> None:
        _insert(self.store, "the token is in the vault")
        _insert(self.store, "the dashboard listens on port 8100")
        described = pm.census(self.store)
        self.assertEqual(described["total"], 2)
        self.assertEqual(described["modes"][pm.PRIVATE]["excluded"], 1)
        self.assertNotIn("vault", json.dumps(described))


class NoWideningTests(BaseCase):
    def test_no_mode_shows_more_than_normal(self) -> None:
        _insert(self.store, "the token is in the vault")
        _insert(self.store, "the mortgage statement arrives on the third")
        _insert(self.store, "my brother called about the boat")
        _insert(self.store, "the dashboard listens on port 8100")
        baseline = self.visible_ids(pm.NORMAL)
        for name in pm.MODES:
            with self.subTest(mode=name):
                self.assertLessEqual(self.visible_ids(name), baseline)

    def test_the_advertised_widening_list_is_empty(self) -> None:
        self.assertEqual(pm.describe()["widening_modes"], [])

    def test_only_normal_reads_and_writes(self) -> None:
        described = {mode["mode"]: mode for mode in pm.describe()["modes"]}
        keeping = [name for name, mode in described.items() if mode["reads"] and mode["writes"]]
        self.assertEqual(keeping, [pm.NORMAL, pm.PRIVATE])
        self.assertFalse(described[pm.OFF_RECORD]["reads"])
        self.assertFalse(described[pm.SEALED]["writes"])

    def test_restrictiveness_order_is_monotonic(self) -> None:
        ranks = [pm.RESTRICTIVENESS[name] for name in pm.MODES]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(pm.MODES[0], pm.NORMAL)
        self.assertEqual(pm.MODES[-1], pm.OFF_RECORD)


class WriteTests(BaseCase):
    def test_off_record_refuses_a_write_with_a_reason(self) -> None:
        decision = pm.screen_write({"privacy_mode": "off_record"}, content="a fact worth keeping")
        self.assertFalse(decision.allowed)
        self.assertIn("off_record retains nothing", decision.reason)

    def test_off_record_stores_nothing(self) -> None:
        pm.screen_write({"privacy_mode": "off_record"}, content="a fact worth keeping")
        count = self.store._conn.execute("SELECT COUNT(*) AS count FROM memories").fetchone()["count"]
        self.assertEqual(int(count), 0)

    def test_sealed_refuses_a_write_because_it_would_be_invisible(self) -> None:
        decision = pm.screen_write({"privacy_mode": "sealed"}, content="a fact worth keeping")
        self.assertFalse(decision.allowed)
        self.assertIn("pinned", decision.reason)

    def test_normal_accepts_a_write(self) -> None:
        self.assertTrue(pm.screen_write({"privacy_mode": "normal"}, content="a fact").allowed)

    def test_a_refusal_never_echoes_the_content(self) -> None:
        decision = pm.screen_write({"privacy_mode": "off_record"}, content="the token is hunter2")
        self.assertNotIn("hunter2", json.dumps(decision.as_dict()))


class BrokenStoreTests(unittest.TestCase):
    def test_a_broken_store_is_reported_as_unreadable_not_empty(self) -> None:
        class Broken:
            _conn = None

        view = pm.view(Broken(), {"privacy_mode": "private"})
        self.assertFalse(view.readable)
        self.assertIn("could not be read", view.statement)
        self.assertIn("not an empty store", view.statement)

    def test_a_broken_store_census_says_unavailable_not_zero(self) -> None:
        class Broken:
            _conn = None

        described = pm.census(Broken())
        self.assertFalse(described["readable"])
        self.assertIsNone(described["total"])

    def test_tagging_a_broken_store_fails_without_raising(self) -> None:
        class Broken:
            _conn = None

        self.assertFalse(pm.tag(Broken(), "id", "finance"))
        self.assertFalse(pm.pin_safe(Broken(), "id", why="test"))

    def test_reading_a_mode_never_creates_schema(self) -> None:
        """A privacy check that writes has a side effect on the store it was asked to protect."""
        with tempfile.TemporaryDirectory() as tmp:
            store = CortexStore(Path(tmp) / "cortex.db")
            try:
                pm.view(store, {"privacy_mode": "sealed"})
                pm.census(store)
                names = {row["name"] for row in store._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
                self.assertNotIn("privacy_tags", names)
                self.assertNotIn("privacy_safe_pins", names)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
