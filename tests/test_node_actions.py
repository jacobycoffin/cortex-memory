"""The action layer for a node or an edge (#167): one catalogue, idempotent, demote-never-delete."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import legal_hold, node_actions
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class Broken:
    """A store whose connection cannot be used - a locked or half-migrated database."""

    _conn = None


class ActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str = "the proxmox box runs 8.4", **kwargs) -> str:
        return self.store.add_memory(
            content=text, kind=kwargs.pop("kind", "semantic"),
            source_type="user_turn", source_category="USER_STATED", **kwargs,
        )[0]

    def row(self, memory_id: str) -> dict:
        return dict(self.store._conn.execute(
            "SELECT id, state, pinned, protected, quarantine_reason FROM memories WHERE id = ?",
            (memory_id,)).fetchone())

    def table_counts(self) -> tuple:
        tables = tuple(sorted(r["name"] for r in self.store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")))
        return tables, tuple(
            int(self.store._conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]) for name in tables)

    # -- the catalogue is complete and handles everything it offers -------------------

    def test_every_catalogue_action_is_handled(self) -> None:
        """A catalogue entry with no implementation is a button that silently does nothing."""
        memory_id = self.memory()
        other = self.memory("the proxmox box runs 8.3")
        self.store.add_edge(memory_id, other, "supports", weight=0.5)
        for action in node_actions.CATALOGUE:
            with self.subTest(action=action.key):
                if action.target == "edge":
                    result = node_actions.apply(
                        self.store, memory_id, action.key, target="edge", dst_id=other,
                        relation="supports", argument="x", consent=True, at=NOW)
                else:
                    result = node_actions.apply(
                        self.store, memory_id, action.key, argument="a correction",
                        consent=True, at=NOW)
                self.assertNotEqual(result.status, "unknown_action", action.key)

    def test_the_catalogue_offers_no_way_to_delete(self) -> None:
        """The store's rule is demote-never-delete, and a missing action cannot be called by mistake."""
        keys = {action.key for action in node_actions.CATALOGUE}
        for forbidden in ("delete", "purge", "destroy", "erase", "forget"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, keys)
        for action in node_actions.CATALOGUE:
            if not action.writes or action.key in {"hold", "release_hold", "correct"}:
                continue  # read-only has nothing to undo; a hold has no state; a correction is superseded
            self.assertTrue(action.inverse, f"{action.key} has no inverse, so it cannot be undone")

    def test_the_catalogue_declares_what_it_needs(self) -> None:
        catalogue = {action.key: action for action in node_actions.CATALOGUE}
        self.assertTrue(catalogue["correct"].requires_argument)
        self.assertTrue(catalogue["retire"].needs_confirmation)
        self.assertFalse(catalogue["explain"].writes)
        self.assertEqual(catalogue["pin"].inverse, "unpin")
        for action in node_actions.CATALOGUE:
            self.assertTrue(action.why, f"{action.key} has no stated purpose")

    # -- idempotent, and says so ------------------------------------------------------

    def test_pinning_twice_reports_already(self) -> None:
        memory_id = self.memory()
        first = node_actions.apply(self.store, memory_id, "pin", at=NOW)
        second = node_actions.apply(self.store, memory_id, "pin", at=NOW)
        self.assertEqual(first.status, "applied")
        self.assertEqual(second.status, "already")
        self.assertEqual(self.row(memory_id)["pinned"], 1, "a replay must not toggle the pin off")

    def test_unpinning_something_not_pinned_reports_already(self) -> None:
        memory_id = self.memory()
        self.assertEqual(node_actions.apply(self.store, memory_id, "unpin", at=NOW).status, "already")

    def test_muting_twice_reports_already(self) -> None:
        memory_id = self.memory()
        self.assertEqual(node_actions.apply(self.store, memory_id, "mute", at=NOW).status, "applied")
        self.assertEqual(node_actions.apply(self.store, memory_id, "mute", at=NOW).status, "already")
        self.assertEqual(self.row(memory_id)["state"], "cold")

    # -- demotion, never deletion -----------------------------------------------------

    def test_retire_archives_and_keeps_the_row(self) -> None:
        memory_id = self.memory()
        result = node_actions.apply(self.store, memory_id, "retire", consent=True, at=NOW)
        self.assertTrue(result.changed)
        self.assertEqual(self.row(memory_id)["state"], "archived")
        self.assertIsNotNone(self.row(memory_id), "retire must not delete")

    def test_restore_brings_a_retired_memory_back(self) -> None:
        memory_id = self.memory()
        node_actions.apply(self.store, memory_id, "retire", consent=True, at=NOW)
        node_actions.apply(self.store, memory_id, "restore", at=NOW)
        self.assertEqual(self.row(memory_id)["state"], "active")

    def test_quarantine_records_its_reason(self) -> None:
        memory_id = self.memory()
        node_actions.apply(self.store, memory_id, "quarantine", consent=True,
                           reason="looks like two facts glued together", at=NOW)
        row = self.row(memory_id)
        self.assertEqual(row["state"], "quarantine")
        self.assertIn("two facts", row["quarantine_reason"])

    # -- protections outrank the action ----------------------------------------------

    def test_a_protected_memory_refuses_every_writing_action_with_a_reason(self) -> None:
        memory_id = self.memory(protected=1)
        for action in node_actions.actions_for("memory"):
            # `hold`/`release_hold` are the protection itself, so they are deliberately not blocked by it
            if not action.writes or action.key in {"hold", "release_hold"}:
                continue
            with self.subTest(action=action.key):
                result = node_actions.apply(self.store, memory_id, action.key,
                                            argument="a correction", consent=True, at=NOW)
                self.assertEqual(result.status, "refused", action.key)
                self.assertIn("protected", result.reason)
        self.assertEqual(self.row(memory_id)["state"], "active")

    def test_a_legal_hold_refuses_and_names_the_hold(self) -> None:
        memory_id = self.memory()
        node_actions.apply(self.store, memory_id, "hold", consent=True, reason="litigation", at=NOW)
        result = node_actions.apply(self.store, memory_id, "retire", consent=True, at=NOW)
        self.assertEqual(result.status, "refused")
        self.assertIn("legal hold", result.reason)

    def test_placing_a_second_hold_reports_already(self) -> None:
        memory_id = self.memory()
        self.assertEqual(node_actions.apply(self.store, memory_id, "hold", consent=True, at=NOW).status,
                         "applied")
        self.assertEqual(node_actions.apply(self.store, memory_id, "hold", consent=True, at=NOW).status,
                         "already")

    def test_releasing_a_hold_lets_the_action_through(self) -> None:
        memory_id = self.memory()
        node_actions.apply(self.store, memory_id, "hold", consent=True, at=NOW)
        node_actions.apply(self.store, memory_id, "release_hold", consent=True, at=NOW)
        self.assertFalse(legal_hold.is_held(self.store, memory_id))
        self.assertEqual(node_actions.apply(self.store, memory_id, "retire", consent=True, at=NOW).status,
                         "applied")

    # -- refusals that name what went wrong ------------------------------------------

    def test_an_unknown_memory_is_a_named_refusal(self) -> None:
        result = node_actions.apply(self.store, "no-such-id", "pin", at=NOW)
        self.assertEqual(result.status, "unknown_target")
        self.assertEqual(result.reason, "unknown memory")

    def test_an_unknown_action_names_the_catalogue(self) -> None:
        result = node_actions.apply(self.store, self.memory(), "delete", at=NOW)
        self.assertEqual(result.status, "unknown_action")
        self.assertIn("catalogue", result.reason)

    def test_an_edge_action_on_a_memory_is_refused(self) -> None:
        result = node_actions.apply(self.store, self.memory(), "reinforce", at=NOW)
        self.assertEqual(result.status, "unknown_action")
        self.assertIn("edge", result.reason)

    def test_a_confirmation_action_refuses_without_consent(self) -> None:
        memory_id = self.memory()
        result = node_actions.apply(self.store, memory_id, "retire", at=NOW)
        self.assertEqual(result.status, "refused")
        self.assertIn("asks first", result.reason)
        self.assertEqual(self.row(memory_id)["state"], "active", "a refusal must not half-apply")

    def test_a_correction_without_text_asks_for_it(self) -> None:
        result = node_actions.apply(self.store, self.memory(), "correct", consent=True, at=NOW)
        self.assertEqual(result.status, "needs_argument")

    def test_a_correction_supersedes_and_points_at_the_new_record(self) -> None:
        memory_id = self.memory()
        result = node_actions.apply(self.store, memory_id, "correct", consent=True,
                                    argument="the proxmox box runs 8.5", at=NOW)
        self.assertEqual(result.status, "applied", result.reason)
        self.assertTrue(result.after.get("correction_id"))

    def test_explain_writes_nothing(self) -> None:
        memory_id = self.memory()
        before = self.table_counts()
        result = node_actions.apply(self.store, memory_id, "explain", at=NOW)
        self.assertEqual(result.status, "already")
        self.assertFalse(result.changed)
        self.assertEqual(before, self.table_counts())

    # -- edges ------------------------------------------------------------------------

    def test_reinforce_and_weaken_move_the_weight_and_clamp(self) -> None:
        src = self.memory()
        dst = self.memory("the box runs 8.3")
        self.store.add_edge(src, dst, "supports", weight=0.5)
        up = node_actions.apply(self.store, src, "reinforce", target="edge", dst_id=dst,
                                relation="supports", at=NOW)
        self.assertEqual(up.status, "applied")
        self.assertAlmostEqual(up.after["weight"], 0.55, places=4)
        for _ in range(20):
            node_actions.apply(self.store, src, "reinforce", target="edge", dst_id=dst,
                               relation="supports", at=NOW)
        weight = self.store._conn.execute(
            "SELECT weight FROM edges WHERE relation = 'supports'").fetchone()[0]
        self.assertLessEqual(weight, node_actions.WEIGHT_CEILING)

    def test_an_unknown_edge_is_a_named_refusal(self) -> None:
        result = node_actions.apply(self.store, self.memory(), "reinforce", target="edge",
                                    dst_id="nope", relation="supports", at=NOW)
        self.assertEqual(result.status, "unknown_target")
        self.assertIn("unknown edge", result.reason)

    def test_an_edge_action_without_its_coordinates_is_refused(self) -> None:
        result = node_actions.apply(self.store, self.memory(), "reinforce", target="edge", at=NOW)
        self.assertEqual(result.status, "unknown_target")
        self.assertIn("dst_id", result.reason)

    # -- what a view renders ----------------------------------------------------------

    def test_available_explains_why_an_action_is_unavailable(self) -> None:
        memory_id = self.memory(protected=1)
        entries = {entry["key"]: entry for entry in node_actions.available(self.store, memory_id)}
        self.assertFalse(entries["retire"]["available"])
        self.assertIn("protected", entries["retire"]["reason"])
        self.assertTrue(entries["explain"]["available"], "a read-only action survives a protection")

    def test_available_says_unknown_for_an_absent_memory(self) -> None:
        entries = node_actions.available(self.store, "no-such-id")
        self.assertTrue(all(not entry["available"] for entry in entries))
        self.assertTrue(all(entry["reason"] == "unknown memory" for entry in entries))

    def test_the_catalogue_renders_its_flags(self) -> None:
        lines = node_actions.render_catalogue("memory")
        self.assertTrue(any("asks first" in line for line in lines))
        self.assertTrue(any("needs text" in line for line in lines))
        self.assertTrue(any("read-only" in line for line in lines))

    def test_a_broken_store_is_a_refusal_not_an_exception(self) -> None:
        result = node_actions.apply(Broken(), "abc", "pin", at=NOW)
        self.assertEqual(result.status, "unknown_target")
        self.assertEqual(node_actions.available(Broken(), "abc")[0]["reason"], "unknown memory")

    def test_a_result_renders_one_line(self) -> None:
        memory_id = self.memory()
        line = node_actions.render(node_actions.apply(self.store, memory_id, "pin", at=NOW))
        self.assertEqual(len(line), 1)
        self.assertIn("pin", line[0])


if __name__ == "__main__":
    unittest.main()
