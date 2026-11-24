"""Merging two entity ids keeps a reviewable diff, consent and an alias trail (#63)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import entity_merge
from cortex.store import CortexStore

try:  # #39 owns the alias table; the merge tool works whether or not it has landed yet.
    from cortex import aliases as _aliases
except Exception:  # noqa: BLE001
    _aliases = None

try:  # #49 owns the node table.
    from cortex import entity_nodes as _entity_nodes
except Exception:  # noqa: BLE001
    _entity_nodes = None

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)


def seed_alias(store, entity_id: str, alias: str, *, hits: int = 1) -> None:
    """Give an entity one alias, with an exact hit count."""
    entity_merge.ensure(store)
    if _aliases is not None:
        for _ in range(hits):
            _aliases.add(store, entity_id, alias, at=NOW)
        return
    stamp = NOW.isoformat()
    with store.transaction() as conn:
        conn.execute(
            """INSERT INTO entity_aliases
               (entity_id, alias_key, alias, source, hits, created_at, updated_at)
               VALUES (?, ?, ?, 'manual', ?, ?, ?)
               ON CONFLICT(entity_id, alias_key) DO UPDATE SET hits = excluded.hits""",
            (entity_id, entity_merge._key(alias), alias, hits, stamp, stamp),
        )


def seed_node(store, entity_id: str, *, kind: str = "unknown", summary: str = "") -> None:
    if _entity_nodes is not None:
        _entity_nodes.create(store, entity_id, entity_id, kind=kind, at=NOW)
        if summary:
            _entity_nodes.set_summary(store, entity_id, summary, at=NOW)
        _entity_nodes.set_kind(store, entity_id, kind, at=NOW)
        return
    stamp = NOW.isoformat()
    with store.transaction() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS entity_nodes (
                   entity_id TEXT PRIMARY KEY,
                   canonical_name TEXT NOT NULL,
                   kind TEXT NOT NULL,
                   summary TEXT NOT NULL DEFAULT '',
                   created_at TEXT NOT NULL,
                   updated_at TEXT NOT NULL
               )"""
        )
        conn.execute(
            """INSERT OR REPLACE INTO entity_nodes
               (entity_id, canonical_name, kind, summary, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (entity_id, entity_id, kind, summary, stamp, stamp),
        )


def alias_rows(store, entity_id: str) -> list[tuple[str, int]]:
    try:
        rows = store._conn.execute(
            "SELECT alias, hits FROM entity_aliases WHERE entity_id = ? ORDER BY alias_key",
            (entity_id,),
        ).fetchall()
    except Exception:  # noqa: BLE001 - the table may not exist yet
        return []
    return [(str(row["alias"]), int(row["hits"])) for row in rows]


def alias_names(store, entity_id: str) -> list[str]:
    return [name for name, _ in alias_rows(store, entity_id)]


def add_memory(store, entities: list[str], content: str = "A fact about a machine.") -> str:
    return store.add_memory(
        content=content,
        kind="semantic",
        source_type="user_turn",
        source_category="USER_EXPLICIT",
        entities=entities,
    )[0]


def memory_names(store, memory_id: str) -> list[str]:
    row = store._conn.execute(
        "SELECT entities_json FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    return json.loads(str(row["entities_json"]))


def memory_count(store) -> int:
    return int(store._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"])


class MergeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        entity_merge.ensure(self.store)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()


class ProposeTests(MergeTestCase):
    def setUp(self) -> None:
        super().setUp()
        seed_alias(self.store, "t3", "t3code")
        seed_alias(self.store, "t3", "t3 box")
        seed_alias(self.store, "CT116", "ct116")

    def test_the_diff_shows_both_sides_aliases(self) -> None:
        plan = entity_merge.propose(self.store, "t3", "CT116")
        self.assertCountEqual(plan["aliases_before"]["source"], ["t3code", "t3 box"])
        self.assertCountEqual(plan["aliases_before"]["target"], ["ct116"])
        self.assertEqual((plan["source_id"], plan["target_id"]), ("t3", "CT116"))

    def test_memories_are_counted_by_the_alias_they_use(self) -> None:
        add_memory(self.store, ["t3code", "Proxmox"])
        add_memory(self.store, ["CT116"])
        add_memory(self.store, ["unrelated"])
        plan = entity_merge.propose(self.store, "t3", "CT116")
        self.assertEqual(plan["memories_touched"], 1)

    def test_a_different_kind_is_a_conflict(self) -> None:
        seed_node(self.store, "t3", kind="device")
        seed_node(self.store, "CT116", kind="service")
        plan = entity_merge.propose(self.store, "t3", "CT116")
        self.assertTrue(any(item.startswith("kind:") for item in plan["conflicts"]))

    def test_two_summaries_are_a_conflict(self) -> None:
        seed_node(self.store, "t3", kind="device", summary="The t3 box.")
        seed_node(self.store, "CT116", kind="device", summary="The Proxmox box.")
        plan = entity_merge.propose(self.store, "t3", "CT116")
        self.assertTrue(any(item.startswith("summary:") for item in plan["conflicts"]))

    def test_the_sentence_is_one_line(self) -> None:
        plan = entity_merge.propose(self.store, "t3", "CT116")
        self.assertIsInstance(plan["sentence"], str)
        self.assertNotIn("\n", plan["sentence"])
        self.assertIn("t3", plan["sentence"])
        self.assertIn("CT116", plan["sentence"])

    def test_unknown_entities_propose_an_empty_diff(self) -> None:
        plan = entity_merge.propose(self.store, "nothing", "here")
        self.assertEqual(plan["aliases_before"], {"source": [], "target": []})
        self.assertEqual((plan["nodes"]["source"], plan["nodes"]["target"]), (None, None))
        self.assertEqual(plan["memories_touched"], 0)
        self.assertEqual(plan["conflicts"], [])


class ConsentTests(MergeTestCase):
    def test_apply_refuses_without_a_named_consenter(self) -> None:
        seed_alias(self.store, "t3", "t3code")
        result = entity_merge.apply(self.store, "t3", "CT116", by="   ")
        self.assertFalse(result["merged"])
        self.assertIn("consent", result["reason"])
        self.assertEqual(entity_merge.history(self.store), [])

    def test_an_entity_cannot_be_merged_into_itself(self) -> None:
        result = entity_merge.apply(self.store, "CT116", "ct116", by="Jacoby")
        self.assertFalse(result["merged"])
        self.assertIn("itself", result["reason"])

    def test_a_refused_merge_changes_nothing(self) -> None:
        seed_alias(self.store, "t3", "t3code")
        seed_alias(self.store, "CT116", "ct116")
        memory_id = add_memory(self.store, ["t3code", "Proxmox"])
        before_aliases = alias_rows(self.store, "t3")
        entity_merge.apply(self.store, "t3", "CT116", by="")
        self.assertEqual(alias_rows(self.store, "t3"), before_aliases)
        self.assertCountEqual(memory_names(self.store, memory_id), ["t3code", "Proxmox"])
        self.assertEqual(entity_merge.history(self.store), [])


class ApplyTests(MergeTestCase):
    def setUp(self) -> None:
        super().setUp()
        seed_alias(self.store, "t3", "t3code")
        seed_alias(self.store, "t3", "t3 box")
        seed_alias(self.store, "CT116", "ct116")

    def test_the_aliases_move_to_the_target(self) -> None:
        result = entity_merge.apply(self.store, "t3", "CT116", by="Jacoby", at=NOW)
        self.assertTrue(result["merged"])
        self.assertEqual(result["aliases_moved"], 2)
        self.assertEqual(alias_rows(self.store, "t3"), [])
        self.assertCountEqual(
            alias_names(self.store, "CT116"), ["ct116", "t3code", "t3 box"]
        )

    def test_the_entity_lists_are_repointed(self) -> None:
        memory_id = add_memory(self.store, ["t3code", "Proxmox"])
        entity_merge.apply(self.store, "t3", "CT116", by="Jacoby", at=NOW)
        self.assertCountEqual(memory_names(self.store, memory_id), ["CT116", "Proxmox"])

    def test_a_name_already_on_the_target_is_not_duplicated(self) -> None:
        memory_id = add_memory(self.store, ["CT116", "t3code", "Proxmox"])
        entity_merge.apply(self.store, "t3", "CT116", by="Jacoby", at=NOW)
        self.assertEqual(len(memory_names(self.store, memory_id)), 2)
        self.assertCountEqual(memory_names(self.store, memory_id), ["CT116", "Proxmox"])

    def test_every_memory_row_survives_and_the_merge_is_recorded(self) -> None:
        memory_id = add_memory(self.store, ["t3code"])
        before = memory_count(self.store)
        result = entity_merge.apply(self.store, "t3", "CT116", by="Jacoby", at=NOW)
        self.assertEqual(memory_count(self.store), before)
        entry = entity_merge.history(self.store)[0]
        self.assertEqual(entry["merge_id"], result["merge_id"])
        self.assertEqual(entry["by"], "Jacoby")
        self.assertEqual(entry["at"], NOW.isoformat())
        self.assertEqual(entry["memories_touched"], 1)
        self.assertIsNone(entry["undone_at"])
        self.assertTrue(memory_names(self.store, memory_id))


class HistoryTests(MergeTestCase):
    def test_history_is_newest_first_and_limited(self) -> None:
        seed_alias(self.store, "t3", "t3code")
        seed_alias(self.store, "CT116", "ct116")
        seed_alias(self.store, "r630", "r630 box")
        first = entity_merge.apply(self.store, "t3", "CT116", by="Jacoby", at=NOW)
        second = entity_merge.apply(
            self.store, "r630", "CT116", by="Jacoby", at=NOW + timedelta(hours=1)
        )
        self.assertEqual(entity_merge.history(self.store, limit=1)[0]["merge_id"],
                         second["merge_id"])
        self.assertEqual(
            [row["merge_id"] for row in entity_merge.history(self.store)],
            [second["merge_id"], first["merge_id"]],
        )


class UndoTests(MergeTestCase):
    def setUp(self) -> None:
        super().setUp()
        seed_alias(self.store, "t3", "t3code")
        seed_alias(self.store, "CT116", "ct116")

    def test_undo_restores_the_entity_list(self) -> None:
        memory_id = add_memory(self.store, ["t3code", "Proxmox"])
        entity_merge.apply(self.store, "t3", "CT116", by="Jacoby", at=NOW)
        result = entity_merge.undo(self.store, at=NOW)
        self.assertTrue(result["undone"])
        self.assertCountEqual(memory_names(self.store, memory_id), ["t3code", "Proxmox"])

    def test_undo_puts_the_alias_rows_back(self) -> None:
        entity_merge.apply(self.store, "t3", "CT116", by="Jacoby", at=NOW)
        self.assertTrue(entity_merge.undo(self.store, at=NOW)["undone"])
        self.assertIn("t3code", alias_names(self.store, "t3"))
        self.assertNotIn("t3code", alias_names(self.store, "CT116"))

    def test_undo_restores_a_folded_alias_and_its_hits(self) -> None:
        seed_alias(self.store, "t3", "the box", hits=2)
        seed_alias(self.store, "CT116", "the box", hits=3)
        entity_merge.apply(self.store, "t3", "CT116", by="Jacoby", at=NOW)
        self.assertEqual(dict(alias_rows(self.store, "CT116"))["the box"], 5)
        entity_merge.undo(self.store, at=NOW)
        self.assertEqual(dict(alias_rows(self.store, "t3"))["the box"], 2)
        self.assertEqual(dict(alias_rows(self.store, "CT116"))["the box"], 3)

    def test_a_second_undo_refuses(self) -> None:
        entity_merge.apply(self.store, "t3", "CT116", by="Jacoby", at=NOW)
        self.assertTrue(entity_merge.undo(self.store, at=NOW)["undone"])
        second = entity_merge.undo(self.store, at=NOW)
        self.assertFalse(second["undone"])
        self.assertIn("already undone", second["reason"])

    def test_undo_can_target_an_earlier_merge(self) -> None:
        seed_alias(self.store, "r630", "r630 box")
        seed_alias(self.store, "X9", "x9 host")
        first = entity_merge.apply(self.store, "t3", "CT116", by="Jacoby", at=NOW)
        second = entity_merge.apply(
            self.store, "r630", "X9", by="Jacoby", at=NOW + timedelta(hours=1)
        )
        self.assertTrue(entity_merge.undo(self.store, first["merge_id"])["undone"])
        rows = {row["merge_id"]: row for row in entity_merge.history(self.store)}
        self.assertIsNotNone(rows[first["merge_id"]]["undone_at"])
        self.assertIsNone(rows[second["merge_id"]]["undone_at"])
        # The first merge is back; the second one is still in force.
        self.assertIn("t3code", alias_names(self.store, "t3"))
        self.assertIn("r630 box", alias_names(self.store, "X9"))


class ReportTests(MergeTestCase):
    def test_report_counts_merges_and_what_is_still_undoable(self) -> None:
        seed_alias(self.store, "t3", "t3code")
        seed_alias(self.store, "CT116", "ct116")
        seed_alias(self.store, "r630", "r630 box")
        entity_merge.apply(self.store, "t3", "CT116", by="Jacoby", at=NOW)
        entity_merge.apply(self.store, "r630", "CT116", by="Jacoby", at=NOW)
        entity_merge.undo(self.store, at=NOW)
        described = entity_merge.report(self.store)
        self.assertEqual(described["merges"], 2)
        self.assertEqual(described["undone"], 1)
        self.assertEqual(described["undoable"], 1)


if __name__ == "__main__":
    unittest.main()
