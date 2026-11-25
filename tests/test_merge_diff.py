"""A merge is reviewable before it happens and reversible afterwards (#168)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import merge_diff
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
KEPT = "The dashboard runs on port 8100 behind nginx."
ABSORBED = "The dashboard runs on port 8100 behind nginx. The staging host is cyt116."


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.survivor = self.memory(KEPT, scope={"project": "dashboard"}, pinned=False)
        self.absorbed = self.memory(ABSORBED)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, scope: dict | None = None, pinned: bool = False,
               valid_to: str | None = None, entities: list[str] | None = None) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT", scope=scope or {}, pinned=pinned,
            valid_to=valid_to, entities=entities or [],
        )[0]

    def row(self, memory_id: str) -> dict:
        return dict(self.store._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone())

    def snapshot(self) -> tuple:
        memories = [tuple(row) for row in self.store._conn.execute(
            "SELECT * FROM memories ORDER BY id").fetchall()]
        edges = [tuple(row) for row in self.store._conn.execute(
            "SELECT * FROM edges ORDER BY src_id, dst_id, relation").fetchall()]
        tables = [row[0] for row in self.store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()]
        return (memories, edges, tables, self.store._conn.total_changes)


class LossTests(Base):
    def test_a_clean_merge_is_not_lossy(self) -> None:
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed,
                                  merged_text=ABSORBED, at=NOW)
        self.assertIsNotNone(diff)
        self.assertFalse(diff.lossy)
        self.assertEqual(diff.lost_facts, [])

    def test_keeping_only_the_survivor_text_is_lossy_and_names_the_fact(self) -> None:
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed, at=NOW)
        self.assertTrue(diff.lossy)
        self.assertTrue(any("cyt116" in fact for fact in diff.lost_facts))
        self.assertIn("LOSSY", diff.describe())

    def test_a_proposed_text_that_drops_one_clause_names_that_clause(self) -> None:
        diff = merge_diff.compute(
            self.store, self.survivor, self.absorbed,
            merged_text="The dashboard runs on port 8100 behind nginx, on cyt116.", at=NOW)
        self.assertTrue(diff.lossy)
        self.assertTrue(any("staging host" in fact for fact in diff.lost_facts))

    def test_lossy_apply_refuses_without_accept_loss(self) -> None:
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed, at=NOW)
        result = merge_diff.apply(self.store, diff, consent=True, consented_by="operator")
        self.assertFalse(result.ok)
        self.assertIn("lossy", result.reason)
        self.assertTrue(result.lost_facts)
        self.assertEqual(self.row(self.absorbed)["state"], "active")

    def test_lossy_apply_proceeds_only_when_loss_is_accepted(self) -> None:
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed, at=NOW)
        result = merge_diff.apply(self.store, diff, consent=True, consented_by="operator",
                                  accept_loss=True, at=NOW)
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(len(result.lost_facts), len(diff.lost_facts))


class FieldTests(Base):
    def test_fields_are_named_by_column_and_table(self) -> None:
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed,
                                  merged_text=ABSORBED, at=NOW)
        names = {item["field"] for item in diff.fields}
        self.assertIn("content", names)
        self.assertIn("pinned", names)
        self.assertTrue(all(item["table"] == "memories" for item in diff.fields))

    def test_pinned_is_carried_from_the_absorbed_row(self) -> None:
        pinned = self.memory("Grafana lives at grafana.internal.", pinned=True)
        diff = merge_diff.compute(self.store, self.survivor, pinned, merged_text=ABSORBED, at=NOW)
        field = next(item for item in diff.fields if item["field"] == "pinned")
        self.assertEqual((field["before"], field["after"], field["changed"]), (0, 1, True))

    def test_scope_is_merged_and_a_conflict_is_reported_not_overwritten(self) -> None:
        other = self.memory("Backups land in the vault.", scope={"project": "storage"})
        diff = merge_diff.compute(self.store, self.survivor, other,
                                  merged_text=KEPT + " Backups land in the vault.", at=NOW)
        scope = next(item for item in diff.fields if item["field"] == "scope_json")
        self.assertIn("dashboard", scope["after"])
        self.assertEqual(diff.conflicts[0]["kept"], "dashboard")
        self.assertTrue(any(loss["kind"] == "scope_conflict"
                            for loss in diff.structural_losses))

    def test_entities_are_a_union_without_duplicates(self) -> None:
        self.store._conn.execute(
            "UPDATE memories SET entities_json = ? WHERE id = ?",
            ('["CT116"]', self.survivor))
        other = self.memory("The box is named cyt116.", entities=["ct116", "nginx"])
        diff = merge_diff.compute(self.store, self.survivor, other,
                                  merged_text=KEPT + " The box is named cyt116.", at=NOW)
        entities = next(item for item in diff.fields if item["field"] == "entities_json")
        self.assertEqual(entities["after"], '["CT116", "nginx"]')

    def test_the_absorbed_row_is_retired_not_deleted(self) -> None:
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed,
                                  merged_text=ABSORBED, at=NOW)
        self.assertEqual(diff.counts["memories:retired"], 1)
        self.assertEqual(diff.retired[0]["id"], self.absorbed)
        self.assertTrue(diff.retired[0]["table"] == "memories")

    def test_every_count_names_the_table_it_came_from(self) -> None:
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed, at=NOW)
        self.assertTrue(diff.counts)
        for key in diff.counts:
            table, _, what = key.partition(":")
            self.assertIn(table, {"memories", "edges"}, key)
            self.assertTrue(what, key)


class EdgeTests(Base):
    def test_an_inbound_edge_is_repointed_to_the_survivor(self) -> None:
        third = self.memory("Nginx terminates TLS for the dashboard.")
        self.store.add_edge(third, self.absorbed, "supports")
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed,
                                  merged_text=ABSORBED, at=NOW)
        move = next(item for item in diff.edge_moves if item["action"] == "repoint_dst")
        self.assertEqual(move["new_dst_id"], self.survivor)
        result = merge_diff.apply(self.store, diff, consent=True, consented_by="operator", at=NOW)
        self.assertTrue(result.ok, result.reason)
        rows = self.store._conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE relation = 'supports'").fetchall()
        self.assertEqual([tuple(row) for row in rows], [(third, self.survivor)])

    def test_an_edge_between_the_pair_is_dropped_and_counted(self) -> None:
        self.store.add_edge(self.survivor, self.absorbed, "co_used")
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed,
                                  merged_text=ABSORBED, at=NOW)
        self.assertEqual(diff.counts["edges:dropped"], 1)
        self.assertTrue(any(loss["kind"] == "edge_self_loop"
                            for loss in diff.structural_losses))
        merge_diff.apply(self.store, diff, consent=True, consented_by="operator", at=NOW)
        left = self.store._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE relation = 'co_used'").fetchone()[0]
        self.assertEqual(left, 0)

    def test_a_repointed_edge_that_already_exists_is_reported_as_folded(self) -> None:
        third = self.memory("The reverse proxy is nginx.")
        self.store.add_edge(third, self.absorbed, "supports", weight=0.4)
        self.store.add_edge(third, self.survivor, "supports", weight=0.6)
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed,
                                  merged_text=ABSORBED, at=NOW)
        self.assertEqual(diff.counts["edges:folded"], 1)
        move = next(item for item in diff.edge_moves if item["action"] == "fold_duplicate")
        self.assertIsNotNone(move["destination_before"])

    def test_counts_separate_moved_repointed_folded_and_dropped(self) -> None:
        third = self.memory("Nginx terminates TLS for the dashboard.")
        self.store.add_edge(self.absorbed, third, "extends")
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed,
                                  merged_text=ABSORBED, at=NOW)
        self.assertEqual(diff.counts["edges:moved"], 1)
        self.assertEqual(diff.counts["edges:repointed"], 1)
        self.assertEqual(diff.counts["edges:folded"], 0)


class WriteTests(Base):
    def test_computing_a_diff_writes_nothing(self) -> None:
        before = self.snapshot()
        merge_diff.compute(self.store, self.survivor, self.absorbed, at=NOW)
        merge_diff.compute(self.store, self.survivor, self.absorbed,
                           merged_text=ABSORBED, at=NOW)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(merge_diff.records(self.store), [])

    def test_apply_needs_consent_then_a_name(self) -> None:
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed,
                                  merged_text=ABSORBED, at=NOW)
        self.assertFalse(merge_diff.apply(self.store, diff).ok)
        self.assertIn("consent", merge_diff.apply(self.store, diff).reason)
        self.assertFalse(merge_diff.apply(self.store, diff, consent=True).ok)
        self.assertEqual(self.row(self.absorbed)["state"], "active")

    def test_apply_requires_a_computed_diff(self) -> None:
        result = merge_diff.apply(self.store, {"survivor_id": self.survivor},  # type: ignore[arg-type]
                                  consent=True, consented_by="operator")
        self.assertFalse(result.ok)
        self.assertIn("diff", result.reason)

    def test_a_stale_diff_is_refused(self) -> None:
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed,
                                  merged_text=ABSORBED, at=NOW)
        self.store._conn.execute("UPDATE memories SET content = ? WHERE id = ?",
                                 ("Changed after review.", self.absorbed))
        self.store._conn.commit()
        result = merge_diff.apply(self.store, diff, consent=True, consented_by="operator")
        self.assertFalse(result.ok)
        self.assertIn("changed since", result.reason)

    def test_unknown_ids_compute_nothing(self) -> None:
        self.assertIsNone(merge_diff.compute(self.store, "nope", self.absorbed))
        self.assertIsNone(merge_diff.compute(self.store, self.survivor, "nope"))
        self.assertIsNone(merge_diff.compute(self.store, self.survivor, self.survivor))

    def test_a_broken_store_returns_no_rows_rather_than_raising(self) -> None:
        class Broken:
            class _Conn:
                def execute(self, *_args, **_kwargs):
                    raise RuntimeError("database is locked")

            _conn = _Conn()

        self.assertIsNone(merge_diff.compute(Broken(), "a", "b"))
        self.assertEqual(merge_diff.records(Broken()), [])
        self.assertIsNone(merge_diff.preview_undo(Broken(), "missing"))


class UndoTests(Base):
    def merge(self) -> str:
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed,
                                  merged_text=ABSORBED, at=NOW)
        result = merge_diff.apply(self.store, diff, consent=True, consented_by="operator", at=NOW)
        self.assertTrue(result.ok, result.reason)
        return result.merge_id

    def test_the_undo_record_carries_ids_and_every_previous_value(self) -> None:
        diff = merge_diff.compute(self.store, self.survivor, self.absorbed,
                                  merged_text=ABSORBED, at=NOW)
        undo = diff.undo
        self.assertEqual(undo["survivor"]["id"], self.survivor)
        self.assertEqual(undo["absorbed"]["id"], self.absorbed)
        for name in ("scope_json", "pinned", "valid_to", "content", "state"):
            self.assertIn(name, undo["survivor"]["before"])
            self.assertIn(name, undo["absorbed"]["before"])
        self.assertEqual(json.loads(undo["survivor"]["before"]["scope_json"]),
                         {"project": "dashboard"})

    def test_undo_restores_both_rows_and_the_edges(self) -> None:
        third = self.memory("Nginx terminates TLS for the dashboard.")
        self.store.add_edge(third, self.absorbed, "supports")
        before_survivor = self.row(self.survivor)
        before_absorbed = self.row(self.absorbed)
        merge_id = self.merge()
        self.assertEqual(self.row(self.absorbed)["state"], "superseded")
        self.assertIsNotNone(self.row(self.absorbed)["valid_to"])
        self.assertIn("cyt116", self.row(self.survivor)["content"])
        result = merge_diff.undo(self.store, merge_id, at=NOW)
        self.assertTrue(result["ok"], result["reason"])
        after_survivor = self.row(self.survivor)
        after_absorbed = self.row(self.absorbed)
        for name in ("content", "scope_json", "pinned", "state", "valid_to", "content_hash"):
            self.assertEqual(after_survivor[name], before_survivor[name], name)
            self.assertEqual(after_absorbed[name], before_absorbed[name], name)
        edges = self.store._conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE relation = 'supports'").fetchall()
        self.assertEqual([tuple(row) for row in edges], [(third, self.absorbed)])

    def test_undo_preview_is_read_only_and_shows_the_reverse(self) -> None:
        merge_id = self.merge()
        before = self.snapshot()
        preview = merge_diff.preview_undo(self.store, merge_id)
        self.assertIsNotNone(preview)
        self.assertTrue(preview["exact"])
        fields = {item["field"] for item in preview["fields"]}
        self.assertIn("content", fields)
        self.assertIn("state", fields)
        self.assertEqual(self.snapshot(), before)

    def test_undo_twice_is_refused(self) -> None:
        merge_id = self.merge()
        self.assertTrue(merge_diff.undo(self.store, merge_id)["ok"])
        second = merge_diff.undo(self.store, merge_id)
        self.assertFalse(second["ok"])
        self.assertIn("already", second["reason"])

    def test_an_unknown_merge_id_undoes_nothing(self) -> None:
        before = self.snapshot()
        result = merge_diff.undo(self.store, "does-not-exist")
        self.assertFalse(result["ok"])
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
