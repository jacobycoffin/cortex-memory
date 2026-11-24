"""Move memory between harnesses without changing what it means (#153)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import portability as po
from cortex.store import CortexStore


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.source = CortexStore(Path(self.tmp.name) / "source.db")
        self.target = CortexStore(Path(self.tmp.name) / "target.db")

    def tearDown(self):
        self.source.close()
        self.target.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, kind: str = "semantic",
               category: str = "USER_STATED") -> str:
        return self.source.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category=category,
        )[0]

    def test_a_bundle_describes_itself(self) -> None:
        self.memory("the dashboard listens on port 8100")
        payload = po.bundle(self.source)
        self.assertEqual(payload["format"], po.FORMAT)
        self.assertEqual(payload["version"], po.VERSION)
        self.assertTrue(payload["guarantees"])

    def test_ids_travel_unchanged(self) -> None:
        memory_id = self.memory("the dashboard listens on port 8100")
        payload = po.bundle(self.source)
        self.assertEqual(payload["memories"][0]["id"], memory_id)

    def test_provenance_and_confidence_travel(self) -> None:
        self.memory("a fact", category="USER_EXPLICIT")
        record = po.bundle(self.source)["memories"][0]
        self.assertEqual(record["source_category"], "USER_EXPLICIT")
        self.assertIn("confidence", record)

    def test_an_unknown_kind_is_not_coerced(self) -> None:
        self.memory("an odd fact", kind="mystery")
        self.assertEqual(po.bundle(self.source)["memories"][0]["kind"], "mystery")

    def test_an_import_adds_the_memory(self) -> None:
        memory_id = self.memory("the dashboard listens on port 8100")
        result = po.import_bundle(self.target, po.bundle(self.source), dry_run=False)
        self.assertTrue(result.ok)
        self.assertEqual(result.added, 1)
        rows = self.target._conn.execute("SELECT * FROM memories WHERE id = ?",
                                         (memory_id,)).fetchall()
        self.assertEqual(len(rows), 1)

    def test_a_dry_run_writes_nothing(self) -> None:
        self.memory("the dashboard listens on port 8100")
        result = po.import_bundle(self.target, po.bundle(self.source))
        self.assertEqual(result.added, 1)
        count = self.target._conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"]
        self.assertEqual(int(count), 0)

    def test_importing_twice_adds_nothing(self) -> None:
        """Idempotent, or a re-import doubles the store."""
        self.memory("the dashboard listens on port 8100")
        payload = po.bundle(self.source)
        po.import_bundle(self.target, payload, dry_run=False)
        second = po.import_bundle(self.target, payload, dry_run=False)
        self.assertEqual(second.added, 0)
        self.assertEqual(len(second.collisions), 1)

    def test_a_collision_is_refused_not_overwritten(self) -> None:
        """Two harnesses with the same id are not the same memory; last-writer-wins is a rumour."""
        memory_id = self.memory("the dashboard listens on port 8100")
        payload = po.bundle(self.source)
        po.import_bundle(self.target, payload, dry_run=False)
        with self.target.transaction() as conn:
            conn.execute("UPDATE memories SET content = 'a different fact entirely' WHERE id = ?",
                         (memory_id,))
        result = po.import_bundle(self.target, payload, dry_run=False)
        self.assertEqual(result.added, 0)
        self.assertIn(memory_id, result.collisions)
        self.assertIn("refusing to overwrite", result.reason)
        kept = self.target._conn.execute("SELECT content FROM memories WHERE id = ?",
                                        (memory_id,)).fetchone()["content"]
        self.assertIn("different fact", kept)

    def test_a_superseded_parent_travels_with_its_child(self) -> None:
        """Without the parent the bundle is unimportable: the target store enforces the reference."""
        old = self.memory("the port is 8100")
        new = self.memory("the port is 8101")
        with self.source.transaction() as conn:
            conn.execute("UPDATE memories SET supersedes_id = ? WHERE id = ?", (old, new))
        payload = po.bundle(self.source, ids=[new], include_edges=False)
        self.assertEqual(len(payload["memories"]), 2)
        result = po.import_bundle(self.target, payload, dry_run=False)
        self.assertEqual(result.added, 2)
        self.assertEqual(result.dangling, [])

    def test_a_link_outside_the_bundle_is_reported_dangling(self) -> None:
        old = self.memory("the port is 8100")
        new = self.memory("the port is 8101")
        with self.source.transaction() as conn:
            conn.execute("UPDATE memories SET supersedes_id = ? WHERE id = ?", (old, new))
        payload = po.bundle(self.source, ids=[new], include_edges=False, include_parents=False)
        result = po.import_bundle(self.target, payload, dry_run=False)
        self.assertIn(new, result.dangling)
        # And the store refuses it, which is why parents travel by default.
        self.assertEqual(result.added, 0)

    def test_a_link_inside_the_bundle_is_not_dangling(self) -> None:
        old = self.memory("the port is 8100")
        new = self.memory("the port is 8101")
        with self.source.transaction() as conn:
            conn.execute("UPDATE memories SET supersedes_id = ? WHERE id = ?", (old, new))
        payload = po.bundle(self.source)
        result = po.import_bundle(self.target, payload, dry_run=False)
        self.assertEqual(result.dangling, [])

    def test_edges_are_exported_only_when_both_ends_travel(self) -> None:
        first = self.memory("a fact about the port")
        second = self.memory("another fact about the port")
        self.source.add_edge(first, second, "related", weight=0.5)
        everything = po.bundle(self.source)
        self.assertEqual(len(everything["edges"]), 1)
        partial = po.bundle(self.source, ids=[first])
        self.assertEqual(partial["edges"], [])

    def test_a_newer_bundle_version_is_refused(self) -> None:
        payload = po.bundle(self.source)
        payload["version"] = po.VERSION + 1
        result = po.import_bundle(self.target, payload, dry_run=False)
        self.assertFalse(result.ok)
        self.assertIn("newer than this importer", result.reason)

    def test_a_foreign_payload_is_refused(self) -> None:
        result = po.import_bundle(self.target, {"memories": []}, dry_run=False)
        self.assertFalse(result.ok)

    def test_a_bundle_missing_a_required_field_is_refused_by_name(self) -> None:
        """A sqlite constraint error is not an explanation; the missing field is."""
        payload = po.bundle(self.source)
        payload["memories"] = [{"id": "x", "kind": "semantic", "content": "a fact"}]
        result = po.import_bundle(self.target, payload, dry_run=False)
        self.assertFalse(result.ok)
        self.assertIn("missing required field", result.reason)

    def test_a_round_trip_changes_nothing(self) -> None:
        memory_id = self.memory("the dashboard listens on port 8100 behind the proxy")
        described = po.round_trip(self.source, memory_id, self.target)
        self.assertTrue(described["ok"], described)
        self.assertEqual(described["differences"], [])

    def test_a_round_trip_of_a_missing_memory_says_so(self) -> None:
        self.assertIn("no such memory", po.round_trip(self.source, "nope", self.target)["reason"])

    def test_the_render_states_the_size_and_the_links(self) -> None:
        self.memory("a fact")
        rendered = "\n".join(po.render(self.source))
        self.assertIn("portability:", rendered)
        self.assertIn("format", rendered)

    def test_a_bundle_is_json_serialisable(self) -> None:
        self.memory("a fact")
        self.assertIn("memories", json.loads(json.dumps(po.bundle(self.source))))

    def test_a_broken_store_exports_an_empty_bundle(self) -> None:
        class Broken:
            _conn = None

        payload = po.bundle(Broken())
        self.assertEqual(payload["memories"], [])
        self.assertFalse(po.import_bundle(Broken(), {"format": po.FORMAT, "version": 1,
                                                     "memories": []}, dry_run=True).ok is None)


if __name__ == "__main__":
    unittest.main()
