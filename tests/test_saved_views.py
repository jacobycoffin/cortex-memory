"""Saved views: a filter with a name, never a script (#163)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import saved_views
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, kind: str = "semantic", pinned: bool = False,
               scope: dict | None = None, confidence: float = 0.9) -> str:
        memory_id = self.store.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category="USER_STATED",
            pinned=pinned, confidence=confidence,
        )[0]
        if scope:
            with self.store.transaction() as conn:
                conn.execute("UPDATE memories SET scope_json = ? WHERE id = ?",
                             (json.dumps(scope), memory_id))
        return memory_id

    def test_a_view_round_trips(self) -> None:
        saved_views.save(self.store, "recent", {"sort": "recent"}, at=NOW)
        found = saved_views.run(self.store, "recent", at=NOW)
        self.assertTrue(found["found"])
        self.assertEqual(found["spec"], {"sort": "recent"})

    def test_an_unknown_field_is_refused(self) -> None:
        """A typo that disappears produces a view that is quietly wrong."""
        with self.assertRaises(saved_views.SpecError):
            saved_views.save(self.store, "bad", {"sort": "recent", "knd": "semantic"})

    def test_a_sql_fragment_is_refused(self) -> None:
        with self.assertRaises(saved_views.SpecError):
            saved_views.save(self.store, "bad", {"query": "x'; DROP TABLE memories; --"})

    def test_an_unknown_sort_is_refused(self) -> None:
        with self.assertRaises(saved_views.SpecError):
            saved_views.save(self.store, "bad", {"sort": "random()"})

    def test_an_unknown_state_is_refused(self) -> None:
        with self.assertRaises(saved_views.SpecError):
            saved_views.save(self.store, "bad", {"state": "everything"})

    def test_a_view_needs_a_name(self) -> None:
        with self.assertRaises(saved_views.SpecError):
            saved_views.save(self.store, "  ", {"limit": 5})

    def test_the_limit_is_clamped(self) -> None:
        saved_views.save(self.store, "big", {"limit": 10 ** 6}, at=NOW)
        spec = saved_views.views(self.store)[0]["spec"]
        self.assertEqual(spec["limit"], saved_views.MAX_LIMIT)

    def test_a_kind_filter_selects_only_that_kind(self) -> None:
        wanted = self.memory("a preference", kind="preference")
        self.memory("a fact", kind="semantic")
        saved_views.save(self.store, "prefs", {"kind": "preference"}, at=NOW)
        found = saved_views.run(self.store, "prefs", at=NOW)
        self.assertEqual([item["id"] for item in found["items"]], [wanted])

    def test_a_project_filter_reads_the_scope_tag(self) -> None:
        wanted = self.memory("a duelist-road fact", scope={"project": "duelist-road"})
        self.memory("an unrelated fact")
        saved_views.save(self.store, "dr", {"project": "duelist-road"}, at=NOW)
        found = saved_views.run(self.store, "dr", at=NOW)
        self.assertEqual([item["id"] for item in found["items"]], [wanted])

    def test_a_pinned_view_only_shows_pins(self) -> None:
        pinned = self.memory("pinned fact", pinned=True)
        self.memory("unpinned fact")
        saved_views.save(self.store, "pins", {"pinned_only": True}, at=NOW)
        self.assertEqual([i["id"] for i in saved_views.run(self.store, "pins", at=NOW)["items"]],
                         [pinned])

    def test_archived_memories_are_hidden_by_default(self) -> None:
        kept = self.memory("kept")
        gone = self.memory("gone")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'archived' WHERE id = ?", (gone,))
        saved_views.save(self.store, "recent", {}, at=NOW)
        self.assertEqual([i["id"] for i in saved_views.run(self.store, "recent", at=NOW)["items"]],
                         [kept])

    def test_an_archived_view_can_ask_for_archived(self) -> None:
        gone = self.memory("gone")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'archived' WHERE id = ?", (gone,))
        saved_views.save(self.store, "gone", {"state": "archived"}, at=NOW)
        self.assertEqual([i["id"] for i in saved_views.run(self.store, "gone", at=NOW)["items"]],
                         [gone])

    def test_the_total_is_a_count_not_a_page(self) -> None:
        for index in range(5):
            self.memory(f"fact {index}")
        saved_views.save(self.store, "few", {"limit": 2}, at=NOW)
        found = saved_views.run(self.store, "few", at=NOW)
        self.assertEqual(len(found["items"]), 2)
        self.assertEqual(found["total"], 5)
        self.assertEqual(found["more"], 3)

    def test_explain_shows_the_clause_and_the_match_count(self) -> None:
        self.memory("a preference", kind="preference")
        saved_views.save(self.store, "prefs", {"kind": "preference"}, at=NOW)
        described = saved_views.explain(self.store, "prefs")
        self.assertIn("kind = ?", described["sql"])
        self.assertEqual(described["matches"], 1)

    def test_explain_binds_its_values(self) -> None:
        saved_views.save(self.store, "q", {"query": "dashboard"}, at=NOW)
        described = saved_views.explain(self.store, "q")
        self.assertEqual(described["params"], ["%dashboard%"])

    def test_running_an_unknown_view_is_reported(self) -> None:
        self.assertFalse(saved_views.run(self.store, "nope", at=NOW)["found"])

    def test_saving_twice_replaces_the_spec_not_the_history(self) -> None:
        saved_views.save(self.store, "v", {"limit": 5}, at=NOW)
        again = saved_views.save(self.store, "v", {"limit": 9}, at=NOW + timedelta(hours=1))
        self.assertTrue(again["replaced"])
        self.assertEqual(again["created_at"], NOW.isoformat())

    def test_use_is_counted(self) -> None:
        saved_views.save(self.store, "v", {}, at=NOW)
        saved_views.run(self.store, "v", at=NOW)
        saved_views.run(self.store, "v", at=NOW)
        self.assertEqual(saved_views.views(self.store)[0]["use_count"], 2)

    def test_unused_views_are_named_with_their_age(self) -> None:
        saved_views.save(self.store, "forgotten", {}, at=NOW - timedelta(days=10))
        unused = saved_views.unused_views(self.store)
        self.assertEqual([row["name"] for row in unused], ["forgotten"])
        self.assertGreater(unused[0]["age_days"], 9)

    def test_a_view_that_was_used_is_not_listed_as_unused(self) -> None:
        saved_views.save(self.store, "v", {}, at=NOW)
        saved_views.run(self.store, "v", at=NOW)
        self.assertEqual(saved_views.unused_views(self.store), [])

    def test_views_are_listed_least_used_first(self) -> None:
        saved_views.save(self.store, "a", {}, at=NOW)
        saved_views.save(self.store, "b", {}, at=NOW)
        saved_views.run(self.store, "b", at=NOW)
        self.assertEqual([row["name"] for row in saved_views.views(self.store)], ["a", "b"])

    def test_delete_is_reported(self) -> None:
        saved_views.save(self.store, "v", {}, at=NOW)
        self.assertTrue(saved_views.delete(self.store, "v"))
        self.assertFalse(saved_views.delete(self.store, "v"))

    def test_the_presets_install_once(self) -> None:
        added = saved_views.install_presets(self.store, at=NOW)
        self.assertEqual(len(added), len(saved_views.presets()))
        self.assertEqual(saved_views.install_presets(self.store, at=NOW), [])

    def test_installing_presets_never_overwrites_a_tuned_view(self) -> None:
        saved_views.install_presets(self.store, at=NOW)
        saved_views.save(self.store, "new", {"limit": 7}, at=NOW)
        saved_views.install_presets(self.store, at=NOW)
        spec = [row["spec"] for row in saved_views.views(self.store) if row["name"] == "new"][0]
        self.assertEqual(spec["limit"], 7)

    def test_an_empty_spec_is_a_valid_recent_list(self) -> None:
        self.memory("anything")
        saved_views.save(self.store, "all", {}, at=NOW)
        self.assertEqual(len(saved_views.run(self.store, "all", at=NOW)["items"]), 1)


if __name__ == "__main__":
    unittest.main()
