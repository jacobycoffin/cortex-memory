"""Phone-first memory surface: list, search, timeline (#159)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import phone_surface
from cortex.store import CortexStore


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, kind: str = "semantic") -> str:
        return self.store.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category="USER_STATED",
        )[0]

    def test_the_landing_view_lists_recent_memories(self) -> None:
        first = self.memory("The dashboard listens on 8100.")
        second = self.memory("Nginx proxies the dashboard.")
        page = phone_surface.recent(self.store)
        self.assertEqual(page["view"], "list")
        self.assertEqual([item["id"] for item in page["items"]], [second, first])

    def test_the_preview_is_bounded(self) -> None:
        self.memory("x" * 400)
        page = phone_surface.recent(self.store)
        self.assertLessEqual(len(page["items"][0]["text"]), phone_surface.PREVIEW)

    def test_the_response_says_how_many_more_there_are(self) -> None:
        for index in range(6):
            self.memory(f"fact {index}")
        page = phone_surface.recent(self.store, limit=2)
        self.assertEqual(len(page["items"]), 2)
        self.assertEqual(page["total"], 6)
        self.assertEqual(page["more"], 4)

    def test_the_total_is_a_count_not_the_page_size(self) -> None:
        for index in range(3):
            self.memory(f"fact {index}")
        page = phone_surface.recent(self.store, limit=1)
        self.assertEqual(page["total"], 3)

    def test_archived_memories_are_not_on_the_first_screen(self) -> None:
        kept = self.memory("kept")
        gone = self.memory("gone")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'archived' WHERE id = ?", (gone,))
        page = phone_surface.recent(self.store)
        self.assertEqual([item["id"] for item in page["items"]], [kept])

    def test_a_kind_filter_is_respected(self) -> None:
        self.memory("a preference", kind="preference")
        semantic = self.memory("a fact", kind="semantic")
        page = phone_surface.recent(self.store, kind="semantic")
        self.assertEqual([item["id"] for item in page["items"]], [semantic])

    def test_search_filters(self) -> None:
        wanted = self.memory("The dashboard listens on 8100.")
        self.memory("The vault indexer runs on a timer.")
        page = phone_surface.search(self.store, "dashboard")
        self.assertEqual([item["id"] for item in page["items"]], [wanted])
        self.assertEqual(page["view"], "search")

    def test_an_empty_search_falls_back_to_the_list(self) -> None:
        self.memory("anything")
        page = phone_surface.search(self.store, "   ")
        self.assertEqual(page["view"], "search")
        self.assertEqual(page["query"], "")
        self.assertEqual(len(page["items"]), 1)

    def test_a_search_with_no_match_is_empty_not_an_error(self) -> None:
        self.memory("anything")
        page = phone_surface.search(self.store, "zzzzz")
        self.assertEqual(page["items"], [])
        self.assertEqual(page["total"], 0)

    def test_the_timeline_groups_by_day(self) -> None:
        self.memory("today one")
        self.memory("today two")
        page = phone_surface.timeline(self.store)
        self.assertEqual(page["view"], "timeline")
        self.assertEqual(len(page["days"]), 1)
        self.assertEqual(len(page["days"][0]["items"]), 2)

    def test_the_timeline_is_newest_day_first(self) -> None:
        first = self.memory("older")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET created_at = '2026-01-01T00:00:00+00:00' WHERE id = ?",
                         (first,))
        second = self.memory("newer")
        page = phone_surface.timeline(self.store)
        # Computed, not hardcoded: a literal "today" makes this test fail the day after it is written,
        # and it must be the UTC day because that is what the surface groups by.
        from datetime import datetime, timezone
        self.assertEqual(page["days"][0]["day"], datetime.now(timezone.utc).date().isoformat())
        self.assertEqual([item["id"] for item in page["days"][0]["items"]], [second])

    def test_detail_returns_one_memory_and_its_neighbours(self) -> None:
        first = self.memory("The dashboard listens on 8100.")
        second = self.memory("Nginx proxies the dashboard.")
        self.store.add_edge(first, second, "co_retrieval", weight=0.8)
        found = phone_surface.detail(self.store, first)
        self.assertTrue(found["found"])
        self.assertEqual(found["memory"]["id"], first)
        self.assertEqual([item["id"] for item in found["neighbours"]], [second])

    def test_detail_says_so_when_the_memory_is_missing(self) -> None:
        found = phone_surface.detail(self.store, "nope")
        self.assertFalse(found["found"])

    def test_only_detail_returns_edges(self) -> None:
        """The graph is a drill-down; a list that carries edges is a graph in disguise."""
        first = self.memory("a")
        second = self.memory("b")
        self.store.add_edge(first, second, "co_retrieval", weight=0.5)
        for page in (phone_surface.recent(self.store), phone_surface.timeline(self.store),
                     phone_surface.search(self.store, "a")):
            self.assertNotIn("neighbours", page)
            self.assertNotIn("edges", page)

    def test_the_payload_router_defaults_to_the_list(self) -> None:
        self.memory("anything")
        self.assertEqual(phone_surface.payload(self.store)["view"], "list")

    def test_the_payload_router_routes_each_view(self) -> None:
        memory_id = self.memory("anything")
        self.assertEqual(phone_surface.payload(self.store, "timeline")["view"], "timeline")
        self.assertEqual(
            phone_surface.payload(self.store, "search", query="anything")["view"], "search")
        self.assertEqual(
            phone_surface.payload(self.store, "detail", memory_id=memory_id)["view"], "detail")

    def test_the_limit_is_clamped(self) -> None:
        for index in range(3):
            self.memory(f"fact {index}")
        self.assertEqual(len(phone_surface.recent(self.store, limit=10 ** 6)["items"]), 3)
        self.assertLessEqual(len(phone_surface.recent(self.store, limit=0)["items"]), 1)

    def test_a_broken_store_renders_an_empty_list(self) -> None:
        class Broken:
            _conn = None

        page = phone_surface.recent(Broken())
        self.assertEqual(page["items"], [])

    def test_the_text_rendering_is_readable(self) -> None:
        memory_id = self.memory("The dashboard listens on 8100.")
        lines = phone_surface.render_text(phone_surface.detail(self.store, memory_id))
        self.assertIn(memory_id[:8], " ".join(lines))
        self.assertIn("dashboard", " ".join(lines))

    def test_the_text_rendering_names_a_missing_memory(self) -> None:
        lines = phone_surface.render_text(phone_surface.detail(self.store, "nope"))
        self.assertIn("not found", lines[0])


class AssetTests(unittest.TestCase):
    """The page itself has to be phone-shaped, not a desktop dashboard with a smaller viewport."""

    def setUp(self) -> None:
        self.html = (Path(phone_surface.__file__).with_name("phone.html")).read_text(
            encoding="utf-8")

    def test_the_page_declares_a_phone_viewport(self) -> None:
        self.assertIn("width=device-width", self.html)

    def test_the_landing_page_has_no_canvas(self) -> None:
        """A force layout is the desktop's job; on a phone it is a pinch target."""
        self.assertNotIn("<canvas", self.html)

    def test_the_tap_targets_are_big_enough(self) -> None:
        self.assertIn("min-height: 44px", self.html)

    def test_the_page_is_small(self) -> None:
        self.assertLess(len(self.html.encode("utf-8")), 20000)

    def test_the_page_talks_to_the_phone_endpoint(self) -> None:
        self.assertIn("/api/phone", self.html)


if __name__ == "__main__":
    unittest.main()


class SheetTests(unittest.TestCase):
    """A bottom sheet has one screen and no back button."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, **kwargs) -> str:
        return self.store.add_memory(
            content=text, kind=kwargs.get("kind", "semantic"), source_type="vault",
            source_category="USER_STATED", confidence=kwargs.get("confidence", 0.86),
        )[0]

    def test_the_sheet_carries_provenance(self) -> None:
        memory_id = self.memory("The dashboard listens on 8100.")
        found = phone_surface.sheet(self.store, memory_id)
        self.assertEqual(found["view"], "sheet")
        self.assertEqual(found["provenance"]["kind"], "semantic")
        self.assertEqual(found["provenance"]["confidence"], 0.86)

    def test_the_provenance_is_quoted_from_the_row(self) -> None:
        memory_id = self.memory("A fact.")
        found = phone_surface.sheet(self.store, memory_id)
        row = self.store._conn.execute(
            "SELECT source_type, created_at FROM memories WHERE id = ?", (memory_id,)).fetchone()
        self.assertEqual(found["provenance"]["source_type"], row["source_type"])
        self.assertTrue(found["provenance"]["created_at"])

    def test_a_missing_memory_is_reported(self) -> None:
        found = phone_surface.sheet(self.store, "nope")
        self.assertFalse(found["found"])

    def test_neighbours_are_included(self) -> None:
        first = self.memory("a")
        second = self.memory("b")
        self.store.add_edge(first, second, "co_retrieval", weight=0.7)
        found = phone_surface.sheet(self.store, first)
        self.assertEqual([item["id"] for item in found["neighbours"]], [second])

    def test_cited_by_is_unknown_when_there_is_no_citation_table(self) -> None:
        memory_id = self.memory("A fact.")
        self.assertIsNone(phone_surface.sheet(self.store, memory_id)["cited_by"])

    def test_the_router_serves_the_sheet_view(self) -> None:
        memory_id = self.memory("A fact.")
        found = phone_surface.payload(self.store, "sheet", memory_id=memory_id)
        self.assertEqual(found["view"], "sheet")


class SheetMarkupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.html = (Path(phone_surface.__file__).with_name("phone.html")).read_text(
            encoding="utf-8")

    def test_the_sheet_is_a_modal_dialog(self) -> None:
        self.assertIn('role="dialog"', self.html)
        self.assertIn('aria-modal="true"', self.html)

    def test_the_sheet_starts_hidden(self) -> None:
        self.assertIn('aria-hidden="true"', self.html)

    def test_there_is_a_close_control_big_enough_for_a_thumb(self) -> None:
        self.assertIn('id="sheet-close"', self.html)
        self.assertIn("min-height: 44px", self.html)

    def test_tapping_a_row_opts_out_of_navigation(self) -> None:
        """The list keeps its scroll position because the sheet is not a page."""
        self.assertIn("preventDefault()", self.html)

    def test_the_sheet_can_be_dismissed_three_ways(self) -> None:
        self.assertIn("sheet-close", self.html)      # the button
        self.assertIn("backdrop.addEventListener", self.html)  # the backdrop
        self.assertIn('event.key === "Escape"', self.html)      # the keyboard

    def test_the_sheet_asks_for_its_own_view(self) -> None:
        self.assertIn("view=sheet", self.html)

    def test_the_page_is_still_small(self) -> None:
        self.assertLess(len(self.html.encode("utf-8")), 26000)
