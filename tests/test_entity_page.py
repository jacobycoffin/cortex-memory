"""One page per entity (#166).

The page's rules are the tests: a section names its table, an empty section is dropped, a
retired belief is shown and marked, the page is bounded and says what it cut, and an unknown
entity is an answer rather than a blank page.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import entity_nodes, entity_page
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class PageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, entities: tuple[str, ...] = ("proxmox",),
               days_ago: int = 0, **kwargs) -> str:
        memory_id = self.store.add_memory(
            content=text, kind=kwargs.pop("kind", "semantic"),
            source_type="user_turn", source_category="USER_STATED",
            entities=list(entities), **kwargs,
        )[0]
        stamp = (NOW - timedelta(days=days_ago)).isoformat()
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET created_at = ?, observed_at = ? WHERE id = ?",
                         (stamp, stamp, memory_id))
        return memory_id

    # -- rule (e): an unknown entity is an answer, not an empty page -------------------

    def test_an_unknown_entity_says_so_rather_than_printing_nothing(self) -> None:
        page = entity_page.build(self.store, "no-such-box")
        self.assertFalse(page.found)
        self.assertIn("unknown entity", page.reason)
        self.assertEqual(page.sections, [])

    def test_the_render_of_an_unknown_entity_names_the_tables_it_asked(self) -> None:
        lines = entity_page.render(self.store, "no-such-box")
        self.assertEqual(len(lines), 1)
        self.assertIn("unknown entity", lines[0])
        self.assertIn("memories", lines[0])

    def test_an_entity_with_no_node_is_resolved_from_its_memories(self) -> None:
        """The node is a summary of the memories; the memories are the evidence."""
        self.memory("the proxmox box runs 8.4")
        page = entity_page.build(self.store, "proxmox")
        self.assertTrue(page.found)
        self.assertEqual(page.resolved_by, "memories")
        self.assertEqual(page.canonical_name, "proxmox")

    def test_a_node_supplies_the_canonical_name_kind_and_aliases(self) -> None:
        self.memory("the proxmox box runs 8.4")
        entity_nodes.create(self.store, "proxmox", "Proxmox VE", kind="service",
                            summary="the hypervisor box")
        page = entity_page.build(self.store, "proxmox")
        self.assertEqual(page.entity_id, "proxmox")
        self.assertEqual(page.canonical_name, "Proxmox VE")
        self.assertEqual(page.kind, "service")
        self.assertEqual(page.summary, "the hypervisor box")

    # -- rule (a): every section states its source table -------------------------------

    def test_every_section_names_a_source_table(self) -> None:
        first = self.memory("the proxmox box runs 8.4")
        second = self.memory("the proxmox box runs 8.3", days_ago=30)
        self.store.add_edge(first, second, "supports", weight=0.6)
        page = entity_page.build(self.store, "proxmox")
        self.assertTrue(page.sections)
        for section in page.sections:
            self.assertTrue(section.source, f"{section.key} has no source")
            self.assertIn(section.source.split(".")[0],
                          ("memories", "edges", "entity_nodes", "aliases"))

    def test_the_render_prints_the_source_of_each_section(self) -> None:
        first = self.memory("the proxmox box runs 8.4")
        second = self.memory("the proxmox box runs 8.3", days_ago=30)
        self.store.add_edge(first, second, "supports", weight=0.6)
        rendered = "\n".join(entity_page.render(self.store, "proxmox"))
        self.assertIn("[source: memories.entities_json]", rendered)
        self.assertIn("[source: edges]", rendered)

    # -- rule (b): an empty section is omitted, not printed empty ----------------------

    def test_a_section_with_no_data_is_omitted(self) -> None:
        self.memory("the proxmox box runs 8.4")
        page = entity_page.build(self.store, "proxmox")
        self.assertIsNotNone(page.section("memories"))
        for key in ("edges", "changes", "contradictions", "retirements"):
            self.assertIsNone(page.section(key), f"{key} should have been omitted")
        self.assertNotIn("retirements", entity_page.build(self.store, "proxmox").as_dict()["sections"][0]["key"])

    def test_the_render_has_no_heading_without_items(self) -> None:
        self.memory("the proxmox box runs 8.4")
        rendered = "\n".join(entity_page.render(self.store, "proxmox"))
        for absent in ("edges:", "changes:", "contradictions:", "retirements:"):
            self.assertNotIn(absent, rendered)

    # -- rule (c): a retired belief is shown and marked, never hidden ------------------

    def test_a_retired_belief_is_shown_and_marked(self) -> None:
        retired = self.memory("the proxmox box runs 8.3", days_ago=40,
                              valid_to=(NOW - timedelta(days=5)).isoformat())
        page = entity_page.build(self.store, "proxmox")
        memories = page.section("memories")
        self.assertEqual(memories.total, 1)
        self.assertEqual(memories.items[0]["id"], retired)
        self.assertTrue(memories.items[0]["retired_at"])
        retirements = page.section("retirements")
        self.assertIsNotNone(retirements, "a retired belief must have a retirement section")
        self.assertEqual(retirements.items[0]["id"], retired)

    def test_the_rendered_retired_line_carries_the_mark(self) -> None:
        self.memory("the proxmox box runs 8.3", valid_to=NOW.isoformat())
        rendered = "\n".join(entity_page.render(self.store, "proxmox"))
        self.assertIn("!retired", rendered)

    def test_a_retired_belief_names_what_replaced_it(self) -> None:
        old = self.memory("the proxmox box runs 8.3", days_ago=40,
                          valid_to=(NOW - timedelta(days=5)).isoformat())
        self.memory("the proxmox box runs 8.4", days_ago=5, supersedes_id=old)
        retirements = entity_page.build(self.store, "proxmox").section("retirements")
        self.assertTrue(retirements.items[0]["replaced_by"])

    # -- bounded page ------------------------------------------------------------------

    def test_memories_are_newest_first(self) -> None:
        self.memory("the proxmox box runs 8.2", days_ago=60)
        self.memory("the proxmox box runs 8.4", days_ago=1)
        self.memory("the proxmox box runs 8.3", days_ago=30)
        items = entity_page.build(self.store, "proxmox").section("memories").items
        self.assertIn("8.4", items[0]["content"])
        self.assertIn("8.2", items[-1]["content"])

    def test_memories_are_capped_and_the_truncation_is_stated(self) -> None:
        for index in range(entity_page.MAX_MEMORIES + 5):
            self.memory(f"the proxmox box runs 8.4 build {index}", days_ago=index)
        section = entity_page.build(self.store, "proxmox").section("memories")
        self.assertEqual(len(section.items), entity_page.MAX_MEMORIES)
        self.assertEqual(section.total, entity_page.MAX_MEMORIES + 5)
        self.assertEqual(section.truncated, 5)
        self.assertIn("5 not shown", "\n".join(entity_page.render(self.store, "proxmox")))

    def test_a_caller_can_widen_one_section(self) -> None:
        for index in range(4):
            self.memory(f"the proxmox box runs 8.{index}", days_ago=index)
        limits = entity_page.Limits(memories=2)
        self.assertEqual(len(entity_page.build(self.store, "proxmox",
                                                limits=limits).section("memories").items), 2)

    # -- edges --------------------------------------------------------------------------

    def test_edges_are_grouped_by_relation_with_weights(self) -> None:
        anchor = self.memory("the proxmox box runs 8.4")
        for index, weight in enumerate((0.2, 0.6)):
            other = self.memory(f"the proxmox backup job number {index}")
            self.store.add_edge(anchor, other, "supports", weight=weight)
        other = self.memory("the proxmox box is in the rack")
        self.store.add_edge(anchor, other, "co_used", weight=0.4)
        section = entity_page.build(self.store, "proxmox").section("edges")
        relations = {item["relation"]: item for item in section.items}
        self.assertEqual(relations["supports"]["count"], 2)
        self.assertAlmostEqual(relations["supports"]["mean_weight"], 0.4, places=3)
        self.assertEqual(relations["co_used"]["count"], 1)
        self.assertEqual(section.total, 3)

    def test_edges_are_found_when_the_entity_is_on_the_destination_side(self) -> None:
        """`add_edge` stores canonical order, so a one-sided query reports an isolated memory."""
        entity_memory = self.memory("the proxmox box runs 8.4")
        other = self.memory("the rack is in the basement", entities=("basement",))
        self.store.add_edge(other, entity_memory, "supports", weight=0.5)
        section = entity_page.build(self.store, "proxmox").section("edges")
        self.assertIsNotNone(section)
        self.assertEqual(section.total, 1)
        self.assertEqual(section.items[0]["items"][0]["other"], other)

    def test_the_strongest_link_is_explained_in_a_sentence(self) -> None:
        anchor = self.memory("the proxmox box runs 8.4")
        other = self.memory("the proxmox backup job runs nightly")
        self.store.add_edge(anchor, other, "supports", weight=0.7,
                            evidence_type="explicit_wikilink")
        section = entity_page.build(self.store, "proxmox").section("edges")
        self.assertTrue(section.items[0]["items"][0].get("sentence"))

    # -- changes -------------------------------------------------------------------------

    def test_a_supersession_pair_becomes_a_change(self) -> None:
        old = self.memory("the proxmox box runs 8.3", days_ago=40)
        self.memory("the proxmox box runs 8.4", days_ago=2, supersedes_id=old)
        section = entity_page.build(self.store, "proxmox").section("changes")
        self.assertIsNotNone(section)
        self.assertEqual(section.total, 1)
        self.assertIn("8.3", section.items[0]["was"])
        self.assertIn("8.4", section.items[0]["now"])

    def test_two_disagreeing_memories_are_not_reported_as_a_change(self) -> None:
        """Only a supersession is a change; a plain disagreement is a contradiction."""
        self.memory("the proxmox box runs 8.3", days_ago=40)
        self.memory("the proxmox box runs 8.4", days_ago=2)
        page = entity_page.build(self.store, "proxmox")
        self.assertIsNone(page.section("changes"))

    # -- contradictions -------------------------------------------------------------------

    def test_a_contradiction_on_the_same_claim_is_surfaced(self) -> None:
        self.memory("the proxmox maintenance window is Sunday",
                    subject="proxmox", predicate="maintenance day", object_value="Sunday")
        self.memory("the proxmox maintenance window is Tuesday",
                    subject="proxmox", predicate="maintenance day", object_value="Tuesday")
        section = entity_page.build(self.store, "proxmox").section("contradictions")
        self.assertIsNotNone(section)
        self.assertEqual(section.total, 1)
        self.assertIn("maintenance day", section.items[0]["reason"])

    def test_the_contradiction_scan_is_bounded(self) -> None:
        for index in range(entity_page.CONFLICT_SCAN_CAP + 5):
            self.memory(f"proxmox note number {index} about nothing in particular",
                        subject=f"subject {index}", predicate="note", object_value=str(index))
        page = entity_page.build(self.store, "proxmox")
        self.assertTrue(page.found)  # quadratic work is capped, not the page

    # -- broken store -----------------------------------------------------------------------

    def test_a_broken_store_is_reported_not_raised(self) -> None:
        class Broken:
            _conn = None

        page = entity_page.build(Broken(), "proxmox")
        self.assertFalse(page.found)
        self.assertIn("could not be read", page.reason)
        self.assertEqual(entity_page.render(Broken(), "proxmox")[0].count("/"), 0)


    def test_an_alias_row_is_used_and_the_page_does_not_create_it(self) -> None:
        """`aliases.TABLE` is `entity_aliases`, not `aliases`: the guard must name the real table."""
        self.memory("the proxmox box runs 8.4")
        before = self.snapshot()[0]
        self.assertNotIn("entity_aliases", before)
        page = entity_page.build(self.store, "proxmox")
        self.assertEqual(before, self.snapshot()[0], "a page must not create the alias table")

        from cortex import aliases
        aliases.add(self.store, "proxmox", "pmx")  # the writer creates it, not the reader
        page = entity_page.build(self.store, "pmx")
        self.assertTrue(page.found)
        self.assertEqual(page.resolved_by, "aliases")
        self.assertEqual(page.entity_id, "proxmox")

    def test_the_page_only_reads(self) -> None:
        first = self.memory("the proxmox box runs 8.4")
        second = self.memory("the proxmox box runs 8.3", days_ago=30)
        self.store.add_edge(first, second, "supports", weight=0.6)
        before = self.snapshot()
        entity_page.build(self.store, "proxmox")
        entity_page.render(self.store, "proxmox")
        self.assertEqual(before, self.snapshot())

    def snapshot(self) -> tuple:
        tables = tuple(sorted(
            str(row["name"]) for row in
            self.store._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        ))
        counts = tuple(
            int(self.store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        )
        return tables, counts

    def test_the_report_states_which_tables_exist(self) -> None:
        described = entity_page.report(self.store)
        self.assertFalse(described["entity_nodes"])
        self.assertTrue(described["edges"])


if __name__ == "__main__":
    unittest.main()
