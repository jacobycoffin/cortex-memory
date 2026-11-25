"""The sources behind a memory (#175): unknown is not zero, and a redaction is never echoed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import sources_panel
from cortex.store import CortexStore


class Broken:
    """A store whose connection cannot be used."""

    _conn = None


class PanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str = "the proxmox box runs 8.4", **kwargs) -> str:
        if "source_type" not in kwargs:
            kwargs["source_type"] = "user_turn"
        if "source_category" not in kwargs:
            kwargs["source_category"] = "USER_STATED"
        return self.store.add_memory(content=text, kind=kwargs.pop("kind", "semantic"), **kwargs)[0]

    def snapshot(self) -> tuple:
        tables = tuple(sorted(r["name"] for r in self.store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")))
        counts = tuple(int(self.store._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                       for t in tables)
        return tables, counts

    # -- provenance -------------------------------------------------------------------

    def test_the_panel_names_the_column_behind_every_value(self) -> None:
        panel = sources_panel.payload(self.store, self.memory())
        for item in panel.provenance:
            self.assertTrue(item.column, item.label)
        labels = {item.label: item for item in panel.provenance}
        self.assertEqual(labels["source type"].value, "user_turn")
        self.assertEqual(labels["source type"].column, "source_type")

    def test_a_missing_field_is_unknown_not_zero_and_not_blank(self) -> None:
        """`retrieved_count = 0` and 'no such column' are different claims; a panel must not merge them."""
        panel = sources_panel.payload(self.store, self.memory())
        counters = {item.label: item for item in panel.counters}
        for label, item in counters.items():
            with self.subTest(label=label):
                self.assertIn(item.value, (0, sources_panel.UNKNOWN))
                if item.value == sources_panel.UNKNOWN:
                    self.assertFalse(item.known)

    def test_a_counter_the_store_does_not_have_reads_unknown(self) -> None:
        panel = sources_panel.payload(self.store, self.memory())
        missing = [item for item in panel.counters if not item.known]
        for item in missing:
            self.assertEqual(item.value, "unknown")
            self.assertNotEqual(item.value, 0, "an absent measurement must not print as zero")

    def test_an_unknown_memory_is_named_not_rendered_blank(self) -> None:
        panel = sources_panel.payload(self.store, "no-such-id")
        self.assertFalse(panel.found)
        self.assertTrue(panel.readable)
        self.assertIn("unknown memory", panel.reason)
        self.assertEqual(sources_panel.render(panel), [panel.reason])

    # -- the store's state, honestly ------------------------------------------------

    def test_a_broken_store_says_it_could_not_be_read(self) -> None:
        panel = sources_panel.payload(Broken(), "abc")
        self.assertFalse(panel.readable)
        self.assertIn("could not be read", panel.reason)
        self.assertEqual(panel.provenance, [], "an unreadable store has no fields, not empty ones")

    def test_reading_a_panel_writes_nothing(self) -> None:
        memory_id = self.memory()
        self.store.add_edge(memory_id, self.memory("the box runs 8.3"), "supports", weight=0.5)
        before = self.snapshot()
        sources_panel.payload(self.store, memory_id)
        sources_panel.render(sources_panel.payload(self.store, memory_id))
        self.assertEqual(before, self.snapshot())

    # -- evidence, bounded -----------------------------------------------------------

    def test_edges_are_bounded_and_the_cut_is_stated(self) -> None:
        memory_id = self.memory()
        for index in range(sources_panel.MAX_EDGES + 4):
            self.store.add_edge(memory_id, self.memory(f"neighbour {index}"), "supports",
                                weight=0.1 * index)
        panel = sources_panel.payload(self.store, memory_id)
        self.assertEqual(len(panel.edges), sources_panel.MAX_EDGES)
        self.assertEqual(panel.edges_total, sources_panel.MAX_EDGES + 4)
        self.assertGreater(panel.edges_total, len(panel.edges), "the panel must say what it cut")

    def test_edges_are_ordered_strongest_first(self) -> None:
        memory_id = self.memory()
        for weight in (0.1, 0.9, 0.5):
            self.store.add_edge(memory_id, self.memory(f"n{weight}"), "supports", weight=weight)
        panel = sources_panel.payload(self.store, memory_id)
        weights = [entry["weight"] for entry in panel.edges]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_an_edge_records_its_direction_and_the_other_end(self) -> None:
        memory_id = self.memory()
        other = self.memory("the box runs 8.3")
        self.store.add_edge(memory_id, other, "supports", weight=0.5)
        panel = sources_panel.payload(self.store, memory_id)
        entry = panel.edges[0]
        self.assertIn(entry["direction"], {"in", "out"})
        self.assertEqual(entry["other"], other)
        self.assertEqual(entry["relation"], "supports")

    def test_usage_is_bounded_and_counted(self) -> None:
        memory_id = self.memory()
        with self.store.transaction() as conn:
            for index in range(sources_panel.MAX_USAGE + 3):
                conn.execute(
                    "INSERT INTO usage_records (usage_id, task_id, memory_id, session_id, query, score, "
                    "selected, used, attribution, outcome, created_at, resolved_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (f"u{index}", "t1", memory_id, "s1", "q", 0.5, 1, 0, "cited", "ok",
                     f"2026-11-2{index % 10}T00:00:00+00:00", None),
                )
        panel = sources_panel.payload(self.store, memory_id)
        self.assertEqual(len(panel.usage), sources_panel.MAX_USAGE)
        self.assertEqual(panel.usage_total, sources_panel.MAX_USAGE + 3)

    def test_a_memory_with_no_usage_says_none_rather_than_unknown(self) -> None:
        panel = sources_panel.payload(self.store, self.memory())
        self.assertEqual(panel.usage_total, 0)
        lines = sources_panel.render(panel)
        self.assertTrue(any("none for this memory" in line for line in lines))

    # -- what it cannot know, and what it must not repeat ---------------------------

    def test_the_panel_states_what_it_cannot_know(self) -> None:
        panel = sources_panel.payload(self.store, self.memory())
        self.assertTrue(panel.cannot_know)
        self.assertTrue(any("still true" in line for line in panel.cannot_know))

    def test_a_quarantine_reason_is_reported_as_a_flag(self) -> None:
        memory_id = self.memory(quarantine_reason="looks like two facts glued together")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'quarantine' WHERE id = ?", (memory_id,))
        panel = sources_panel.payload(self.store, memory_id)
        self.assertTrue(any("quarantined" in flag for flag in panel.flags))

    def test_a_redaction_is_named_and_its_value_is_not_echoed(self) -> None:
        """A source panel is exactly where a leaked key would be quoted back at a reader."""
        memory_id = self.memory(
            text="the deploy token is [REDACTED:openai_key] and it lives in the vault",
        )
        panel = sources_panel.payload(self.store, memory_id)
        rendered = "\n".join(sources_panel.render(panel))
        self.assertIn("redacted", rendered.lower())
        self.assertIn("openai_key", rendered, "the pattern is named; the value is not")
        self.assertNotIn("the deploy token is", rendered, "the panel quotes no content at all")

    def test_a_protected_memory_is_flagged(self) -> None:
        memory_id = self.memory(protected=1)
        panel = sources_panel.payload(self.store, memory_id)
        self.assertTrue(any("protected" in flag for flag in panel.flags))

    # -- the supersession chain ------------------------------------------------------

    def test_a_supersession_chain_lists_what_it_replaced_and_what_replaced_it(self) -> None:
        old = self.memory("the proxmox box runs 8.3")
        new = self.memory("the proxmox box runs 8.4", supersedes_id=old)
        panel = sources_panel.payload(self.store, new)
        ids = {entry["id"] for entry in panel.chain}
        self.assertIn(old, ids)
        self.assertIn(new, ids)
        self.assertGreaterEqual(panel.chain_total, 2)

    # -- rendering -------------------------------------------------------------------

    def test_render_names_an_unknown_field_with_its_column(self) -> None:
        panel = sources_panel.payload(self.store, self.memory())
        rendered = "\n".join(sources_panel.render(panel))
        if any("unknown" in line for line in rendered.splitlines()):
            self.assertIn("not recorded", rendered)

    def test_render_is_bounded(self) -> None:
        panel = sources_panel.payload(self.store, self.memory())
        self.assertLessEqual(len(sources_panel.render(panel, limit=3)), 12)


if __name__ == "__main__":
    unittest.main()
