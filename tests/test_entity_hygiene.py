"""Entity hygiene (#116): the score never travels without its parts, and only active rows count."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import entity_hygiene
from cortex.store import CortexStore

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


class HygieneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, entities: tuple[str, ...] = ("proxmox",), **kwargs) -> str:
        return self.store.add_memory(
            content=text, kind="semantic", source_type="user_turn", source_category="USER_STATED",
            entities=list(entities), **kwargs,
        )[0]

    def age(self, memory_id: str, days: float) -> None:
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET created_at = ?, observed_at = ? WHERE id = ?",
                         ((NOW - timedelta(days=days)).isoformat(),) * 2 + (memory_id,))

    def entity(self, report, name: str):
        return next(item for item in report.entities if item.entity == name)

    # -- the parts travel with the score ---------------------------------------------

    def test_the_score_is_accompanied_by_the_terms_that_produced_it(self) -> None:
        self.memory("the proxmox box runs 8.4")
        report = entity_hygiene.report(self.store, now=NOW)
        item = self.entity(report, "proxmox")
        parts = item.parts()
        self.assertEqual(sum(parts.values()) + item.hygiene, 1.0)
        self.assertTrue(any(value for value in parts.values()))
        self.assertIn("thin", item.render())

    def test_a_healthy_entity_scores_one(self) -> None:
        for index in range(3):
            memory_id = self.memory(f"proxmox fact number {index}")
            with self.store.transaction() as conn:
                conn.execute("UPDATE memories SET used_count = 2 WHERE id = ?", (memory_id,))
        report = entity_hygiene.report(self.store, now=NOW)
        item = self.entity(report, "proxmox")
        self.assertEqual(item.verdict, "healthy")
        self.assertEqual(item.hygiene, 1.0)

    # -- each term -------------------------------------------------------------------

    def test_a_single_memory_is_thin_not_wrong(self) -> None:
        self.memory("proxmox mentioned once")
        item = self.entity(entity_hygiene.report(self.store, now=NOW), "proxmox")
        self.assertEqual(item.verdict, "thin")
        self.assertAlmostEqual(item.hygiene, 1.0 - entity_hygiene.W_THIN - entity_hygiene.W_UNEXERCISED, places=4)

    def test_an_old_record_is_stale(self) -> None:
        for index in range(2):
            memory_id = self.memory(f"proxmox fact {index}")
            self.age(memory_id, 400)
            with self.store.transaction() as conn:
                conn.execute("UPDATE memories SET used_count = 1 WHERE id = ?", (memory_id,))
        item = self.entity(entity_hygiene.report(self.store, now=NOW), "proxmox")
        self.assertEqual(item.verdict, "stale")

    def test_disagreeing_active_objects_are_a_contradiction(self) -> None:
        self.memory("the proxmox box runs 8.4", subject="proxmox", predicate="runs", object_value="8.4")
        self.memory("the proxmox box runs 8.3", subject="proxmox", predicate="runs", object_value="8.3")
        item = self.entity(entity_hygiene.report(self.store, now=NOW), "proxmox")
        self.assertEqual(item.verdict, "contradictory")
        self.assertEqual(item.contradictions, 1)

    def test_a_retired_side_is_history_not_a_contradiction(self) -> None:
        first = self.memory("the box runs 8.4", subject="proxmox", predicate="runs", object_value="8.4")
        self.memory("the box runs 8.3", subject="proxmox", predicate="runs", object_value="8.3")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'archived' WHERE id = ?", (first,))
        item = self.entity(entity_hygiene.report(self.store, now=NOW), "proxmox")
        self.assertEqual(item.contradictions, 0, "counting a corrected side punishes the store for being fixed")

    def test_never_used_costs_the_least(self) -> None:
        for index in range(2):
            self.memory(f"proxmox fact {index}")
        item = self.entity(entity_hygiene.report(self.store, now=NOW), "proxmox")
        self.assertEqual(item.verdict, "unexercised")
        self.assertAlmostEqual(item.hygiene, 1.0 - entity_hygiene.W_UNEXERCISED, places=4)

    # -- what is never scored --------------------------------------------------------

    def test_archived_memories_do_not_make_an_entity(self) -> None:
        memory_id = self.memory("an archived mention")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'archived' WHERE id = ?", (memory_id,))
        report = entity_hygiene.report(self.store, now=NOW)
        self.assertEqual(report.considered, 0)

    def test_a_memory_with_no_entity_scores_nothing(self) -> None:
        self.memory("a fact about nothing in particular", entities=())
        self.assertEqual(entity_hygiene.report(self.store, now=NOW).considered, 0)

    # -- sorting and bounds ----------------------------------------------------------

    def test_the_worst_entity_comes_first(self) -> None:
        for index in range(3):
            memory_id = self.memory(f"healthy fact {index}", entities=("good",))
            with self.store.transaction() as conn:
                conn.execute("UPDATE memories SET used_count = 1 WHERE id = ?", (memory_id,))
        self.memory("thin fact", entities=("bad",))
        report = entity_hygiene.report(self.store, now=NOW)
        self.assertEqual(report.entities[0].entity, "bad")

    def test_the_report_is_bounded_and_says_what_it_hid(self) -> None:
        for index in range(6):
            self.memory(f"fact about entity {index}", entities=(f"thing{index}",))
        report = entity_hygiene.report(self.store, now=NOW, limit=2)
        self.assertEqual(len(report.entities), 2)
        self.assertEqual(report.hidden, 4)
        self.assertTrue(any("4 more" in line for line in report.render(limit=2)))

    def test_the_headline_counts_the_verdicts(self) -> None:
        self.memory("a", entities=("one",))
        for index in range(3):
            memory_id = self.memory(f"b{index}", entities=("two",))
            with self.store.transaction() as conn:
                conn.execute("UPDATE memories SET used_count = 1 WHERE id = ?", (memory_id,))
        head = entity_hygiene.report(self.store, now=NOW).render()[0]
        self.assertIn("1 thin", head)
        self.assertIn("1 healthy", head)

    def test_every_verdict_is_from_the_named_set(self) -> None:
        self.memory("a", entities=("one",))
        for item in entity_hygiene.report(self.store, now=NOW).entities:
            self.assertIn(item.verdict, entity_hygiene.VERDICTS)

    def test_an_empty_store_says_so_instead_of_scoring_nothing(self) -> None:
        lines = entity_hygiene.report(self.store, now=NOW).render()
        self.assertEqual(len(lines), 1)
        self.assertIn("No entity", lines[0])

    def test_a_broken_store_says_it_could_not_read(self) -> None:
        class Broken:
            _conn = None

        report = entity_hygiene.report(Broken(), now=NOW)
        self.assertFalse(report.readable)
        self.assertIn("could not be read", report.render()[0])

    def test_scoring_writes_nothing(self) -> None:
        self.memory("a fact")
        tables = tuple(sorted(r["name"] for r in self.store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")))
        counts = tuple(int(self.store._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                       for t in tables)
        entity_hygiene.report(self.store, now=NOW)
        self.assertEqual(counts, tuple(int(self.store._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                                       for t in tables))


if __name__ == "__main__":
    unittest.main()
