"""Which injected memories did the reply actually use (#145)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import used_resolution as ur
from cortex.store import CortexStore


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str) -> str:
        return self.store.add_memory(
            content=text, kind="semantic", source_type="user_turn", source_category="USER_STATED",
        )[0]

    def filler(self, count: int = 30) -> None:
        """Background so inverse document frequency has something to be inverse of."""
        for index in range(count):
            self.memory(f"unrelated note number {index} about {index} things in the garage")

    def test_a_reply_that_leans_on_one_memory_resolves_it(self) -> None:
        self.filler()
        target = self.memory("the zigbee coordinator sits on the second shelf of the rack")
        other = self.memory("the printer queue name is officejet")
        found = ur.resolve(self.store, "the zigbee coordinator is on the second shelf", [target, other])
        self.assertEqual(found.resolved, [target])
        self.assertIn(other, found.unresolved)

    def test_a_memory_that_was_not_injected_is_not_evidence(self) -> None:
        """Counting it would credit the store for something it did not supply."""
        self.filler()
        injected = self.memory("the printer queue name is officejet")
        mentioned = self.memory("the zigbee coordinator sits on the second shelf")
        found = ur.resolve(self.store, "the zigbee coordinator is on the second shelf", [injected])
        self.assertNotIn(mentioned, found.resolved)

    def test_common_words_carry_no_weight(self) -> None:
        """A term in most of the store proves nothing on its own."""
        for index in range(40):
            self.memory(f"the hermes dashboard note number {index} about the dashboard")
        target = self.memory("the hermes dashboard note that is the target")
        found = ur.resolve(self.store, "the hermes dashboard is fine", [target])
        self.assertIn(target, found.unresolved)

    def test_a_tie_is_ambiguous_and_nobody_is_counted(self) -> None:
        self.filler()
        first = self.memory("the fibre modem syncs at 2.5 gigabit on the north wall")
        second = self.memory("the fibre modem syncs at 2.5 gigabit on the south wall")
        found = ur.resolve(self.store, "the fibre modem syncs at 2.5 gigabit", [first, second])
        self.assertEqual(found.ambiguous, sorted([first, second]))
        self.assertEqual(found.resolved, [])
        self.assertIn("equally", found.note)

    def test_no_match_is_unresolved_not_unused(self) -> None:
        self.filler()
        target = self.memory("the zigbee coordinator sits on the second shelf")
        found = ur.resolve(self.store, "completely different subject matter entirely", [target])
        self.assertEqual(found.resolved, [])
        self.assertEqual(found.unresolved, [target])

    def test_an_empty_candidate_list_says_so(self) -> None:
        found = ur.resolve(self.store, "a reply", [])
        self.assertIn("nothing to resolve", found.note)

    def test_an_unknown_id_is_unresolved(self) -> None:
        found = ur.resolve(self.store, "a reply about the zigbee coordinator", ["nope"])
        self.assertEqual(found.unresolved, ["nope"])

    def test_a_reply_without_content_words_resolves_nothing(self) -> None:
        target = self.memory("the zigbee coordinator sits on the second shelf")
        found = ur.resolve(self.store, "the and for with that", [target])
        self.assertEqual(found.resolved, [])
        self.assertIn("no content words", found.note)

    def test_applying_counts_the_memory(self) -> None:
        self.filler()
        target = self.memory("the zigbee coordinator sits on the second shelf")
        found = ur.resolve(self.store, "the zigbee coordinator is on the second shelf", [target],
                           turn_id="t1")
        result = ur.apply(self.store, found)
        self.assertEqual(result["applied"], [target])
        row = self.store._conn.execute("SELECT used_count FROM memories WHERE id = ?",
                                       (target,)).fetchone()
        self.assertEqual(int(row["used_count"]), 1)

    def test_applying_twice_does_not_double_count(self) -> None:
        """A counter that increments on every re-run is a fiction."""
        self.filler()
        target = self.memory("the zigbee coordinator sits on the second shelf")
        found = ur.resolve(self.store, "the zigbee coordinator is on the second shelf", [target],
                           turn_id="t1")
        ur.apply(self.store, found)
        second = ur.apply(self.store, found)
        self.assertEqual(second["applied"], [])
        self.assertEqual(second["skipped"], [target])
        row = self.store._conn.execute("SELECT used_count FROM memories WHERE id = ?",
                                       (target,)).fetchone()
        self.assertEqual(int(row["used_count"]), 1)

    def test_ambiguous_memories_are_never_counted(self) -> None:
        self.filler()
        first = self.memory("the fibre modem syncs at 2.5 gigabit on the north wall")
        second = self.memory("the fibre modem syncs at 2.5 gigabit on the south wall")
        found = ur.resolve(self.store, "the fibre modem syncs at 2.5 gigabit", [first, second],
                           turn_id="t2")
        result = ur.apply(self.store, found)
        self.assertEqual(result["applied"], [])

    def test_a_dry_run_counts_nothing(self) -> None:
        self.filler()
        target = self.memory("the zigbee coordinator sits on the second shelf")
        found = ur.resolve(self.store, "the zigbee coordinator is on the second shelf", [target],
                           turn_id="t1")
        result = ur.apply(self.store, found, dry_run=True)
        self.assertTrue(result["dry_run"])
        row = self.store._conn.execute("SELECT used_count FROM memories WHERE id = ?",
                                       (target,)).fetchone()
        self.assertEqual(int(row["used_count"] or 0), 0)

    def test_the_last_used_timestamp_is_written(self) -> None:
        self.filler()
        target = self.memory("the zigbee coordinator sits on the second shelf")
        found = ur.resolve(self.store, "the zigbee coordinator is on the second shelf", [target],
                           turn_id="t1")
        ur.apply(self.store, found)
        row = self.store._conn.execute("SELECT last_used_at FROM memories WHERE id = ?",
                                       (target,)).fetchone()
        self.assertTrue(row["last_used_at"])

    def test_the_summary_states_why_resolution_runs_at_turn_time(self) -> None:
        described = ur.summary(self.store)
        self.assertIn("not what was said", described["note"])

    def test_the_render_states_the_coverage(self) -> None:
        self.filler()
        self.memory("a fact")
        rendered = "\n".join(ur.render(self.store))
        self.assertIn("used-id resolution", rendered)
        self.assertIn("injected ids seen", rendered)

    def test_a_broken_store_resolves_nothing(self) -> None:
        class Broken:
            _conn = None

        found = ur.resolve(Broken(), "a reply about the zigbee coordinator", ["x"])
        self.assertEqual(found.resolved, [])
        self.assertEqual(ur.summary(Broken())["traces"], 0)


if __name__ == "__main__":
    unittest.main()
