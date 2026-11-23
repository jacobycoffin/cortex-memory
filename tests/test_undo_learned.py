"""One command to take back what a consolidation run learned (#213)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import undo_learned as ul
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
LATER = datetime(2026, 11, 22, 12, 5, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.source = self.memory("The dashboard runs on port 8100.")
        self.target = self.memory("Nginx proxies the dashboard on port 8100.")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def supersede(self, memory_id: str, content: str | None = None) -> None:
        with self.store.transaction() as conn:
            if content is None:
                conn.execute(
                    "UPDATE memories SET state = 'superseded' WHERE id = ?", (memory_id,)
                )
            else:
                conn.execute(
                    "UPDATE memories SET state = 'superseded', content = ? WHERE id = ?",
                    (content, memory_id),
                )

    def state(self, memory_id: str) -> str:
        return str(
            self.store._conn.execute(
                "SELECT state FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()[0]
        )

    def content(self, memory_id: str) -> str:
        return str(
            self.store._conn.execute(
                "SELECT content FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()[0]
        )

    def test_a_step_with_no_change_undoes_nothing(self) -> None:
        run = ul.begin(self.store, "consolidation", [self.source, self.target], at=NOW)
        described = ul.commit(self.store, run, at=LATER)
        self.assertEqual(described["changed"], [])
        result = ul.undo(self.store, run, at=LATER)
        self.assertEqual(result["undone"], 0)
        self.assertIn("nothing", result["sentence"])

    def test_undo_restores_a_superseded_state(self) -> None:
        run = ul.begin(self.store, "consolidation", [self.source], at=NOW)
        self.supersede(self.source)
        self.assertEqual(ul.commit(self.store, run, at=LATER)["changed"], [self.source])
        self.assertEqual(self.state(self.source), "superseded")
        result = ul.undo(self.store, run, at=LATER)
        self.assertEqual(result["restored"], [self.source])
        self.assertEqual(self.state(self.source), "active")

    def test_undo_restores_rewritten_content(self) -> None:
        original = self.content(self.target)
        run = ul.begin(self.store, "consolidation", [self.target], at=NOW)
        self.supersede(self.target, content="a merged version nobody asked for")
        ul.commit(self.store, run, at=LATER)
        ul.undo(self.store, run, at=LATER)
        self.assertEqual(self.content(self.target), original)

    def test_the_undo_is_one_command_for_the_whole_run(self) -> None:
        run = ul.begin(self.store, "consolidation", [self.source, self.target], at=NOW)
        self.supersede(self.source)
        self.supersede(self.target)
        ul.commit(self.store, run, at=LATER)
        result = ul.undo(self.store, run, at=LATER)
        self.assertEqual(result["undone"], 2)
        self.assertEqual(self.state(self.source), "active")
        self.assertEqual(self.state(self.target), "active")

    def test_undo_without_a_run_id_reverses_the_latest_step(self) -> None:
        first = ul.begin(self.store, "consolidation", [self.source], at=NOW)
        self.supersede(self.source)
        ul.commit(self.store, first, at=NOW)
        second = ul.begin(self.store, "triage", [self.target], at=LATER)
        self.supersede(self.target)
        ul.commit(self.store, second, at=LATER)
        self.assertEqual(ul.undo(self.store, at=LATER)["module"], "triage")
        self.assertEqual(self.state(self.target), "active")
        self.assertEqual(self.state(self.source), "superseded")

    def test_a_step_is_never_undone_twice(self) -> None:
        run = ul.begin(self.store, "consolidation", [self.source], at=NOW)
        self.supersede(self.source)
        ul.commit(self.store, run, at=LATER)
        ul.undo(self.store, run, at=LATER)
        second = ul.undo(self.store, run, at=LATER)
        self.assertEqual(second["undone"], 0)
        self.assertIn("already undone", second["reason"])

    def test_undo_touches_only_the_selected_ids(self) -> None:
        untouched = self.memory("A memory nobody selected.")
        run = ul.begin(self.store, "consolidation", [self.source], at=NOW)
        self.supersede(self.source)
        self.supersede(untouched)
        ul.commit(self.store, run, at=LATER)
        ul.undo(self.store, run, at=LATER)
        self.assertEqual(self.state(self.source), "active")
        self.assertEqual(self.state(untouched), "superseded")

    def test_undo_does_not_roll_the_clock_back(self) -> None:
        """A step restores values, not history: recreating rows is out of scope by design."""
        run = ul.begin(self.store, "consolidation", [self.source], at=NOW)
        self.supersede(self.source)
        ul.commit(self.store, run, at=LATER)
        ul.undo(self.store, run, at=LATER)
        self.assertIsNotNone(self.store.get_memory(self.source))

    def test_an_unknown_run_reports_cleanly(self) -> None:
        described = ul.commit(self.store, "no-such-run")
        self.assertEqual(described["changed"], [])
        self.assertIn("unknown run", described["reason"])

    def test_missing_ids_are_tolerated(self) -> None:
        run = ul.begin(self.store, "consolidation", ["not-a-real-id"], at=NOW)
        self.assertEqual(ul.commit(self.store, run, at=LATER)["changed"], [])

    def test_steps_are_listed_newest_first_and_undone_ones_hidden(self) -> None:
        first = ul.begin(self.store, "consolidation", [self.source], at=NOW)
        ul.commit(self.store, first, at=NOW)
        second = ul.begin(self.store, "triage", [self.target], at=LATER)
        ul.commit(self.store, second, at=LATER)
        listed = ul.steps(self.store)
        self.assertEqual([row["module"] for row in listed], ["triage", "consolidation"])
        ul.undo(self.store, second, at=LATER)
        self.assertEqual([row["module"] for row in ul.steps(self.store)], ["consolidation"])

    def test_the_report_counts_undoable_and_undone(self) -> None:
        run = ul.begin(self.store, "consolidation", [self.source], at=NOW)
        self.supersede(self.source)
        ul.commit(self.store, run, at=LATER)
        self.assertEqual(ul.report(self.store)["undoable"], 1)
        ul.undo(self.store, run, at=LATER)
        described = ul.report(self.store)
        self.assertEqual(described["undone"], 1)
        self.assertEqual(described["undoable"], 0)

    def test_an_empty_store_is_quiet(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            self.assertEqual(ul.steps(other), [])
            self.assertEqual(ul.undo(other)["undone"], 0)
            self.assertEqual(ul.report(other)["steps"], 0)
        finally:
            other.close()


if __name__ == "__main__":
    unittest.main()
