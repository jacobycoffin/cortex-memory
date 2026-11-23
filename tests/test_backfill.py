"""When a field gains meaning, re-derive it for the rows written before (#45)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import backfill, scope_capture
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, **kwargs) -> str:
        kwargs.setdefault("kind", "semantic")
        kwargs.setdefault("source_type", "user_turn")
        kwargs.setdefault("source_category", "USER_EXPLICIT")
        return self.store.add_memory(content=content, **kwargs)[0]

    def scope(self, memory_id: str) -> str:
        return str(self.store._conn.execute(
            "SELECT scope_json FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()[0] or "")

    # --- plan -----------------------------------------------------------------------------------

    def test_the_plan_counts_the_rows_that_need_work(self) -> None:
        self.memory("Jacoby runs the dashboard on r630.")
        planned = backfill.plan(self.store, only=["scope"])
        line = planned["derivations"][0]
        self.assertEqual(line["name"], "scope")
        self.assertEqual(line["needs"], 1)

    def test_the_plan_changes_nothing(self) -> None:
        memory_id = self.memory("Jacoby runs the dashboard on r630.")
        backfill.plan(self.store, only=["scope"])
        self.assertIn(self.scope(memory_id), {"", "{}"})

    def test_a_not_derivable_field_is_declared_not_approximated(self) -> None:
        planned = backfill.plan(self.store, only=["extraction_confidence"])
        self.assertTrue(planned["derivations"][0]["not_derivable"])
        self.assertIsNone(planned["derivations"][0]["needs"])

    def test_an_unknown_name_selects_nothing(self) -> None:
        self.memory("Jacoby runs the dashboard on r630.")
        self.assertEqual(backfill.plan(self.store, only=["no-such-thing"])["total"], 0)

    # --- run ------------------------------------------------------------------------------------

    def test_a_run_fills_the_missing_field(self) -> None:
        memory_id = self.memory("Jacoby runs the dashboard on r630.")
        outcome = backfill.run(self.store, only=["scope"], at=NOW)
        self.assertEqual(outcome.filled, 1)
        self.assertIn("jacoby", self.scope(memory_id))

    def test_a_second_run_fills_nothing(self) -> None:
        """Idempotence is the whole safety property: the predicate is "missing", not "unrun"."""
        self.memory("Jacoby runs the dashboard on r630.")
        backfill.run(self.store, only=["scope"], at=NOW)
        second = backfill.run(self.store, only=["scope"], at=NOW)
        self.assertEqual(second.filled, 0)
        self.assertEqual(second.scanned, 0)

    def test_a_fact_with_nothing_to_fill_is_not_rescanned(self) -> None:
        """Otherwise "missing" and "not applicable" are the same predicate, forever."""
        self.memory("Backups run nightly.")
        first = backfill.run(self.store, only=["scope"], at=NOW)
        second = backfill.run(self.store, only=["scope"], at=NOW)
        self.assertEqual(first.filled, 0)
        self.assertEqual(second.scanned, 0)
        self.assertIn(backfill.verify(self.store, derivations=[
            derivation for derivation in backfill.default_derivations()
            if derivation.name == "scope"
        ])["remaining"]["scope"], (0,))

    def test_a_marked_row_is_not_retried(self) -> None:
        memory_id = self.memory("Backups run nightly.")
        backfill.run(self.store, only=["scope"], at=NOW)
        marks = self.store._conn.execute(
            "SELECT COUNT(*) FROM backfill_marks WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]
        self.assertEqual(marks, 1)

    def test_every_run_is_recorded(self) -> None:
        self.memory("Jacoby runs the dashboard on r630.")
        outcome = backfill.run(self.store, only=["scope"], at=NOW)
        recorded = backfill.history(self.store)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]["run_id"], outcome.run_id)
        self.assertEqual(recorded[0]["derivation"], "scope")
        self.assertEqual(recorded[0]["filled"], 1)

    def test_one_run_can_cover_several_derivations(self) -> None:
        """The ledger is per derivation, so a run id alone cannot be the primary key."""
        self.memory("Jacoby runs the dashboard on r630, per the Proxmox box.")
        outcome = backfill.run(self.store, at=NOW)
        self.assertGreaterEqual(len(outcome.details), 2)
        recorded = backfill.history(self.store, limit=50)
        self.assertGreaterEqual(len(recorded), 2)
        self.assertEqual({row["run_id"] for row in recorded}, {outcome.run_id})

    def test_the_run_carries_the_schema_version_it_ran_for(self) -> None:
        backfill.set_schema_version(self.store, 3)
        self.memory("Jacoby runs the dashboard on r630.")
        backfill.run(self.store, only=["scope"], at=NOW)
        self.assertEqual(backfill.history(self.store)[0]["schema_version"], 3)

    def test_the_importance_derivation_writes_a_score(self) -> None:
        memory_id = self.memory("Jacoby prefers the dashboard on one page.", kind="preference")
        backfill.run(self.store, only=["importance"], at=NOW)
        stored = self.store._conn.execute(
            "SELECT importance FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        self.assertGreater(float(stored), 0.0)

    def test_a_raising_derivation_is_reported_not_fatal(self) -> None:
        def needs(store, limit):
            raise RuntimeError("boom")

        broken = backfill.Derivation("broken", "x", 1, needs, lambda store, row: False)
        outcome = backfill.run(self.store, derivations=[broken], at=NOW)
        self.assertTrue(any("error" in detail for detail in outcome.details))
        self.assertEqual(outcome.filled, 0)

    def test_one_bad_row_does_not_stop_the_pass(self) -> None:
        self.memory("Jacoby runs the dashboard on r630.")
        self.memory("Kaya listens on the VPS.")

        def needs(store, limit):
            return [{"id": "missing", "content": "x"}, {"id": "missing-2", "content": "y"}]

        def derive(store, row):
            raise ValueError("bad row")

        broken = backfill.Derivation("broken", "x", 1, needs, derive)
        outcome = backfill.run(self.store, derivations=[broken], at=NOW)
        self.assertEqual(outcome.failed, 2)

    def test_a_not_derivable_field_is_skipped_by_a_run(self) -> None:
        outcome = backfill.run(self.store, only=["extraction_confidence"], at=NOW)
        self.assertEqual(outcome.filled, 0)
        self.assertTrue(any(detail.get("skipped") == "not derivable"
                            for detail in outcome.details))

    # --- verify ---------------------------------------------------------------------------------

    def test_verify_is_clean_after_a_run(self) -> None:
        self.memory("Jacoby runs the dashboard on r630.")
        backfill.run(self.store, only=["scope"], at=NOW)
        checked = backfill.verify(self.store, derivations=[
            derivation for derivation in backfill.default_derivations()
            if derivation.name == "scope"
        ])
        self.assertTrue(checked["clean"])

    def test_verify_reports_what_is_left(self) -> None:
        self.memory("Jacoby runs the dashboard on r630.")
        checked = backfill.verify(self.store, derivations=[
            derivation for derivation in backfill.default_derivations()
            if derivation.name == "scope"
        ])
        self.assertFalse(checked["clean"])
        self.assertEqual(checked["remaining"]["scope"], 1)

    # --- version --------------------------------------------------------------------------------

    def test_the_schema_version_starts_at_zero(self) -> None:
        self.assertEqual(backfill.schema_version(self.store), 0)

    def test_the_schema_version_round_trips(self) -> None:
        backfill.set_schema_version(self.store, 2)
        self.assertEqual(backfill.schema_version(self.store), 2)
        backfill.set_schema_version(self.store, 4)
        self.assertEqual(backfill.schema_version(self.store), 4)

    def test_the_report_carries_the_recent_runs(self) -> None:
        self.memory("Jacoby runs the dashboard on r630.")
        backfill.run(self.store, only=["scope"], at=NOW)
        described = backfill.report(self.store)
        self.assertEqual(len(described["runs"]), 1)

    def test_the_derivations_declare_what_they_fill(self) -> None:
        names = {derivation.name for derivation in backfill.default_derivations()}
        self.assertIn("scope", names)
        self.assertIn("entities", names)
        for derivation in backfill.default_derivations():
            self.assertTrue(derivation.fills)

    def test_scope_capture_still_behaves_when_called_through_the_backfill(self) -> None:
        memory_id = self.memory("Backups run nightly.")
        backfill.run(self.store, only=["scope"], at=NOW)
        # nothing to scope, and the row is not corrupted by the attempt
        self.assertIn(self.scope(memory_id), {"", "{}"})
        self.assertTrue(scope_capture.is_global({}))


if __name__ == "__main__":
    unittest.main()
