"""Record who, where and when a fact applies, at write time (#30)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import scope_capture
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class InferTests(unittest.TestCase):
    def test_a_named_person_is_who(self) -> None:
        self.assertEqual(scope_capture.who("Jacoby prefers terse answers."), ["jacoby"])

    def test_two_people_are_both_recorded(self) -> None:
        self.assertEqual(scope_capture.who("Hermes and Jacoby share one scope."),
                         ["jacoby", "hermes"])

    def test_a_capitalised_day_is_not_a_person(self) -> None:
        self.assertEqual(scope_capture.who("Monday backups run at three."), [])

    def test_a_host_is_where(self) -> None:
        self.assertIn("r630", scope_capture.where("The Proxmox host r630 runs the containers."))

    def test_a_project_is_where(self) -> None:
        self.assertIn("vps", scope_capture.where("The cortex-memory plugin ships on the VPS."))

    def test_project_inference_needs_a_store_and_is_optional(self) -> None:
        self.assertEqual(scope_capture.where("Nothing named here at all."), [])

    def test_an_iso_date_is_when(self) -> None:
        self.assertIn("2026-11-22", scope_capture.when("As of 2026-11-22 the store is unencrypted."))

    def test_a_relative_marker_is_when(self) -> None:
        self.assertTrue(scope_capture.when("Backups moved to the new host since Tuesday."))

    def test_a_recurring_marker_is_when(self) -> None:
        self.assertTrue(scope_capture.when("The sweep runs on weekdays."))

    def test_a_plain_fact_infers_nothing(self) -> None:
        self.assertEqual(scope_capture.infer("Backups run nightly."), {})

    def test_infer_combines_the_dimensions(self) -> None:
        captured = scope_capture.infer(
            "Jacoby moved the cortex-memory store on r630 to 2026-11-22."
        )
        self.assertIn("who", captured)
        self.assertIn("where", captured)
        self.assertIn("when", captured)


class RiskTests(unittest.TestCase):
    def test_no_scope_is_global(self) -> None:
        self.assertTrue(scope_capture.is_global({}))

    def test_a_scope_with_who_is_not_global(self) -> None:
        self.assertFalse(scope_capture.is_global({"who": ["jacoby"]}))

    def test_an_existing_project_counts_as_scope(self) -> None:
        self.assertFalse(scope_capture.is_global({"active_project": "cortex"}))

    def test_empty_lists_are_still_global(self) -> None:
        self.assertTrue(scope_capture.is_global({"who": [], "where": []}))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def scope(self, memory_id: str) -> dict:
        raw = self.store._conn.execute(
            "SELECT scope_json FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        return json.loads(raw or "{}")

    def test_capture_fills_the_missing_keys(self) -> None:
        memory_id = self.memory("Jacoby runs the dashboard on r630.")
        written = scope_capture.capture(self.store, memory_id, at=NOW)
        self.assertIn("jacoby", written["who"])
        self.assertIn("r630", written["where"])

    def test_capture_never_overwrites_an_existing_scope(self) -> None:
        memory_id = self.memory("Jacoby runs the dashboard on r630.")
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET scope_json = ? WHERE id = ?",
                (json.dumps({"active_project": "cortex", "who": ["someone-else"]}), memory_id),
            )
        scope_capture.capture(self.store, memory_id, at=NOW)
        stored = self.scope(memory_id)
        self.assertEqual(stored["active_project"], "cortex")
        self.assertEqual(stored["who"], ["someone-else"])
        self.assertTrue(stored["where"])

    def test_capture_records_when_it_was_captured(self) -> None:
        memory_id = self.memory("Jacoby prefers terse answers.")
        scope_capture.capture(self.store, memory_id, at=NOW)
        self.assertEqual(self.scope(memory_id)["captured_at"], NOW.isoformat())

    def test_a_second_capture_is_a_no_op(self) -> None:
        memory_id = self.memory("Jacoby prefers terse answers.")
        first = scope_capture.capture(self.store, memory_id, at=NOW)
        second = scope_capture.capture(self.store, memory_id, at=NOW)
        self.assertEqual(first, second)

    def test_a_fact_with_no_qualifiers_stays_empty(self) -> None:
        memory_id = self.memory("Backups run nightly.")
        written = scope_capture.capture(self.store, memory_id, at=NOW)
        self.assertEqual(written, {})
        self.assertTrue(scope_capture.is_global(self.scope(memory_id)))

    def test_capture_reads_the_stored_content_when_none_is_given(self) -> None:
        memory_id = self.memory("Kaya listens on the VPS.")
        written = scope_capture.capture(self.store, memory_id, at=NOW)
        self.assertIn("kaya", written["who"])

    def test_capture_of_a_missing_memory_is_quiet(self) -> None:
        self.assertEqual(scope_capture.capture(self.store, "no-such-id"), {})

    def test_coverage_counts_the_dimensions(self) -> None:
        one = self.memory("Jacoby runs the dashboard on r630.")
        self.memory("Backups run nightly.")
        scope_capture.capture(self.store, one, at=NOW)
        counts = scope_capture.coverage(self.store)
        self.assertEqual(counts["memories"], 2)
        self.assertEqual(counts["global"], 1)
        self.assertGreaterEqual(counts["who"], 1)

    def test_backfill_scopes_the_unscoped(self) -> None:
        memory_id = self.memory("Jacoby runs the dashboard on r630.")
        described = scope_capture.backfill(self.store)
        self.assertEqual(described["scanned"], 1)
        self.assertTrue(self.scope(memory_id))

    def test_backfill_does_not_repeat_work(self) -> None:
        self.memory("Jacoby runs the dashboard on r630.")
        scope_capture.backfill(self.store)
        self.assertEqual(scope_capture.backfill(self.store)["scanned"], 0)

    def test_the_report_carries_the_share_of_global_facts(self) -> None:
        self.memory("Backups run nightly.")
        described = scope_capture.report(self.store)
        self.assertEqual(described["global_share"], 1.0)

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(scope_capture.coverage(self.store)["memories"], 0)
        self.assertEqual(scope_capture.backfill(self.store)["scanned"], 0)


if __name__ == "__main__":
    unittest.main()
