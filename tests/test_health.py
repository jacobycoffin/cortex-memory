"""One screen that says whether this brain is healthy (#169)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import health
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, kind: str = "semantic", pinned: bool = False) -> str:
        return self.store.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category="USER_STATED",
            pinned=pinned,
        )[0]

    def recall(self, prepare_ms: float, *, abstained: bool = False) -> None:
        self.store.record_recall_run(
            session_id="s", query="q", mode="auto", reason="test", requested_limit=6,
            token_budget=700, candidate_count=1, selected_count=0 if abstained else 1,
            estimated_tokens=10, prepare_ms=prepare_ms, abstained=abstained, task_id=None,
        )

    def find(self, items, name):
        return [item for item in items if item.name == name][0]

    def test_an_unmeasurable_check_is_unknown_not_zero(self) -> None:
        """A missing table is an absence of evidence, not a healthy zero."""
        items = health.checks(self.store)
        self.assertEqual(self.find(items, "latency").status, health.UNKNOWN)
        self.assertIsNone(self.find(items, "latency").value)

    def test_fast_runs_are_ok(self) -> None:
        for _ in range(5):
            self.recall(40.0)
        self.assertEqual(health.latency(self.store).status, health.OK)

    def test_slow_runs_fail(self) -> None:
        for _ in range(5):
            self.recall(4000.0)
        self.assertEqual(health.latency(self.store).status, health.FAIL)

    def test_latency_reports_both_percentiles(self) -> None:
        self.recall(10.0)
        self.recall(1000.0)
        found = health.latency(self.store)
        self.assertIn("p50", str(found.value))
        self.assertIn("p95", str(found.value))

    def test_the_latency_check_counts_runs_over_target(self) -> None:
        self.recall(10.0)
        self.recall(900.0)
        self.assertIn("over the p50 target", health.latency(self.store).note)

    def test_the_window_is_stated(self) -> None:
        for _ in range(3):
            self.recall(50.0)
        self.assertEqual(health.latency(self.store).window, "last 3")

    def test_the_threshold_is_printed_with_the_reading(self) -> None:
        for _ in range(2):
            self.recall(50.0)
        self.assertIn("p50", health.latency(self.store).threshold)

    def test_storage_reports_megabytes(self) -> None:
        self.memory("a fact")
        found = health.storage(self.store)
        self.assertIn(found.status, (health.OK, health.WARN, health.FAIL))
        self.assertIsNotNone(found.value)

    def test_abstention_is_unknown_without_runs(self) -> None:
        self.assertEqual(health.abstention(self.store).status, health.UNKNOWN)

    def test_many_abstentions_warn(self) -> None:
        for _ in range(4):
            self.recall(50.0, abstained=True)
        self.recall(50.0)
        self.assertEqual(health.abstention(self.store).status, health.WARN)

    def test_mostly_abstaining_fails(self) -> None:
        for _ in range(9):
            self.recall(50.0, abstained=True)
        self.recall(50.0)
        self.assertEqual(health.abstention(self.store).status, health.FAIL)

    def test_the_queue_check_is_unknown_before_anything_is_queued(self) -> None:
        self.assertEqual(health.queue_health(self.store).status, health.UNKNOWN)

    def test_a_parked_item_fails_the_queue_check(self) -> None:
        from cortex import consolidation_queue

        memory_id = self.memory("a fact")
        consolidation_queue.enqueue(self.store, [memory_id], at=NOW)
        with self.store.transaction() as conn:
            conn.execute("UPDATE consolidation_queue SET state = 'parked'")
        self.assertEqual(health.queue_health(self.store).status, health.FAIL)

    def test_template_heavy_content_fails_the_data_quality_check(self) -> None:
        self.memory("Vault note: X Path: x.md Section: X - body one")
        self.memory("Vault note: Y Path: y.md Section: Y - body two")
        self.memory("a clean fact")
        self.assertEqual(health.data_quality(self.store).status, health.FAIL)

    def test_clean_content_passes_the_data_quality_check(self) -> None:
        self.memory("a clean fact")
        self.assertEqual(health.data_quality(self.store).status, health.OK)

    def test_few_pinned_facts_warn(self) -> None:
        self.memory("one pinned fact", pinned=True)
        self.assertEqual(health.coverage(self.store).status, health.WARN)

    def test_enough_pinned_facts_pass(self) -> None:
        for index in range(3):
            self.memory(f"pinned {index}", pinned=True)
        self.assertEqual(health.coverage(self.store).status, health.OK)

    def test_the_worst_check_decides_the_verdict(self) -> None:
        items = [
            health.Check("a", health.OK), health.Check("b", health.WARN),
            health.Check("c", health.FAIL),
        ]
        self.assertEqual(health.verdict(items), health.FAIL)

    def test_unknown_outranks_ok(self) -> None:
        self.assertEqual(health.verdict([health.Check("a", health.OK),
                                         health.Check("b", health.UNKNOWN)]), health.UNKNOWN)

    def test_the_render_names_what_needs_attention(self) -> None:
        for _ in range(9):
            self.recall(50.0, abstained=True)
        self.recall(50.0)
        rendered = "\n".join(health.render(self.store))
        self.assertIn("needs attention", rendered)

    def test_a_check_line_shows_mark_value_and_source(self) -> None:
        line = health.Check("latency", health.WARN, "p50 300 / p95 900", "ms", "recall_runs",
                            "last 10", "p50 ≤ 150").line()
        for fragment in ("!", "latency", "300", "recall_runs", "p50 ≤ 150"):
            self.assertIn(fragment, line)

    def test_the_report_carries_every_check(self) -> None:
        described = health.report(self.store)
        self.assertEqual(len(described["checks"]), len(health.checks(self.store)))
        self.assertIn("never zero", described["note"])

    def test_a_broken_store_yields_unknown_checks_rather_than_raising(self) -> None:
        class Broken:
            _conn = None

        items = health.checks(Broken())
        self.assertTrue(all(item.status == health.UNKNOWN for item in items))


if __name__ == "__main__":
    unittest.main()
