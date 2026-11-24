"""What the render budget is spent on, and what nobody is measuring (#172)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import render_budget
from cortex.store import CortexStore


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def run_with(self, *, budget: int = 620, selected: int = 500) -> None:
        self.store.record_recall_run(
            session_id="s", query="q", mode="auto", reason="test", requested_limit=6,
            token_budget=budget, candidate_count=1, selected_count=1,
            estimated_tokens=selected, prepare_ms=20.0, abstained=False, task_id=None,
        )

    def test_the_budget_registry_has_no_unbounded_entry(self) -> None:
        for item in render_budget.budgets():
            self.assertGreater(item.cap, 0, f"{item.name} has no cap")
            self.assertTrue(item.source, f"{item.name} has no source")

    def test_the_registry_names_the_known_blocks(self) -> None:
        names = {item.name for item in render_budget.budgets()}
        self.assertIn("constraints block", names)
        self.assertIn("bootstrap block", names)
        self.assertIn("memory budget", names)

    def test_the_selection_share_is_measured(self) -> None:
        self.run_with(budget=1000, selected=250)
        described = render_budget.spend(self.store)
        self.assertAlmostEqual(described.selected_share_p50, 0.25, places=4)

    def test_a_selection_over_budget_is_capped_at_one(self) -> None:
        self.run_with(budget=100, selected=500)
        self.assertLessEqual(render_budget.spend(self.store).selected_share_p50, 1.0)

    def test_an_unrecorded_render_side_is_flagged(self) -> None:
        """`rendered_tokens = 0` may mean 'nobody wrote it' or 'nothing was rendered'."""
        self.run_with()
        described = render_budget.spend(self.store)
        self.assertFalse(described.rendered_recorded)
        self.assertIn("unmeasured", described.note)

    def test_a_recorded_render_side_is_recognised(self) -> None:
        self.run_with()
        with self.store.transaction() as conn:
            conn.execute("UPDATE recall_runs SET rendered_tokens = 480, withheld_count = 2")
        described = render_budget.spend(self.store)
        self.assertTrue(described.rendered_recorded)
        self.assertTrue(described.withheld_recorded)
        self.assertEqual(described.note, "")

    def test_withheld_alone_is_still_a_gap(self) -> None:
        self.run_with()
        with self.store.transaction() as conn:
            conn.execute("UPDATE recall_runs SET withheld_count = 1")
        described = render_budget.spend(self.store)
        self.assertFalse(described.rendered_recorded)
        self.assertIn("rendered_tokens", described.note)

    def test_no_runs_is_reported(self) -> None:
        described = render_budget.spend(self.store)
        self.assertEqual(described.runs, 0)
        self.assertIn("no recall runs", described.note)

    def test_the_token_estimate_is_labelled_as_an_estimate(self) -> None:
        self.assertEqual(render_budget.estimate_tokens("a" * 400), 100)

    def test_an_empty_block_estimates_zero(self) -> None:
        self.assertEqual(render_budget.estimate_tokens(""), 0)

    def test_a_block_reports_its_share_of_the_cap(self) -> None:
        described = render_budget.block_spend("x" * 160, 320)
        self.assertAlmostEqual(described["share"], 0.5, places=4)
        self.assertFalse(described["over"])

    def test_a_block_over_its_cap_says_so(self) -> None:
        self.assertTrue(render_budget.block_spend("x" * 400, 320)["over"])

    def test_the_verdict_is_unmeasured_without_render_telemetry(self) -> None:
        self.run_with()
        self.assertEqual(render_budget.report(self.store)["verdict"], "unmeasured")

    def test_the_verdict_is_measured_once_the_render_side_is_written(self) -> None:
        self.run_with()
        with self.store.transaction() as conn:
            conn.execute("UPDATE recall_runs SET rendered_tokens = 400, withheld_count = 0")
        self.assertEqual(render_budget.report(self.store)["verdict"], "measured")

    def test_the_report_carries_every_budget(self) -> None:
        self.assertEqual(len(render_budget.report(self.store)["budgets"]),
                         len(render_budget.budgets()))

    def test_the_render_states_the_gap(self) -> None:
        self.run_with()
        rendered = "\n".join(render_budget.render(self.store))
        self.assertIn("NOT RECORDED", rendered)
        self.assertIn("caps:", rendered)

    def test_a_broken_store_reports_nothing_rather_than_raising(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(render_budget.spend(Broken()).runs, 0)


if __name__ == "__main__":
    unittest.main()
