"""A procedure is offered only after N consecutive successes (#115)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import procedure_confidence, procedure_validation as pv
from cortex.store import CortexStore

BASE = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def runs(self, procedure_id: str, outcomes: str, *, note: str = "") -> None:
        """Record outcomes in order, one minute apart, so the order is unambiguous.

        `s`/`f` are shorthand for the store's own outcome values; a rejected shorthand would leave the table empty and
        every verdict would quietly read `unvalidated`.
        """
        mapping = {"s": procedure_confidence.SUCCESS, "f": procedure_confidence.FAILURE}
        for index, letter in enumerate(outcomes):
            result = procedure_confidence.record(
                self.store, procedure_id, mapping[letter],
                at=BASE + timedelta(minutes=index), note=note,
            )
            self.assertTrue(result["ok"], result.get("reason"))


class VerdictTests(Base):
    def test_three_consecutive_successes_offer_a_procedure(self) -> None:
        self.runs("verify-on-mac", "sss")
        verdict = pv.validate(self.store, "verify-on-mac")
        self.assertEqual(verdict.verdict, pv.OFFERED)
        self.assertTrue(verdict.may_offer)
        self.assertEqual(verdict.streak, 3)

    def test_two_successes_are_not_enough_and_say_so(self) -> None:
        self.runs("verify-on-mac", "ss")
        verdict = pv.validate(self.store, "verify-on-mac")
        self.assertEqual(verdict.verdict, pv.WITHHELD)
        self.assertFalse(verdict.may_offer)
        self.assertIn("2 of 3", verdict.reason, "a withheld verdict must read as progress, not breakage")

    def test_a_failure_resets_the_streak(self) -> None:
        """Three successes then a failure is not a validated procedure: the failure is the newer evidence."""
        self.runs("verify-on-mac", "sssf")
        verdict = pv.validate(self.store, "verify-on-mac")
        self.assertEqual(verdict.verdict, pv.WITHHELD)
        self.assertEqual(verdict.streak, 0)
        self.assertEqual(verdict.successes, 3, "the successes are still counted, they just do not carry the verdict")

    def test_successes_after_a_failure_rebuild_the_streak(self) -> None:
        self.runs("verify-on-mac", "sssfsss")
        verdict = pv.validate(self.store, "verify-on-mac")
        self.assertEqual(verdict.verdict, pv.OFFERED)
        self.assertEqual(verdict.streak, 3)
        self.assertEqual(verdict.failures, 1)

    def test_the_last_failure_travels_with_the_verdict(self) -> None:
        self.runs("git-worktree-remove", "ssf", note="polkit refused")
        verdict = pv.validate(self.store, "git-worktree-remove")
        self.assertEqual(verdict.verdict, pv.WITHHELD)
        self.assertIn("polkit refused", verdict.reason)
        self.assertTrue(verdict.last_failure_at)

    def test_no_evidence_is_its_own_verdict(self) -> None:
        verdict = pv.validate(self.store, "never-run")
        self.assertEqual(verdict.verdict, pv.UNVALIDATED)
        self.assertFalse(verdict.may_offer)

    def test_an_unreadable_store_is_not_an_untested_procedure(self) -> None:
        class Broken:
            _conn = None

        verdict = pv.validate(Broken(), "verify-on-mac")
        self.assertEqual(verdict.verdict, pv.UNREADABLE)
        self.assertNotEqual(verdict.verdict, pv.UNVALIDATED)

    def test_reading_never_creates_schema(self) -> None:
        tables_before = self.store._conn.execute(
            "SELECT COUNT(*) c FROM sqlite_master WHERE type='table'").fetchone()["c"]
        pv.validate(self.store, "never-run")
        tables_after = self.store._conn.execute(
            "SELECT COUNT(*) c FROM sqlite_master WHERE type='table'").fetchone()["c"]
        self.assertEqual(tables_before, tables_after)

    def test_the_bar_is_configurable_and_reported(self) -> None:
        self.runs("verify-on-mac", "sss")
        verdict = pv.validate(self.store, "verify-on-mac", required=5)
        self.assertEqual(verdict.verdict, pv.WITHHELD)
        self.assertEqual(verdict.required, 5)
        self.assertIn("3 of 5", verdict.reason)


class GateTests(Base):
    def test_the_gate_separates_offered_from_withheld(self) -> None:
        self.runs("ready", "sss")
        self.runs("not-ready", "ss")
        gate = pv.gate(self.store, ["ready", "not-ready", "never-run"])
        self.assertEqual([v.procedure_id for v in gate.offered], ["ready"])
        self.assertEqual([v.procedure_id for v in gate.withheld], ["not-ready"])
        self.assertIn("unvalidated", [v.verdict for v in gate.verdicts])

    def test_the_gate_keeps_the_caller_order_and_drops_duplicates(self) -> None:
        self.runs("a", "sss")
        self.runs("b", "sss")
        gate = pv.gate(self.store, ["b", "a", "b", "a"])
        self.assertEqual([v.procedure_id for v in gate.verdicts], ["b", "a"])

    def test_the_render_states_the_counts_and_the_bar(self) -> None:
        self.runs("ready", "sss")
        lines = pv.gate(self.store, ["ready"]).render()
        self.assertIn("offered", lines[0])

    def test_known_procedures_lists_what_has_evidence(self) -> None:
        self.runs("a", "s")
        self.runs("b", "sf")
        self.assertEqual(sorted(pv.known_procedures(self.store)), ["a", "b"])

    def test_known_procedures_is_empty_without_history(self) -> None:
        self.assertEqual(pv.known_procedures(self.store), [])

    def test_the_gate_writes_nothing(self) -> None:
        self.runs("a", "ss")
        before = self.store._conn.execute(
            "SELECT COUNT(*) c FROM procedure_outcomes").fetchone()["c"]
        pv.gate(self.store, ["a"])
        after = self.store._conn.execute(
            "SELECT COUNT(*) c FROM procedure_outcomes").fetchone()["c"]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
