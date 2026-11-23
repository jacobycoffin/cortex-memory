"""Cheap pass proposes, strong pass accepts, disagreements queue for review (#27)."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import two_pass
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
TEXT = (
    "The dashboard runs on port 8100. "
    "Jacoby prefers short answers. "
    "Is the vault encrypted?"
)


class FlagTests(unittest.TestCase):
    def test_off_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(two_pass.ENABLED_ENV, None)
            self.assertFalse(two_pass.enabled())

    def test_switchable(self) -> None:
        with mock.patch.dict(os.environ, {two_pass.ENABLED_ENV: "1"}):
            self.assertTrue(two_pass.enabled())


class ProposerTests(unittest.TestCase):
    def test_statements_are_proposed(self) -> None:
        proposed = two_pass.heuristic_propose(TEXT)
        self.assertEqual(len(proposed), 2)
        self.assertTrue(all(not item.endswith("?") for item in proposed))

    def test_a_question_is_not_a_fact(self) -> None:
        self.assertEqual(two_pass.heuristic_propose("Is the vault encrypted?"), [])

    def test_short_text_proposes_nothing(self) -> None:
        self.assertEqual(two_pass.heuristic_propose("ok"), [])


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def run_two_pass(self, accept=None, also_propose=None) -> two_pass.Outcome:
        return two_pass.run(self.store, TEXT, accept=accept, also_propose=also_propose, at=NOW)

    def test_agreement_writes_the_fact_and_queues_nothing(self) -> None:
        outcome = self.run_two_pass(accept=lambda candidate: True)
        self.assertEqual(len(outcome.accepted), 2)
        self.assertEqual(outcome.disagreements, 0)
        self.assertEqual(two_pass.queue(self.store), [])

    def test_an_over_reach_is_queued_not_discarded(self) -> None:
        outcome = self.run_two_pass(accept=lambda candidate: "prefers" not in candidate)
        self.assertEqual(len(outcome.rejected), 1)
        self.assertEqual(outcome.disagreements, 1)
        queued = two_pass.queue(self.store)
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0]["kind"], "over_reach")
        self.assertEqual(queued[0]["cheap_verdict"], "propose")
        self.assertEqual(queued[0]["strong_verdict"], "reject")

    def test_a_miss_is_queued_too(self) -> None:
        outcome = self.run_two_pass(
            accept=lambda candidate: True,
            also_propose=lambda text: [*two_pass.heuristic_propose(text),
                                       "The vault key lives outside the store."],
        )
        self.assertEqual(len(outcome.missed), 1)
        self.assertEqual(outcome.disagreements, 1)
        self.assertEqual(two_pass.queue(self.store)[0]["kind"], "miss")

    def test_a_failing_gate_is_a_refusal_not_an_acceptance(self) -> None:
        def broken(candidate: str) -> bool:
            raise RuntimeError("the strong model is down")

        outcome = self.run_two_pass(accept=broken)
        self.assertEqual(outcome.accepted, [])
        self.assertEqual(len(outcome.rejected), 2)
        self.assertEqual(len(two_pass.queue(self.store)), 2)

    def test_the_accepted_fact_is_the_cheap_pass_text(self) -> None:
        outcome = self.run_two_pass(accept=lambda candidate: "port 8100" in candidate)
        self.assertEqual(len(outcome.accepted), 1)
        self.assertIn("port 8100", outcome.accepted[0])

    def test_an_empty_text_produces_nothing(self) -> None:
        outcome = two_pass.run(self.store, "", at=NOW)
        self.assertEqual((outcome.accepted, outcome.disagreements), ([], 0))

    def test_the_queue_hides_resolved_items_by_default(self) -> None:
        self.run_two_pass(accept=lambda candidate: False)
        before = len(two_pass.queue(self.store))
        self.assertEqual(before, 2)
        item = two_pass.queue(self.store)[0]
        self.assertTrue(two_pass.resolve(self.store, item["disagreement_id"], "accept", at=NOW))
        self.assertEqual(len(two_pass.queue(self.store)), before - 1)
        self.assertEqual(len(two_pass.queue(self.store, unresolved_only=False)), before)

    def test_resolving_twice_is_refused(self) -> None:
        self.run_two_pass(accept=lambda candidate: False)
        item = two_pass.queue(self.store)[0]
        two_pass.resolve(self.store, item["disagreement_id"], "accept", at=NOW)
        self.assertFalse(two_pass.resolve(self.store, item["disagreement_id"], "reject", at=NOW))

    def test_resolving_an_unknown_item_is_false(self) -> None:
        self.assertFalse(two_pass.resolve(self.store, "nope", "accept"))

    def test_the_report_separates_the_two_kinds(self) -> None:
        self.run_two_pass(accept=lambda candidate: "prefers" not in candidate)
        described = two_pass.report(self.store)
        self.assertEqual(described["over_reach"], 1)
        self.assertEqual(described["missed"], 0)
        self.assertEqual(described["queued"], 1)

    def test_the_outcome_describes_itself(self) -> None:
        outcome = self.run_two_pass(accept=lambda candidate: "prefers" not in candidate)
        self.assertIn("queued for review", outcome.describe())

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(two_pass.queue(self.store), [])
        self.assertEqual(two_pass.report(self.store)["total"], 0)


if __name__ == "__main__":
    unittest.main()
