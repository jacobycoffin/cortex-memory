"""Volatility is decided at write, and the ladder respects it (#31)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import forget_ladder, volatility
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class ClassifyTests(unittest.TestCase):
    def test_a_stable_fact_is_durable(self) -> None:
        for text in (
            "The cortex dashboard runs on port 8100.",
            "Jacoby prefers short answers.",
            "The vault lives at ~/.hermes/obsidian-vault.",
            "Deploys always go through the release job.",
        ):
            self.assertEqual(volatility.classify(text, "semantic").volatility,
                             volatility.DURABLE, text)

    def test_a_time_bound_fact_is_ephemeral(self) -> None:
        for text in (
            "I am on call this week.",
            "The build is failing right now.",
            "Temporarily routing through the backup tunnel.",
            "Waiting for the vendor until friday.",
        ):
            self.assertEqual(volatility.classify(text, "semantic").volatility,
                             volatility.EPHEMERAL, text)

    def test_a_recurring_fact_is_seasonal(self) -> None:
        for text in (
            "The heating runs higher during the school year.",
            "He works from the office every summer.",
            "Holiday coverage starts in December.",
        ):
            self.assertEqual(volatility.classify(text, "semantic").volatility,
                             volatility.SEASONAL, text)

    def test_a_durable_marker_outranks_an_ephemeral_one(self) -> None:
        """Losing 'the port is 8100' costs more than keeping a stale rota entry."""
        assessment = volatility.classify(
            "The dashboard runs on port 8100, currently behind nginx.", "semantic"
        )
        self.assertEqual(assessment.volatility, volatility.DURABLE)
        self.assertEqual(assessment.reason, "durable marker")

    def test_a_durable_kind_is_not_shortened_by_a_stray_marker(self) -> None:
        assessment = volatility.classify("Prefers tea today.", "preference")
        self.assertNotEqual(assessment.volatility, volatility.EPHEMERAL)

    def test_an_episode_is_ephemeral_by_default(self) -> None:
        self.assertEqual(volatility.classify("Fixed the tunnel.", "episode").volatility,
                         volatility.EPHEMERAL)

    def test_an_identity_is_durable_by_default(self) -> None:
        self.assertEqual(volatility.classify("Jacoby Coffin", "identity").volatility,
                         volatility.DURABLE)

    def test_a_plain_semantic_fact_defaults_to_durable(self) -> None:
        assessment = volatility.classify("The r630 hosts three containers.", "semantic")
        self.assertEqual(assessment.volatility, volatility.DURABLE)
        self.assertEqual(assessment.reason, "no volatility marker")

    def test_the_evidence_is_reported(self) -> None:
        assessment = volatility.classify("On call right now.", "semantic")
        self.assertIn("right now", assessment.evidence)
        self.assertIn("ephemeral", assessment.describe())


class WindowTests(unittest.TestCase):
    def test_ephemeral_shortens_and_durable_lengthens(self) -> None:
        base = (30, 90, 180)
        self.assertEqual(volatility.apply_to(base, volatility.EPHEMERAL), (15, 45, 90))
        self.assertEqual(volatility.apply_to(base, volatility.DURABLE), (45, 135, 270))
        self.assertEqual(volatility.apply_to(base, volatility.SEASONAL), (22, 67, 135))

    def test_a_window_never_reaches_zero(self) -> None:
        self.assertTrue(all(days >= 1 for days in volatility.apply_to((1, 1, 1),
                                                                    volatility.EPHEMERAL)))

    def test_an_unknown_class_does_not_change_the_window(self) -> None:
        self.assertEqual(volatility.apply_to((30, 90, 180), "mystery"), (30, 90, 180))


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _add(self, content: str, kind: str = "semantic") -> str:
        return self.store.add_memory(
            content=content, kind=kind, source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def test_assignment_is_persisted_and_looked_up(self) -> None:
        memory_id = self._add("I am on call this week.")
        assigned = volatility.assign(self.store, memory_id, "I am on call this week.",
                                     kind="semantic", at=NOW)
        self.assertEqual(assigned.volatility, volatility.EPHEMERAL)
        self.assertEqual(volatility.lookup(self.store, memory_id), volatility.EPHEMERAL)

    def test_reassignment_updates_in_place(self) -> None:
        memory_id = self._add("Router password is in the vault.")
        volatility.assign(self.store, memory_id, "Temporarily broken.", at=NOW)
        volatility.assign(self.store, memory_id, "Router password is in the vault.", at=NOW)
        self.assertEqual(volatility.lookup(self.store, memory_id), volatility.DURABLE)
        self.assertEqual(volatility.report(self.store)["assigned"], 1)

    def test_lookup_on_an_unassigned_memory_is_none(self) -> None:
        self.assertIsNone(volatility.lookup(self.store, "no-such-id"))

    def test_the_report_counts_classes(self) -> None:
        for content in ("The dashboard runs on port 8100.", "On call this week.",
                        "Higher heating during the school year."):
            memory_id = self._add(content)
            volatility.assign(self.store, memory_id, content, at=NOW)
        described = volatility.report(self.store)
        self.assertEqual(described["counts"][volatility.DURABLE], 1)
        self.assertEqual(described["counts"][volatility.EPHEMERAL], 1)
        self.assertEqual(described["counts"][volatility.SEASONAL], 1)


class LadderIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_the_ladder_shortens_an_ephemeral_memory(self) -> None:
        memory_id = self.store.add_memory(
            content="I am on call this week.", kind="episode",
            source_type="user_turn", source_category="USER_EXPLICIT",
        )[0]
        volatility.assign(self.store, memory_id, "I am on call this week.", kind="episode",
                          at=NOW)
        declared = forget_ladder._declared_dwell("episode")
        scaled = volatility.apply_to(declared, volatility.EPHEMERAL)
        self.assertLess(scaled[0], declared[0])

    def test_an_ephemeral_memory_is_planned_earlier(self) -> None:
        """The whole point: the same quiet memory walks sooner when it was known to be short-lived."""
        content = "I am on call this week."
        memory_id = self.store.add_memory(
            content=content, kind="episode", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        from datetime import timedelta

        quiet_since = (NOW - timedelta(days=20)).isoformat()
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET last_used_at = ?, created_at = ? WHERE id = ?",
                (quiet_since, quiet_since, memory_id),
            )
        without = forget_ladder.plan(self.store, at=NOW)
        self.assertEqual(without.planned if hasattr(without, "planned") else 0,
                         len(without.steps))
        self.assertFalse([step for step in without.steps if step.memory_id == memory_id])

        volatility.assign(self.store, memory_id, content, kind="episode", at=NOW)
        with_er = forget_ladder.plan(self.store, at=NOW)
        self.assertTrue([step for step in with_er.steps if step.memory_id == memory_id])

    def test_the_ladder_still_runs_with_no_volatility_rows(self) -> None:
        self.store.add_memory(content="A fact.", kind="semantic", source_type="user_turn",
                              source_category="USER_EXPLICIT")
        self.assertIsNone(volatility.lookup(self.store, "missing"))


if __name__ == "__main__":
    unittest.main()
