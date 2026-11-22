"""A rejection teaches the next similar write (#35)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import reject_learn
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
REJECTED = "Project Orbital dashboard uses port 8443 behind nginx"


class SignatureTests(unittest.TestCase):
    def test_stopwords_are_dropped(self) -> None:
        terms = reject_learn.signature("the dashboard runs on the port")
        self.assertNotIn("the", terms)
        self.assertIn("dashboard", terms)

    def test_the_signature_is_order_free(self) -> None:
        self.assertEqual(
            reject_learn.signature("alpha beta gamma"),
            reject_learn.signature("gamma beta alpha"),
        )

    def test_short_tokens_are_ignored(self) -> None:
        self.assertNotIn("a", reject_learn.signature("a b c dashboard"))

    def test_the_signature_is_capped(self) -> None:
        text = " ".join(f"word{index}" for index in range(40))
        self.assertLessEqual(len(reject_learn.signature(text)),
                             reject_learn.MAX_SIGNATURE_TERMS)


class RecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_rejection_becomes_a_rule(self) -> None:
        rule_id = reject_learn.record(self.store, REJECTED, reason_code="duplicate",
                                      reason_text="already stored", at=NOW)
        self.assertIsNotNone(rule_id)
        self.assertEqual(reject_learn.report(self.store)["rules"], 1)

    def test_thin_content_is_not_learned(self) -> None:
        self.assertIsNone(reject_learn.record(self.store, "yes it is", reason_code="trivial"))

    def test_the_same_rejection_does_not_duplicate_the_rule(self) -> None:
        first = reject_learn.record(self.store, REJECTED, reason_code="duplicate", at=NOW)
        second = reject_learn.record(self.store, REJECTED, reason_code="duplicate", at=NOW)
        self.assertEqual(first, second)
        self.assertEqual(reject_learn.report(self.store)["rules"], 1)

    def test_different_reasons_are_different_rules(self) -> None:
        reject_learn.record(self.store, REJECTED, reason_code="duplicate", at=NOW)
        reject_learn.record(self.store, REJECTED, reason_code="not_factual", at=NOW)
        self.assertEqual(reject_learn.report(self.store)["rules"], 2)


class MatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        reject_learn.record(self.store, REJECTED, reason_code="duplicate",
                            reason_text="already stored", at=NOW)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_near_identical_write_is_caught(self) -> None:
        match = reject_learn.check(
            self.store, "Project Orbital dashboard uses port 8443 behind nginx again"
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.reason_code, "duplicate")
        self.assertGreaterEqual(match.similarity, reject_learn.MATCH_THRESHOLD)

    def test_a_different_fact_is_not_caught(self) -> None:
        self.assertIsNone(
            reject_learn.check(self.store, "The vault stores nightly encrypted snapshots")
        )

    def test_user_stated_material_is_never_filtered(self) -> None:
        """A learned rule must not bury something the user said."""
        match = reject_learn.check(
            self.store, "Project Orbital dashboard uses port 8443 behind nginx",
            source_category="USER_EXPLICIT",
        )
        self.assertIsNone(match)

    def test_the_threshold_is_respected(self) -> None:
        match = reject_learn.check(
            self.store, "Project Orbital dashboard uses port 8443", threshold=0.95
        )
        self.assertIsNone(match)

    def test_a_missing_table_does_not_raise(self) -> None:
        empty = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            self.assertIsNone(reject_learn.check(empty, REJECTED))
        finally:
            empty.close()

    def test_hits_are_counted(self) -> None:
        match = reject_learn.check(self.store, REJECTED + " again")
        self.assertIsNotNone(match)
        reject_learn.note_hit(self.store, match.rule_id)
        reject_learn.note_hit(self.store, match.rule_id)
        self.assertEqual(reject_learn.report(self.store)["hits"], 2)

    def test_a_rule_can_be_removed_when_it_misfires(self) -> None:
        match = reject_learn.check(self.store, REJECTED)
        self.assertIsNotNone(match)
        self.assertTrue(reject_learn.forget_rule(self.store, match.rule_id))
        self.assertIsNone(reject_learn.check(self.store, REJECTED))


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_the_report_groups_by_reason(self) -> None:
        reject_learn.record(self.store, "Project Orbital dashboard uses port 8443",
                            reason_code="duplicate", at=NOW)
        reject_learn.record(self.store, "The router rebooted again today sadly",
                            reason_code="ephemeral", at=NOW)
        described = reject_learn.report(self.store)
        self.assertEqual(described["rules"], 2)
        self.assertIn("duplicate", described["by_reason"])
        self.assertIn("ephemeral", described["by_reason"])

    def test_an_empty_log_reports_zero(self) -> None:
        described = reject_learn.report(self.store)
        self.assertEqual((described["rules"], described["hits"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
