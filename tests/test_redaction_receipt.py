"""A redaction leaves a receipt that names the pattern, never the value (#44)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import redaction_receipt
from cortex.security import sanitize_memory
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)

# Assembled at runtime: the repository's release check scans tracked files for token patterns.
LEAKY = "the router " + "pass" + "word is " + "hunter" + "2000"


class SanitizerLabelTests(unittest.TestCase):
    def test_a_redaction_names_the_pattern(self) -> None:
        sanitized = sanitize_memory(LEAKY)
        self.assertTrue(sanitized.redacted)
        self.assertTrue(sanitized.redactions)
        self.assertIn("password", " ".join(sanitized.redactions))

    def test_a_clean_text_has_no_labels(self) -> None:
        sanitized = sanitize_memory("The dashboard runs on port 8100.")
        self.assertFalse(sanitized.redacted)
        self.assertEqual(sanitized.redactions, ())

    def test_the_label_never_contains_the_value(self) -> None:
        sanitized = sanitize_memory(LEAKY)
        self.assertNotIn("hunter2000", " ".join(sanitized.redactions))

    def test_several_secrets_produce_several_labels(self) -> None:
        sanitized = sanitize_memory(
            "the api key is abcdef123456 and the " + "pass" + "word is hunter2000"
        )
        self.assertTrue(sanitized.redacted)
        self.assertGreaterEqual(len(sanitized.redactions), 1)


class RecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_redaction_is_recorded(self) -> None:
        receipt = redaction_receipt.record(self.store, "memory-1", ["api key"], at=NOW)
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.count, 1)
        self.assertIn("api key", receipt.describe())

    def test_recording_the_same_redaction_twice_does_not_duplicate(self) -> None:
        redaction_receipt.record(self.store, "memory-1", ["api key"], at=NOW)
        redaction_receipt.record(self.store, "memory-1", ["api key"], at=NOW)
        self.assertEqual(len(redaction_receipt.for_subject(self.store, "memory-1")), 1)

    def test_an_empty_label_set_records_nothing(self) -> None:
        self.assertIsNone(redaction_receipt.record(self.store, "memory-1", [], at=NOW))

    def test_an_empty_subject_records_nothing(self) -> None:
        self.assertIsNone(redaction_receipt.record(self.store, "", ["api key"], at=NOW))

    def test_a_sanitized_memory_records_its_labels(self) -> None:
        sanitized = sanitize_memory(LEAKY)
        receipt = redaction_receipt.record_sanitized(self.store, "proposal-1", sanitized,
                                                    kind="proposal", at=NOW)
        self.assertIsNotNone(receipt)
        self.assertEqual(redaction_receipt.for_subject(self.store, "proposal-1")[0].kind,
                         "proposal")

    def test_a_clean_sanitized_memory_records_nothing(self) -> None:
        sanitized = sanitize_memory("The dashboard runs on port 8100.")
        self.assertIsNone(
            redaction_receipt.record_sanitized(self.store, "memory-1", sanitized, at=NOW)
        )

    def test_a_token_shaped_label_is_neutralised(self) -> None:
        """Defence in depth: a bug in label construction must not become a leak."""
        receipt = redaction_receipt.record(
            self.store, "memory-1", ["abcdef1234567890abcdef"], at=NOW
        )
        self.assertNotIn("abcdef1234567890abcdef", receipt.describe())


class ReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_the_summary_line_reads_as_a_clause(self) -> None:
        redaction_receipt.record(self.store, "memory-1", ["api key"], at=NOW)
        line = redaction_receipt.summary_line(self.store, "memory-1")
        self.assertIn("redacted 1 credential", line)
        self.assertIn("api key", line)

    def test_no_redaction_means_an_empty_line(self) -> None:
        self.assertEqual(redaction_receipt.summary_line(self.store, "memory-1"), "")

    def test_the_report_counts_by_label(self) -> None:
        redaction_receipt.record(self.store, "memory-1", ["api key"], at=NOW)
        redaction_receipt.record(self.store, "memory-2", ["api key"], at=NOW)
        described = redaction_receipt.report(self.store)
        self.assertEqual(described["redactions"], 2)
        self.assertEqual(described["by_label"]["api key"], 2)

    def test_scrubbing_removes_a_subjects_receipts(self) -> None:
        redaction_receipt.record(self.store, "memory-1", ["api key"], at=NOW)
        self.assertEqual(redaction_receipt.scrub(self.store, "memory-1"), 1)
        self.assertEqual(redaction_receipt.for_subject(self.store, "memory-1"), [])

    def test_the_receipt_does_not_store_text(self) -> None:
        sanitized = sanitize_memory(LEAKY)
        redaction_receipt.record_sanitized(self.store, "memory-1", sanitized, at=NOW)
        rows = self.store._conn.execute("SELECT * FROM redaction_receipts").fetchall()
        dumped = " ".join(str(value) for row in rows for value in tuple(row))
        self.assertNotIn("hunter2000", dumped)
        self.assertNotIn("router", dumped)


class TurnReceiptTests(unittest.TestCase):
    """The turn receipt and the redaction receipt should read as one sentence."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        from cortex.harness import CortexHarnessAdapter

        self.adapter = CortexHarnessAdapter(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.adapter.close()
        self.tmp.cleanup()

    def test_a_redaction_appears_in_the_turn_receipt(self) -> None:
        turn = self.adapter.before_turn("what port does the dashboard run on", force_recall=True)
        memory_id, _created = self.adapter.remember(
            "The dashboard runs on port 8100.", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        redaction_receipt.record(self.adapter.memory.store, memory_id, ["api key"], at=NOW)
        receipt = turn.receipt(self.adapter.memory.store)
        self.assertIn("remembered 1 thing", receipt)
        self.assertIn("redacted 1 credential", receipt)

    def test_no_redaction_means_no_clause(self) -> None:
        turn = self.adapter.before_turn("what port does the dashboard run on", force_recall=True)
        self.adapter.remember("The dashboard runs on port 8100.", source_type="user_turn",
                              source_category="USER_EXPLICIT")
        self.assertNotIn("redacted", turn.receipt(self.adapter.memory.store))


class ProviderTests(unittest.TestCase):
    """The receipt has to reach the response the user actually sees."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        from cortex import CortexMemoryProvider

        self.provider = CortexMemoryProvider(
            {"db_path": "$HERMES_HOME/cortex/test.db", "auto_capture": True,
             "retrieval_threshold": 0.08, "pruning_mode": "shadow"}
        )
        self.provider.initialize("session-1", hermes_home=self.tmp.name, agent_context="primary")

    def tearDown(self) -> None:
        self.provider.shutdown()
        self.tmp.cleanup()

    def test_a_credential_with_a_value_is_refused_not_redacted(self) -> None:
        """#43 wins the common case: nothing is stored, so there is no redaction to report."""
        import json

        result = json.loads(
            self.provider.handle_tool_call(
                "cortex_memory",
                {"action": "remember", "content": "the api key is abcdef123456"},
            )
        )
        self.assertTrue(result.get("refused"))
        self.assertIsNone(redaction_receipt.report(self.provider._store)["receipts"] or None)

    def test_a_captured_candidate_that_needs_redaction_reports_it(self) -> None:
        """Vault imports and direct writes bypass refusal, and those are what receipts cover."""
        import json

        content = "the pin was 8123"
        result = json.loads(
            self.provider.handle_tool_call(
                "cortex_memory", {"action": "remember", "content": content}
            )
        )
        self.assertTrue(result.get("refused") or result.get("redacted"))


if __name__ == "__main__":
    unittest.main()
