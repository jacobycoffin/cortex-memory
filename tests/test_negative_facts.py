"""A negative is a recorded fact, not a hole (#58)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import negative_facts
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class PolarityTests(unittest.TestCase):
    def test_plain_negations_are_negative(self) -> None:
        for text in (
            "The box does not run Debian.",
            "The dashboard is not behind Cloudflare.",
            "We never deploy on Fridays.",
            "The tunnel no longer works.",
            "He stopped using the old vault.",
            "The plugin is disabled.",
        ):
            self.assertEqual(negative_facts.polarity(text), negative_facts.NEGATIVE, text)

    def test_affirmative_statements_stay_affirmative(self) -> None:
        for text in (
            "The box runs Debian.",
            "The dashboard is behind Cloudflare.",
            "We deploy on Mondays.",
        ):
            self.assertEqual(negative_facts.polarity(text), negative_facts.POSITIVE, text)

    def test_a_question_is_not_a_negation(self) -> None:
        self.assertEqual(negative_facts.polarity("Is the box not running Debian?"),
                         negative_facts.POSITIVE)

    def test_an_empty_string_is_affirmative(self) -> None:
        self.assertEqual(negative_facts.polarity(""), negative_facts.POSITIVE)

    def test_no_longer_wins_over_no(self) -> None:
        recorded = negative_facts.polarity("The tunnel no longer works.")
        self.assertEqual(recorded, negative_facts.NEGATIVE)


class StatementTests(unittest.TestCase):
    """A passage that mentions "not" is not a negative fact."""

    def test_a_long_document_chunk_is_unclassified(self) -> None:
        chunk = "Vault note: Home Assistant Path: Home.md Section: Status: not yet configured. " * 3
        self.assertEqual(negative_facts.classify(chunk), negative_facts.UNCLASSIFIED)

    def test_a_document_prefix_is_unclassified(self) -> None:
        self.assertEqual(
            negative_facts.classify("Vault note: the tunnel is not reachable"),
            negative_facts.UNCLASSIFIED,
        )

    def test_a_multi_paragraph_passage_is_unclassified(self) -> None:
        passage = "One. Two. Three. Four sentences about nothing in particular."
        self.assertEqual(negative_facts.classify(passage), negative_facts.UNCLASSIFIED)

    def test_a_short_statement_is_classified(self) -> None:
        self.assertEqual(negative_facts.classify("The box does not run Debian."),
                         negative_facts.NEGATIVE)

    def test_an_unclassified_memory_is_not_answered_from(self) -> None:
        import tempfile
        from pathlib import Path

        from cortex.store import CortexStore

        with tempfile.TemporaryDirectory() as tmp:
            store = CortexStore(Path(tmp) / "cortex.db")
            try:
                memory_id = store.add_memory(
                    content="Vault note: the box does not run Debian.",
                    kind="semantic", source_type="doc", source_category="DOCUMENT_EXTRACTED",
                )[0]
                negative_facts.record(store, memory_id,
                                      "Vault note: the box does not run Debian.", at=NOW)
                self.assertEqual(negative_facts.of(store, memory_id),
                                 negative_facts.UNCLASSIFIED)
                self.assertEqual(
                    negative_facts.answer(store, "The box runs Debian.")["verdict"], "unknown"
                )
            finally:
                store.close()


class ClaimTests(unittest.TestCase):
    def test_opposite_polarity_on_the_same_topic_is_a_claim(self) -> None:
        self.assertTrue(
            negative_facts.same_claim(
                "The box runs Debian.", "The box does not run Debian."
            )
        )

    def test_the_same_polarity_is_not_a_claim(self) -> None:
        self.assertFalse(
            negative_facts.same_claim("The box runs Debian.", "The box runs Debian.")
        )

    def test_unrelated_statements_are_not_a_claim(self) -> None:
        self.assertFalse(
            negative_facts.same_claim(
                "The box runs Debian.", "The vault requires a nightly backup."
            )
        )

    def test_overlap_ignores_polarity(self) -> None:
        self.assertGreater(
            negative_facts.topic_overlap(
                "The dashboard runs on port 8100.", "The dashboard does not run on port 8100."
            ),
            0.6,
        )


class RecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def add(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def test_a_negative_is_recorded_with_its_negator(self) -> None:
        memory_id = self.add("The box does not run Debian.")
        recorded = negative_facts.record(self.store, memory_id, "The box does not run Debian.",
                                         at=NOW)
        self.assertEqual(recorded.polarity, negative_facts.NEGATIVE)
        self.assertEqual(recorded.negator, "does not")

    def test_an_affirmative_is_recorded_too(self) -> None:
        memory_id = self.add("The box runs Debian.")
        negative_facts.record(self.store, memory_id, "The box runs Debian.", at=NOW)
        self.assertEqual(negative_facts.of(self.store, memory_id), negative_facts.POSITIVE)

    def test_recording_is_idempotent(self) -> None:
        memory_id = self.add("The box does not run Debian.")
        negative_facts.record(self.store, memory_id, "The box does not run Debian.", at=NOW)
        negative_facts.record(self.store, memory_id, "The box does not run Debian.", at=NOW)
        self.assertEqual(negative_facts.report(self.store)["recorded"], 1)

    def test_an_unrecorded_memory_reports_nothing(self) -> None:
        self.assertIsNone(negative_facts.of(self.store, "no-such-id"))

    def test_the_report_counts_both_polarities(self) -> None:
        self.add("The box runs Debian.")
        first = self.add("The box does not run Debian.")
        negative_facts.record(self.store, first, "The box does not run Debian.", at=NOW)
        described = negative_facts.report(self.store)
        self.assertEqual(described["negative"], 1)
        self.assertEqual(described["affirmative"], 0)


class AnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def add(self, content: str) -> str:
        memory_id = self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        negative_facts.record(self.store, memory_id, content, at=NOW)
        return memory_id

    def test_a_stored_negative_answers_no(self) -> None:
        self.add("The box does not run Debian.")
        result = negative_facts.answer(self.store, "The box runs Debian.")
        self.assertEqual(result["verdict"], "negative")
        self.assertEqual(len(result["deny"]), 1)

    def test_a_stored_positive_answers_yes(self) -> None:
        self.add("The box runs Debian.")
        self.assertEqual(negative_facts.answer(self.store, "The box runs Debian.")["verdict"],
                         "affirmative")

    def test_nothing_stored_answers_unknown(self) -> None:
        self.assertEqual(
            negative_facts.answer(self.store, "The box runs Debian.")["verdict"], "unknown"
        )

    def test_both_answers_both_rather_than_picking(self) -> None:
        """Holding the affirmative and the negative is a dispute, not a coin flip (#48)."""
        self.add("The box runs Debian.")
        self.add("The box does not run Debian.")
        result = negative_facts.answer(self.store, "The box runs Debian.")
        self.assertEqual(result["verdict"], "both")
        self.assertEqual((len(result["support"]), len(result["deny"])), (1, 1))

    def test_a_suppressed_negative_does_not_answer(self) -> None:
        from cortex import suppression

        memory_id = self.add("The box does not run Debian.")
        suppression.suppress(self.store, memory_id, reason="user asked")
        self.assertEqual(
            negative_facts.answer(self.store, "The box runs Debian.")["verdict"], "unknown"
        )


class AbsenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_missing_positive_is_not_inferred_as_a_negative(self) -> None:
        """The one inference a memory store must never make."""
        self.store.add_memory(
            content="The vault requires a nightly backup.", kind="semantic",
            source_type="user_turn", source_category="USER_EXPLICIT",
        )
        self.assertEqual(
            negative_facts.answer(self.store, "The vault does not require a nightly backup.")
            ["verdict"],
            "unknown",
        )

    def test_negatives_lists_only_recorded_negatives(self) -> None:
        for content in ("The box does not run Debian.", "The box runs Debian."):
            memory_id = self.store.add_memory(
                content=content, kind="semantic", source_type="user_turn",
                source_category="USER_EXPLICIT",
            )[0]
            negative_facts.record(self.store, memory_id, content, at=NOW)
        listed = negative_facts.negatives(self.store)
        self.assertEqual(len(listed), 1)
        self.assertIn("does not", listed[0]["content"])


if __name__ == "__main__":
    unittest.main()
