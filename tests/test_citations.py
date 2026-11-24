"""Which memories the answer actually repeated (#201)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import citations
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str) -> str:
        return self.store.add_memory(
            content=text, kind="semantic", source_type="user_turn",
            source_category="USER_STATED",
        )[0]

    def test_a_repeated_phrase_is_a_citation(self) -> None:
        memory_id = self.memory("The dashboard listens on port 8100 behind nginx in the homelab.")
        found = citations.detect(
            self.store,
            "It listens on port 8100 behind nginx in the homelab, as you said.",
            [memory_id],
        )
        self.assertEqual([item.memory_id for item in found.cited], [memory_id])
        self.assertEqual(found.cited[0].method, "phrase")

    def test_an_explicit_marker_is_a_citation(self) -> None:
        memory_id = self.memory("The dashboard listens on port 8100.")
        found = citations.detect(self.store, f"See [m:{memory_id}] for the port.", [memory_id])
        self.assertEqual(found.cited[0].method, "marker")

    def test_a_marker_beats_a_phrase(self) -> None:
        memory_id = self.memory("The dashboard listens on port 8100 behind nginx.")
        reply = f"The dashboard listens on port 8100 behind nginx [m:{memory_id}]"
        self.assertEqual(citations.detect(self.store, reply, [memory_id]).cited[0].method, "marker")

    def test_a_marker_for_an_unknown_memory_is_ignored(self) -> None:
        memory_id = self.memory("The dashboard listens on port 8100.")
        found = citations.detect(self.store, "See [m:deadbeef-0000-0000-0000-000000000000].",
                                 [memory_id])
        self.assertEqual(found.cited, [])

    def test_one_shared_word_is_not_a_citation(self) -> None:
        """'backup' and 'portal' are coincidence; a citation needs a run of words."""
        memory_id = self.memory("The nightly backup runs at four in the morning on r630.")
        found = citations.detect(self.store, "Your backup is fine, nothing to do.", [memory_id])
        self.assertEqual(found.cited, [])

    def test_a_memory_not_in_context_is_never_cited(self) -> None:
        self.memory("The dashboard listens on port 8100 behind nginx.")
        found = citations.detect(self.store, "It listens on port 8100 behind nginx.", [])
        self.assertEqual(found.cited, [])
        self.assertEqual(found.considered, 0)

    def test_a_phrase_from_a_memory_outside_the_candidates_is_ignored(self) -> None:
        cited = self.memory("The dashboard listens on port 8100 behind nginx in the homelab.")
        other = self.memory("Something completely unrelated about printer drivers.")
        found = citations.detect(
            self.store, "It listens on port 8100 behind nginx in the homelab.", [other]
        )
        self.assertEqual(found.cited, [])

    def test_the_threshold_is_configurable(self) -> None:
        memory_id = self.memory("The nightly backup runs at four in the morning on r630.")
        short = citations.detect(self.store, "the nightly backup runs at four", [memory_id],
                                 min_words=4)
        strict = citations.detect(self.store, "the nightly backup runs at four", [memory_id],
                                  min_words=9)
        self.assertEqual(len(short.cited), 1)
        self.assertEqual(strict.cited, [])

    def test_the_citation_keeps_the_phrase(self) -> None:
        memory_id = self.memory("The dashboard listens on port 8100 behind nginx in the homelab.")
        found = citations.detect(self.store, "It listens on port 8100 behind nginx.", [memory_id])
        self.assertTrue(found.cited[0].phrase)

    def test_recording_persists_one_row_per_turn(self) -> None:
        memory_id = self.memory("The dashboard listens on port 8100 behind nginx in the homelab.")
        reply = "It listens on port 8100 behind nginx in the homelab."
        citations.record(self.store, "turn-1", reply, [memory_id], at=NOW)
        citations.record(self.store, "turn-1", reply, [memory_id], at=NOW)
        self.assertEqual(len(citations.citations(self.store, memory_id)), 1)

    def test_two_turns_are_two_citations(self) -> None:
        memory_id = self.memory("The dashboard listens on port 8100 behind nginx in the homelab.")
        reply = "It listens on port 8100 behind nginx in the homelab."
        citations.record(self.store, "turn-1", reply, [memory_id], at=NOW)
        citations.record(self.store, "turn-2", reply, [memory_id], at=NOW)
        self.assertEqual(len(citations.citations(self.store, memory_id)), 2)

    def test_recording_nothing_writes_nothing(self) -> None:
        memory_id = self.memory("The dashboard listens on port 8100.")
        found = citations.record(self.store, "turn-1", "No idea, sorry.", [memory_id], at=NOW)
        self.assertEqual(found.cited, [])
        self.assertEqual(citations.report(self.store)["citations"], 0)

    def test_the_report_counts_by_method(self) -> None:
        first = self.memory("The dashboard listens on port 8100 behind nginx in the homelab.")
        second = self.memory("The nightly backup runs at four in the morning on r630 every day.")
        citations.record(self.store, "t1", "It listens on port 8100 behind nginx in the homelab.",
                         [first], at=NOW)
        citations.record(self.store, "t2", f"See [m:{second}].", [second], at=NOW)
        described = citations.report(self.store)
        self.assertEqual(described["citations"], 2)
        self.assertEqual(described["by_method"], {"marker": 1, "phrase": 1})

    def test_the_result_describes_itself(self) -> None:
        memory_id = self.memory("The dashboard listens on port 8100 behind nginx in the homelab.")
        found = citations.detect(self.store, "It listens on port 8100 behind nginx.", [memory_id])
        self.assertIn("cited out of 1", found.describe())
        empty = citations.detect(self.store, "unrelated", [memory_id])
        self.assertIn("nothing cited", empty.describe())

    def test_a_missing_memory_does_not_raise(self) -> None:
        found = citations.detect(self.store, "anything at all here", ["no-such-id"])
        self.assertEqual(found.cited, [])

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(citations.report(self.store)["citations"], 0)
        self.assertEqual(citations.citations(self.store, "nope"), [])
        self.assertEqual(citations.detect(self.store, "", []).cited, [])


class PhraseTests(unittest.TestCase):
    def test_phrases_are_contiguous_runs(self) -> None:
        self.assertIn("the port is 8100", citations.phrases("the port is 8100", size=4))

    def test_a_short_text_has_no_phrases(self) -> None:
        self.assertEqual(citations.phrases("too short", size=5), set())

    def test_words_lowercase_and_strip_punctuation(self) -> None:
        self.assertEqual(citations.words("Port 8100, behind Nginx!"), ["port", "8100", "behind",
                                                                     "nginx"])


if __name__ == "__main__":
    unittest.main()
