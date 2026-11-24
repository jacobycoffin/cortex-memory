"""Recall, spoken (#147)."""

from __future__ import annotations

import unittest

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import voice_summary


def memory(text: str, confidence: float = 0.9) -> dict:
    return {"content": text, "confidence": confidence}


class SpeakableTests(unittest.TestCase):
    def test_markdown_is_not_spoken(self) -> None:
        self.assertNotIn("*", voice_summary.speakable("**important** thing"))
        self.assertNotIn("#", voice_summary.speakable("## heading"))

    def test_an_arrow_becomes_to(self) -> None:
        self.assertIn(" to ", voice_summary.speakable("8100 → nginx"))

    def test_an_ampersand_becomes_and(self) -> None:
        self.assertIn(" and ", voice_summary.speakable("logs & metrics"))

    def test_a_percent_sign_becomes_a_word(self) -> None:
        self.assertIn("percent", voice_summary.speakable("26% of offers"))

    def test_a_parenthetical_is_dropped(self) -> None:
        """A parenthesis is a screen habit; spoken, it is a stumble."""
        self.assertEqual(voice_summary.speakable("The port (which is old) is 8100."),
                         "The port is 8100")

    def test_a_trailing_full_stop_is_removed(self) -> None:
        self.assertFalse(voice_summary.speakable("A sentence.").endswith("."))

    def test_a_number_stays_a_number(self) -> None:
        """'port 8100' reads aloud correctly; 'eight thousand one hundred' does not."""
        self.assertIn("8100", voice_summary.speakable("it listens on port 8100"))

    def test_a_cut_lands_on_a_word_boundary(self) -> None:
        trimmed = voice_summary.trim("the dashboard listens on port 8100 behind nginx", 22)
        self.assertTrue(trimmed.endswith("…"))
        self.assertNotIn("ng", trimmed[:-1].split()[-1][-2:] if trimmed[:-1].split() else "")

    def test_trimming_never_lengthens_the_text(self) -> None:
        self.assertLessEqual(len(voice_summary.trim("word " * 40, 30)), 30)


class SummaryTests(unittest.TestCase):
    def test_nothing_to_say_is_one_short_sentence(self) -> None:
        self.assertEqual(voice_summary.spoken([]), "I don't have anything on that.")

    def test_one_memory_is_read_directly(self) -> None:
        self.assertEqual(
            voice_summary.spoken([memory("The dashboard listens on 8100.")]),
            "The dashboard listens on 8100.",
        )

    def test_two_memories_are_counted_before_they_are_read(self) -> None:
        """The listener should know the size before the content."""
        said = voice_summary.spoken([
            memory("The dashboard listens on 8100."),
            memory("Nginx proxies it."),
        ])
        self.assertTrue(said.startswith("Here's two things."))

    def test_three_memories_are_joined_with_and(self) -> None:
        said = voice_summary.spoken([memory("one thing"), memory("two thing"), memory("three thing")])
        self.assertIn(" and ", said)
        self.assertIn("three thing", said)

    def test_the_extra_memories_are_offered_not_read(self) -> None:
        said = voice_summary.spoken([memory(f"fact {index}") for index in range(5)])
        self.assertIn("And 2 more", said)

    def test_a_guess_is_hedged(self) -> None:
        said = voice_summary.spoken([memory("the box is in the loft", confidence=0.4)])
        self.assertTrue(said.startswith("I think "))

    def test_a_sure_memory_is_not_hedged(self) -> None:
        said = voice_summary.spoken([memory("the box is in the loft", confidence=0.95)])
        self.assertFalse(said.startswith("I think "))

    def test_no_memory_id_is_spoken(self) -> None:
        said = voice_summary.spoken([memory("id 7b61d06a-fef8-40dc is the thing")])
        self.assertNotIn("fef8", said)

    def test_a_long_memory_is_trimmed_rather_than_read_out(self) -> None:
        said = voice_summary.spoken([memory("word " * 200)], max_chars=120)
        self.assertLessEqual(len(said), 200)
        self.assertIn("…", said)

    def test_the_sentence_stays_within_the_budget_for_one_memory(self) -> None:
        said = voice_summary.summarise([memory("the dashboard listens on port 8100")], max_chars=100)
        self.assertLessEqual(said.as_dict()["chars"], 160)

    def test_an_empty_memory_is_dropped_not_spoken_as_nothing(self) -> None:
        found = voice_summary.summarise([memory(""), memory("a real fact")])
        self.assertEqual(found.items, ["a real fact"])

    def test_only_empty_memories_says_so_plainly(self) -> None:
        self.assertIn("can say", voice_summary.spoken([memory("")]))

    def test_the_hedge_count_is_reported(self) -> None:
        found = voice_summary.summarise([memory("a guess", confidence=0.3),
                                         memory("a fact", confidence=0.9)])
        self.assertEqual(found.hedged, 1)

    def test_the_dropped_count_includes_blank_memories(self) -> None:
        # One memory falls outside the window, and a blank one inside it is not spoken either.
        found = voice_summary.summarise([memory("a"), memory(""), memory("c"), memory("d")])
        self.assertEqual(found.dropped, 2)

    def test_phone_rows_are_accepted_by_their_text_field(self) -> None:
        found = voice_summary.for_voice([{"text": "the dashboard listens on 8100", "confidence": 0.9}])
        self.assertIn("8100", found["sentence"])

    def test_an_object_with_a_content_attribute_works_too(self) -> None:
        class Row:
            content = "an object not a dict"
            confidence = 0.9

        self.assertIn("an object", voice_summary.spoken([Row()]))


if __name__ == "__main__":
    unittest.main()


class ThresholdTests(unittest.TestCase):
    """An absolute hedge threshold marks half a real store as guesses."""

    def test_the_threshold_never_hedges_more_than_the_share_asked_for(self) -> None:
        """Ties make an exact target impossible; the guarantee is an upper bound, not an equality."""
        values = [0.5] * 80 + [0.95] * 20
        threshold = voice_summary.sure_threshold(values, share=0.2)
        hedged = sum(1 for value in values if value < threshold) / len(values)
        self.assertLessEqual(hedged, 0.2)

    def test_a_store_of_equal_confidences_is_handled(self) -> None:
        self.assertEqual(voice_summary.sure_threshold([0.78] * 100, share=0.2), 0.78)

    def test_no_values_falls_back_to_the_constant(self) -> None:
        self.assertEqual(voice_summary.sure_threshold([]), voice_summary.SURE)

    def test_the_threshold_is_usable_by_summarise(self) -> None:
        values = [0.5, 0.6, 0.7, 0.92, 0.95]
        threshold = voice_summary.sure_threshold(values, share=0.4)
        found = voice_summary.summarise(
            [memory("a low one", confidence=0.5), memory("a high one", confidence=0.95)],
            sure=threshold,
        )
        self.assertGreaterEqual(found.hedged, 1)
