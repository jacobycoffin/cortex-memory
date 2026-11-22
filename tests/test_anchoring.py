"""Relative time is anchored to a real date at write (#40)."""

from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import anchoring

# Wednesday 2026-11-18. Deliberately midweek so "last Tuesday" is not "yesterday".
WEDNESDAY = date(2026, 11, 18)


class RelativeDayTests(unittest.TestCase):
    def test_yesterday_becomes_a_date(self) -> None:
        result = anchoring.anchor("The deploy broke yesterday.", at=WEDNESDAY)
        self.assertIn("yesterday (2026-11-17)", result.text)

    def test_today_and_tomorrow(self) -> None:
        self.assertIn("today (2026-11-18)", anchoring.anchored("On call today.", at=WEDNESDAY))
        self.assertIn("tomorrow (2026-11-19)", anchoring.anchored("Swap it tomorrow.", at=WEDNESDAY))

    def test_the_day_before_yesterday(self) -> None:
        text = anchoring.anchored("It started the day before yesterday.", at=WEDNESDAY)
        self.assertIn("the day before yesterday (2026-11-16)", text)

    def test_n_days_ago(self) -> None:
        self.assertIn("3 days ago (2026-11-15)", anchoring.anchored("Down 3 days ago.", at=WEDNESDAY))

    def test_in_n_days(self) -> None:
        self.assertIn("in 5 days (2026-11-23)", anchoring.anchored("Renewal in 5 days.", at=WEDNESDAY))


class WeekdayTests(unittest.TestCase):
    def test_last_tuesday_is_the_previous_one_not_today(self) -> None:
        text = anchoring.anchored("last Tuesday the deploy broke", at=WEDNESDAY)
        self.assertIn("last Tuesday (2026-11-17)", text)

    def test_last_weekday_on_the_same_weekday_goes_back_a_week(self) -> None:
        text = anchoring.anchored("last Wednesday it failed", at=WEDNESDAY)
        self.assertIn("(2026-11-11)", text)

    def test_next_weekday_goes_forward(self) -> None:
        text = anchoring.anchored("next Monday the review", at=WEDNESDAY)
        self.assertIn("(2026-11-23)", text)

    def test_week_and_month_phrases(self) -> None:
        self.assertIn("last week (2026-11-09)", anchoring.anchored("last week we moved", at=WEDNESDAY))
        self.assertIn("last month (2026-10-18)", anchoring.anchored("last month we moved", at=WEDNESDAY))


class MonthDayTests(unittest.TestCase):
    def test_a_past_month_day_resolves_this_year(self) -> None:
        self.assertIn("October 3 (2026-10-03)", anchoring.anchored("Since October 3 it works", at=WEDNESDAY))

    def test_a_future_month_day_resolves_last_year(self) -> None:
        self.assertIn("December 4 (2025-12-04)", anchoring.anchored("Since December 4", at=WEDNESDAY))

    def test_an_impossible_day_is_left_alone(self) -> None:
        self.assertEqual(anchoring.anchored("February 30 is a joke", at=WEDNESDAY),
                         "February 30 is a joke")


class IdempotenceTests(unittest.TestCase):
    def test_an_anchored_phrase_is_not_anchored_again(self) -> None:
        once = anchoring.anchored("The deploy broke yesterday.", at=WEDNESDAY)
        twice = anchoring.anchored(once, at=WEDNESDAY)
        self.assertEqual(once, twice)
        self.assertEqual(twice.count("2026-11-17"), 1)

    def test_re_anchoring_at_a_later_date_does_not_move_it(self) -> None:
        once = anchoring.anchored("The deploy broke yesterday.", at=WEDNESDAY)
        later = anchoring.anchored(once, at=date(2026, 12, 1))
        self.assertIn("(2026-11-17)", later)

    def test_a_phrase_with_no_relative_reference_is_untouched(self) -> None:
        text = "The dashboard runs on port 8100."
        self.assertEqual(anchoring.anchored(text, at=WEDNESDAY), text)

    def test_an_ambiguous_phrase_is_left_alone(self) -> None:
        for text in ("the other week it broke", "sometime recently", "a while back"):
            self.assertEqual(anchoring.anchored(text, at=WEDNESDAY), text, text)

    def test_a_time_of_day_is_not_invented(self) -> None:
        text = anchoring.anchored("It broke yesterday at 3pm.", at=WEDNESDAY)
        self.assertIn("yesterday (2026-11-17)", text)
        self.assertNotIn("T15", text)


class ReportTests(unittest.TestCase):
    def test_the_report_names_what_it_resolved(self) -> None:
        result = anchoring.anchor("yesterday and last Tuesday", at=WEDNESDAY)
        self.assertTrue(result.changed)
        self.assertIn("yesterday -> 2026-11-17", result.describe())
        self.assertEqual(len(result.resolved), 2)

    def test_an_unchanged_text_reports_nothing(self) -> None:
        self.assertEqual(anchoring.anchor("nothing relative here").describe(),
                         "nothing to anchor")

    def test_needs_anchor_answers_the_capture_question(self) -> None:
        self.assertTrue(anchoring.needs_anchor("happened yesterday"))
        self.assertFalse(anchoring.needs_anchor("port 8100 is open"))

    def test_phrases_lists_the_matched_expressions(self) -> None:
        self.assertEqual(anchoring.phrases("yesterday and 3 days ago", ), ["yesterday", "3 days ago"])

    def test_a_datetime_is_accepted_as_the_reference(self) -> None:
        moment = datetime(2026, 11, 18, 23, 30, tzinfo=timezone.utc)
        self.assertIn("(2026-11-17)", anchoring.anchored("yesterday it broke", at=moment))


class ExtractionTests(unittest.TestCase):
    """The real write path: a captured candidate carries the date, not just the phrase."""

    def test_a_captured_candidate_is_anchored(self) -> None:
        from cortex.extraction import extract_candidates

        candidates = extract_candidates(
            "The deploy broke yesterday and the rollback requires a verified snapshot."
        )
        joined = " ".join(candidate.content for candidate in candidates)
        self.assertIn("yesterday (", joined)

    def test_a_fact_without_relative_time_is_unchanged(self) -> None:
        from cortex.extraction import extract_candidates

        candidates = extract_candidates("The dashboard requires port 8100 to be open.")
        self.assertIn("port 8100", candidates[0].content)
        self.assertNotIn("(", candidates[0].content)


if __name__ == "__main__":
    unittest.main()
