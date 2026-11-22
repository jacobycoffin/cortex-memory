"""Three facts in one message become three records (#34)."""

from __future__ import annotations

import unittest

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import splitting
from cortex.extraction import extract_candidates


class SentenceTests(unittest.TestCase):
    def test_several_sentences_split(self) -> None:
        facts = splitting.split_facts(
            "The dashboard runs on port 8100. The vault lives in the home directory."
        )
        self.assertEqual(len(facts), 2)
        self.assertTrue(facts[0].startswith("The dashboard"))

    def test_a_single_fact_is_untouched(self) -> None:
        self.assertEqual(
            splitting.split_facts("The dashboard runs on port 8100."),
            ["The dashboard runs on port 8100."],
        )

    def test_blank_input_yields_nothing(self) -> None:
        self.assertEqual(splitting.split_facts("   "), [])

    def test_order_is_preserved(self) -> None:
        facts = splitting.split_facts(
            "Alpha is one. Bravo is two. Charlie is three."
        )
        self.assertEqual([fact.split()[0] for fact in facts], ["Alpha", "Bravo", "Charlie"])


class CompoundTests(unittest.TestCase):
    def test_two_independent_clauses_split(self) -> None:
        facts = splitting.split_compound(
            "The dashboard runs on port 8100 and the vault lives in the home directory."
        )
        self.assertEqual(len(facts), 2)
        self.assertIn("8100", facts[0])
        self.assertTrue(facts[1].startswith("the vault"))

    def test_a_comma_list_is_one_fact(self) -> None:
        text = "Ports 8100, 8101 and 8102 are open."
        self.assertEqual(len(splitting.split_compound(text)), 1)

    def test_a_short_side_blocks_the_split(self) -> None:
        text = "The dashboard runs on port 8100 and so on."
        self.assertEqual(len(splitting.split_compound(text)), 1)

    def test_a_continuation_clause_blocks_the_split(self) -> None:
        text = "The dashboard runs on port 8100 and it serves the vault."
        self.assertEqual(len(splitting.split_compound(text)), 1)

    def test_a_clause_without_a_verb_blocks_the_split(self) -> None:
        text = "The dashboard and the vault in the home directory."
        self.assertEqual(len(splitting.split_compound(text)), 1)

    def test_a_conjunction_inside_a_protected_region_is_not_a_boundary(self) -> None:
        """The split happens at the real conjunction, never inside the URL/path/code/quote."""
        cases = (
            ("The repo lives at https://example.com/alpha-and-beta and the branch is main.",
             "https://example.com/alpha-and-beta"),
            ("Run `cortex and mine --all` and the port is 8100.", "`cortex and mine --all`"),
            ("The note says (alpha and beta) and the port is 8100.", "(alpha and beta)"),
            ("The vault lives at ~/hermes/vault and the port is 8100.", "~/hermes/vault"),
        )
        for text, token in cases:
            facts = splitting.split_compound(text)
            self.assertEqual(len(facts), 2, text)
            self.assertIn(token, facts[0], text)

    def test_a_conjunction_only_inside_a_protected_region_never_splits(self) -> None:
        text = "The repo lives at https://example.com/alpha-and-beta today."
        self.assertEqual(len(splitting.split_compound(text)), 1)

    def test_an_ip_or_time_is_not_a_conjunction(self) -> None:
        self.assertEqual(
            len(splitting.split_compound("The host 192.168.1.116 serves on 8100.")), 1
        )

    def test_a_three_fact_compound_splits_thrice(self) -> None:
        facts = splitting.split_compound(
            "The dashboard runs on 8100 and the vault lives at home and the timezone is Eastern."
        )
        self.assertEqual(len(facts), 3)


class CapTests(unittest.TestCase):
    def test_the_cap_holds(self) -> None:
        text = " ".join(f"Sentence {index} is a fact about the system." for index in range(12))
        self.assertLessEqual(len(splitting.split_facts(text)), splitting.MAX_FACTS)

    def test_the_report_flags_the_cap(self) -> None:
        text = " ".join(f"Sentence {index} is a fact about the system." for index in range(12))
        self.assertTrue(splitting.describe(text)["at_cap"])

    def test_a_single_fact_is_not_multi(self) -> None:
        self.assertFalse(splitting.describe("The port is 8100.")["multi"])


class ExtractionIntegrationTests(unittest.TestCase):
    def test_a_compound_message_produces_two_candidates(self) -> None:
        candidates = extract_candidates(
            "The dashboard uses port 8100 and the vault requires a nightly backup."
        )
        self.assertGreaterEqual(len(candidates), 2)
        contents = " ".join(candidate.content for candidate in candidates)
        self.assertIn("8100", contents)
        self.assertIn("vault", contents)

    def test_a_comma_list_does_not_become_fragments(self) -> None:
        candidates = extract_candidates("Ports 8100, 8101 and 8102 are open on the host.")
        self.assertTrue(all("8101 and 8102" not in candidate.content for candidate in candidates))

    def test_a_single_sentence_still_yields_one_candidate(self) -> None:
        candidates = extract_candidates("The dashboard runs on port 8100.")
        self.assertEqual(len(candidates), 1)


if __name__ == "__main__":
    unittest.main()
