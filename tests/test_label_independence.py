"""Labels you can trust are labels somebody else made (#207)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import label_independence as labels


class FileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name: str, rows: list[dict]) -> Path:
        path = self.root / name
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_a_file_is_read_with_its_labeller(self) -> None:
        path = self.write("a.jsonl", [
            {"case_id": "c1", "relevant_memory_ids": ["m1", "m2"], "labeller": "jacoby"},
        ])
        found = labels.read(path)
        self.assertEqual(found.cases["c1"], {"m1", "m2"})
        self.assertEqual(found.labeller, "jacoby")

    def test_a_file_without_a_labeller_is_unattributed(self) -> None:
        path = self.write("a.jsonl", [{"case_id": "c1", "relevant_memory_ids": ["m1"]}])
        self.assertEqual(labels.read(path).labeller, labels.UNATTRIBUTED)

    def test_the_caller_can_name_the_labeller(self) -> None:
        path = self.write("a.jsonl", [{"case_id": "c1", "relevant_memory_ids": ["m1"]}])
        self.assertEqual(labels.read(path, labeller="second-pass").labeller, "second-pass")

    def test_a_missing_file_is_a_gap_not_a_crash(self) -> None:
        found = labels.read(self.root / "nope.jsonl")
        self.assertEqual(len(found), 0)

    def test_a_broken_line_is_skipped(self) -> None:
        path = self.root / "a.jsonl"
        path.write_text('{"case_id": "c1", "relevant_memory_ids": ["m1"]}\nnot json\n',
                        encoding="utf-8")
        self.assertEqual(sorted(labels.read(path).cases), ["c1"])

    def test_a_row_without_a_case_id_is_skipped(self) -> None:
        path = self.write("a.jsonl", [{"relevant_memory_ids": ["m1"]}])
        self.assertEqual(len(labels.read(path)), 0)


class AgreementTests(unittest.TestCase):
    def make(self, name: str, cases: dict) -> labels.LabelSet:
        return labels.LabelSet(labeller=name, cases={key: set(value) for key, value in cases.items()})

    def test_identical_labels_agree_completely(self) -> None:
        left = self.make("a", {"c1": ["m1", "m2"], "c2": ["m3"]})
        right = self.make("b", {"c1": ["m1", "m2"], "c2": ["m3"]})
        found = labels.agreement(left, right)
        self.assertEqual(found["mean_jaccard"], 1.0)
        self.assertEqual(found["exact"], 2)

    def test_partial_overlap_is_measured_not_rounded(self) -> None:
        left = self.make("a", {"c1": ["m1", "m2"]})
        right = self.make("b", {"c1": ["m2", "m3"]})
        self.assertAlmostEqual(labels.agreement(left, right)["mean_jaccard"], 1 / 3, places=4)

    def test_no_common_case_means_undefined_agreement(self) -> None:
        """Nobody checked is not everybody agreed."""
        left = self.make("a", {"c1": ["m1"]})
        right = self.make("b", {"c2": ["m1"]})
        found = labels.agreement(left, right)
        self.assertEqual(found["cases"], 0)
        self.assertIsNone(found["mean_jaccard"])
        self.assertIn("undefined", found["note"])

    def test_one_side_empty_is_a_disagreement(self) -> None:
        left = self.make("a", {"c1": ["m1"]})
        right = self.make("b", {"c1": []})
        found = labels.agreement(left, right)
        self.assertEqual(found["mean_jaccard"], 0.0)
        self.assertEqual(found["zero_overlap"], 1)

    def test_two_empty_answers_agree(self) -> None:
        left = self.make("a", {"c1": []})
        right = self.make("b", {"c1": []})
        self.assertEqual(labels.agreement(left, right)["mean_jaccard"], 1.0)

    def test_disagreements_name_what_each_labeller_added(self) -> None:
        left = self.make("a", {"c1": ["m1", "m2"]})
        right = self.make("b", {"c1": ["m2", "m3"]})
        rows = labels.disagreements(left, right)
        self.assertEqual(rows[0]["only_first"], ["m1"])
        self.assertEqual(rows[0]["only_second"], ["m3"])

    def test_agreements_are_not_listed_as_disagreements(self) -> None:
        left = self.make("a", {"c1": ["m1"]})
        right = self.make("b", {"c1": ["m1"]})
        self.assertEqual(labels.disagreements(left, right), [])

    def test_coverage_counts_independent_cases(self) -> None:
        first = self.make("a", {"c1": ["m1"], "c2": ["m2"]})
        second = self.make("b", {"c1": ["m1"]})
        found = labels.coverage([first, second])
        self.assertEqual(found["cases"], 2)
        self.assertEqual(found["independent"], 1)
        self.assertEqual(found["single"], 1)
        self.assertAlmostEqual(found["independent_share"], 0.5)

    def test_a_single_labeller_is_flagged_not_applauded(self) -> None:
        found = labels.coverage([self.make("a", {"c1": ["m1"]})])
        self.assertEqual(found["independent"], 0)
        self.assertIn("unverified", found["note"])

    def test_a_case_labelled_by_one_side_is_never_dropped_from_coverage(self) -> None:
        first = self.make("a", {"c1": ["m1"], "c2": ["m2"], "c3": ["m3"]})
        second = self.make("b", {"c1": ["m1"]})
        found = labels.coverage([first, second])
        self.assertEqual(found["cases"], 3)
        self.assertEqual(found["single"], 2)

    def test_reusing_one_labeller_name_does_not_count_as_independence(self) -> None:
        """Two files, one person: same name, so the case is still single-labelled."""
        first = self.make("jacoby", {"c1": ["m1"]})
        second = self.make("jacoby", {"c1": ["m2"]})
        found = labels.coverage([first, second])
        self.assertEqual(found["independent"], 0)


class OrderingTests(unittest.TestCase):
    def test_the_blind_order_is_reproducible(self) -> None:
        ids = [f"c{index}" for index in range(20)]
        self.assertEqual(labels.blind_order(ids, seed="x"), labels.blind_order(ids, seed="x"))

    def test_the_blind_order_is_not_the_input_order(self) -> None:
        ids = [f"c{index}" for index in range(20)]
        self.assertNotEqual(labels.blind_order(ids, seed="x"), ids)

    def test_the_blind_order_keeps_every_case(self) -> None:
        ids = [f"c{index}" for index in range(20)]
        self.assertEqual(sorted(labels.blind_order(ids)), sorted(ids))

    def test_a_different_seed_gives_a_different_order(self) -> None:
        ids = [f"c{index}" for index in range(20)]
        self.assertNotEqual(labels.blind_order(ids, seed="x"), labels.blind_order(ids, seed="y"))


class AdversarialTests(unittest.TestCase):
    def test_a_confident_miss_is_selected(self) -> None:
        rows = labels.adversarial([{"case_id": "c1", "gap": 0.2, "correct": False}])
        self.assertEqual([row["case_id"] for row in rows], ["c1"])

    def test_a_confident_hit_is_not_adversarial(self) -> None:
        self.assertEqual(
            labels.adversarial([{"case_id": "c1", "gap": 0.2, "correct": True}]), []
        )

    def test_a_coin_toss_is_not_adversarial(self) -> None:
        """A wrong pick with no margin proves nothing about the ranking."""
        rows = labels.adversarial([{"case_id": "c1", "gap": 0.001, "correct": False}], min_gap=0.05)
        self.assertEqual(rows, [])

    def test_the_widest_margin_comes_first(self) -> None:
        rows = labels.adversarial([
            {"case_id": "c1", "gap": 0.06, "correct": False},
            {"case_id": "c2", "gap": 0.4, "correct": False},
        ])
        self.assertEqual([row["case_id"] for row in rows], ["c2", "c1"])

    def test_a_missing_gap_is_skipped_rather_than_treated_as_zero(self) -> None:
        self.assertEqual(labels.adversarial([{"case_id": "c1", "correct": False}]), [])


class SummaryTests(unittest.TestCase):
    def make(self, name: str, cases: dict) -> labels.LabelSet:
        return labels.LabelSet(labeller=name, cases={key: set(value) for key, value in cases.items()})

    def test_one_label_set_reports_undefined_agreement(self) -> None:
        found = labels.summary([self.make("a", {"c1": ["m1"]})])
        self.assertIsNone(found["agreement"]["mean_jaccard"])
        self.assertIn("undefined", found["agreement"]["note"])

    def test_two_label_sets_report_agreement_and_disagreements(self) -> None:
        found = labels.summary([
            self.make("a", {"c1": ["m1"], "c2": ["m2"]}),
            self.make("b", {"c1": ["m1"], "c2": ["m9"]}),
        ])
        self.assertEqual(found["agreement"]["cases"], 2)
        self.assertEqual(len(found["disagreements"]), 1)

    def test_an_empty_summary_is_quiet(self) -> None:
        found = labels.summary([])
        self.assertEqual(found["coverage"]["cases"], 0)


if __name__ == "__main__":
    unittest.main()


class ReportTests(unittest.TestCase):
    """The adversarial slice has to come from what a run actually recorded."""

    def test_a_returned_rank_beyond_k_is_a_near_miss(self) -> None:
        found = labels.from_report([
            {"case_index": 1, "hit_at_k": False, "first_relevant_rank": 8},
        ], top_k=6)
        self.assertEqual(found["near_misses"], 1)
        self.assertEqual(found["selected"], ["1"])

    def test_a_returned_rank_far_down_is_not_a_near_miss(self) -> None:
        found = labels.from_report([
            {"case_index": 1, "hit_at_k": False, "first_relevant_rank": 40},
        ], top_k=6)
        self.assertEqual(found["near_misses"], 0)

    def test_a_hit_is_never_selected_as_adversarial(self) -> None:
        found = labels.from_report([{"case_index": 1, "hit_at_k": True, "first_relevant_rank": 2}])
        self.assertEqual(found["selected"], [])

    def test_a_miss_with_no_rank_is_counted_and_explained(self) -> None:
        """The report does not record where an unreturned memory would have ranked."""
        found = labels.from_report([{"case_index": 1, "hit_at_k": False,
                                     "first_relevant_rank": None}])
        self.assertEqual(found["misses"], 1)
        self.assertEqual(found["misses_with_a_recorded_rank"], 0)
        self.assertIn("cannot be told apart", found["note"])

    def test_counts_add_up(self) -> None:
        found = labels.from_report([
            {"case_index": 1, "hit_at_k": True, "first_relevant_rank": 1},
            {"case_index": 2, "hit_at_k": False, "first_relevant_rank": 7},
            {"case_index": 3, "hit_at_k": False, "first_relevant_rank": None},
        ])
        self.assertEqual((found["cases"], found["hits"], found["misses"]), (3, 1, 2))
