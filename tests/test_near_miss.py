"""Near-miss pairs are similar on purpose, and wrong together (#80)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import near_miss as nm
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)

LEFT = "The dashboard runs on port 8100 behind nginx."
RIGHT = "The dashboard runs on port 8100 behind nginx in staging."
FAR = "The sourdough starter is fed at eight in the morning."


class DiscriminatorTests(unittest.TestCase):
    def test_a_differing_number_is_the_discriminator(self) -> None:
        found = nm.discriminator("The dashboard runs on port 8100.", "The dashboard runs on port 8200.")
        self.assertIn("8100", found)
        self.assertIn("8200", found)

    def test_a_negation_is_a_discriminator(self) -> None:
        self.assertIn("negation", nm.discriminator("Backups run nightly.", "Backups never run nightly."))

    def test_a_qualifier_is_a_discriminator(self) -> None:
        self.assertIn("staging", nm.discriminator(LEFT, RIGHT))

    def test_an_unrelated_pair_still_reports_its_word_difference(self) -> None:
        self.assertTrue(nm.discriminator(LEFT, FAR))

    def test_identical_text_has_no_discriminator(self) -> None:
        self.assertEqual(nm.discriminator(LEFT, LEFT), [])

    def test_similarity_lands_in_the_band_for_a_qualifier(self) -> None:
        self.assertGreaterEqual(nm.similarity(LEFT, RIGHT), nm.BAND_LOW)
        self.assertLess(nm.similarity(LEFT, RIGHT), nm.BAND_HIGH)

    def test_unrelated_text_scores_low(self) -> None:
        self.assertLess(nm.similarity(LEFT, FAR), nm.BAND_LOW)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.left = self.memory(LEFT)
        self.right = self.memory(RIGHT)
        self.far = self.memory(FAR)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def test_recording_a_pair_works_either_way_round(self) -> None:
        self.assertTrue(nm.record(self.store, self.left, self.right, at=NOW))
        self.assertEqual(len(nm.pairs(self.store)), 1)
        self.assertTrue(nm.record(self.store, self.right, self.left, at=NOW))
        self.assertEqual(len(nm.pairs(self.store)), 1)

    def test_the_same_memory_is_refused(self) -> None:
        self.assertFalse(nm.record(self.store, self.left, self.left))

    def test_an_empty_id_is_refused(self) -> None:
        self.assertFalse(nm.record(self.store, self.left, ""))

    def test_recording_does_not_touch_either_memory(self) -> None:
        before = self.store.get_memory(self.left)["content"]
        nm.record(self.store, self.left, self.right, at=NOW)
        self.assertEqual(self.store.get_memory(self.left)["content"], before)
        self.assertEqual(self.store.get_memory(self.left)["state"], "active")

    def test_a_recorded_pair_is_suppressed(self) -> None:
        nm.record(self.store, self.left, self.right, at=NOW)
        self.assertEqual(nm.suppressed(self.store, [self.left, self.right]),
                         {(min(self.left, self.right), max(self.left, self.right))})

    def test_a_pair_outside_the_set_is_not_suppressed(self) -> None:
        nm.record(self.store, self.left, self.right, at=NOW)
        self.assertEqual(nm.suppressed(self.store, [self.left]), set())

    def test_the_weaker_half_is_dropped(self) -> None:
        nm.record(self.store, self.left, self.right, at=NOW)
        ranked = [(self.left, 0.9), (self.right, 0.4)]
        dropped = nm.demote(self.store, ranked, score_of=lambda item: item[1],
                            id_of=lambda item: item[0])
        self.assertEqual(dropped, [1])

    def test_the_stronger_half_survives_whichever_order(self) -> None:
        nm.record(self.store, self.left, self.right, at=NOW)
        ranked = [(self.right, 0.9), (self.left, 0.4)]
        dropped = nm.demote(self.store, ranked, score_of=lambda item: item[1],
                            id_of=lambda item: item[0])
        self.assertEqual(dropped, [1])

    def test_nothing_is_dropped_when_only_one_of_the_pair_is_present(self) -> None:
        nm.record(self.store, self.left, self.right, at=NOW)
        ranked = [(self.left, 0.9), (self.far, 0.5)]
        self.assertEqual(nm.demote(self.store, ranked, score_of=lambda item: item[1],
                                   id_of=lambda item: item[0]), [])

    def test_release_lets_the_pair_surface_together_again(self) -> None:
        nm.record(self.store, self.left, self.right, at=NOW)
        self.assertEqual(nm.release(self.store, self.left, self.right), 1)
        self.assertEqual(nm.suppressed(self.store, [self.left, self.right]), set())

    def test_detection_proposes_a_qualifier_pair(self) -> None:
        proposals = nm.detect(self.store, scan=50)
        ids = {tuple(sorted((item.left_id, item.right_id))) for item in proposals}
        self.assertIn(tuple(sorted((self.left, self.right))), ids)

    def test_detection_never_applies_anything(self) -> None:
        nm.detect(self.store, scan=50)
        self.assertEqual(nm.pairs(self.store), [])

    def test_detection_skips_an_already_recorded_pair(self) -> None:
        nm.record(self.store, self.left, self.right, at=NOW)
        proposals = nm.detect(self.store, scan=50)
        self.assertNotIn(tuple(sorted((self.left, self.right))),
                         {tuple(sorted((item.left_id, item.right_id))) for item in proposals})

    def test_the_report_counts_recorded_and_proposed(self) -> None:
        nm.record(self.store, self.left, self.right, at=NOW)
        described = nm.report(self.store)
        self.assertEqual(described["recorded"], 1)
        self.assertEqual(described["by_user"], 1)

    def test_a_machine_recorded_pair_is_not_counted_as_a_user_one(self) -> None:
        nm.record(self.store, self.left, self.right, recorded_by="autojudge", at=NOW)
        self.assertEqual(nm.report(self.store)["by_user"], 0)

    def test_reading_pairs_never_creates_the_table(self) -> None:
        """A read path that commits is a read path that can change what it observes."""
        other = CortexStore(Path(self.tmp.name) / "readonly.db")
        try:
            nm.suppressed(other, ["a", "b"])
            nm.demote(other, [("a", 1.0), ("b", 0.5)], score_of=lambda i: i[1],
                      id_of=lambda i: i[0])
            row = other._conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = ?", (nm.TABLE,)
            ).fetchone()
            self.assertEqual(int(row[0]), 0)
        finally:
            other.close()

    def test_an_empty_store_is_quiet(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            self.assertEqual(nm.pairs(other), [])
            self.assertEqual(nm.detect(other), [])
            self.assertEqual(nm.demote(other, [("a", 1.0)], score_of=lambda i: i[1],
                                       id_of=lambda i: i[0]), [])
        finally:
            other.close()


if __name__ == "__main__":
    unittest.main()
