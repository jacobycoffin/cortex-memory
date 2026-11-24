"""Say no more than the memory can support (#176)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import confidence_phrasing as cp
from cortex.store import CortexStore


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, confidence: float = 0.8, kind: str = "semantic") -> str:
        memory_id = self.store.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category="USER_STATED",
        )[0]
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET confidence = ? WHERE id = ?", (confidence, memory_id))
        return memory_id

    def test_the_bands_come_from_the_store_not_a_constant(self) -> None:
        """A constant here is a guess about a distribution (#147)."""
        for index in range(20):
            self.memory(f"fact {index}", confidence=0.30 + index * 0.03)
        ladder = cp.bands(self.store)
        self.assertGreater(ladder[1].floor, 0.30)
        self.assertLess(ladder[2].floor, 0.90)

    def test_a_flat_distribution_still_produces_ordered_bands(self) -> None:
        for index in range(5):
            self.memory(f"fact {index}", confidence=0.78)
        ladder = cp.bands(self.store)
        floors = [band.floor for band in ladder]
        self.assertEqual(floors, sorted(floors))
        self.assertGreater(ladder[2].floor, ladder[1].floor)

    def test_an_empty_store_falls_back_to_the_documented_ceilings(self) -> None:
        self.assertEqual([band.floor for band in cp.bands(self.store)], [0.0, 0.6, 0.85, 0.95])

    def test_the_audit_counts_absolute_words(self) -> None:
        self.memory("the backup always runs at 3am")
        self.memory("the dashboard listens on 8100")
        described = cp.audit(self.store)
        self.assertEqual(described.examined, 2)
        self.assertEqual(described.overclaiming, 1)

    def test_the_audit_groups_by_band(self) -> None:
        for index in range(10):
            self.memory(f"x{index}", confidence=0.2 + index * 0.08)
        self.memory("it never fails", confidence=0.2)
        described = cp.audit(self.store)
        self.assertTrue(described.by_band)

    def test_a_pile_up_falls_back_instead_of_splitting_it(self) -> None:
        """p25 and p75 of a degenerate distribution are 0.02 apart and separate nothing."""
        for index in range(20):
            self.memory(f"fact {index}", confidence=0.78 if index % 2 else 0.80)
        floors = [band.floor for band in cp.bands(self.store)]
        self.assertEqual(floors, [0.0, 0.6, 0.85, 0.95])

    def test_a_spread_distribution_supplies_its_own_cut_points(self) -> None:
        for index in range(20):
            self.memory(f"fact {index}", confidence=0.20 + index * 0.04)
        floors = [band.floor for band in cp.bands(self.store)]
        self.assertNotEqual(floors, [0.0, 0.6, 0.85, 0.95])

    def test_the_audit_says_it_is_a_lexical_screen(self) -> None:
        self.memory("it always works")
        self.assertIn("lexical screen", cp.audit(self.store).note)

    def test_the_audit_ranks_the_words(self) -> None:
        self.memory("it never fails and it always works")
        self.memory("that always happens")
        described = cp.audit(self.store)
        self.assertEqual(described.worst_words["always"], 2)

    def test_a_clean_store_says_so(self) -> None:
        self.memory("the dashboard listens on 8100")
        self.assertIn("no absolute words", cp.audit(self.store).note)

    def test_the_note_never_claims_the_audit_rewrote_anything(self) -> None:
        """The stored content is the owner's text; a checker does not get to edit it."""
        self.memory("it always works")
        self.assertIn("edit to the owner's text", cp.audit(self.store).note)

    def test_an_empty_window_is_reported_not_zero(self) -> None:
        described = cp.audit(self.store)
        self.assertEqual(described.examined, 0)
        self.assertIsNone(described.as_dict()["share"])

    def test_the_render_states_the_bands_and_the_share(self) -> None:
        self.memory("it always works")
        rendered = "\n".join(cp.render(self.store))
        self.assertIn("overclaim", rendered)
        self.assertIn("bands from the store", rendered)

    def test_a_broken_store_audits_nothing(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(cp.audit(Broken()).examined, 0)


class PhraseTests(unittest.TestCase):
    LADDER = [cp.Band("unsure", 0.0, "I'm not sure, but I have a note that"),
              cp.Band("likely", 0.6, "I think"),
              cp.Band("confident", 0.85, "I'm fairly sure"),
              cp.Band("verified", 0.95, "I know")]

    def test_a_verified_fact_is_stated_plainly(self) -> None:
        self.assertTrue(cp.phrase(0.97, band_list=self.LADDER, text="the port is 8100")
                        .startswith("I know"))

    def test_a_weak_fact_is_not_stated_plainly(self) -> None:
        self.assertTrue(cp.phrase(0.55, band_list=self.LADDER, text="the port is 8100")
                        .startswith("I'm not sure"))

    def test_the_phrase_never_adds_certainty(self) -> None:
        for confidence in (0.1, 0.55, 0.78, 0.9, 0.99):
            spoken = cp.phrase(confidence, band_list=self.LADDER, text="x")
            self.assertNotIn("definitely", spoken.lower())
            self.assertNotIn("always", spoken.lower())

    def test_unknown_confidence_is_phrased_as_unknown(self) -> None:
        """The average is a guess about a distribution, so it is not used here."""
        self.assertIn("unknown", cp.phrase(None, band_list=self.LADDER, text="x"))

    def test_a_stale_fact_carries_its_date(self) -> None:
        spoken = cp.phrase(0.9, band_list=self.LADDER, text="the port is 8100",
                           days_old=120, kind="semantic")
        self.assertIn("as of 120 days ago", spoken)

    def test_a_fresh_fact_carries_no_date(self) -> None:
        spoken = cp.phrase(0.9, band_list=self.LADDER, text="the port is 8100",
                           days_old=5, kind="semantic")
        self.assertNotIn("as of", spoken)

    def test_a_kind_without_a_window_is_never_dated(self) -> None:
        spoken = cp.phrase(0.9, band_list=self.LADDER, text="x", days_old=900, kind="mystery")
        self.assertNotIn("as of", spoken)

    def test_an_operational_fact_goes_stale_sooner_than_an_identity(self) -> None:
        self.assertLess(cp.STALE_AFTER_DAYS["operational"], cp.STALE_AFTER_DAYS["identity"])

    def test_the_body_is_not_dropped(self) -> None:
        self.assertIn("the port is 8100", cp.phrase(0.9, band_list=self.LADDER,
                                                    text="the port is 8100"))

    def test_an_empty_body_still_phrases(self) -> None:
        self.assertEqual(cp.phrase(0.9, band_list=self.LADDER, text=""), "I'm fairly sure")

    def test_absolute_words_are_found_case_insensitively(self) -> None:
        self.assertEqual(cp.overclaims("It NEVER fails"), ["never"])

    def test_ordinary_words_are_not_overclaims(self) -> None:
        self.assertEqual(cp.overclaims("the dashboard listens on 8100"), [])


if __name__ == "__main__":
    unittest.main()
