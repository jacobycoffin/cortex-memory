"""Confirmation must not look like freshness (#68)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import prediction_error as pe
from cortex.store import CortexStore


class SlotTests(unittest.TestCase):
    def test_a_port_is_a_slot(self) -> None:
        self.assertEqual(pe.slots("The dashboard listens on port 8100."), {"port": {"8100"}})

    def test_a_colon_port_is_a_slot(self) -> None:
        self.assertIn("8100", pe.slots("it runs at localhost:8100").get("port", set()))

    def test_a_clock_time_is_not_a_port(self) -> None:
        """`:NNN` matched timestamps on the live store and invented 3,728 contradictions."""
        for text in ("entries at 18:48 and 21:03", "the run finished at 12:31", "18:48"):
            self.assertNotIn("port", pe.slots(text), text)

    def test_a_host_with_a_port_is_a_port(self) -> None:
        self.assertIn("3773", pe.slots("t3 serve on 127.0.0.1:3773").get("port", set()))
        self.assertIn("3000", pe.slots("the app listens at dashboard.local:3000").get("port", set()))

    def test_a_port_does_not_also_count_as_a_bare_number(self) -> None:
        found = pe.slots("The dashboard listens on port 8100.")
        self.assertNotIn("number", found)

    def test_an_address_is_a_slot(self) -> None:
        found = pe.slots("CT117 is 100.73.162.114")
        self.assertEqual(found["address"], {"100.73.162.114"})
        # The host name is a slot too, and it is recorded separately: one sentence can fix two
        # things, and a gate that collapsed them could not say which one changed.
        self.assertEqual(found["host"], {"CT117"})

    def test_a_path_is_a_slot(self) -> None:
        self.assertIn("/srv/example/store", pe.slots("the store lives at /srv/example/store").get(
            "path", set()))

    def test_a_host_is_a_slot(self) -> None:
        self.assertIn("CT117", pe.slots("The live host is CT117").get("host", set()))

    def test_a_date_is_a_slot(self) -> None:
        self.assertEqual(pe.slots("the cutover was 2026-11-01"), {"date": {"2026-11-01"}})


class CompareTests(unittest.TestCase):
    stored = "The dashboard listens on port 8100 behind nginx."

    def test_restating_a_fact_confirms_it(self) -> None:
        found = pe.compare(self.stored, "The dashboard listens on port 8100 behind nginx.")
        self.assertEqual(found.verdict, pe.CONFIRM)
        self.assertFalse(found.touches_the_memory)

    def test_confirmation_says_leave_it_alone(self) -> None:
        found = pe.compare(self.stored, "the dashboard listens on port 8100 behind nginx")
        self.assertIn("leave the memory alone", found.describe())

    def test_a_changed_port_contradicts(self) -> None:
        found = pe.compare(self.stored, "The dashboard listens on port 8200 behind nginx.")
        self.assertEqual(found.verdict, pe.CONTRADICT)
        self.assertEqual(found.contradicted_slots.get("port"), ["8200"])
        self.assertTrue(found.touches_the_memory)

    def test_a_contradiction_says_replace_not_overwrite(self) -> None:
        found = pe.compare(self.stored, "The dashboard listens on port 8200 behind nginx.")
        self.assertIn("replace, do not overwrite", found.describe())

    def test_a_new_slot_refines(self) -> None:
        found = pe.compare(self.stored, "The dashboard listens on port 8100 behind nginx at /srv/app")
        self.assertEqual(found.verdict, pe.REFINE)
        self.assertIn("path", found.new_slots)

    def test_a_refine_says_the_memory_became_more_specific(self) -> None:
        found = pe.compare(self.stored, "The dashboard listens on port 8100 behind nginx at /srv/app")
        self.assertIn("more specific", found.describe())

    def test_an_unrelated_sentence_is_novel(self) -> None:
        found = pe.compare(self.stored, "The backup window moved to four in the morning.")
        self.assertEqual(found.verdict, pe.NOVEL)
        self.assertFalse(found.touches_the_memory)

    def test_several_slot_kinds_changing_at_once_is_a_rewrite_not_a_correction(self) -> None:
        """A note that shifts its date, its path and a number was edited, not corrected.

        Live data forced this: of 5,725 vault re-import pairs, 95.4% differed in some slot, and
        calling each one "contradicts port X — replace" would have been a lie about a note being
        rewritten.
        """
        stored = "The nightly job moved to 2026-11-01 and writes /srv/backup on port 8100."
        found = pe.compare(stored, "The nightly job moved to 2026-11-08 and writes /srv/backups on "
                                   "port 8200.")
        self.assertEqual(found.verdict, pe.REFINE)
        self.assertGreater(len(found.contradicted_slots), 1)
        self.assertIn("rewritten rather than corrected", found.reason)

    def test_a_rewrite_says_so_in_its_sentence(self) -> None:
        stored = "The nightly job moved to 2026-11-01 and writes /srv/backup on port 8100."
        found = pe.compare(stored, "The nightly job moved to 2026-11-08 and writes /srv/backups on "
                                   "port 8200.")
        self.assertIn("not a correction", found.describe())

    def test_one_slot_kind_changing_is_still_a_contradiction(self) -> None:
        stored = "The nightly job writes /srv/backup on port 8100."
        found = pe.compare(stored, "The nightly job writes /srv/backup on port 8200.")
        self.assertEqual(found.verdict, pe.CONTRADICT)

    def test_a_date_that_moved_is_not_a_contradiction(self) -> None:
        """Vault notes carry "as of" dates that shift on every re-import without being wrong."""
        stored = "Vault note: Current Projects - last checked 2026-07-30, all gates pending."
        found = pe.compare(stored, "Vault note: Current Projects - last checked 2026-08-02, all "
                                   "gates pending.")
        self.assertEqual(found.verdict, pe.REFINE)
        self.assertIn("when-written label", found.reason)

    def test_a_refresh_timestamp_is_not_a_contradiction(self) -> None:
        """Live: "last refreshed on July 15, 2026 at 4:37 PM ET" produced a number diff every refresh."""
        stored = "Vault note: Build Info - last refreshed on July 15, 2026 at 4:37 PM ET."
        found = pe.compare(stored, "Vault note: Build Info - last refreshed on July 15, 2026 at " \
                                   "5:09 PM ET.")
        self.assertNotEqual(found.verdict, pe.CONTRADICT)
        self.assertEqual(found.contradicted_slots.get("time", [])[0][:5], "5:09 ")

    def test_a_date_change_still_counts_as_a_touch(self) -> None:
        stored = "Vault note: Current Projects - last checked 2026-07-30, all gates pending."
        found = pe.compare(stored, "Vault note: Current Projects - last checked 2026-08-02, all "
                                   "gates pending.")
        self.assertTrue(found.touches_the_memory)

    def test_a_contradiction_needs_the_same_subject(self) -> None:
        """A different number in an unrelated sentence is not a change to this memory."""
        found = pe.compare(self.stored, "The nightly job runs on port 8200 for the printer.")
        self.assertNotEqual(found.verdict, pe.CONTRADICT)

    def test_similarity_is_reported(self) -> None:
        found = pe.compare(self.stored, self.stored)
        self.assertAlmostEqual(found.similarity, 1.0, places=4)

    def test_the_judgement_round_trips(self) -> None:
        payload = pe.compare(self.stored, "The dashboard listens on port 8200.").as_dict()
        for key in ("verdict", "similarity", "contradicted_slots", "touches_the_memory", "sentence"):
            self.assertIn(key, payload)

    def test_an_empty_observation_is_novel(self) -> None:
        self.assertEqual(pe.compare(self.stored, "").verdict, pe.NOVEL)

    def test_two_empty_strings_confirm_nothing(self) -> None:
        """No overlap is no evidence: an empty observation must not be read as agreement."""
        self.assertEqual(pe.compare("", "").verdict, pe.NOVEL)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_STATED",
        )[0]

    def test_judge_reads_the_stored_text(self) -> None:
        memory_id = self.memory("The dashboard listens on port 8100 behind nginx.")
        found = pe.judge(self.store, memory_id, "dashboard on port 8100 behind nginx")
        self.assertEqual(found.verdict, pe.CONFIRM)
        self.assertEqual(found.memory_id, memory_id)

    def test_an_unknown_memory_is_not_a_crash(self) -> None:
        found = pe.judge(self.store, "no-such-id", "anything at all")
        self.assertEqual(found.verdict, pe.NOVEL)
        self.assertIn("no such memory", found.reason)

    def test_the_gate_prefers_a_contradiction_over_a_confirmation(self) -> None:
        """When one memory agrees and another disagrees, the disagreement is the news."""
        agreeing = self.memory("The dashboard listens on port 8100 behind nginx.")
        disagreeing = self.memory("The dashboard listens on port 8200 behind nginx.")
        decided = pe.gate(self.store, "The dashboard listens on port 8100 behind nginx",
                          [agreeing, disagreeing])
        self.assertEqual(decided["verdict"], pe.CONTRADICT)
        self.assertEqual(decided["memory_id"], disagreeing)

    def test_the_gate_says_when_nothing_should_be_touched(self) -> None:
        memory_id = self.memory("The dashboard listens on port 8100 behind nginx.")
        decided = pe.gate(self.store, "The dashboard listens on port 8100 behind nginx", [memory_id])
        self.assertFalse(decided["touches_the_memory"])
        self.assertIn("leave the memory alone", decided["sentence"])

    def test_the_gate_can_pick_its_own_candidates(self) -> None:
        self.memory("The dashboard listens on port 8100 behind nginx.")
        decided = pe.gate(self.store, "The dashboard listens on port 8200 behind nginx")
        self.assertEqual(decided["verdict"], pe.CONTRADICT)

    def test_the_gate_shows_the_shortlist(self) -> None:
        for port in (8100, 8101, 8102):
            self.memory(f"The dashboard listens on port {port} behind nginx.")
        decided = pe.gate(self.store, "The dashboard listens on port 8100 behind nginx")
        self.assertEqual(len(decided["candidates"]), 3)

    def test_an_empty_store_decides_novel(self) -> None:
        self.assertEqual(pe.gate(self.store, "anything")["verdict"], pe.NOVEL)


if __name__ == "__main__":
    unittest.main()
