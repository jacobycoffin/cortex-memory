"""A failure is remembered with its host and its workaround, and stops being offered once resolved (#133)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import failure_modes as fm
from cortex.store import CortexStore

CONTEXT = "systemd-inhibit --mode=block over ssh"
SIGNAL = "polkit refused: interactive authentication required"


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()


class RecordTests(Base):
    def test_a_failure_with_a_context_is_recorded(self) -> None:
        result = fm.record(self.store, context=CONTEXT, signal=SIGNAL,
                           workaround="run it from the GUI session, not over ssh", host="omarchy")
        self.assertTrue(result["ok"], result.get("reason"))
        catalogue = fm.lookup(self.store)
        self.assertEqual(len(catalogue.modes), 1)
        self.assertEqual(catalogue.modes[0].host, "omarchy")

    def test_a_failure_without_a_context_is_refused(self) -> None:
        result = fm.record(self.store, context="", signal=SIGNAL)
        self.assertFalse(result["ok"])
        self.assertIn("context", result["reason"])

    def test_a_failure_without_a_signal_is_refused(self) -> None:
        result = fm.record(self.store, context=CONTEXT, signal="")
        self.assertFalse(result["ok"])
        self.assertIn("signal", result["reason"])

    def test_a_repeat_increments_the_count(self) -> None:
        fm.record(self.store, context=CONTEXT, signal=SIGNAL)
        fm.record(self.store, context=CONTEXT, signal=SIGNAL)
        mode = fm.lookup(self.store).modes[0]
        self.assertEqual(mode.occurrences, 2)

    def test_a_workaround_can_be_added_later_and_is_not_erased_by_silence(self) -> None:
        fm.record(self.store, context=CONTEXT, signal=SIGNAL)
        fm.record(self.store, context=CONTEXT, signal=SIGNAL, workaround="run from the GUI session")
        self.assertEqual(fm.lookup(self.store).modes[0].workaround, "run from the GUI session")
        fm.record(self.store, context=CONTEXT, signal=SIGNAL)  # no workaround given this time
        self.assertEqual(fm.lookup(self.store).modes[0].workaround, "run from the GUI session",
                         "silence must not delete what was learned")

    def test_the_signal_is_redacted_before_it_is_stored(self) -> None:
        secret = "sk" + "-" + "a" * 24
        fm.record(self.store, context=CONTEXT, signal=f"auth failed with {secret}", workaround="")
        stored = fm.lookup(self.store).modes[0].signal
        self.assertNotIn(secret, stored)
        self.assertIn("[redacted]", stored)

    def test_a_credential_assignment_is_redacted(self) -> None:
        fm.record(self.store, context=CONTEXT, signal="curl failed: Authorization: Bearer abc123def456")
        stored = fm.lookup(self.store).modes[0].signal
        self.assertNotIn("abc123def456", stored)


class LookupTests(Base):
    def test_a_mode_without_a_workaround_is_unexplained(self) -> None:
        fm.record(self.store, context=CONTEXT, signal=SIGNAL)
        mode = fm.lookup(self.store).modes[0]
        self.assertFalse(mode.explained)
        self.assertEqual(fm.workaround_for(self.store, CONTEXT), "",
                         "an unexplained failure is worth knowing, never offered as guidance")

    def test_a_workaround_is_returned_for_its_context(self) -> None:
        fm.record(self.store, context=CONTEXT, signal=SIGNAL, workaround="run from the GUI session")
        self.assertEqual(fm.workaround_for(self.store, CONTEXT), "run from the GUI session")

    def test_a_workaround_does_not_leak_to_another_context(self) -> None:
        fm.record(self.store, context=CONTEXT, signal=SIGNAL, workaround="run from the GUI session")
        self.assertEqual(fm.workaround_for(self.store, "docker compose up"), "")

    def test_a_resolved_mode_stops_being_offered_but_is_kept(self) -> None:
        fm.record(self.store, context=CONTEXT, signal=SIGNAL, workaround="run from the GUI session")
        fm.resolve(self.store, context=CONTEXT, signal=SIGNAL)
        self.assertEqual(fm.workaround_for(self.store, CONTEXT), "")
        self.assertEqual(fm.lookup(self.store).modes, [])
        kept = fm.lookup(self.store, include_resolved=True).modes
        self.assertEqual(len(kept), 1, "the history stays readable after the advice retires")
        self.assertTrue(kept[0].resolved)

    def test_a_recurrence_reopens_a_resolved_mode(self) -> None:
        fm.record(self.store, context=CONTEXT, signal=SIGNAL, workaround="run from the GUI session")
        fm.resolve(self.store, context=CONTEXT, signal=SIGNAL)
        fm.record(self.store, context=CONTEXT, signal=SIGNAL, workaround="run from the GUI session")
        self.assertEqual(fm.workaround_for(self.store, CONTEXT), "run from the GUI session",
                         "it failed again: the workaround is live again")

    def test_resolving_an_unknown_mode_says_so(self) -> None:
        result = fm.resolve(self.store, context="nothing", signal="never seen")
        self.assertFalse(result["ok"])
        self.assertIn("no such failure mode", result["reason"])

    def test_an_empty_catalogue_reads_as_empty_not_broken(self) -> None:
        catalogue = fm.lookup(self.store)
        self.assertTrue(catalogue.readable)
        self.assertEqual(catalogue.modes, [])

    def test_a_broken_store_is_unreadable(self) -> None:
        class Broken:
            _conn = None

        catalogue = fm.lookup(Broken())
        self.assertFalse(catalogue.readable)
        self.assertIn("could not be read", catalogue.render()[0])

    def test_the_render_counts_the_states(self) -> None:
        fm.record(self.store, context=CONTEXT, signal=SIGNAL, workaround="run from the GUI session")
        fm.record(self.store, context="other", signal="boom")
        lines = fm.lookup(self.store).render()
        self.assertIn("with a workaround", lines[0])
        self.assertIn("unexplained", lines[0])


if __name__ == "__main__":
    unittest.main()
