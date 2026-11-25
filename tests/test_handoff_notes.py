"""Handoff notes (#143): typed, shape-bound, bounded, and superseded rather than deleted."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import handoff_notes as hn
from cortex.store import CortexStore


class HandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def add(self, kind="precondition", body="check the runner is awake before starting", pid="verify-on-mac", **kw):
        return hn.note(self.store, pid, kind, body, **kw)

    # -- typing ------------------------------------------------------------------------

    def test_a_note_has_a_kind(self) -> None:
        self.assertTrue(self.add()["ok"])
        kinds = [item.kind for item in hn.notes(self.store, "verify-on-mac")]
        self.assertEqual(kinds, ["precondition"])

    def test_an_untyped_note_is_refused(self) -> None:
        result = hn.note(self.store, "p", "advice", "be careful out there")
        self.assertFalse(result["ok"])
        self.assertIn("kind must be one of", result["reason"])

    def test_a_note_must_say_something_checkable(self) -> None:
        self.assertFalse(hn.note(self.store, "p", "gotcha", "careful")["ok"])

    def test_an_empty_procedure_is_refused(self) -> None:
        self.assertFalse(hn.note(self.store, "  ", "env", "the runner has 8 cores")["ok"])

    # -- superseding -------------------------------------------------------------------

    def test_a_newer_note_of_the_same_kind_supersedes_the_older(self) -> None:
        self.add(body="check the runner is awake before starting")
        self.add(body="check the runner is awake AND on tailscale")
        current = hn.notes(self.store, "verify-on-mac")
        self.assertEqual(len(current), 1)
        self.assertIn("AND on tailscale", current[0].body)
        history = hn.notes(self.store, "verify-on-mac", include_superseded=True)
        self.assertEqual(len(history), 2, "notes are superseded, never deleted")

    def test_a_note_of_a_different_kind_does_not_supersede(self) -> None:
        self.add(kind="precondition")
        self.add(kind="gotcha", body="a suspend looks like a broken runner")
        self.assertEqual(len(hn.notes(self.store, "verify-on-mac")), 2)

    # -- the handoff order -------------------------------------------------------------

    def test_a_fresh_failure_outranks_older_advice(self) -> None:
        self.add(kind="env", body="the runner is an 8-core apple silicon box")
        self.add(kind="precondition", body="confirm tailscale shows the runner online")
        self.add(kind="last_failure", body="polkit refused the inhibit over ssh")
        lines = hn.handoff(self.store, "verify-on-mac")
        self.assertIn("last_failure", lines[1])
        self.assertIn("precondition", lines[2])
        self.assertIn("env", lines[3])

    def test_a_handoff_is_at_most_four_lines_by_construction(self) -> None:
        """One note per kind, so the type system bounds the handoff instead of a truncation rule."""
        for kind in hn.KINDS:
            hn.note(self.store, "p", kind, f"a {kind} worth reading before starting")
        self.assertEqual(len(hn.notes(self.store, "p")), 4)
        self.assertEqual(len(hn.handoff(self.store, "p")), 5, "head + one line per kind")

    def test_a_cut_below_the_natural_bound_states_what_it_left_out(self) -> None:
        for kind in hn.KINDS:
            hn.note(self.store, "p", kind, f"a {kind} worth reading before starting")
        lines = hn.handoff(self.store, "p", limit=2)
        self.assertEqual(len(lines), 4, "head + 2 notes + the cut")
        self.assertIn("2 more", lines[-1])

    def test_an_unnoted_procedure_says_so(self) -> None:
        self.assertEqual(hn.handoff(self.store, "unknown"), ["No handoff notes for unknown."])

    # -- shape binding -----------------------------------------------------------------

    def test_a_note_written_against_an_old_shape_is_marked_stale(self) -> None:
        hn.note(self.store, "bar-module", "gotcha", "edit the waybar config to add a module",
                tool="omarchy-bar", shape="waybar")
        lines = hn.handoff(self.store, "bar-module", current_shapes={"omarchy-bar": "quickshell"})
        self.assertIn("stale", lines[1])

    def test_a_note_matching_the_current_shape_is_not_stale(self) -> None:
        hn.note(self.store, "bar-module", "gotcha", "plugins live under ~/.config/omarchy/plugins",
                tool="omarchy-bar", shape="quickshell")
        lines = hn.handoff(self.store, "bar-module", current_shapes={"omarchy-bar": "quickshell"})
        self.assertNotIn("stale", lines[1])

    def test_stale_and_superseded_are_different_marks(self) -> None:
        hn.note(self.store, "p", "gotcha", "the old way of doing this was wrong", tool="t3", shape="0.0.45")
        hn.note(self.store, "p", "gotcha", "the new way of doing this works", tool="t3", shape="0.1.4")
        history = hn.notes(self.store, "p", include_superseded=True)
        self.assertTrue(any(item.superseded for item in history))

    def test_a_note_without_a_shape_is_never_stale(self) -> None:
        hn.note(self.store, "p", "env", "this host has no /usr/bin/time", tool="omarchy")
        lines = hn.handoff(self.store, "p", current_shapes={"omarchy": "something-new"})
        self.assertNotIn("stale", lines[1])

    # -- reading -----------------------------------------------------------------------

    def test_reading_writes_nothing(self) -> None:
        self.add()
        before = self.store._conn.execute(f"SELECT COUNT(*), SUM(superseded_at IS NOT NULL) FROM {hn.NOTE_TABLE}").fetchone()
        hn.notes(self.store, "verify-on-mac")
        hn.handoff(self.store, "verify-on-mac")
        after = self.store._conn.execute(f"SELECT COUNT(*), SUM(superseded_at IS NOT NULL) FROM {hn.NOTE_TABLE}").fetchone()
        self.assertEqual(tuple(before), tuple(after))

    def test_an_empty_store_has_no_notes_rather_than_an_error(self) -> None:
        self.assertEqual(hn.notes(self.store, "anything"), [])

    def test_every_note_renders_its_kind(self) -> None:
        self.add(kind="env", body="the runner has 8 cores and 15GB")
        self.assertIn("[env]", hn.notes(self.store, "verify-on-mac")[0].render())


if __name__ == "__main__":
    unittest.main()
