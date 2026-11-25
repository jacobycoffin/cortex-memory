"""Tool deprecation tracking (#138): a shape change demotes, it never deletes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import tool_deprecation
from cortex.store import CortexStore


class DeprecationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def proc(self, pid="p1", tool="t3", shape="0.0.45", note=""):
        return tool_deprecation.record_procedure(self.store, pid, tool, shape=shape, note=note)

    # -- observing a shape ------------------------------------------------------------

    def test_the_first_observation_invalidates_nothing(self) -> None:
        result = tool_deprecation.observe(self.store, "t3", "0.1.4")
        self.assertFalse(result.changed)
        self.assertEqual(result.invalidated, [])
        self.assertIn("first observation", result.render())

    def test_the_same_shape_twice_is_a_heartbeat_not_a_change(self) -> None:
        tool_deprecation.observe(self.store, "t3", "0.1.4")
        self.proc(shape="0.1.4")
        again = tool_deprecation.observe(self.store, "t3", "0.1.4")
        self.assertTrue(again.already_known)
        self.assertFalse(again.changed)
        self.assertEqual(again.invalidated, [], "a heartbeat must not demote anything")
        self.assertEqual(tool_deprecation.status(self.store).counts.get("current"), 1)

    def test_a_new_shape_demotes_the_procedures_written_against_the_old_one(self) -> None:
        tool_deprecation.observe(self.store, "t3", "0.0.45")
        self.proc("p1", shape="0.0.45")
        self.proc("p2", shape="0.0.45")
        result = tool_deprecation.observe(self.store, "t3", "0.1.4")
        self.assertTrue(result.changed)
        self.assertEqual(sorted(result.invalidated), ["p1", "p2"])
        counts = tool_deprecation.status(self.store).counts
        self.assertEqual(counts.get("stale"), 2)
        self.assertIsNone(counts.get("current"))

    def test_the_previous_shape_is_kept_for_a_rollback(self) -> None:
        tool_deprecation.observe(self.store, "t3", "0.0.45")
        tool_deprecation.observe(self.store, "t3", "0.1.4")
        self.assertEqual(tool_deprecation.status(self.store).tools["t3"]["previous"], "0.0.45")

    def test_a_rollback_revalidates_nothing_automatically(self) -> None:
        """Going back to the old shape restores the possibility of validity, not the claim of it."""
        tool_deprecation.observe(self.store, "t3", "0.0.45")
        self.proc(shape="0.0.45")
        tool_deprecation.observe(self.store, "t3", "0.1.4")
        tool_deprecation.observe(self.store, "t3", "0.0.45")
        self.assertEqual(tool_deprecation.status(self.store).counts.get("stale"), 1)

    # -- invalidation is per tool ------------------------------------------------------

    def test_a_change_in_one_tool_leaves_another_alone(self) -> None:
        tool_deprecation.observe(self.store, "gh", "2.0")
        tool_deprecation.observe(self.store, "pytest", "9.2")
        self.proc("g1", tool="gh", shape="2.0")
        self.proc("y1", tool="pytest", shape="9.2")
        tool_deprecation.observe(self.store, "gh", "3.0")
        status = tool_deprecation.status(self.store)
        self.assertEqual(status.tools["gh"]["procedures"].get("stale"), 1)
        self.assertEqual(status.tools["pytest"]["procedures"].get("current"), 1,
                         "over-broad invalidation teaches people to ignore the flag")

    # -- pinned, unpinned, stale -------------------------------------------------------

    def test_a_procedure_without_a_shape_is_unpinned_not_current(self) -> None:
        tool_deprecation.observe(self.store, "t3", "0.1.4")
        result = tool_deprecation.record_procedure(self.store, "p1", "t3")
        self.assertEqual(result["state"], "unpinned")

    def test_a_procedure_pinned_to_a_stale_shape_starts_stale(self) -> None:
        tool_deprecation.observe(self.store, "t3", "0.1.4")
        self.assertEqual(self.proc(shape="0.0.45")["state"], "stale")

    def test_a_procedure_pinned_to_the_current_shape_is_current(self) -> None:
        tool_deprecation.observe(self.store, "t3", "0.1.4")
        self.assertEqual(self.proc(shape="0.1.4")["state"], "current")

    def test_registering_needs_an_id_and_a_tool(self) -> None:
        self.assertFalse(tool_deprecation.record_procedure(self.store, "", "t3")["ok"])
        self.assertFalse(tool_deprecation.record_procedure(self.store, "p1", " ")["ok"])

    # -- re-verification ---------------------------------------------------------------

    def test_revalidating_without_evidence_is_refused(self) -> None:
        tool_deprecation.observe(self.store, "t3", "0.1.4")
        self.proc(shape="0.0.45")
        result = tool_deprecation.revalidate(self.store, "p1")
        self.assertFalse(result["ok"])
        self.assertIn("evidence", result["reason"])

    def test_revalidating_with_evidence_restores_current(self) -> None:
        tool_deprecation.observe(self.store, "t3", "0.1.4")
        self.proc(shape="0.0.45")
        result = tool_deprecation.revalidate(self.store, "p1", evidence="ran t3 thread list --json: V2 rows returned")
        self.assertTrue(result["ok"])
        self.assertEqual(result["shape"], "0.1.4")
        self.assertEqual(tool_deprecation.status(self.store).counts.get("current"), 1)

    def test_revalidating_an_unknown_procedure_is_refused(self) -> None:
        self.assertFalse(tool_deprecation.revalidate(self.store, "nope", evidence="x")["ok"])

    def test_revalidating_a_tool_with_no_observed_shape_is_refused(self) -> None:
        tool_deprecation.record_procedure(self.store, "p1", "ghost-tool")
        result = tool_deprecation.revalidate(self.store, "p1", evidence="x")
        self.assertFalse(result["ok"])
        self.assertIn("no observed shape", result["reason"])

    # -- reading -----------------------------------------------------------------------

    def test_status_names_the_shape_and_the_counts(self) -> None:
        tool_deprecation.observe(self.store, "t3", "0.1.4")
        self.proc("p1", shape="0.1.4")
        self.proc("p2", shape="0.0.45")
        lines = tool_deprecation.status(self.store).render()
        self.assertIn("t3 @ 0.1.4", lines[1])
        self.assertIn("current 1", lines[1])
        self.assertIn("stale 1", lines[1])

    def test_an_empty_store_says_nothing_observed(self) -> None:
        self.assertEqual(tool_deprecation.status(self.store).render(), ["No tool shape has been observed yet."])

    def test_a_broken_store_says_it_could_not_read(self) -> None:
        class Broken:
            _conn = None

        status = tool_deprecation.status(Broken())
        self.assertFalse(status.readable)
        self.assertIn("could not be read", status.render()[0])


if __name__ == "__main__":
    unittest.main()
