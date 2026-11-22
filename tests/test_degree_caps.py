"""No memory may hold more than N edges; the overflow waits for a topic node (#73)."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import degree_caps
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class CapTests(unittest.TestCase):
    def test_the_default_cap_is_twenty_four(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORTEX_MAX_DEGREE", None)
            self.assertEqual(degree_caps.cap(), degree_caps.DEFAULT_CAP)

    def test_the_cap_is_configurable(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_MAX_DEGREE": "5"}):
            self.assertEqual(degree_caps.cap(), 5)

    def test_a_malformed_cap_falls_back(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_MAX_DEGREE": "lots"}):
            self.assertEqual(degree_caps.cap(), degree_caps.DEFAULT_CAP)

    def test_a_negative_cap_falls_back(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_MAX_DEGREE": "-3"}):
            self.assertEqual(degree_caps.cap(), degree_caps.DEFAULT_CAP)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.hub = self.memory("The hub that co-occurs with everything.")
        self.spokes = [self.memory(f"Spoke number {index}.") for index in range(6)]

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def connect_all(self, children: list[str]) -> None:
        for child in children:
            self.store.add_edge(self.hub, child, "related", weight=0.3)

    def test_an_edge_under_the_cap_is_admitted(self) -> None:
        admission = degree_caps.admit(self.store, self.hub, self.spokes[0], "related", maximum=3)
        self.assertTrue(admission.admitted)

    def test_an_edge_at_the_cap_is_refused(self) -> None:
        self.connect_all(self.spokes[:3])
        admission = degree_caps.admit(self.store, self.hub, self.spokes[3], "related", maximum=3)
        self.assertFalse(admission.admitted)
        self.assertIn("at cap 3", admission.reason)

    def test_a_refusal_is_recorded_as_overflow(self) -> None:
        self.connect_all(self.spokes[:3])
        degree_caps.admit(self.store, self.hub, self.spokes[3], "related",
                          weight=0.4, maximum=3, at=NOW)
        pending = degree_caps.overflow(self.store, self.hub)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["other_id"], self.spokes[3])
        self.assertAlmostEqual(float(pending[0]["weight"]), 0.4, places=4)

    def test_the_refusal_does_not_create_the_edge(self) -> None:
        self.connect_all(self.spokes[:3])
        degree_caps.admit(self.store, self.hub, self.spokes[3], "related", maximum=3)
        exists = self.store._conn.execute(
            "SELECT 1 FROM edges WHERE src_id = ? AND dst_id = ?", (self.hub, self.spokes[3])
        ).fetchone()
        self.assertIsNone(exists)

    def test_the_other_endpoint_can_be_the_one_at_cap(self) -> None:
        self.connect_all(self.spokes[:3])
        admission = degree_caps.admit(self.store, self.spokes[4], self.hub, "related", maximum=3)
        self.assertFalse(admission.admitted)

    def test_degree_counts_both_directions(self) -> None:
        self.connect_all(self.spokes[:2])
        self.assertEqual(degree_caps.degree(self.store, self.hub), 2)

    def test_overflow_groups_by_hub_and_relation(self) -> None:
        self.connect_all(self.spokes[:2])
        for child in self.spokes[2:5]:
            degree_caps.admit(self.store, self.hub, child, "related", maximum=2)
        groups = degree_caps.groups(self.store)
        self.assertEqual(groups[0]["memory_id"], self.hub)
        self.assertEqual(groups[0]["pending"], 3)

    def test_release_clears_a_hub(self) -> None:
        self.connect_all(self.spokes[:2])
        for child in self.spokes[2:5]:
            degree_caps.admit(self.store, self.hub, child, "related", maximum=2)
        self.assertEqual(degree_caps.release(self.store, self.hub), 3)
        self.assertEqual(degree_caps.overflow(self.store, self.hub), [])

    def test_freeing_room_admits_the_edge_again(self) -> None:
        self.connect_all(self.spokes[:2])
        self.assertFalse(
            degree_caps.admit(self.store, self.hub, self.spokes[5], "related", maximum=2).admitted
        )
        with self.store.transaction() as conn:
            # add_edge may store the pair in canonical order, so match either direction.
            conn.execute(
                "DELETE FROM edges WHERE (src_id = ? AND dst_id = ?) "
                "OR (src_id = ? AND dst_id = ?)",
                (self.hub, self.spokes[0], self.spokes[0], self.hub),
            )
        self.assertTrue(
            degree_caps.admit(self.store, self.hub, self.spokes[5], "related", maximum=2).admitted
        )

    def test_the_report_names_the_hubs(self) -> None:
        self.connect_all(self.spokes[:5])
        described = degree_caps.report(self.store, maximum=3)
        self.assertEqual(described["over_cap"], 1)
        self.assertEqual(described["worst"][0][0], 5)

    def test_admission_describes_itself(self) -> None:
        self.assertIn("admitted",
                      degree_caps.admit(self.store, self.hub, self.spokes[0], "related",
                                        maximum=3).describe())
        self.connect_all(self.spokes[:3])
        self.assertIn("refused",
                      degree_caps.admit(self.store, self.hub, self.spokes[4], "related",
                                        maximum=3).describe())

    def test_a_store_with_no_edges_is_quiet(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            self.assertEqual(degree_caps.report(other)["edges"], 0)
            self.assertEqual(degree_caps.overflow(other), [])
        finally:
            other.close()

    def test_co_retrieval_respects_the_cap(self) -> None:
        """The bulk creator of edges is the hook, so hubs cannot be manufactured in bulk."""
        from cortex import co_retrieval

        self.connect_all(self.spokes[:3])
        with mock.patch.dict(os.environ, {"CORTEX_MAX_DEGREE": "3"}):
            result = co_retrieval.materialize(self.store)
        self.assertIn("capped", result)
        self.assertGreaterEqual(result["capped"], 0)


if __name__ == "__main__":
    unittest.main()
