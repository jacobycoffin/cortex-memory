"""Score transparency: the arithmetic of a recall must survive the turn (item #2)."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.recall_explain import MAX_ROWS, load_breakdown, record_breakdown
from cortex.store import CortexStore


@dataclass
class Fake:
    memory: dict[str, Any]
    score: float
    components: dict[str, float] = field(default_factory=dict)


class BreakdownTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_top_five_are_kept_with_components(self) -> None:
        results = [
            Fake({"id": f"m{i}"}, 1.0 - i / 10, {"relevance": 0.5, "use_boost": 0.03})
            for i in range(8)
        ]
        written = record_breakdown(self.store, "task-1", results)
        self.assertEqual(written, MAX_ROWS)
        rows = load_breakdown(self.store, "task-1")
        self.assertEqual(len(rows), MAX_ROWS)
        self.assertEqual(rows[0]["memory_id"], "m0")
        self.assertEqual(rows[0]["components"]["use_boost"], 0.03)

    def test_shadow_scores_are_filtered_out(self) -> None:
        result = Fake({"id": "m1"}, 0.4, {"relevance": 0.3, "shadow_score": 0.9})
        record_breakdown(self.store, "task-2", [result])
        rows = load_breakdown(self.store, "task-2")
        self.assertNotIn("shadow_score", rows[0]["components"])
        self.assertIn("relevance", rows[0]["components"])

    def test_recording_the_same_turn_twice_replaces_it(self) -> None:
        record_breakdown(self.store, "task-3", [Fake({"id": "a"}, 0.9, {})])
        record_breakdown(self.store, "task-3", [Fake({"id": "b"}, 0.8, {})])
        rows = load_breakdown(self.store, "task-3")
        self.assertEqual([r["memory_id"] for r in rows], ["b"])

    def test_unknown_task_returns_empty(self) -> None:
        self.assertEqual(load_breakdown(self.store, "nope"), [])

    def test_empty_results_write_nothing(self) -> None:
        self.assertEqual(record_breakdown(self.store, "task-4", []), 0)

    def test_components_are_json_round_trippable(self) -> None:
        result = Fake({"id": "m"}, 0.5, {"alpha": 0.25, "beta": 1})
        record_breakdown(self.store, "task-5", [result])
        row = load_breakdown(self.store, "task-5")[0]
        self.assertEqual(row["components"], {"alpha": 0.25, "beta": 1.0})
        json.dumps(row)  # must be serialisable for the report script


if __name__ == "__main__":
    unittest.main()
