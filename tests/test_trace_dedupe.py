"""Regression tests for the trace-event dedupe (2026-09-10).

`record_memory_trace_decision` used to persist the full candidate list twice:
once into `memory_traces.candidate_memories_json` and again inside the
`retrieval_decision` event payload.  On the live database that duplication was
~430MB — 97% of the decision ledger's bytes, storing identical data.

Events now carry a compact payload and `memory_trace_jsonl()` re-attaches the
candidate list from the trace row, so the exported ledger is unchanged.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.store import CortexStore


def _candidates(count: int) -> list[dict]:
    return [
        {
            "memory_id": f"mem-{index:03d}",
            "selected": index % 3 == 0,
            "reason": (
                "matched the goal" if index % 3 == 0 else "insufficient direct relevance"
            ),
            "score": round(0.9 - index * 0.01, 4),
            "components": {
                "lexical": 0.5,
                "graph": 0.2,
                "context_gate": 1.0,
                "activation": 0.31,
            },
        }
        for index in range(count)
    ]


class TraceEventDedupeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _record(self, task_id: str, count: int = 40) -> str:
        return self.store.record_memory_trace_decision(
            task_id=task_id,
            session_id="session-dedupe",
            goal="Synthetic dedupe fixture goal",
            context_summary="Synthetic dedupe fixture context",
            task_type="general",
            recall_mode="adaptive",
            retrieval_used=True,
            retrieval_reason="Synthetic reason for the dedupe fixture.",
            queries=["dedupe fixture query"],
            candidate_memories=_candidates(count),
        )

    def _event_payload(self, task_id: str) -> dict:
        with self.store._lock:
            row = self.store._conn.execute(
                "SELECT payload_json FROM memory_trace_events "
                "WHERE task_id=? AND event_type='retrieval_decision'",
                (task_id,),
            ).fetchone()
        self.assertIsNotNone(row, "the decision event must be appended")
        return json.loads(str(row["payload_json"]))

    # -- the duplication itself ------------------------------------------

    def test_event_payload_does_not_duplicate_candidate_memories(self) -> None:
        self._record("task-dedupe-1")
        payload = self._event_payload("task-dedupe-1")
        self.assertNotIn("candidate_memories", payload)
        # The rest of the operational explanation is still recorded.
        for key in (
            "goal",
            "context_summary",
            "retrieval_context",
            "task_type",
            "recall_mode",
            "retrieval_used",
            "retrieval_reason",
            "queries",
            "selected_memory_ids",
            "rejected_memory_ids",
        ):
            self.assertIn(key, payload)

    def test_trace_row_keeps_the_candidate_detail(self) -> None:
        self._record("task-dedupe-2", count=40)
        trace = self.store.memory_traces(task_id="task-dedupe-2")[0]
        self.assertEqual(len(trace["candidate_memories"]), 40)
        # Components survive on the trace row — nothing is lost, only un-duplicated.
        self.assertTrue(trace["candidate_memories"][0]["components"])

    def test_event_payload_is_dramatically_smaller(self) -> None:
        self._record("task-dedupe-3", count=40)
        with self.store._lock:
            event_size = self.store._conn.execute(
                "SELECT LENGTH(payload_json) FROM memory_trace_events WHERE task_id=?",
                ("task-dedupe-3",),
            ).fetchone()[0]
            trace_size = self.store._conn.execute(
                "SELECT LENGTH(candidate_memories_json) FROM memory_traces WHERE task_id=?",
                ("task-dedupe-3",),
            ).fetchone()[0]
        self.assertGreater(trace_size, 0)
        self.assertLess(
            event_size,
            trace_size // 2,
            f"event payload {event_size}B should be far smaller than the "
            f"trace candidate payload {trace_size}B",
        )

    # -- the export contract is preserved --------------------------------

    def test_jsonl_export_reattaches_candidate_memories(self) -> None:
        self._record("task-dedupe-4", count=7)
        lines = self.store.memory_trace_jsonl(task_id="task-dedupe-4").splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])["payload"]
        self.assertIn("candidate_memories", payload)
        self.assertEqual(len(payload["candidate_memories"]), 7)
        self.assertEqual(payload["candidate_memories"][0]["memory_id"], "mem-000")

    def test_jsonl_export_reattaches_components(self) -> None:
        self._record("task-dedupe-6", count=4)
        payload = json.loads(self.store.memory_trace_jsonl(task_id="task-dedupe-6"))[
            "payload"
        ]
        self.assertTrue(payload["candidate_memories"][0]["components"])

    def test_jsonl_export_keeps_legacy_inline_payloads(self) -> None:
        """An event written BEFORE the dedupe keeps its inline candidates."""
        legacy = {"goal": "legacy", "candidate_memories": [{"memory_id": "old-1"}]}
        with self.store._lock:
            self.store._conn.execute(
                "INSERT INTO memory_trace_events"
                "(event_id,task_id,event_type,payload_json,created_at) "
                "VALUES(?,?,?,?,?)",
                (
                    "legacy-event-id",
                    "task-legacy",
                    "retrieval_decision",
                    json.dumps(legacy),
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            self.store._conn.commit()

        payload = json.loads(self.store.memory_trace_jsonl(task_id="task-legacy"))[
            "payload"
        ]
        self.assertEqual(payload["candidate_memories"], [{"memory_id": "old-1"}])

    def test_jsonl_export_without_a_trace_row_still_lists_the_event(self) -> None:
        """A missing trace row must not drop the event from the export."""
        with self.store._lock:
            self.store._conn.execute(
                "INSERT INTO memory_trace_events"
                "(event_id,task_id,event_type,payload_json,created_at) "
                "VALUES(?,?,?,?,?)",
                (
                    "orphan-event-id",
                    "task-orphan",
                    "retrieval_decision",
                    json.dumps({"goal": "orphan"}),
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            self.store._conn.commit()

        lines = self.store.memory_trace_jsonl(task_id="task-orphan").splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])["payload"]
        self.assertEqual(payload["goal"], "orphan")
        self.assertNotIn("candidate_memories", payload)


if __name__ == "__main__":
    unittest.main()
