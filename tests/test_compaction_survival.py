"""What has to survive a compaction, and how you know it did (#152)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import compaction_survival as cs
from cortex.store import CortexStore

SESSION = "local-abc123"


class VerifyTests(unittest.TestCase):
    def test_nothing_lost_is_ok(self) -> None:
        described = cs.verify(["a", "b"], ["a", "b", "c"])
        self.assertTrue(described["ok"])
        self.assertEqual(described["lost"], [])

    def test_a_loss_is_named(self) -> None:
        described = cs.verify(["a", "b", "c"], ["a"])
        self.assertFalse(described["ok"])
        self.assertEqual(described["lost"], ["b", "c"])

    def test_the_loss_share_is_computed(self) -> None:
        self.assertAlmostEqual(cs.verify(["a", "b", "c", "d"], ["a", "b"])["loss_share"], 0.5)

    def test_an_empty_before_list_proves_nothing(self) -> None:
        """`0 lost of 0` must not read as a clean bill of health."""
        described = cs.verify([], ["a"])
        self.assertIsNone(described["loss_share"])
        self.assertIn("nothing to lose", described["note"])


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, kind: str = "semantic", pinned: bool = False,
               category: str = "USER_STATED") -> str:
        memory_id = self.store.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category=category,
        )[0]
        if pinned:
            with self.store.transaction() as conn:
                conn.execute("UPDATE memories SET pinned = 1 WHERE id = ?", (memory_id,))
        return memory_id

    def trace(self, ids: list[str], *, session: str = SESSION) -> None:
        self.store.record_memory_trace(
            task_id="t1", session_id=session, goal="g", retrieval_used=True,
            selected_memory_ids=ids, candidate_memories=[], rejected_memory_ids=[],
        ) if hasattr(self.store, "record_memory_trace") else None

    def seed_trace(self, ids: list[str], *, session: str = SESSION) -> None:
        """Write a trace row directly: the API surface differs across versions."""
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO memory_traces(trace_id, task_id, session_id, goal, context_summary, "
                "task_type, recall_mode, retrieval_used, retrieval_reason, "
                "selected_memory_ids_json, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))",
                (f"trace-{session}-{len(ids)}", "t1", session, "g", "", "general", "auto", 1,
                 "test", json.dumps(ids)),
            )

    def test_a_correction_survives_first(self) -> None:
        old = self.memory("the port is 8100")
        from cortex import correction_memory as cm

        cm.record(self.store, supersedes=old, content="the port is 8101")
        described = cs.rebuild(self.store, SESSION)
        self.assertIn("Corrections:", described.text)
        self.assertEqual(described.sections.get("corrections"), 1)

    def test_a_pinned_memory_survives(self) -> None:
        self.memory("answer in metric units", pinned=True)
        self.assertIn("Pinned:", cs.rebuild(self.store, SESSION).text)

    def test_a_used_memory_survives(self) -> None:
        memory_id = self.memory("the zigbee coordinator sits on the second shelf")
        with self.store.transaction() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS used_resolutions (turn_id TEXT, memory_id TEXT, "
                         "resolved_at TEXT, score REAL, PRIMARY KEY(turn_id, memory_id))")
            conn.execute("INSERT INTO used_resolutions VALUES(?,?,datetime('now'),1.0)",
                         (SESSION, memory_id))
        described = cs.rebuild(self.store, SESSION)
        self.assertIn("Used in this session:", described.text)
        self.assertIn(memory_id, described.ids)

    def test_a_retrieved_but_unused_memory_is_lower_priority(self) -> None:
        used = self.memory("the zigbee coordinator sits on the second shelf")
        only_seen = self.memory("the printer queue name is officejet")
        self.seed_trace([used, only_seen])
        with self.store.transaction() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS used_resolutions (turn_id TEXT, memory_id TEXT, "
                         "resolved_at TEXT, score REAL, PRIMARY KEY(turn_id, memory_id))")
            conn.execute("INSERT INTO used_resolutions VALUES(?,?,datetime('now'),1.0)",
                         (SESSION, used))
        described = cs.rebuild(self.store, SESSION)
        # The block carries 8-char tags, not full ids.
        self.assertLess(described.text.index(used[:8]), described.text.index(only_seen[:8]))

    def test_the_post_compaction_budget_is_smaller_than_the_prompt_budget(self) -> None:
        self.assertLess(cs.COMPACTION_BUDGET_CHARS, 1800)

    def test_the_block_fits_the_budget(self) -> None:
        for index in range(10):
            self.memory(f"a pinned instruction number {index} with a good many words in it",
                        pinned=True)
        described = cs.rebuild(self.store, SESSION, budget=300)
        self.assertLessEqual(described.chars, 300)

    def test_drops_are_counted(self) -> None:
        for index in range(10):
            self.memory(f"a pinned instruction number {index} with a good many words in it",
                        pinned=True)
        described = cs.rebuild(self.store, SESSION, budget=300)
        self.assertGreater(described.dropped, 0)

    def test_the_rebuild_is_deterministic(self) -> None:
        self.memory("answer in metric units", pinned=True)
        first = cs.rebuild(self.store, SESSION)
        second = cs.rebuild(self.store, SESSION)
        self.assertEqual(first.digest, second.digest)

    def test_every_line_carries_a_tag(self) -> None:
        memory_id = self.memory("answer in metric units", pinned=True)
        self.assertIn(f"[{memory_id[:8]}]", cs.rebuild(self.store, SESSION).text)

    def test_an_empty_session_rebuilds_to_nothing(self) -> None:
        described = cs.rebuild(self.store, "no-such-session")
        self.assertEqual(described.text, "")
        self.assertIn("nothing to rebuild", "\n".join(cs.render(self.store, "no-such-session")))

    def test_the_verify_round_trip_catches_a_dropped_memory(self) -> None:
        first = self.memory("answer in metric units and never use imperial measurements anywhere",
                            pinned=True)
        second = self.memory("the dashboard listens on port 8100 behind the reverse proxy on r630",
                             pinned=True)
        described = cs.rebuild(self.store, SESSION, budget=120)
        checked = cs.verify([first, second], described.ids)
        self.assertFalse(checked["ok"])
        self.assertTrue(checked["lost"])

    def test_the_render_states_the_budget_and_digest(self) -> None:
        self.memory("answer in metric units", pinned=True)
        rendered = "\n".join(cs.render(self.store, SESSION))
        self.assertIn("compaction rebuild", rendered)
        self.assertIn("digest", rendered)

    def test_a_broken_store_rebuilds_to_nothing(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(cs.rebuild(Broken(), SESSION).text, "")
        self.assertEqual(cs.session_ids(Broken(), SESSION), {"selected": [], "used": []})


if __name__ == "__main__":
    unittest.main()
