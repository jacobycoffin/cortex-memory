"""Link a memory to the tool call it informed (#70)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import tool_links
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str = "The dashboard listens on port 8100.") -> str:
        return self.store.add_memory(
            content=content, kind="procedure", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def execution(self, tool: str = "ssh", *, success: int = 1, execution_id: str = "") -> str:
        identifier = execution_id or f"exec-{tool}-{success}-{len(tool_links.report(self.store)['by_tool'])}"
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO tool_executions
                   (execution_id, session_id, task_type, tool_name, argument_keys, success,
                    created_at)
                   VALUES (?, 's1', 'general', ?, '[]', ?, ?)""",
                (identifier, tool, success, NOW.isoformat()),
            )
        return identifier

    def test_a_link_is_recorded(self) -> None:
        memory_id = self.memory()
        execution_id = self.execution()
        made = tool_links.link(self.store, memory_id, execution_id, at=NOW)
        self.assertTrue(made["linked"])
        self.assertEqual(len(tool_links.for_memory(self.store, memory_id)), 1)

    def test_the_tool_name_is_read_from_the_execution(self) -> None:
        memory_id = self.memory()
        execution_id = self.execution("proxmox")
        tool_links.link(self.store, memory_id, execution_id, at=NOW)
        self.assertEqual(tool_links.for_memory(self.store, memory_id)[0].tool_name, "proxmox")

    def test_an_unknown_memory_is_refused(self) -> None:
        execution_id = self.execution()
        made = tool_links.link(self.store, "no-such-memory", execution_id)
        self.assertFalse(made["linked"])
        self.assertIn("unknown memory", made["reason"])

    def test_an_unknown_execution_is_refused(self) -> None:
        memory_id = self.memory()
        made = tool_links.link(self.store, memory_id, "no-such-execution")
        self.assertFalse(made["linked"])
        self.assertIn("unknown execution", made["reason"])

    def test_an_empty_side_is_refused(self) -> None:
        self.assertFalse(tool_links.link(self.store, "", "x")["linked"])

    def test_relinking_the_same_pair_is_a_no_op(self) -> None:
        memory_id = self.memory()
        execution_id = self.execution()
        first = tool_links.link(self.store, memory_id, execution_id, at=NOW)
        second = tool_links.link(self.store, memory_id, execution_id, at=NOW)
        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(len(tool_links.for_memory(self.store, memory_id)), 1)

    def test_one_memory_can_inform_many_calls(self) -> None:
        memory_id = self.memory()
        for index in range(3):
            tool_links.link(self.store, memory_id, self.execution(execution_id=f"e{index}"), at=NOW)
        self.assertEqual(len(tool_links.for_memory(self.store, memory_id)), 3)

    def test_link_many_links_a_whole_turn(self) -> None:
        first = self.memory("The dashboard listens on port 8100.")
        second = self.memory("Nginx proxies the dashboard.")
        execution_id = self.execution()
        made = tool_links.link_many(self.store, [first, second, "missing"], execution_id, at=NOW)
        self.assertEqual(made["linked"], 2)
        self.assertEqual(len(made["refused"]), 1)

    def test_for_memory_reports_the_outcome(self) -> None:
        memory_id = self.memory()
        tool_links.link(self.store, memory_id, self.execution(success=0), at=NOW)
        self.assertFalse(tool_links.for_memory(self.store, memory_id)[0].success)

    def test_for_tool_lists_the_memories_it_used(self) -> None:
        memory_id = self.memory()
        tool_links.link(self.store, memory_id, self.execution("ssh"), at=NOW)
        self.assertIn(memory_id, tool_links.for_tool(self.store, "ssh"))

    def test_by_tool_reports_the_success_share(self) -> None:
        first = self.memory("The dashboard listens on port 8100.")
        second = self.memory("Nginx proxies the dashboard.")
        tool_links.link(self.store, first, self.execution("ssh", success=1,
                                                          execution_id="a1"), at=NOW)
        tool_links.link(self.store, second, self.execution("ssh", success=0,
                                                           execution_id="a2"), at=NOW)
        detail = tool_links.by_tool(self.store)["ssh"]
        self.assertEqual(detail["calls"], 2)
        self.assertAlmostEqual(detail["success_share"], 0.5, places=4)

    def test_by_tool_counts_a_call_once_even_with_two_memories(self) -> None:
        first = self.memory("The dashboard listens on port 8100.")
        second = self.memory("Nginx proxies the dashboard.")
        execution_id = self.execution("ssh")
        tool_links.link(self.store, first, execution_id, at=NOW)
        tool_links.link(self.store, second, execution_id, at=NOW)
        detail = tool_links.by_tool(self.store)["ssh"]
        self.assertEqual(detail["links"], 2)
        self.assertEqual(detail["calls"], 1)

    def test_a_use_describes_itself(self) -> None:
        memory_id = self.memory()
        tool_links.link(self.store, memory_id, self.execution("ssh", success=0), at=NOW)
        self.assertIn("failed", tool_links.for_memory(self.store, memory_id)[0].describe())

    def test_the_report_counts_links_and_tools(self) -> None:
        first = self.memory("The dashboard listens on port 8100.")
        second = self.memory("Nginx proxies the dashboard.")
        tool_links.link(self.store, first, self.execution("ssh", execution_id="r1"), at=NOW)
        tool_links.link(self.store, second, self.execution("docker", execution_id="r2"), at=NOW)
        described = tool_links.report(self.store)
        self.assertEqual(described["links"], 2)
        self.assertEqual(described["memories"], 2)
        self.assertEqual(described["tools"], 2)

    def test_the_round_trip(self) -> None:
        memory_id = self.memory()
        tool_links.link(self.store, memory_id, self.execution(), at=NOW)
        payload = tool_links.for_memory(self.store, memory_id)[0].as_dict()
        for key in ("memory_id", "execution_id", "tool_name", "success", "created_at"):
            self.assertIn(key, payload)

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(tool_links.for_memory(self.store, "nothing"), [])
        self.assertEqual(tool_links.by_tool(self.store), {})
        self.assertEqual(tool_links.report(self.store)["links"], 0)


if __name__ == "__main__":
    unittest.main()
