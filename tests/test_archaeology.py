"""What a memory actually taught the agent (#81)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import archaeology, outcomes, tool_links
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

    def event(self, memory_id: str, event: str, *, at: datetime = NOW,
              query: str = "how do I deploy", session: str = "s1") -> None:
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO access_log (memory_id, event, query, session_id, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (memory_id, event, query, session, at.isoformat()),
            )

    def execution(self, tool: str = "ssh", *, execution_id: str = "e1") -> str:
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO tool_executions
                   (execution_id, session_id, task_type, tool_name, argument_keys, success,
                    created_at)
                   VALUES (?, 's1', 'general', ?, '[]', 1, ?)""",
                (execution_id, tool, NOW.isoformat()),
            )
        return execution_id

    def test_injected_but_never_used_taught_nothing(self) -> None:
        memory_id = self.memory()
        for _ in range(4):
            self.event(memory_id, "injected")
        found = archaeology.effects(self.store, memory_id, at=NOW)
        self.assertEqual(found.injected, 4)
        self.assertEqual(found.used, 0)
        self.assertFalse(found.taught_something)
        self.assertIn("taught nothing", found.describe())

    def test_a_used_memory_taught_something(self) -> None:
        memory_id = self.memory()
        self.event(memory_id, "injected")
        self.event(memory_id, "used", query="deploy the dashboard")
        found = archaeology.effects(self.store, memory_id, at=NOW)
        self.assertTrue(found.taught_something)
        self.assertIn("deploy the dashboard", found.queries)

    def test_sessions_are_counted(self) -> None:
        memory_id = self.memory()
        self.event(memory_id, "used", session="a")
        self.event(memory_id, "used", session="b")
        self.assertEqual(archaeology.effects(self.store, memory_id, at=NOW).sessions, 2)

    def test_events_outside_the_window_are_excluded(self) -> None:
        memory_id = self.memory()
        self.event(memory_id, "used", at=NOW - timedelta(days=200))
        self.assertEqual(archaeology.effects(self.store, memory_id, days=90, at=NOW).used, 0)

    def test_the_window_is_part_of_the_answer(self) -> None:
        memory_id = self.memory()
        self.event(memory_id, "used", at=NOW - timedelta(days=200))
        found = archaeology.effects(self.store, memory_id, days=365, at=NOW)
        self.assertEqual(found.used, 1)
        self.assertIn("365 days", found.describe())

    def test_a_tool_call_it_informed_is_an_effect(self) -> None:
        memory_id = self.memory()
        tool_links.link(self.store, memory_id, self.execution(), at=NOW)
        found = archaeology.effects(self.store, memory_id, at=NOW)
        self.assertEqual(found.tool_calls_informed, 1)
        self.assertTrue(found.taught_something)

    def test_task_outcomes_are_reported(self) -> None:
        memory_id = self.memory()
        outcomes.record(self.store, memory_id, "t1", "success", at=NOW)
        outcomes.record(self.store, memory_id, "t2", "failure", at=NOW)
        found = archaeology.effects(self.store, memory_id, at=NOW)
        self.assertEqual((found.tasks_succeeded, found.tasks_failed), (1, 1))

    def test_a_correction_that_replaced_it_is_an_effect(self) -> None:
        from cortex import self_correction

        memory_id = self.memory("The dashboard listens on port 8100.")
        self_correction.intercept(self.store, "I meant 8200.", target_id=memory_id, at=NOW)
        found = archaeology.effects(self.store, memory_id, at=NOW)
        self.assertEqual(found.corrections_replacing_it, 1)
        self.assertIn("corrected", found.describe())

    def test_associations_count_as_an_effect(self) -> None:
        first = self.memory("The dashboard listens on port 8100.")
        second = self.memory("Nginx proxies the dashboard.")
        self.store.add_edge(first, second, "co_observed", weight=0.4,
                            evidence_type="co_retrieval", evidence_key="k",
                            explanation="seen together")
        self.assertGreaterEqual(archaeology.effects(self.store, first, at=NOW).associations_formed,
                                1)

    def test_the_digest_carries_the_memory_text(self) -> None:
        memory_id = self.memory("The dashboard listens on port 8100.")
        payload = archaeology.digest(self.store, memory_id, at=NOW)
        self.assertIn("port 8100", payload["content"])
        self.assertEqual(payload["kind"], "procedure")

    def test_the_influential_list_is_ranked_by_use(self) -> None:
        quiet = self.memory("A fact nobody needed.")
        used = self.memory("A fact that mattered.")
        self.event(quiet, "injected")
        for _ in range(3):
            self.event(used, "used")
        ranked = archaeology.most_influential(self.store, at=NOW)
        self.assertEqual(ranked[0]["memory_id"], used)

    def test_a_tool_call_outweighs_a_bare_use(self) -> None:
        used = self.memory("A fact that was read.")
        informed = self.memory("A fact that changed a call.")
        self.event(used, "used")
        self.event(informed, "used")
        tool_links.link(self.store, informed, self.execution(execution_id="x1"), at=NOW)
        ranked = archaeology.most_influential(self.store, at=NOW)
        self.assertEqual(ranked[0]["memory_id"], informed)

    def test_the_useless_list_is_the_counterweight(self) -> None:
        ignored = self.memory("Offered many times and never used.")
        for _ in range(6):
            self.event(ignored, "injected")
        listed = archaeology.silently_useless(self.store, at=NOW)
        self.assertEqual([item["memory_id"] for item in listed], [ignored])

    def test_a_memory_used_once_is_not_listed_as_useless(self) -> None:
        memory_id = self.memory("Used once.")
        for _ in range(6):
            self.event(memory_id, "injected")
        self.event(memory_id, "used")
        self.assertEqual(archaeology.silently_useless(self.store, at=NOW), [])

    def test_the_report_counts_offered_against_used(self) -> None:
        memory_id = self.memory()
        self.event(memory_id, "injected")
        self.event(memory_id, "used")
        described = archaeology.report(self.store, at=NOW)
        self.assertEqual(described["injected"], 1)
        self.assertEqual(described["used"], 1)
        self.assertAlmostEqual(described["use_share"], 1.0, places=4)

    def test_the_round_trip(self) -> None:
        memory_id = self.memory()
        self.event(memory_id, "used")
        payload = archaeology.effects(self.store, memory_id, at=NOW).as_dict()
        for key in ("memory_id", "days", "used", "injected_but_unused", "taught_something",
                    "sentence"):
            self.assertIn(key, payload)

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertFalse(archaeology.effects(self.store, "nothing", at=NOW).taught_something)
        self.assertEqual(archaeology.most_influential(self.store, at=NOW), [])
        self.assertEqual(archaeology.silently_useless(self.store, at=NOW), [])
        self.assertEqual(archaeology.report(self.store, at=NOW)["memories"], 0)


if __name__ == "__main__":
    unittest.main()
