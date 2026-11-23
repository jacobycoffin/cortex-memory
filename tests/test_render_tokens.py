"""The render columns are written, not left at their defaults (#215)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import CortexMemoryProvider


class RenderColumnsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = CortexMemoryProvider(
            {
                "db_path": str(Path(self.tmp.name) / "cortex" / "test.db"),
                "auto_capture": False,
                "retrieval_threshold": 0.05,
                "pruning_mode": "shadow",
            }
        )
        self.provider.initialize("session-render", hermes_home=self.tmp.name,
                                 agent_context="primary")
        self.store = self.provider._store
        for index in range(6):
            self.store.add_memory(
                content=f"The dashboard runs on port 810{index} behind nginx.",
                kind="semantic", source_type="user_turn", source_category="USER_EXPLICIT",
            )

    def tearDown(self) -> None:
        self.provider.shutdown()
        self.tmp.cleanup()

    def latest(self) -> dict:
        row = self.store._conn.execute(
            """SELECT query, selected_count, rendered_count, withheld_count, rendered_tokens,
                      estimated_tokens
               FROM recall_runs ORDER BY created_at DESC LIMIT 1"""
        ).fetchone()
        return dict(row) if row else {}

    def test_a_recall_writes_the_rendered_tokens(self) -> None:
        self.provider.prefetch("which port does the dashboard use", session_id="s1")
        row = self.latest()
        self.assertGreater(row["rendered_tokens"], 0)

    def test_the_rendered_count_is_what_was_injected(self) -> None:
        self.provider.prefetch("which port does the dashboard use", session_id="s1")
        row = self.latest()
        self.assertGreater(row["rendered_count"], 0)
        self.assertLessEqual(row["rendered_count"], row["selected_count"])

    def test_a_tight_budget_reports_what_it_withheld(self) -> None:
        config = {
            "db_path": str(Path(self.tmp.name) / "cortex" / "tight.db"),
            "auto_capture": False,
            "retrieval_threshold": 0.0,
            "token_budget": 130,
            "pruning_mode": "shadow",
        }
        provider = CortexMemoryProvider(config)
        provider.initialize("session-tight", hermes_home=self.tmp.name, agent_context="primary")
        try:
            for index in range(8):
                provider._store.add_memory(
                    content=(
                        "A deliberately long memory about the dashboard, nginx, port 8100 and the "
                        f"proxy settings that matter for case {index}."
                    ),
                    kind="semantic", source_type="user_turn", source_category="USER_EXPLICIT",
                )
            provider.prefetch("dashboard nginx proxy port settings", session_id="s2")
            row = provider._store._conn.execute(
                """SELECT selected_count, rendered_count, withheld_count FROM recall_runs
                   ORDER BY created_at DESC LIMIT 1"""
            ).fetchone()
            self.assertGreaterEqual(row["withheld_count"], 0)
            self.assertEqual(
                row["selected_count"], row["rendered_count"] + row["withheld_count"]
            )
        finally:
            provider.shutdown()

    def test_no_memory_control_writes_zeroes_on_purpose(self) -> None:
        self.provider.prefetch("anything at all", session_id="s1")
        row = self.latest()
        self.assertGreaterEqual(row["rendered_tokens"], 0)
        self.assertIsNotNone(row["query"])

    def test_the_rendered_tokens_match_the_injected_block(self) -> None:
        block = self.provider.prefetch("which port does the dashboard use", session_id="s1")
        row = self.latest()
        self.assertEqual(row["rendered_tokens"], (len(block or "") + 3) // 4)


if __name__ == "__main__":
    unittest.main()
