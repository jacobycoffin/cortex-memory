"""`cortex-memory status` and `cortex-memory why` (the two questions a person asks).

The point of these commands is that they never lie about what they cannot measure, and never crash on a
store that is empty, missing tables, or not there at all. So the tests are about the honest paths first:
an absent table prints `not measurable yet`, an empty store prints `0 (the store is empty)`, and a wrong
path prints a sentence instead of creating a database that reads as a working one.
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import plain_summary
from cortex.cli import main as cli_main
from cortex.store import CortexStore


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "cortex.db"
        self.store = CortexStore(self.db)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def cli(self, *argv: str) -> tuple[int, str]:
        output = io.StringIO()
        with patch.object(sys, "argv", ["cortex-memory", "--db", str(self.db), *argv]), redirect_stdout(output):
            code = cli_main()
        return code, output.getvalue()

    def memory(self, text: str = "Production deploys need a health check.", *, kind: str = "procedure") -> str:
        return self.store.add_memory(
            content=text,
            kind=kind,
            source_type="user_turn",
            source_category="USER_STATED",
        )[0]

    def graded_answer(self, memory_id: str, outcome: str = "helpful") -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO usage_records(usage_id,task_id,memory_id,selected,used,outcome,created_at) "
                "VALUES(?,?,?,1,1,?,datetime('now'))",
                ("usage-1", "task-1", memory_id, outcome),
            )

    def procedure_outcome(self, memory_id: str) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS procedure_outcomes ("
                "procedure_id TEXT NOT NULL, outcome TEXT NOT NULL, at TEXT NOT NULL, "
                "note TEXT NOT NULL DEFAULT '', PRIMARY KEY(procedure_id, outcome, at))"
            )
            conn.execute(
                "INSERT INTO procedure_outcomes VALUES(?,?,datetime('now'),'')",
                (memory_id, "success"),
            )

    def breakdown(self, memory_id: str, components: dict | None = None) -> None:
        payload = components if components is not None else {"semantic": 0.18, "usage_weight": 0.09, "recency": 0.06}
        with self.store.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS recall_score_breakdown ("
                "task_id TEXT NOT NULL, rank INTEGER NOT NULL, memory_id TEXT NOT NULL, "
                "score REAL NOT NULL, components_json TEXT NOT NULL DEFAULT '{}', "
                "created_at TEXT NOT NULL, PRIMARY KEY(task_id, rank))"
            )
            conn.execute(
                "INSERT INTO recall_score_breakdown VALUES('task-1',2,?,0.41,?,datetime('now'))",
                (memory_id, json.dumps(payload)),
            )


class StatusTests(_Base):
    def test_empty_store_says_not_measurable_instead_of_zero(self) -> None:
        code, out = self.cli("status")
        self.assertEqual(code, 0)
        self.assertIn("0 (the store is empty)", out)
        # Four things that have never been measured are absent evidence, not zero evidence.
        self.assertEqual(out.count("not measurable yet"), 4)
        self.assertNotIn("answers graded: 0", out)
        self.assertNotIn("procedures graded: 0", out)

    def test_status_counts_what_is_stored(self) -> None:
        first = self.memory()
        self.memory("The router lives in the hall cupboard.", kind="semantic")
        self.store.log_access(first, "retrieved", query="deploy")
        self.store.log_access(first, "used", query="deploy")
        self.graded_answer(first)
        self.procedure_outcome(first)
        code, out = self.cli("status")
        self.assertEqual(code, 0)
        self.assertIn("memories: 2 (2 active)", out)
        self.assertIn("recalled in the last 7 days: 1 memory", out)
        self.assertIn("answers graded: 1", out)
        self.assertIn("procedures graded: 1", out)
        self.assertIn("last recall: today", out)

    def test_status_survives_missing_tables(self) -> None:
        self.memory()
        self.store._conn.executescript("DROP TABLE access_log; DROP TABLE usage_records;")
        self.store._conn.commit()
        code, out = self.cli("status")
        self.assertEqual(code, 0, out)
        self.assertIn("memories: 1 (1 active)", out)
        self.assertEqual(out.count("not measurable yet"), 4)

    def test_status_stays_short(self) -> None:
        self.memory()
        _code, out = self.cli("status")
        self.assertLessEqual(len(out.strip().splitlines()), 12)

    def test_status_is_read_only(self) -> None:
        memory_id = self.memory()
        self.store.log_access(memory_id, "retrieved", query="deploy")
        self.store.close()
        before = sqlite3.connect(self.db)
        schema = before.execute("SELECT name FROM sqlite_master ORDER BY name").fetchall()
        counts = before.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        before.close()

        code, _out = self.cli("status")
        self.assertEqual(code, 0)
        self.cli("why", memory_id[:8])

        after = sqlite3.connect(self.db)
        self.assertEqual(after.execute("SELECT name FROM sqlite_master ORDER BY name").fetchall(), schema)
        self.assertEqual(after.execute("SELECT COUNT(*) FROM memories").fetchone()[0], counts)
        after.close()

    def test_status_does_not_create_a_database_at_a_wrong_path(self) -> None:
        missing = Path(self.tmp.name) / "typo.db"
        output = io.StringIO()
        with patch.object(sys, "argv", ["cortex-memory", "--db", str(missing), "status"]), redirect_stdout(output):
            code = cli_main()
        self.assertEqual(code, 0)
        self.assertIn("No Cortex database", output.getvalue())
        self.assertFalse(missing.exists(), "a read-only command must not create the store it cannot find")


class WhyTests(_Base):
    def test_why_explains_one_memory_without_jargon(self) -> None:
        memory_id = self.memory()
        self.store.log_access(memory_id, "retrieved", query="deploy")
        self.graded_answer(memory_id)
        self.breakdown(memory_id)
        code, out = self.cli("why", memory_id[:8])
        self.assertEqual(code, 0, out)
        self.assertIn("a procedure, from a conversation", out)
        self.assertIn("Production deploys need a health check.", out)
        self.assertIn("Last recalled: today", out)
        self.assertIn("Used in an answer: yes, once", out)
        self.assertIn("Why it ranked: #2 (score 0.41)", out)
        # The signal names are translated; the raw component keys are not dumped.
        self.assertIn("wording match", out)
        self.assertIn("past usefulness", out)
        self.assertNotIn("usage_weight", out)
        self.assertNotIn("semantic_fusion", out)

    def test_why_says_when_a_memory_was_never_recalled(self) -> None:
        memory_id = self.memory("A fact nobody has asked about yet.", kind="semantic")
        code, out = self.cli("why", memory_id)
        self.assertEqual(code, 0, out)
        self.assertIn("Last recalled: never, since it was saved", out)
        self.assertIn("Used in an answer: no, never credited", out)
        self.assertIn("Why it ranked: not recorded yet", out)

    def test_why_returns_nonzero_for_an_unknown_memory(self) -> None:
        code, out = self.cli("why", "does-not-exist")
        self.assertEqual(code, 1)
        self.assertIn("No memory matches", out)

    def test_why_survives_missing_recall_and_breakdown_tables(self) -> None:
        memory_id = self.memory()
        self.store._conn.executescript("DROP TABLE access_log;")
        self.store._conn.commit()
        code, out = self.cli("why", memory_id[:8])
        self.assertEqual(code, 0, out)
        self.assertIn("Last recalled: not measurable yet", out)
        self.assertIn("Why it ranked: not recorded yet", out)

    def test_why_survives_a_database_with_no_memories_table(self) -> None:
        other = Path(self.tmp.name) / "elsewhere.db"
        conn = sqlite3.connect(other)
        conn.execute("CREATE TABLE notes (id TEXT)")
        conn.commit()
        conn.close()
        output = io.StringIO()
        with patch.object(sys, "argv", ["cortex-memory", "--db", str(other), "why", "abc"]), redirect_stdout(output):
            code = cli_main()
        self.assertEqual(code, 1)
        self.assertIn("no readable memories table", output.getvalue())


class ModuleTests(_Base):
    def test_plain_summary_opening_a_missing_path_returns_none(self) -> None:
        self.assertIsNone(plain_summary.open_read_only(Path(self.tmp.name) / "absent.db"))

    def test_component_names_are_translated_with_a_readable_fallback(self) -> None:
        self.assertEqual(plain_summary._component_words("semantic_fusion"), "found by meaning")
        self.assertEqual(plain_summary._component_words("some_new_signal"), "some new signal")


if __name__ == "__main__":
    unittest.main()
