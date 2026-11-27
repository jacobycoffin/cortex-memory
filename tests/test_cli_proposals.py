"""`cortex-memory proposals` and the review-queue lines in `cortex-memory status`.

The failure this fixes is not a crash, it is a silence: the nightly dream wrote proposals into
`sleep_proposals` and the only surface that could decide one was the dashboard, so a queue nobody
was ever shown piled up and then `cortex_sleep_retention.py` expired it - 1,829 proposals between
July and September 2026, not one of them decided. So the tests are about visibility first
(the queue is visible, expiry is loud) and about decisions second.
"""

from __future__ import annotations

import io
import json
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
        with patch.object(sys, "argv", ["cortex-memory", "--db", str(self.db), *argv]):
            with redirect_stdout(output):
                try:
                    code = cli_main()
                except SystemExit as exc:  # argparse / explicit exits
                    code = int(exc.code or 0)
        return code, output.getvalue()

    def memory(self, text: str, kind: str = "semantic") -> str:
        memory_id, _ = self.store.add_memory(text, kind=kind, source_category="TOOL_VERIFIED")
        return memory_id

    def proposal(self, kind: str = "association", *, src: str | None = None,
                 dst: str | None = None, status: str = "proposed",
                 created_at: str = "2026-11-27T10:00:00.000+00:00") -> str:
        src = src or self.memory("The deploy target is the workhorse MacBook.")
        dst = dst or self.memory("Test suites run on the MacBook, not the VPS.")
        run_id = "run-" + kind
        with self.store._lock:
            # Fill the run row against its real columns: the schema has grown over time and a
            # hardcoded column list would silently drift out of date.
            columns = self.store._conn.execute("PRAGMA table_info(sleep_runs)").fetchall()
            wanted = {
                "run_id": run_id,
                "mode": "shadow",
                "status": "completed",
                "cutoff_at": created_at,
                "started_at": created_at,
            }
            values: dict[str, object] = {}
            for _cid, name, col_type, not_null, default, _pk in columns:
                if name in wanted:
                    values[name] = wanted[name]
                elif not_null and default is None:
                    values[name] = 0 if "INT" in (col_type or "").upper() or "REAL" in (col_type or "").upper() else ""
            names = ", ".join(values)
            marks = ", ".join("?" for _ in values)
            self.store._conn.execute(
                f"INSERT OR IGNORE INTO sleep_runs({names}) VALUES({marks})", tuple(values.values())
            )
            proposal_id = "prop-" + kind
            self.store._conn.execute(
                """INSERT INTO sleep_proposals(proposal_id,run_id,kind,src_id,dst_id,status,
                       score,evidence_count,rationale,details_json,created_at)
                   VALUES(?,?,?,?,?,?,0.7,2,?,?,?)""",
                (proposal_id, run_id, kind, src, dst, status,
                 "Both memories were used in the same three turns.", "{}", created_at),
            )
            self.store._conn.commit()
        return proposal_id


class ListTests(_Base):
    def test_empty_queue_says_so_plainly(self) -> None:
        code, out = self.cli("proposals")
        self.assertEqual(code, 0)
        self.assertIn("Nothing is waiting for review.", out)

    def test_pending_proposal_is_listed_with_its_reason(self) -> None:
        proposal_id = self.proposal()
        code, out = self.cli("proposals")
        self.assertEqual(code, 0)
        self.assertIn(proposal_id[:8], out)
        self.assertIn("association", out)
        self.assertIn("same three turns", out)

    def test_json_payload_carries_a_schema_version(self) -> None:
        self.proposal()
        code, out = self.cli("proposals", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["proposals"][0]["kind"], "association")

    def test_kind_filter_is_honoured(self) -> None:
        self.proposal(kind="association")
        self.proposal(kind="lifecycle")
        code, out = self.cli("proposals", "--kind", "lifecycle", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["proposals"][0]["kind"], "lifecycle")

    def test_decided_proposals_leave_the_queue(self) -> None:
        proposal_id = self.proposal()
        self.cli("proposals", "skip", proposal_id)
        code, out = self.cli("proposals")
        self.assertEqual(code, 0)
        self.assertIn("Nothing is waiting for review.", out)


class DecisionTests(_Base):
    def status_of(self, proposal_id: str) -> str:
        with self.store._lock:
            return self.store._conn.execute(
                "SELECT status FROM sleep_proposals WHERE proposal_id=?", (proposal_id,)
            ).fetchone()[0]

    def test_approving_a_connection_creates_the_edge(self) -> None:
        proposal_id = self.proposal(kind="association")
        code, out = self.cli("proposals", "approve", proposal_id, "--reason", "clearly related")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.status_of(proposal_id), "operator_approved")
        with self.store._lock:
            edges = self.store._conn.execute(
                "SELECT COUNT(*) FROM edges WHERE relation='operator_link'"
            ).fetchone()[0]
        self.assertEqual(edges, 1)

    def test_skipping_a_connection_denies_it_without_touching_memories(self) -> None:
        proposal_id = self.proposal(kind="association")
        code, out = self.cli("proposals", "skip", proposal_id)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.status_of(proposal_id), "operator_denied")

    def test_a_memory_state_kind_refuses_a_bare_approve(self) -> None:
        """Archiving or trashing a memory is destructive: the operator must name the action."""
        proposal_id = self.proposal(kind="lifecycle")
        code, out = self.cli("proposals", "approve", proposal_id)
        self.assertEqual(code, 2)
        self.assertIn("name the action", out)
        self.assertEqual(self.status_of(proposal_id), "proposed")

    def test_a_memory_state_kind_accepts_an_explicit_action(self) -> None:
        proposal_id = self.proposal(kind="lifecycle")
        code, out = self.cli("proposals", "approve", proposal_id, "--action", "archive")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.status_of(proposal_id), "operator_approved")

    def test_unknown_prefix_is_refused_rather_than_guessed(self) -> None:
        self.proposal()
        code, out = self.cli("proposals", "approve", "zzz")
        self.assertEqual(code, 2)
        self.assertIn("matches 0 pending proposals", out)

    def test_decide_without_an_id_says_what_is_missing(self) -> None:
        code, out = self.cli("proposals", "approve")
        self.assertEqual(code, 2)
        self.assertIn("needs one or more proposal ids", out)


class StatusVisibilityTests(_Base):
    def test_status_reports_the_waiting_queue_and_how_to_see_it(self) -> None:
        self.proposal()
        code, out = self.cli("status")
        self.assertEqual(code, 0)
        self.assertIn("waiting for review:", out)
        self.assertIn("1 proposals", out)
        self.assertIn("cortex-memory proposals", out)

    def test_status_reports_expiry_as_a_warning_not_a_clean_zero(self) -> None:
        self.proposal(status="expired", created_at="2026-11-20T10:00:00.000+00:00")
        lines = plain_summary.status_lines(self.store._conn)
        expired_line = next(line for line in lines if "expired unreviewed" in line)
        self.assertIn("timed out without a decision", expired_line)

    def test_old_expiry_falls_outside_the_window(self) -> None:
        """created_at is ISO text, so a naive `>= datetime('now', ...)` comparison would
        wrongly count rows from any month (an ISO 'T' sorts above a space)."""
        self.proposal(status="expired", created_at="2026-07-13T17:49:15.305+00:00")
        lines = plain_summary.status_lines(self.store._conn)
        expired_line = next(line for line in lines if "expired unreviewed" in line)
        self.assertIn("0 in the last", expired_line)

    def test_missing_table_is_not_measurable_rather_than_zero(self) -> None:
        import sqlite3

        conn = sqlite3.connect(":memory:")
        try:
            lines = plain_summary.status_lines(conn)
        finally:
            conn.close()
        queue_line = next(line for line in lines if "waiting for review" in line)
        self.assertIn("not measurable yet", queue_line)


if __name__ == "__main__":
    unittest.main()
