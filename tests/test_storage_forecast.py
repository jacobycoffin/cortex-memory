"""Storage growth forecast: the projection arithmetic must be checkable.

``scripts/storage_forecast.py`` answers "when does the disk fill?" from a linear
rate. A plausible-looking forecast that silently multiplies the wrong numbers is
worse than no forecast, so these tests pin the rate window and the projection
arithmetic against a synthetic database with a known number of rows over a known
window.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    """Load scripts/storage_forecast.py without importing the Cortex package."""
    path = _REPO_ROOT / "scripts" / "storage_forecast.py"
    spec = importlib.util.spec_from_file_location("storage_forecast_under_test", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


forecast = _load_module()

TODAY = datetime.now(timezone.utc).date()
MIB = 1024 * 1024


def _make_db(path: Path, rows_per_day: int = 10, days: int = 30) -> None:
    """Four tracked tables; ``memories`` grows at exactly ``rows_per_day``."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT, created_at TEXT NOT NULL);
        CREATE TABLE edges (id INTEGER PRIMARY KEY, src TEXT, created_at TEXT NOT NULL);
        CREATE TABLE access_log (id INTEGER PRIMARY KEY, memory_id TEXT, created_at TEXT NOT NULL);
        CREATE TABLE episodes (id INTEGER PRIMARY KEY, summary TEXT, created_at TEXT NOT NULL);
        CREATE INDEX idx_memories_created ON memories(created_at);
        """
    )
    content = "x" * 200
    for day_offset in range(days):
        stamp_day = TODAY - timedelta(days=days - 1 - day_offset)
        for _ in range(rows_per_day):
            conn.execute(
                "INSERT INTO memories (content, created_at) VALUES (?, ?)",
                (content, f"{stamp_day.isoformat()}T12:00:00.000+00:00"),
            )
    # A single edge/episode/access row: present but not growing materially.
    stale = f"{TODAY.isoformat()}T00:00:00.000+00:00"
    conn.execute("INSERT INTO edges (src, created_at) VALUES ('a', ?)", (stale,))
    conn.execute("INSERT INTO access_log (memory_id, created_at) VALUES ('a', ?)", (stale,))
    conn.execute("INSERT INTO episodes (summary, created_at) VALUES ('s', ?)", (stale,))
    conn.commit()
    conn.close()


class ProjectionArithmeticTests(unittest.TestCase):
    def test_linear_projection_is_exact(self) -> None:
        current = 100 * MIB
        per_day = 1 * MIB
        self.assertEqual(forecast.project_bytes(current, per_day, 30), 130 * MIB)
        self.assertEqual(forecast.project_bytes(current, per_day, 90), 190 * MIB)
        self.assertEqual(forecast.project_bytes(current, per_day, 365), 465 * MIB)

    def test_zero_growth_projects_flat(self) -> None:
        self.assertEqual(forecast.project_bytes(50 * MIB, 0.0, 365), 50 * MIB)


class InsertRateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "cortex.db"
        _make_db(self.db)
        self.conn = sqlite3.connect(self.db)

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def test_rate_uses_created_at_window(self) -> None:
        rate = forecast.insert_rate(self.conn, "memories", TODAY, window_days=30)
        self.assertEqual(rate["rows_in_window"], 300)
        self.assertEqual(rate["days"], 30.0)
        self.assertEqual(rate["rows_per_day"], 10.0)

    def test_young_table_is_not_diluted_by_the_full_window(self) -> None:
        # 5 rows, all created today: the rate is 5/day, not 5/30.
        stamp = f"{TODAY.isoformat()}T01:00:00.000+00:00"
        for _ in range(5):
            self.conn.execute(
                "INSERT INTO edges (src, created_at) VALUES ('a', ?)", (stamp,)
            )
        self.conn.commit()
        rate = forecast.insert_rate(self.conn, "edges", TODAY, window_days=30)
        self.assertEqual(rate["rows_in_window"], 6)  # 5 new + the stale row from setup
        self.assertEqual(rate["days"], 1.0)
        self.assertEqual(rate["rows_per_day"], 6.0)

    def test_older_rows_fall_outside_the_window(self) -> None:
        old = f"{(TODAY - timedelta(days=90)).isoformat()}T01:00:00.000+00:00"
        self.conn.execute(
            "INSERT INTO episodes (summary, created_at) VALUES ('old', ?)", (old,)
        )
        self.conn.commit()
        rate = forecast.insert_rate(self.conn, "episodes", TODAY, window_days=30)
        self.assertEqual(rate["rows_in_window"], 1)  # the stale row, not the old one

    def test_missing_table_reports_no_growth(self) -> None:
        rate = forecast.insert_rate(self.conn, "not_a_table", TODAY, window_days=30)
        self.assertEqual(rate["rows_in_window"], 0)
        self.assertEqual(rate["rows_per_day"], 0.0)


class FullReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "cortex.db"
        _make_db(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _report(self, method: str = "auto") -> dict:
        return forecast.build_report(self.db, method=method, window_days=30, now=None)

    def test_projection_follows_rate_times_bytes_per_row(self) -> None:
        report = self._report()
        memory = next(g for g in report["growth"] if g["table"] == "memories")
        self.assertEqual(memory["rows_per_day"], 10.0)
        self.assertGreater(memory["bytes_per_row"], 0)

        expected_per_day = sum(g["rows_per_day"] * g["bytes_per_row"] for g in report["growth"])
        self.assertAlmostEqual(report["bytes_per_day"], expected_per_day, places=6)

        for horizon in forecast.HORIZONS:
            projected = next(p for p in report["projections"] if p["horizon_days"] == horizon)
            expected = round(report["file_bytes"] + report["bytes_per_day"] * horizon)
            self.assertEqual(projected["bytes"], expected, f"{horizon}-day projection")

    def test_known_rows_add_a_known_amount(self) -> None:
        """300 memories in 30 days is 10/day; every horizon delta is rate x days."""
        report = self._report()
        memory = next(g for g in report["growth"] if g["table"] == "memories")
        per_day = report["bytes_per_day"]
        thirty = next(p for p in report["projections"] if p["horizon_days"] == 30)
        ninety = next(p for p in report["projections"] if p["horizon_days"] == 90)
        year = next(p for p in report["projections"] if p["horizon_days"] == 365)
        self.assertAlmostEqual(memory["rows_per_day"] * memory["bytes_per_row"],
                               memory["bytes_per_day"], places=6)
        self.assertAlmostEqual(ninety["bytes"] - thirty["bytes"], per_day * 60, delta=2)
        self.assertAlmostEqual(year["bytes"] - thirty["bytes"], per_day * 335, delta=2)
        self.assertGreaterEqual(report["file_bytes"], len(report["tables"]))

    def test_sizes_are_attributed(self) -> None:
        report = self._report()
        attributed = sum(t["bytes"] for t in report["tables"])
        self.assertLessEqual(attributed, report["file_bytes"])
        self.assertGreaterEqual(report["unattributed_bytes"], 0)
        memories = next(t for t in report["tables"] if t["name"] == "memories")
        self.assertEqual(memories["rows"], 300)
        self.assertGreater(memories["bytes"], 0)
        self.assertIn(report["size_method"], {"dbstat", "page_count"})

    def test_page_fallback_is_labelled(self) -> None:
        report = self._report(method="pages")
        self.assertEqual(report["size_method"], "page_count")
        self.assertIn("--size-method pages", report["size_method_note"])
        self.assertGreater(sum(t["bytes"] for t in report["tables"]), 0)


class CommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "cortex.db"
        _make_db(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, argv: list[str]) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = forecast.main(argv)
        return code, out.getvalue()

    def test_json_output_is_machine_readable(self) -> None:
        code, out = self._run(["--db", str(self.db), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["db"], str(self.db))
        self.assertEqual({p["horizon_days"] for p in payload["projections"]}, {30, 90, 365})
        self.assertIn("assumption", payload)
        self.assertIsNone(payload["sweep_trigger_mb"])

    def test_sweep_trigger_exits_one_when_crossed(self) -> None:
        report = forecast.build_report(self.db)
        ninety_mb = next(
            p for p in report["projections"] if p["horizon_days"] == 90
        )["mb"]
        code, _ = self._run(["--db", str(self.db), "--sweep-trigger-mb", str(ninety_mb - 1)])
        self.assertEqual(code, 1)
        code, _ = self._run(["--db", str(self.db), "--sweep-trigger-mb", str(ninety_mb + 1)])
        self.assertEqual(code, 0)

    def test_missing_database_exits_two(self) -> None:
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = forecast.main(["--db", str(Path(self._tmp.name) / "nope.db")])
        self.assertEqual(code, 2)
        self.assertIn("no such database", err.getvalue())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
