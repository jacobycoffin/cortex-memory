"""Tests for scripts/retention_check.py — the policy must be executable, not aspirational."""

from __future__ import annotations

import importlib.util
import sqlite3
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "retention_check.py"


def load_module():
    spec = importlib.util.spec_from_file_location("retention_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCHEMA = """
CREATE TABLE memories (id TEXT PRIMARY KEY, kind TEXT, created_at TEXT, state TEXT,
                       pinned INTEGER DEFAULT 0);
CREATE TABLE access_log (id INTEGER PRIMARY KEY, memory_id TEXT, event TEXT, created_at TEXT);
CREATE TABLE edges (src_id TEXT, dst_id TEXT, relation TEXT, evidence_count INTEGER DEFAULT 1);
CREATE TABLE prospective_items (memory_id TEXT, status TEXT);
"""


def days_ago(n: int) -> str:
    return f"datetime('now', '-{n} day')"


class RetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(self.id().replace(".", "_") + ".db")
        self.tmp.unlink(missing_ok=True)
        conn = sqlite3.connect(self.tmp)
        conn.executescript(SCHEMA)
        self.conn = conn

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.unlink(missing_ok=True)

    def add(self, mid: str, kind: str, age: int, pinned: int = 0, used_days_ago: int | None = None):
        self.conn.execute(
            f"INSERT INTO memories VALUES (?, ?, {days_ago(age)}, 'active', ?)",
            (mid, kind, pinned),
        )
        if used_days_ago is not None:
            self.conn.execute(
                f"INSERT INTO access_log (memory_id, event, created_at) VALUES (?, 'used', {days_ago(used_days_ago)})",
                (mid,),
            )

    def test_windows_by_kind(self) -> None:
        module = load_module()
        self.add("ep-old", "episode", 120)          # archive (>90)
        self.add("ep-mid", "episode", 45)           # cool (>30)
        self.add("ep-new", "episode", 5)            # retained
        self.add("sem-old", "semantic", 200)        # archive (>180)
        self.add("pref-old", "preference", 800)     # never auto-archives
        self.conn.commit()
        report = module.classify(self.conn)
        self.assertEqual(report["would_archive"], 2)
        # ep-mid cools at 45d; the 800-day preference cools at its 365-day threshold but
        # never archives, which is the whole point of the per-kind window
        self.assertEqual(report["would_cool"], 2)
        self.assertEqual(report["retained"], 1)

    def test_use_resets_the_clock(self) -> None:
        module = load_module()
        self.add("ep", "episode", 200, used_days_ago=3)
        self.conn.commit()
        report = module.classify(self.conn)
        self.assertEqual(report["would_archive"], 0)
        self.assertEqual(report["retained"], 1)

    def test_protections(self) -> None:
        module = load_module()
        self.add("pinned", "episode", 900, pinned=1)
        self.add("feedback", "episode", 900)
        self.conn.execute(
            "INSERT INTO access_log (memory_id, event, created_at) VALUES ('feedback', 'wrong', datetime('now'))"
        )
        self.add("corroborated", "episode", 900)
        self.conn.execute(
            "INSERT INTO edges VALUES ('corroborated', 'other', 'related', 2)"
        )
        self.add("bare-edge", "episode", 900)
        self.conn.execute("INSERT INTO edges VALUES ('bare-edge', 'other', 'related', 1)")
        self.conn.commit()
        report = module.classify(self.conn)
        # three are protected; the bare edge is not evidence, so it still archives
        self.assertEqual(report["would_archive"], 1)
        self.assertEqual(report["protected"]["pinned"], 1)
        self.assertEqual(report["protected"]["has feedback"], 1)
        self.assertEqual(report["protected"]["corroborated edge"], 1)

    def test_a_bare_edge_does_not_protect_the_store(self) -> None:
        """3,704 of 3,818 live memories sit on an evidence-free edge; they must not all be safe."""
        module = load_module()
        for i in range(10):
            self.add(f"m{i}", "episode", 400)
            self.conn.execute(f"INSERT INTO edges VALUES ('m{i}', 'x', 'related', 1)")
        self.conn.commit()
        report = module.classify(self.conn)
        self.assertEqual(report["would_archive"], 10)


if __name__ == "__main__":
    unittest.main()
