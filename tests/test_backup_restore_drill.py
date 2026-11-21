"""The restore drill has to prove a backup is restorable, not merely assert it.

A backup that has never been restored is a hope. These tests pin the drill's
contract: an intact database round-trips to exit 0 with matching counts, a
corrupted copy fails with exit 1, unreadable input exits 2, and the source file
is never touched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DRILL = _REPO_ROOT / "scripts" / "backup_restore_drill.py"


def _load_drill():
    """Load scripts/backup_restore_drill.py without importing the Cortex package."""

    spec = importlib.util.spec_from_file_location("backup_restore_drill_under_test", _DRILL)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"could not load {_DRILL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


drill = _load_drill()


def _build_database(path: Path, memories: int = 400, edges: int = 300) -> None:
    """A small Cortex-shaped database: memories with a text id, edges without one."""

    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT NOT NULL, tags TEXT)")
        conn.execute(
            "CREATE TABLE edges (src_id TEXT NOT NULL, dst_id TEXT NOT NULL, relation TEXT NOT NULL, "
            "weight REAL NOT NULL DEFAULT 0.25, PRIMARY KEY (src_id, dst_id, relation))"
        )
        conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, summary TEXT)")
        for index in range(1, memories + 1):
            conn.execute(
                "INSERT INTO memories (id, content, tags) VALUES (?, ?, ?)",
                (f"mem-{index:04d}", f"memory body {index} " + "x" * 120, "alpha,beta"),
            )
        for index in range(1, edges + 1):
            conn.execute(
                "INSERT INTO edges (src_id, dst_id, relation, weight) VALUES (?, ?, ?, ?)",
                (f"mem-{index:04d}", f"mem-{index + 1:04d}", "relates_to", 0.5),
            )
        for index in range(1, 21):
            conn.execute("INSERT INTO episodes (id, summary) VALUES (?, ?)", (index, f"episode {index}"))
        conn.commit()
    finally:
        conn.close()


def _corrupt_table_root_page(path: Path, table: str) -> None:
    """Zero the b-tree root page of ``table``: valid header, malformed body."""

    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT rootpage FROM sqlite_master WHERE name = ?", (table,)).fetchone()
        if row is None:
            raise AssertionError(f"table {table!r} not found")
        root_page = int(row[0])
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
    finally:
        conn.close()
    data = bytearray(path.read_bytes())
    start = (root_page - 1) * page_size
    data[start:start + page_size] = b"\x00" * page_size
    path.write_bytes(data)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DrillCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.db = self.root / "source.db"
        _build_database(self.db)
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(_DRILL), *args],
            capture_output=True,
            text=True,
            timeout=180,
        )

    def test_intact_database_passes_and_cleans_up(self) -> None:
        before = _sha256(self.db)
        result = self._run(str(self.db), "--scratch", str(self.scratch))
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("RESULT: PASS", result.stdout)
        self.assertIn("integrity_check(restore): ok", result.stdout)
        for table in ("memories", "edges", "episodes"):
            self.assertRegex(result.stdout, rf"(?m)^\s*{table}\s+\d[\d,]*\s+\d[\d,]*\s+match$")
        self.assertNotIn("MISMATCH", result.stdout)
        self.assertEqual(_sha256(self.db), before, "the drill must not modify the source")
        self.assertEqual(list(self.scratch.glob("cortex-restore-drill-*")), [], "--keep was not passed")

    def test_corrupted_copy_exits_1(self) -> None:
        copy = self.root / "corrupted.db"
        copy.write_bytes(self.db.read_bytes())
        _corrupt_table_root_page(copy, "edges")

        result = self._run(str(copy), "--scratch", str(self.scratch))
        self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)
        self.assertIn("RESULT: FAIL", result.stdout)

    def test_missing_source_exits_2(self) -> None:
        result = self._run(str(self.root / "absent.db"), "--scratch", str(self.scratch))
        self.assertEqual(result.returncode, 2, msg=result.stdout + result.stderr)
        self.assertIn("does not exist", result.stderr)

    def test_non_sqlite_file_exits_2(self) -> None:
        junk = self.root / "notes.txt"
        junk.write_text("this is not a database\n")
        result = self._run(str(junk), "--scratch", str(self.scratch))
        self.assertEqual(result.returncode, 2, msg=result.stdout + result.stderr)
        self.assertIn("not a SQLite database", result.stderr)

    def test_keep_retains_a_readable_restore(self) -> None:
        result = self._run(str(self.db), "--scratch", str(self.scratch), "--keep")
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        kept = [path for path in self.scratch.glob("cortex-restore-drill-*") if path.is_dir()]
        self.assertEqual(len(kept), 1, msg=result.stdout)
        restored = kept[0] / "restore" / "cortex.db"
        self.assertTrue(restored.is_file())
        conn = sqlite3.connect(f"file:{restored.as_posix()}?mode=ro", uri=True)
        try:
            self.assertEqual(conn.execute("SELECT count(*) FROM memories").fetchone()[0], 400)
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        finally:
            conn.close()


class SampleDigestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _database(self, name: str, mutate: bool = False) -> Path:
        path = self.root / name
        _build_database(path, memories=30, edges=20)
        if mutate:
            conn = sqlite3.connect(path)
            try:
                conn.execute("UPDATE memories SET content = 'rewritten' WHERE id = 'mem-0007'")
                conn.commit()
            finally:
                conn.close()
        return path

    def test_identical_copies_hash_identically(self) -> None:
        first = sqlite3.connect(self._database("a.db"))
        second = sqlite3.connect(self._database("b.db"))
        try:
            self.assertEqual(
                drill.sample_digest(first, "memories", 200),
                drill.sample_digest(second, "memories", 200),
            )
        finally:
            first.close()
            second.close()

    def test_same_row_count_different_content_hashes_differently(self) -> None:
        good = sqlite3.connect(self._database("good.db"))
        altered = sqlite3.connect(self._database("altered.db", mutate=True))
        try:
            rows, good_digest = drill.sample_digest(good, "memories", 200)
            altered_rows, altered_digest = drill.sample_digest(altered, "memories", 200)
        finally:
            good.close()
            altered.close()
        self.assertEqual(rows, altered_rows)
        self.assertNotEqual(good_digest, altered_digest)

    def test_edges_without_an_id_column_is_sampled_in_a_stable_order(self) -> None:
        conn = sqlite3.connect(self._database("edges.db"))
        try:
            columns = [row[1] for row in conn.execute("PRAGMA table_info(edges)")]
            self.assertNotIn("id", columns)
            self.assertEqual(drill.order_by_clause(conn, "edges"), '"src_id", "dst_id", "relation"')
            self.assertEqual(drill.sample_digest(conn, "edges", 200), drill.sample_digest(conn, "edges", 200))
            self.assertEqual(drill.sample_digest(conn, "edges", 200)[0], 20)
        finally:
            conn.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
