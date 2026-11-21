"""Readiness checks for encryption at rest (backlog item #194)."""

from __future__ import annotations

import importlib.util
import os
import stat
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "encryption_readiness.py"


def load_module():
    spec = importlib.util.spec_from_file_location("encryption_readiness", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DatabaseStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self) -> None:
        self.dir.cleanup()

    def test_plaintext_detected_from_header(self) -> None:
        db = self.root / "plain.db"
        db.write_bytes(b"SQLite format 3\x00" + b"\x00" * 64)
        state, _ = self.module.database_state(db)
        self.assertEqual(state, "plaintext")

    def test_encrypted_heuristic(self) -> None:
        db = self.root / "enc.db"
        db.write_bytes(b"\x00" * 128)
        state, note = self.module.database_state(db)
        self.assertEqual(state, "encrypted")
        self.assertIn("likely encrypted", note)

    def test_missing_file(self) -> None:
        state, _ = self.module.database_state(self.root / "nope.db")
        self.assertEqual(state, "missing")

    def test_a_real_sqlite_database_reads_as_plaintext(self) -> None:
        import sqlite3

        db = self.root / "real.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE t(a)")
        conn.commit()
        conn.close()
        state, _ = self.module.database_state(db)
        self.assertEqual(state, "plaintext")


class KeyFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self) -> None:
        self.dir.cleanup()

    def test_missing_key(self) -> None:
        ok, note = self.module.key_file_state(self.root / "absent.key")
        self.assertFalse(ok)
        self.assertIn("no key file", note)

    def test_world_readable_key_is_refused(self) -> None:
        key = self.root / "store.key"
        key.write_bytes(os.urandom(48))
        key.chmod(0o644)
        ok, note = self.module.key_file_state(key)
        self.assertFalse(ok)
        self.assertIn("0600", note)

    def test_short_key_is_refused(self) -> None:
        key = self.root / "store.key"
        key.write_bytes(b"short")
        key.chmod(stat.S_IRUSR | stat.S_IWUSR)
        ok, note = self.module.key_file_state(key)
        self.assertFalse(ok)
        self.assertIn("32+ bytes", note)

    def test_good_key_passes(self) -> None:
        key = self.root / "store.key"
        key.write_bytes(os.urandom(48))
        key.chmod(stat.S_IRUSR | stat.S_IWUSR)
        ok, note = self.module.key_file_state(key)
        self.assertTrue(ok, note)
        self.assertNotIn("secret", note.lower())

    def test_note_never_contains_key_material(self) -> None:
        key = self.root / "store.key"
        payload = b"W" * 48
        key.write_bytes(payload)
        key.chmod(stat.S_IRUSR | stat.S_IWUSR)
        _, note = self.module.key_file_state(key)
        self.assertNotIn("W" * 8, note)


class CipherTests(unittest.TestCase):
    def test_reports_a_definite_answer(self) -> None:
        module = load_module()
        supported, note = module.cipher_support()
        self.assertIsInstance(supported, bool)
        self.assertTrue(note)
        if not supported:
            self.assertIn("no encryption extension", note)


if __name__ == "__main__":
    unittest.main()
