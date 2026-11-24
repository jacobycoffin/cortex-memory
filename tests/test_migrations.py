"""Tests for ``cortex.migrations``: reversible, backup-gated schema changes.

The module under test is the one place in the repository where a bug can
damage a live store, so these tests care as much about the refusal and failure
paths as about the happy path:

  * an apply without a usable backup changes nothing;
  * an apply that fails halfway leaves neither schema nor ledger behind;
  * a second apply changes nothing at all (not merely "does not raise");
  * reverse restores the previous schema and refuses when it cannot.

Most tests use a small throwaway SQLite store that copies ``CortexStore``'s
``_conn`` + ``transaction()`` contract. Two tests use the real store to prove
the built-in migrations fit the actual schema.

Run: /root/workspace/jev/.venv/bin/python -m pytest tests/test_migrations.py -q
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import migrations
from cortex.migrations import Migration, MigrationError
from cortex.store import CortexStore


class _TempStore:
    """A minimal stand-in with the same migration surface as ``CortexStore``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(
            """
            CREATE TABLE memories (
                id         TEXT PRIMARY KEY,
                session_id TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE memory_embeddings (
                memory_id  TEXT NOT NULL,
                model_id   TEXT NOT NULL,
                dim        INTEGER NOT NULL,
                vector     BLOB NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (memory_id, model_id)
            );
            """
        )
        self._conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        self._conn.close()


def _backup(store: _TempStore, path: Path) -> Path:
    """A consistent copy of the store, made through SQLite's backup API."""

    destination = sqlite3.connect(str(path))
    try:
        store._conn.backup(destination)
    finally:
        destination.close()
    return path


class MigrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_store(self) -> _TempStore:
        return _TempStore(self.root / "cortex.db")

    def backup_path(self) -> Path:
        return self.root / "cortex-backup.db"

    def schema_objects(self, store: _TempStore) -> set[str]:
        return {
            str(row[0])
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
            )
        }

    def columns(self, store: _TempStore, table: str) -> set[str]:
        return {row[1] for row in store._conn.execute(f"PRAGMA table_info({table})")}


class RegistryTests(MigrationTestCase):
    def test_names_are_unique_and_orderable(self) -> None:
        names = [migration.name for migration in migrations.registry()]
        self.assertGreaterEqual(len(names), 2)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names, sorted(names))

    def test_every_builtin_migration_declares_a_reverse(self) -> None:
        for migration in migrations.registry():
            with self.subTest(migration=migration.name):
                self.assertTrue(migration.reversible, f"{migration.name} has no reverse SQL")
                self.assertTrue(migration.statements)
                self.assertTrue(migration.note)


class PendingStatusTests(MigrationTestCase):
    def test_status_starts_with_everything_pending(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)

        self.assertEqual(migrations.status(store)["applied"], [])
        self.assertEqual(migrations.status(store)["reversed"], [])
        self.assertEqual(
            migrations.status(store)["pending"],
            [migration.name for migration in migrations.registry()],
        )

    def test_pending_follows_registry_order_and_does_not_create_the_ledger(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)

        self.assertEqual(
            [migration.name for migration in migrations.pending(store)],
            [migration.name for migration in migrations.registry()],
        )
        subset = migrations.registry()[:1]
        self.assertEqual(migrations.pending(store, subset), subset)
        # Reading state must not write: preflight/pending may run against a
        # store the caller never intends to change.
        self.assertNotIn("schema_migrations", self.schema_objects(store))


class PreflightTests(MigrationTestCase):
    def test_ok_with_a_real_backup(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)
        _backup(store, self.backup_path())

        result = migrations.preflight(store, backup_path=str(self.backup_path()))
        self.assertTrue(result["ok"], result["problems"])
        self.assertEqual(result["problems"], [])

    def test_reports_a_missing_backup(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)

        result = migrations.preflight(store)
        self.assertFalse(result["ok"])
        self.assertTrue(any("backup_path" in problem for problem in result["problems"]))

    def test_reports_empty_and_live_backup_paths(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)
        empty = self.root / "empty.db"
        empty.write_bytes(b"")

        for candidate, needle in (
            (empty, "empty"),
            (store.path, "live database"),
            (self.root / "missing.db", "not a file"),
        ):
            with self.subTest(candidate=str(candidate)):
                result = migrations.preflight(store, backup_path=str(candidate))
                self.assertFalse(result["ok"])
                self.assertTrue(any(needle in problem for problem in result["problems"]))

    def test_reports_a_closed_connection(self) -> None:
        store = self.make_store()
        store.close()

        result = migrations.preflight(store, backup_path=str(self.backup_path()))
        self.assertFalse(result["ok"])
        self.assertTrue(any("did not answer" in problem for problem in result["problems"]))


class ApplyTests(MigrationTestCase):
    def test_refuses_without_a_backup_and_changes_nothing(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)
        before = self.schema_objects(store)

        with self.assertRaises(MigrationError):
            migrations.apply(store, "0001_memory_embeddings_created_index")

        self.assertEqual(self.schema_objects(store), before)
        self.assertNotIn("schema_migrations", self.schema_objects(store))

    def test_applies_and_records_the_migration(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)
        _backup(store, self.backup_path())
        stamp = "2026-01-02T03:04:05+00:00"

        result = migrations.apply(
            store,
            "0001_memory_embeddings_created_index",
            backup_path=str(self.backup_path()),
            at=stamp,
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["applied_at"], stamp)
        self.assertEqual(result["backup_path"], str(self.backup_path()))
        self.assertIn("idx_memory_embeddings_created", self.schema_objects(store))
        row = store._conn.execute(
            "SELECT applied_at, reversed_at FROM schema_migrations WHERE name=?",
            ("0001_memory_embeddings_created_index",),
        ).fetchone()
        self.assertEqual(row["applied_at"], stamp)
        self.assertIsNone(row["reversed_at"])
        self.assertEqual(
            [migration.name for migration in migrations.pending(store)],
            ["0002_memory_retention_classes", "0003_memories_retention_class_column"],
        )

    def test_applying_twice_is_a_noop(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)
        _backup(store, self.backup_path())
        name = "0002_memory_retention_classes"
        migrations.apply(store, name, backup_path=str(self.backup_path()), at="2026-01-01T00:00:00+00:00")
        store._conn.execute(
            "INSERT INTO memory_retention_classes(name, description) VALUES('audit', 'kept')"
        )
        store._conn.commit()

        # No backup on the second call: nothing is going to change, so the
        # refusal must not fire.
        result = migrations.apply(store, name)

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "already applied")
        self.assertEqual(result["applied_at"], "2026-01-01T00:00:00+00:00")
        rows = store._conn.execute(
            "SELECT COUNT(*) AS n FROM memory_retention_classes"
        ).fetchone()
        self.assertEqual(rows["n"], 1)
        ledger = store._conn.execute(
            "SELECT COUNT(*) AS n FROM schema_migrations WHERE name=?", (name,)
        ).fetchone()
        self.assertEqual(ledger["n"], 1)

    def test_guard_skips_statements_when_the_column_already_exists(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)
        _backup(store, self.backup_path())
        store._conn.execute("ALTER TABLE memories ADD COLUMN retention_class TEXT")
        store._conn.commit()

        result = migrations.apply(
            store,
            "0003_memories_retention_class_column",
            backup_path=str(self.backup_path()),
        )

        self.assertTrue(result["applied"])
        self.assertIn("retention_class", self.columns(store, "memories"))
        ledger = store._conn.execute(
            "SELECT COUNT(*) AS n FROM schema_migrations WHERE name=?",
            ("0003_memories_retention_class_column",),
        ).fetchone()
        self.assertEqual(ledger["n"], 1)

    def test_failure_rolls_back_schema_and_ledger(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)
        _backup(store, self.backup_path())
        broken = Migration(
            name="9999_broken_example",
            statements=[
                "CREATE INDEX IF NOT EXISTS idx_broken_probe ON memories(session_id)",
                "INSERT INTO table_that_does_not_exist VALUES (1)",
            ],
            reverse=["DROP INDEX IF EXISTS idx_broken_probe"],
            note="test-only: first statement succeeds, second fails",
        )
        before = self.schema_objects(store)

        with mock.patch.object(migrations, "registry", return_value=[broken]):
            with self.assertRaises(MigrationError) as caught:
                migrations.apply(
                    store, broken.name, backup_path=str(self.backup_path())
                )

        self.assertIn("rolled back", str(caught.exception))
        self.assertEqual(self.schema_objects(store), before)
        self.assertNotIn("idx_broken_probe", self.schema_objects(store))

    def test_failure_keeps_earlier_ledger_rows(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)
        _backup(store, self.backup_path())
        first = migrations.registry()[0]
        migrations.apply(store, first.name, backup_path=str(self.backup_path()))
        broken = Migration(
            name="9999_broken_example",
            statements=["INSERT INTO table_that_does_not_exist VALUES (1)"],
            reverse=[],
            note="test-only: always fails",
        )

        with mock.patch.object(migrations, "registry", return_value=[first, broken]):
            with self.assertRaises(MigrationError):
                migrations.apply(
                    store, broken.name, backup_path=str(self.backup_path())
                )

        names = {
            row[0]
            for row in store._conn.execute("SELECT name FROM schema_migrations")
        }
        self.assertEqual(names, {first.name})
        self.assertEqual(migrations.status(store)["applied"], [first.name])

    def test_assume_copy_allows_a_backup_free_apply(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)

        result = migrations.apply(
            store,
            "0002_memory_retention_classes",
            assume_copy=True,
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["backup_path"], "")
        self.assertIn("memory_retention_classes", self.schema_objects(store))

    def test_unknown_migration_name_refuses(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)

        with self.assertRaises(MigrationError):
            migrations.apply(store, "0000_not_here", assume_copy=True)


class ReverseTests(MigrationTestCase):
    def test_reverse_restores_schema_and_marks_the_ledger(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)
        _backup(store, self.backup_path())
        name = "0001_memory_embeddings_created_index"
        migrations.apply(store, name, backup_path=str(self.backup_path()))
        self.assertIn("idx_memory_embeddings_created", self.schema_objects(store))

        result = migrations.reverse(store, name, at="2026-02-03T04:05:06+00:00")

        self.assertTrue(result["reversed"])
        self.assertNotIn("idx_memory_embeddings_created", self.schema_objects(store))
        self.assertEqual(migrations.status(store)["reversed"], [name])
        self.assertIn(name, migrations.status(store)["pending"])
        self.assertEqual(migrations.status(store)["applied"], [])

    def test_reverse_refuses_when_not_applied_or_already_reversed(self) -> None:
        store = self.make_store()
        self.addCleanup(store.close)

        with self.assertRaises(MigrationError) as caught:
            migrations.reverse(store, "0001_memory_embeddings_created_index")
        self.assertIn("not applied", str(caught.exception))

        _backup(store, self.backup_path())
        name = "0002_memory_retention_classes"
        migrations.apply(store, name, backup_path=str(self.backup_path()))
        migrations.reverse(store, name)

        with self.assertRaises(MigrationError) as caught:
            migrations.reverse(store, name)
        self.assertIn("already reversed", str(caught.exception))

    def test_reverse_refuses_an_irreversible_migration(self) -> None:
        irreversible = Migration(
            name="9999_drop_audit_history",
            statements=["DROP TABLE IF EXISTS access_history"],
            note="test-only: no undo exists for a dropped audit table",
        )
        self.assertFalse(irreversible.reversible)

        with mock.patch.object(migrations, "registry", return_value=[irreversible]):
            with self.assertRaises(MigrationError) as caught:
                migrations.reverse(object(), irreversible.name)
        self.assertIn("irreversible", str(caught.exception))


class RealStoreTests(MigrationTestCase):
    def test_round_trip_against_a_real_cortex_store(self) -> None:
        store = CortexStore(self.root / "real.db")
        self.addCleanup(store.close)
        backup = self.root / "real-backup.db"
        destination = sqlite3.connect(str(backup))
        try:
            store._conn.backup(destination)
        finally:
            destination.close()

        for migration in migrations.registry():
            result = migrations.apply(store, migration.name, backup_path=str(backup))
            self.assertTrue(result["applied"], migration.name)

        self.assertIn("idx_memory_embeddings_created", self.schema_objects(store))
        self.assertIn("memory_retention_classes", self.schema_objects(store))
        self.assertIn("retention_class", self.columns(store, "memories"))
        self.assertEqual(migrations.pending(store), [])

        migrations.reverse(store, "0003_memories_retention_class_column")
        self.assertNotIn("retention_class", self.columns(store, "memories"))
        self.assertEqual(
            [migration.name for migration in migrations.pending(store)],
            ["0003_memories_retention_class_column"],
        )


if __name__ == "__main__":  # pragma: no cover - manual convenience
    unittest.main()
