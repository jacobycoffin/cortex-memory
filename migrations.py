"""Reversible schema migrations with an applied-migrations ledger.

WHY: Cortex already upgrades its schema opportunistically on every store open
(``cortex_schema._migrate_columns``). That is the right shape for adding a
column every writer can ignore, but it is the wrong tool for anything larger:
the work repeats on every open, nothing records what ran or when, and there is
no way to undo it. This module is the deliberate path for structural change:
describe the forward SQL once, describe the undo SQL beside it, verify the
backup before touching anything, run the change and its ledger row in one
transaction, and leave a row that says when it was applied and when (if ever)
it was reversed.

The contract is deliberately small:

  * A migration is applied at most once. Re-applying is a no-op, not an
    error; the ledger row keeps its original timestamp.
  * A failed apply leaves neither partial schema nor a ledger row. SQLite DDL
    is transactional, so the transaction is the safety net: every statement
    and the ledger insert commit or roll back together. Statements are
    executed one by one rather than with ``executescript``, which would commit
    the earlier statements before the later one failed.
  * ``apply`` refuses without a verified backup unless the caller passes
    ``assume_copy=True`` and thereby promises the store is disposable.
  * ``reverse`` refuses when the migration declares no undo statements. Even
    when it has them, reverse can only remove structure: it cannot restore
    data written into a column after the forward step, nor the objects a
    dropped table contained.

The migrations built into :func:`registry` are examples of the three shapes
this store plausibly needs: an index, a lookup table, and a nullable column.
All three are additive and reversible. None of them are irreversible by
nature; the honest limit is that reversing the column migration discards any
values written into that column, so reverse is only safe before writers start
using it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "LEDGER_TABLE",
    "Migration",
    "MigrationError",
    "apply",
    "pending",
    "preflight",
    "registry",
    "reverse",
    "status",
]


# Where applied/reversed migrations are recorded. It lives in the store it
# describes, so a restored copy carries its own history; restoring an older
# copy therefore reports the migrations that copy still needs.
LEDGER_TABLE = "schema_migrations"

_LEDGER_DDL = f"""
    CREATE TABLE IF NOT EXISTS {LEDGER_TABLE} (
        name        TEXT PRIMARY KEY,
        applied_at  TEXT NOT NULL,
        reversed_at TEXT
    )
"""

_SQLITE_HEADER = b"SQLite format 3\x00"


class MigrationError(RuntimeError):
    """Raised when a migration cannot be applied or reversed safely."""


@dataclass(frozen=True)
class Migration:
    """One named forward change and the SQL that undoes it.

    ``statements`` run in order inside a single transaction, so a later
    failure rolls back an earlier statement. ``reverse`` may be empty when the
    change genuinely cannot be undone; :func:`reverse` then refuses instead of
    pretending. ``guard`` exists because SQLite has no
    ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``: it is asked, inside the
    transaction, whether the forward statements still need to run. A guard
    returning ``False`` means the schema is already in the target shape and
    the migration is recorded without re-running SQL.
    """

    name: str
    statements: Sequence[str]
    reverse: Sequence[str] = ()
    note: str = ""
    guard: Callable[[sqlite3.Connection], bool] | None = None

    @property
    def reversible(self) -> bool:
        """Whether this migration declares undo statements."""

        return bool(self.reverse)


def _missing_column(table: str, column: str) -> Callable[[sqlite3.Connection], bool]:
    """Build a guard that is true while ``table`` lacks ``column``.

    Table and column names here are module constants, never caller input.
    """

    def guard(conn: sqlite3.Connection) -> bool:
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        return column not in columns

    return guard


def registry() -> list[Migration]:
    """The built-in migrations, in the order they should be applied.

    These are intentionally small and additive. Each is safe to run against a
    current store: the index and table are ``IF NOT EXISTS`` and the column
    addition is PRAGMA-guarded. Each has a reverse that restores the previous
    schema exactly, apart from data written into the added column.
    """

    return [
        Migration(
            name="0001_memory_embeddings_created_index",
            statements=[
                "CREATE INDEX IF NOT EXISTS idx_memory_embeddings_created "
                "ON memory_embeddings(created_at)",
            ],
            reverse=[
                "DROP INDEX IF EXISTS idx_memory_embeddings_created",
            ],
            note=(
                "Retention and backfill scans walk embeddings oldest-first; without this "
                "index each sweep reads the whole table."
            ),
        ),
        Migration(
            name="0002_memory_retention_classes",
            statements=[
                """
                CREATE TABLE IF NOT EXISTS memory_retention_classes (
                    name        TEXT PRIMARY KEY,
                    description TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_memory_retention_classes_created "
                "ON memory_retention_classes(created_at)",
            ],
            reverse=[
                "DROP INDEX IF EXISTS idx_memory_retention_classes_created",
                "DROP TABLE IF EXISTS memory_retention_classes",
            ],
            note=(
                "A lookup table keeps retention vocabulary editable without a schema "
                "change per label. Reversing drops the rows an operator may have added."
            ),
        ),
        Migration(
            name="0003_memories_retention_class_column",
            statements=[
                "ALTER TABLE memories ADD COLUMN retention_class TEXT",
            ],
            reverse=[
                "ALTER TABLE memories DROP COLUMN retention_class",
            ],
            note=(
                "Nullable, so existing rows and older writers are unaffected. The reverse "
                "statement needs SQLite 3.35+ and discards any values written meanwhile."
            ),
            guard=_missing_column("memories", "retention_class"),
        ),
    ]


def _coerce_migrations(
    migrations: Iterable[Migration] | None,
) -> list[Migration]:
    selected = list(registry() if migrations is None else migrations)
    names = [migration.name for migration in selected]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise MigrationError(f"duplicate migration names: {', '.join(duplicates)}")
    return selected


def _lookup(name: str, migrations: Iterable[Migration] | None = None) -> Migration:
    for migration in _coerce_migrations(migrations):
        if migration.name == name:
            return migration
    known = ", ".join(m.name for m in _coerce_migrations(migrations))
    raise MigrationError(f"unknown migration {name!r}; known migrations: {known}")


def _connection(store: Any) -> sqlite3.Connection:
    conn = getattr(store, "_conn", None)
    if conn is None:
        raise MigrationError("store has no sqlite connection (`_conn`) to migrate")
    return conn


def _stamp(at: Any) -> str:
    """Normalise a caller timestamp to an aware ISO-8601 string in UTC."""

    if at is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    if isinstance(at, datetime):
        moment = at if at.tzinfo is not None else at.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    if isinstance(at, str):
        try:
            moment = datetime.fromisoformat(at)
        except ValueError as exc:
            raise MigrationError(f"invalid timestamp {at!r}: {exc}") from exc
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    raise MigrationError(f"invalid timestamp {at!r}: expected a datetime or ISO-8601 string")


def _ledger_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (LEDGER_TABLE,),
    ).fetchone()
    return row is not None


def _ensure_ledger(conn: sqlite3.Connection) -> None:
    conn.execute(_LEDGER_DDL)


def _ledger_state(conn: sqlite3.Connection) -> dict[str, str | None]:
    """Map migration name -> reversed_at, reading but never creating the ledger."""

    if not _ledger_exists(conn):
        return {}
    return {
        str(row[0]): row[1]
        for row in conn.execute(f"SELECT name, reversed_at FROM {LEDGER_TABLE}")
    }


def _applied_names(conn: sqlite3.Connection) -> set[str]:
    return {name for name, reversed_at in _ledger_state(conn).items() if reversed_at is None}


def pending(store: Any, migrations: Iterable[Migration] | None = None) -> list[Migration]:
    """Return the migrations that are not currently applied, in registry order.

    A migration that was applied and later reversed is pending again: the
    ledger keeps the reversal for audit, but the schema no longer has the
    change.
    """

    selected = _coerce_migrations(migrations)
    try:
        applied = _applied_names(_connection(store))
    except sqlite3.Error as exc:
        raise MigrationError(f"could not read the migration ledger: {exc}") from exc
    return [migration for migration in selected if migration.name not in applied]


def status(store: Any) -> dict[str, list[str]]:
    """Report applied, pending and reversed migration names.

    Reversed names are those the ledger still knows about with a
    ``reversed_at`` stamp. Names in the ledger that are no longer in the
    registry are included in ``applied``/``reversed`` so an operator can see a
    rollback of a migration this build no longer ships.
    """

    selected = _coerce_migrations(None)
    known = {migration.name for migration in selected}
    try:
        state = _ledger_state(_connection(store))
    except sqlite3.Error as exc:
        raise MigrationError(f"could not read the migration ledger: {exc}") from exc

    applied = sorted(
        (name for name, reversed_at in state.items() if reversed_at is None),
        key=lambda name: name,
    )
    reversed_names = sorted(
        (name for name, reversed_at in state.items() if reversed_at is not None),
        key=lambda name: name,
    )
    pending_names = [
        migration.name for migration in selected if migration.name not in set(applied)
    ]
    # Anything applied outside the registry is still reported, but listed after
    # the migrations this build can reason about.
    ordered_applied = [name for name in applied if name in known] + [
        name for name in applied if name not in known
    ]
    return {
        "applied": ordered_applied,
        "pending": pending_names,
        "reversed": reversed_names,
    }


def _schema_problems(store: Any) -> list[str]:
    """Checks that must hold before anything is applied."""

    problems: list[str] = []
    conn = getattr(store, "_conn", None)
    if conn is None:
        return ["store has no sqlite connection (`_conn`) to migrate"]
    try:
        conn.execute("SELECT 1").fetchone()
    except sqlite3.Error as exc:
        return [f"database did not answer: {exc}"]

    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.execute("PRAGMA user_version").fetchone()
    except sqlite3.Error as exc:
        return [f"schema is not readable: {exc}"]
    if not tables:
        problems.append("schema is empty: there are no tables to migrate")
    if _ledger_exists(conn):
        try:
            ledger_columns = {row[1] for row in conn.execute(f"PRAGMA table_info({LEDGER_TABLE})")}
        except sqlite3.Error as exc:
            return problems + [f"migration ledger is not readable: {exc}"]
        expected = {"name", "applied_at", "reversed_at"}
        missing = sorted(expected - ledger_columns)
        if missing:
            problems.append(
                f"migration ledger {LEDGER_TABLE!r} is missing columns: {', '.join(missing)}"
            )
    return problems


def _backup_problem(store: Any, backup_path: str) -> str:
    """Return why ``backup_path`` is not a usable backup, or ``""`` if it is."""

    if not str(backup_path or "").strip():
        return "no backup_path supplied: copy the database and pass the copy's path"
    path = Path(backup_path).expanduser()
    try:
        if not path.is_file():
            return f"backup path is not a file: {path}"
        if path.stat().st_size == 0:
            return f"backup is empty: {path}"
    except OSError as exc:
        return f"backup path could not be inspected: {exc}"

    store_path = getattr(store, "path", None)
    if store_path:
        try:
            if path.resolve() == Path(store_path).expanduser().resolve():
                return "backup path points at the live database"
        except OSError:
            pass

    try:
        with path.open("rb") as handle:
            header = handle.read(len(_SQLITE_HEADER))
    except OSError as exc:
        return f"backup could not be read: {exc}"
    if header != _SQLITE_HEADER:
        # Not a SQLite file (a tarball, a filesystem snapshot, ...). Existence
        # and non-emptiness are the checks this module can honestly make.
        return ""
    try:
        probe = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            result = probe.execute("PRAGMA quick_check").fetchone()
        finally:
            probe.close()
    except sqlite3.Error as exc:
        return f"backup is not a readable SQLite database: {exc}"
    verdict = str(result[0]) if result else "no result"
    if verdict != "ok":
        return f"backup failed SQLite quick_check: {verdict}"
    return ""


def preflight(store: Any, *, backup_path: str = "") -> dict[str, Any]:
    """Check the store and backup before a migration is applied.

    Returns ``{"ok": bool, "problems": [str, ...]}``. It never writes: a
    missing ledger is reported as nothing (it will be created by the apply
    transaction), and a missing or unusable backup is a problem because the
    caller is expected to pass one.
    """

    problems = _schema_problems(store)
    backup_problem = _backup_problem(store, backup_path)
    if backup_problem:
        problems.append(backup_problem)
    return {"ok": not problems, "problems": problems}


def apply(
    store: Any,
    name: str,
    *,
    backup_path: str = "",
    at: Any = None,
    assume_copy: bool = False,
) -> dict[str, Any]:
    """Apply one migration, recording it in the ledger.

    Refuses unless a usable ``backup_path`` was supplied. ``assume_copy=True``
    is the explicit escape hatch for a store the caller knows is a disposable
    copy (a restore drill, a test fixture); it skips the backup requirement
    but still checks that the database answers and its schema is readable.

    Returns ``{"name", "applied", "reason", "applied_at", "backup_path"}``.
    Applying an already-applied migration is a successful no-op with
    ``applied=False``: nothing is executed and the original ledger timestamp
    is preserved.
    """

    migration = _lookup(name)
    conn = _connection(store)
    try:
        state = _ledger_state(conn)
    except sqlite3.Error as exc:
        raise MigrationError(f"could not read the migration ledger: {exc}") from exc
    if name in state and state[name] is None:
        try:
            row = conn.execute(
                f"SELECT applied_at FROM {LEDGER_TABLE} WHERE name=?", (name,)
            ).fetchone()
            applied_at = str(row[0]) if row is not None else ""
        except sqlite3.Error:
            applied_at = ""
        return {
            "name": name,
            "applied": False,
            "reason": "already applied",
            "applied_at": applied_at,
            "backup_path": "",
        }

    if assume_copy:
        problems = _schema_problems(store)
        if problems:
            raise MigrationError(f"preflight failed for {name}: {'; '.join(problems)}")
    else:
        check = preflight(store, backup_path=backup_path)
        if not check["ok"]:
            raise MigrationError(f"preflight failed for {name}: {'; '.join(check['problems'])}")

    stamp = _stamp(at)
    try:
        with store.transaction() as tx:
            _ensure_ledger(tx)
            row = tx.execute(
                f"SELECT applied_at, reversed_at FROM {LEDGER_TABLE} WHERE name=?",
                (name,),
            ).fetchone()
            if row is not None and row[1] is None:
                # Another writer won the race: treat it exactly like the
                # pre-transaction check, without running the statements.
                return {
                    "name": name,
                    "applied": False,
                    "reason": "already applied",
                    "applied_at": str(row[0]),
                    "backup_path": "",
                }
            should_run = migration.guard is None or migration.guard(tx)
            if should_run:
                for statement in migration.statements:
                    tx.execute(statement)
            tx.execute(
                f"INSERT INTO {LEDGER_TABLE}(name, applied_at, reversed_at) VALUES (?, ?, NULL) "
                "ON CONFLICT(name) DO UPDATE SET applied_at=excluded.applied_at, "
                "reversed_at=NULL",
                (name, stamp),
            )
    except sqlite3.Error as exc:
        raise MigrationError(
            f"migration {name} failed and was rolled back: {exc}"
        ) from exc

    return {
        "name": name,
        "applied": True,
        "reason": "",
        "applied_at": stamp,
        "backup_path": "" if assume_copy else str(backup_path),
    }


def reverse(store: Any, name: str, *, at: Any = None) -> dict[str, Any]:
    """Undo one applied migration and stamp its ledger row as reversed.

    Refuses when the migration declares no reverse statements, when it is not
    (or no longer) applied, and when it was already reversed. Undo runs in one
    transaction with the ledger update, so a failure leaves both untouched.
    """

    migration = _lookup(name)
    if not migration.reverse:
        raise MigrationError(
            f"migration {name} declares no reverse statements and is irreversible"
        )
    _connection(store)
    stamp = _stamp(at)
    try:
        with store.transaction() as tx:
            _ensure_ledger(tx)
            row = tx.execute(
                f"SELECT applied_at, reversed_at FROM {LEDGER_TABLE} WHERE name=?",
                (name,),
            ).fetchone()
            if row is None:
                raise MigrationError(f"migration {name} is not applied")
            if row[1] is not None:
                raise MigrationError(f"migration {name} was already reversed at {row[1]}")
            for statement in migration.reverse:
                tx.execute(statement)
            tx.execute(
                f"UPDATE {LEDGER_TABLE} SET reversed_at=? WHERE name=?",
                (stamp, name),
            )
    except sqlite3.Error as exc:
        raise MigrationError(
            f"reverse of {name} failed and was rolled back: {exc}"
        ) from exc

    return {"name": name, "reversed": True, "reversed_at": stamp}
