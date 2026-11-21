#!/usr/bin/env python3
"""Prove a Cortex backup actually restores, instead of assuming it would.

The drill is end to end:

1. take an **online snapshot** of the source database through sqlite3's backup
   API (a page-consistent copy even while another process holds the file open --
   never ``cp`` a live database: a plain file copy can capture a torn WAL);
2. **restore** that snapshot into a scratch directory, again via the backup API,
   which also proves the snapshot is a valid database another process can read;
3. **verify** the restore: per-table row counts, ``PRAGMA integrity_check``, and a
   SHA-256 over a deterministic sample (``ORDER BY id LIMIT 200``, or the table's
   primary key when it has no ``id`` -- ``edges`` is keyed by
   ``(src_id, dst_id, relation)``) of ``memories`` and ``edges``, so an
   equal-count-but-different-content copy cannot pass.

The source is opened read-only (``mode=ro`` URI); the drill never writes to it. The
row counts are read inside the same read transaction that produces the snapshot,
because a live database keeps taking writes while the drill runs.

Exit codes:
    0  every table matched, integrity ok, sample digests identical
    1  a mismatch, a corrupt snapshot/restore, or a failed integrity check
    2  the input is unreadable (missing, not a SQLite file, cannot be opened)

Usage:
    python3 scripts/backup_restore_drill.py [SOURCE_DB] [--scratch DIR] [--keep]
        [--sample-size N] [--quiet]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DB = Path(os.environ.get("CORTEX_DB", "~/.cortex/cortex.db")).expanduser()
SAMPLE_TABLES = ("memories", "edges")
SAMPLE_SIZE = 200
SQLITE_HEADER = b"SQLite format 3\x00"


class UnreadableInput(Exception):
    """The source cannot be opened as a SQLite database at all."""


@dataclass
class TableRow:
    """One table's comparison line in the report."""

    name: str
    source_count: int | None
    restore_count: int | None
    error: str | None = None

    @property
    def match(self) -> bool:
        return self.error is None and self.source_count == self.restore_count


@dataclass
class SampleRow:
    """One sampled-table digest comparison."""

    table: str
    rows: int
    source_digest: str | None
    restore_digest: str | None
    error: str | None = None

    @property
    def match(self) -> bool:
        return (
            self.error is None
            and self.source_digest is not None
            and self.source_digest == self.restore_digest
        )


@dataclass
class Report:
    """Everything the drill observed, plus the exit code it implies."""

    source: Path
    snapshot: Path | None = None
    restored: Path | None = None
    snapshot_sha256: str | None = None
    source_bytes: int = 0
    tables: list[TableRow] = field(default_factory=list)
    samples: list[SampleRow] = field(default_factory=list)
    integrity: list[str] = field(default_factory=list)
    missing_tables: list[str] = field(default_factory=list)
    extra_tables: list[str] = field(default_factory=list)
    error: str | None = None
    unreadable_reason: str | None = None
    kept_dir: Path | None = None
    cleaned: bool = False
    elapsed_s: float = 0.0

    @property
    def integrity_ok(self) -> bool:
        return self.integrity == ["ok"]

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and self.integrity_ok
            and not self.missing_tables
            and not self.extra_tables
            and all(row.match for row in self.tables)
            and all(sample.match for sample in self.samples)
        )

    @property
    def exit_code(self) -> int:
        if self.error is not None:
            return 2 if self.unreadable_reason is not None else 1
        return 0 if self.ok else 1


def _open_readonly(path: Path) -> sqlite3.Connection:
    """Open ``path`` read-only so the drill can never mutate the source."""

    uri = f"file:{path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=30.0)


def check_source(path: Path) -> None:
    """Raise :class:`UnreadableInput` unless ``path`` is a readable SQLite file."""

    if not path.exists():
        raise UnreadableInput(f"source database does not exist: {path}")
    if not path.is_file():
        raise UnreadableInput(f"source is not a regular file: {path}")
    try:
        with path.open("rb") as handle:
            header = handle.read(len(SQLITE_HEADER))
    except OSError as exc:
        raise UnreadableInput(f"cannot read source: {exc}") from exc
    if header != SQLITE_HEADER:
        raise UnreadableInput(f"not a SQLite database (bad header): {path}")
    try:
        conn = _open_readonly(path)
        try:
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise UnreadableInput(f"cannot query source: {exc}") from exc


def snapshot_database(source_conn: sqlite3.Connection, snapshot: Path) -> int:
    """Copy the source connection's current snapshot with the backup API.

    The caller keeps a read transaction open on ``source_conn`` while this runs,
    so the copy is the same view of the database the row counts were read from.
    A live Cortex database keeps taking writes during a drill; reading counts
    from a *new* connection afterwards compares two different points in time and
    reports phantom mismatches.
    """

    snapshot.parent.mkdir(parents=True, exist_ok=True)
    dst = sqlite3.connect(snapshot)
    try:
        source_conn.backup(dst)
        pages = dst.execute("PRAGMA page_count").fetchone()[0]
    finally:
        dst.close()
    os.chmod(snapshot, 0o600)
    return int(pages)


def restore_database(snapshot: Path, restored: Path) -> None:
    """Restore a snapshot into ``restored`` using the backup API."""

    restored.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{snapshot.as_posix()}?mode=ro", uri=True, timeout=30.0)
    try:
        dst = sqlite3.connect(restored)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    os.chmod(restored, 0o600)


def table_names(conn: sqlite3.Connection) -> list[str]:
    """User tables, excluding SQLite's internal ``sqlite_*`` bookkeeping."""

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row[0] for row in rows]


def row_counts(conn: sqlite3.Connection, names: list[str]) -> dict[str, tuple[int | None, str | None]]:
    """Row count per table; a malformed table is reported, not raised."""

    counts: dict[str, tuple[int | None, str | None]] = {}
    for name in names:
        try:
            counts[name] = (int(conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]), None)
        except sqlite3.Error as exc:
            counts[name] = (None, str(exc))
    return counts


def order_by_clause(conn: sqlite3.Connection, table: str) -> str:
    """A deterministic ordering for a table: ``id``, else its primary key.

    ``memories`` has an ``id``; ``edges`` is keyed by ``(src_id, dst_id,
    relation)`` and has no ``id`` column at all, so the drill orders by whatever
    the table declares instead of assuming one shape for every table.
    """

    columns = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    names = [column[1] for column in columns]
    if "id" in names:
        return '"id"'
    primary_key = [column[1] for column in sorted(columns, key=lambda column: column[5]) if column[5]]
    if primary_key:
        return ", ".join(f'"{name}"' for name in primary_key)
    return "rowid"


def sample_digest(conn: sqlite3.Connection, table: str, limit: int) -> tuple[int, str]:
    """SHA-256 over a deterministic sample of ``limit`` rows, canonicalised per row.

    The digest also covers the column names and the row count, so a restore that
    is missing rows, has reordered columns, or holds different values in the same
    number of rows all produce a different hash.
    """

    order_by = order_by_clause(conn, table)
    cursor = conn.execute(f'SELECT * FROM "{table}" ORDER BY {order_by} LIMIT ?', (limit,))
    columns = [description[0] for description in cursor.description or ()]
    digest = hashlib.sha256()
    digest.update(("\x1f".join(columns) + "\x1e").encode("utf-8"))
    count = 0
    for row in cursor:
        digest.update(_canonical_row(row))
        count += 1
    digest.update(f"\x1ecount={count}".encode("utf-8"))
    return count, digest.hexdigest()


def _canonical_row(row: tuple) -> bytes:
    """Stable, type-tagged encoding so text/blob/None cannot collide."""

    parts: list[bytes] = []
    for value in row:
        if value is None:
            parts.append(b"N")
        elif isinstance(value, (bytes, bytearray, memoryview)):
            parts.append(b"B" + bytes(value).hex().encode("ascii"))
        else:
            parts.append(b"T" + str(value).encode("utf-8"))
    return b"\x1f".join(parts) + b"\x1e"


def integrity_report(conn: sqlite3.Connection) -> list[str]:
    """``PRAGMA integrity_check`` results; ``["ok"]`` means sound."""

    try:
        return [str(row[0]) for row in conn.execute("PRAGMA integrity_check")]
    except sqlite3.Error as exc:
        return [f"integrity_check failed: {exc}"]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_drill(
    source: Path,
    *,
    scratch_root: Path | None = None,
    keep: bool = False,
    sample_size: int = SAMPLE_SIZE,
    sample_tables: tuple[str, ...] = SAMPLE_TABLES,
) -> Report:
    """Run the full snapshot -> restore -> verify cycle against ``source``."""

    started = time.monotonic()
    report = Report(source=source)
    check_source(source)
    report.source_bytes = source.stat().st_size

    root = Path(tempfile.mkdtemp(prefix="cortex-restore-drill-", dir=scratch_root))
    snapshot = root / "cortex-snapshot.db"
    restored = root / "restore" / "cortex.db"
    report.snapshot = snapshot
    report.restored = restored

    phase = "snapshot"
    try:
        source_conn = _open_readonly(source)
        source_conn.isolation_level = None  # explicit transaction control
        try:
            source_conn.execute("BEGIN")
            # Establish the read snapshot before anything is copied: every count
            # and every sampled row then describes the same instant as the backup.
            source_conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
            source_tables = table_names(source_conn)
            source_counts = row_counts(source_conn, source_tables)
            source_samples: dict[str, tuple[int, str]] = {}
            for table in sample_tables:
                if table in source_tables:
                    try:
                        source_samples[table] = sample_digest(source_conn, table, sample_size)
                    except sqlite3.Error:
                        pass
            snapshot_database(source_conn, snapshot)
            source_conn.execute("COMMIT")
        finally:
            source_conn.close()

        report.snapshot_sha256 = _hash_file(snapshot)
        phase = "restore"
        restore_database(snapshot, restored)

        restored_conn = sqlite3.connect(f"file:{restored.as_posix()}?mode=ro", uri=True, timeout=30.0)
        try:
            restored_tables = table_names(restored_conn)
            report.missing_tables = sorted(set(source_tables) - set(restored_tables))
            report.extra_tables = sorted(set(restored_tables) - set(source_tables))
            shared = [name for name in source_tables if name in set(restored_tables)]

            restored_counts = row_counts(restored_conn, shared)
            for name in shared:
                source_count, source_error = source_counts[name]
                restore_count, restore_error = restored_counts[name]
                report.tables.append(
                    TableRow(
                        name=name,
                        source_count=source_count,
                        restore_count=restore_count,
                        error=source_error or restore_error,
                    )
                )

            for table in sample_tables:
                if table not in shared:
                    report.samples.append(
                        SampleRow(
                            table=table,
                            rows=0,
                            source_digest=None,
                            restore_digest=None,
                            error="table not present in source",
                        )
                    )
                    continue
                source_sample = source_samples.get(table)
                if source_sample is None:
                    report.samples.append(
                        SampleRow(
                            table=table,
                            rows=0,
                            source_digest=None,
                            restore_digest=None,
                            error="sample unreadable in source",
                        )
                    )
                    continue
                rows, source_digest = source_sample
                try:
                    _, restore_digest = sample_digest(restored_conn, table, sample_size)
                    report.samples.append(
                        SampleRow(
                            table=table,
                            rows=rows,
                            source_digest=source_digest,
                            restore_digest=restore_digest,
                        )
                    )
                except sqlite3.Error as exc:
                    report.samples.append(
                        SampleRow(
                            table=table,
                            rows=rows,
                            source_digest=source_digest,
                            restore_digest=None,
                            error=str(exc),
                        )
                    )

            report.integrity = integrity_report(restored_conn)
        finally:
            restored_conn.close()
    except sqlite3.Error as exc:
        message = str(exc)
        report.error = f"{phase} failed: {message}"
        # Exit 2 means the operator pointed the drill at something that is not a
        # readable database. A backup that fails *after* a clean snapshot is a
        # corrupt artifact, which is a failed verification (exit 1), not bad input.
        if phase == "snapshot" and (
            "not a database" in message.lower() or "unable to open" in message.lower()
        ):
            report.unreadable_reason = message
    finally:
        report.elapsed_s = time.monotonic() - started
        if keep:
            report.kept_dir = root
        else:
            shutil.rmtree(root, ignore_errors=True)
            report.cleaned = True
    return report


def _fmt_count(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def format_report(report: Report, sample_size: int = SAMPLE_SIZE) -> str:
    lines = [f"source:   {report.source} ({report.source_bytes:,} bytes)"]
    suffix = " (deleted after verification)" if report.cleaned else ""
    if report.snapshot is not None:
        lines.append(f"snapshot: {report.snapshot}{suffix}")
        lines.append(f"restore:  {report.restored}{suffix}")
    if report.snapshot_sha256:
        lines.append(f"snapshot sha256: {report.snapshot_sha256}")
    if report.error:
        lines.append(f"error:    {report.error}")
    lines.append("")

    width = max([len(row.name) for row in report.tables] + [5])
    lines.append(f"{'table'.ljust(width)}  {'source':>10}  {'restore':>10}  result")
    lines.append(f"{'-' * width}  {'-' * 10}  {'-' * 10}  ------")
    for row in report.tables:
        result = "match" if row.match else "MISMATCH"
        lines.append(
            f"{row.name.ljust(width)}  {_fmt_count(row.source_count):>10}  "
            f"{_fmt_count(row.restore_count):>10}  {result}"
        )
        if row.error:
            lines.append(f"{' ' * width}  error: {row.error}")
    for name in report.missing_tables:
        lines.append(f"{name.ljust(width)}  {'-':>10}  {'-':>10}  MISSING in restore")
    for name in report.extra_tables:
        lines.append(f"{name.ljust(width)}  {'-':>10}  {'-':>10}  EXTRA in restore")

    lines.append("")
    for sample in report.samples:
        if sample.error:
            lines.append(f"sample {sample.table} (limit {sample_size}): ERROR {sample.error}")
            continue
        verdict = "match" if sample.match else "MISMATCH"
        lines.append(
            f"sample {sample.table} (limit {sample_size}, {sample.rows} rows): "
            f"{(sample.source_digest or '')[:16]} vs {(sample.restore_digest or '')[:16]}  {verdict}"
        )
    lines.append(f"integrity_check(restore): {', '.join(report.integrity) or 'not run'}")

    total = sum(row.restore_count or 0 for row in report.tables)
    verdict = "PASS" if report.ok else "FAIL"
    lines.append("")
    lines.append(
        f"RESULT: {verdict} -- {len(report.tables)} tables, {total:,} rows restored, "
        f"{report.elapsed_s:.2f}s"
    )
    kept = report.kept_dir
    if kept is not None:
        lines.append(f"scratch kept: {kept}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Snapshot a Cortex database, restore it into a scratch directory, and verify it."
    )
    parser.add_argument(
        "source", nargs="?", type=Path, default=DEFAULT_DB, help=f"database to drill (default: {DEFAULT_DB})"
    )
    parser.add_argument("--scratch", type=Path, default=None, help="parent directory for the scratch copy")
    parser.add_argument("--keep", action="store_true", help="keep the restored scratch copy instead of deleting it")
    parser.add_argument(
        "--sample-size", type=int, default=SAMPLE_SIZE, help=f"rows per sampled table (default: {SAMPLE_SIZE})"
    )
    parser.add_argument("--quiet", action="store_true", help="print only the result line")
    args = parser.parse_args(argv)

    if args.sample_size < 1:
        parser.error("--sample-size must be >= 1")

    try:
        report = run_drill(args.source, scratch_root=args.scratch, keep=args.keep, sample_size=args.sample_size)
    except UnreadableInput as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.quiet:
        print("PASS" if report.ok else f"FAIL: {report.error or 'verification mismatch'}")
    else:
        print(format_report(report, args.sample_size))
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
