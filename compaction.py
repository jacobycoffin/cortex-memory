"""Archive compaction: collapse archived memories into summaries, keep originals cold.

Archived memories are out of recall but still occupy the ``memories`` table one
row per note — thousands of rows nobody reads. This module turns them into a
small number of per-group summaries and moves the originals, together with
their edges, access history and version rows, into a gzipped cold-storage file
with a sha256 manifest. ``restore`` puts every byte back.

Rules this follows (see PLAN.md):
  - archive, don't erase: nothing is deleted without a manifest that restores it;
  - one command back: ``python -m cortex.compaction restore <manifest>``;
  - never touch a live fact: only rows already ``state='archived'`` and not
    pinned or protected are candidates, and the delete re-checks that predicate
    so a memory reactivated mid-run survives.

Usage:
  python -m cortex.compaction --db cortex.db                 # dry run, prints plan
  python -m cortex.compaction --db cortex.db --apply --vacuum
  python -m cortex.compaction restore --db cortex.db cold/archive-...manifest.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# Tables exported alongside the memory row. Every one of them cascades from
# ``memories``, so a delete that skips the export is a delete that loses data.
EXPORT_TABLES = ("edges", "access_log", "memory_versions")
SUMMARY_SOURCE_TYPE = "archive_summary"
MANIFEST_VERSION = 1


# ---------------------------------------------------------------- utilities


def _now(now: datetime | None = None) -> datetime:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")


def _stamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%SZ")


def _content_hash(content: str) -> str:
    return hashlib.sha256(" ".join((content or "").split()).casefold().encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]


def _db_bytes(conn: sqlite3.Connection) -> dict[str, int]:
    page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
    page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
    freelist = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
    return {
        "page_size": page_size,
        "pages": page_count,
        "free_pages": freelist,
        "bytes": page_size * (page_count - freelist),
    }


def _require_foreign_keys(conn: sqlite3.Connection) -> None:
    if not int(conn.execute("PRAGMA foreign_keys").fetchone()[0]):
        raise RuntimeError("foreign_keys is off: cascades would not fire, refusing to compact")


# ---------------------------------------------------------------- planning


def _candidate_ids(
    conn: sqlite3.Connection,
    *,
    older_than_days: int | None,
    limit: int | None,
    now: datetime,
) -> list[str]:
    where = [
        "state='archived'",
        "pinned=0",
        "protected=0",
        "coalesce(source_type,'')<>?",
    ]
    params: list[Any] = [SUMMARY_SOURCE_TYPE]
    if older_than_days is not None:
        where.append("updated_at <= ?")
        params.append(_iso(now - timedelta(days=int(older_than_days))))
    sql = f"SELECT id FROM memories WHERE {' AND '.join(where)} ORDER BY created_at, id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    return [str(row[0]) for row in conn.execute(sql, params)]


def _group_key(row: dict[str, Any]) -> str:
    month = str(row.get("created_at") or "")[:7] or "unknown"
    return f"{month}|{row.get('kind') or 'unknown'}|{row.get('source_type') or 'unknown'}"


def plan(
    conn: sqlite3.Connection,
    *,
    older_than_days: int | None = None,
    limit: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Describe what compaction would do, changing nothing."""
    moment = _now(now)
    ids = _candidate_ids(conn, older_than_days=older_than_days, limit=limit, now=moment)
    if not ids:
        return {"candidates": 0, "groups": {}, "content_chars": 0, "ids": [], "bytes": _db_bytes(conn)}
    rows = _fetch_memories(conn, ids)
    groups: dict[str, int] = {}
    chars = 0
    for row in rows:
        groups[_group_key(row)] = groups.get(_group_key(row), 0) + 1
        chars += len(str(row.get("content") or ""))
    return {
        "candidates": len(ids),
        "groups": groups,
        "content_chars": chars,
        "ids": ids,
        "bytes": _db_bytes(conn),
    }


# ---------------------------------------------------------------- export


def _fetch_memories(conn: sqlite3.Connection, ids: Sequence[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for start in range(0, len(ids), 400):
        chunk = list(ids[start : start + 400])
        placeholders = ",".join("?" for _ in chunk)
        out.extend(
            {k: row[k] for k in row.keys()}
            for row in conn.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", chunk)
        )
    return out


def _fetch_related(conn: sqlite3.Connection, ids: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    related: dict[str, list[dict[str, Any]]] = {table: [] for table in EXPORT_TABLES}
    for start in range(0, len(ids), 400):
        chunk = list(ids[start : start + 400])
        placeholders = ",".join("?" for _ in chunk)
        related["edges"].extend(
            {k: row[k] for k in row.keys()}
            for row in conn.execute(
                f"SELECT * FROM edges WHERE src_id IN ({placeholders}) OR dst_id IN ({placeholders})",
                chunk + chunk,
            )
        )
        for table in ("access_log", "memory_versions"):
            related[table].extend(
                {k: row[k] for k in row.keys()}
                for row in conn.execute(f"SELECT * FROM {table} WHERE memory_id IN ({placeholders})", chunk)
            )
    return related


def _write_cold(cold_path: Path, memories: list[dict[str, Any]], related: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counts = {"memories": len(memories), **{k: len(v) for k, v in related.items()}}
    cold_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cold_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in memories:
            handle.write(json.dumps({"table": "memories", "row": row}, ensure_ascii=False, default=str) + "\n")
        for table, rows in related.items():
            for row in rows:
                handle.write(json.dumps({"table": table, "row": row}, ensure_ascii=False, default=str) + "\n")
    return counts


def _summary_content(group: str, rows: list[dict[str, Any]], manifest_name: str, digest: str) -> str:
    month, kind, source_type = (group.split("|", 2) + ["", "", ""])[:3]
    examples = [str(row.get("content") or "").strip() for row in rows[:3] if str(row.get("content") or "").strip()]
    sample = "; ".join(example[:80] for example in examples)
    return (
        f"Archive summary {month}: {len(rows)} archived {kind} memories from {source_type} "
        f"were compacted on {_stamp(_now())}. Originals and their edges, access history and versions are "
        f"preserved in cold storage file {manifest_name} (ids digest {digest}). "
        f"Restore with: python -m cortex.compaction restore {manifest_name.replace('.manifest.json', '.jsonl.gz')}"
        + (f". Examples: {sample}" if sample else "")
    )


def _insert_summaries(conn: sqlite3.Connection, grouped: dict[str, list[dict[str, Any]]], manifest_path: Path, digest: str, now: datetime) -> list[str]:
    summary_ids: list[str] = []
    for group, rows in sorted(grouped.items()):
        summary_id = f"archive-summary-{hashlib.sha256((manifest_path.name + group).encode()).hexdigest()[:24]}"
        content = _summary_content(group, rows, manifest_path.name, digest)
        conn.execute(
            """INSERT OR IGNORE INTO memories(
                   id, kind, content, content_hash, source_type, source_category,
                   origin_source_category, created_at, updated_at, observed_at,
                   extraction_method, confidence, importance, trust, state,
                   record_role, role_method, pinned, protected
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0)""",
            (
                summary_id,
                "operational",
                content,
                _content_hash(content),
                SUMMARY_SOURCE_TYPE,
                "SYSTEM",
                "SYSTEM",
                _iso(now),
                _iso(now),
                _iso(now),
                "archive_compaction",
                0.9,
                0.1,
                0.9,
                "archived",
                "canonical",
                "archive_compaction",
            ),
        )
        summary_ids.append(summary_id)
    return summary_ids


def _fk_edges(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """Every (table, column, on_delete) that points at ``memories.id``, read from the schema.

    Done live rather than hardcoded: the schema grows, and a hardcoded cascade list
    would silently orphan whatever table someone adds next.
    """
    tables = [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    edges: list[tuple[str, str, str]] = []
    for table in tables:
        for fk in conn.execute(f"PRAGMA foreign_key_list({table})"):
            if str(fk[2]) != "memories":
                continue
            edges.append((table, str(fk[3]), str(fk[6] or "NO ACTION").upper()))
    return edges


def _blocked_ids(conn: sqlite3.Connection, ids: Sequence[str]) -> set[str]:
    """Candidates another table holds with RESTRICT/NO ACTION: kept, never force-deleted."""
    edges = _fk_edges(conn)
    blocked: set[str] = set()
    for start in range(0, len(ids), 400):
        chunk = list(ids[start : start + 400])
        marks = ",".join("?" for _ in chunk)
        for table, column, action in edges:
            if action not in {"RESTRICT", "NO ACTION"}:
                continue
            if table == "memories" and column == "supersedes_id":
                # A chain inside the candidate set is fine; a reference from a
                # memory that keeps living is not.
                sql = (
                    f"SELECT DISTINCT supersedes_id FROM memories WHERE supersedes_id IN ({marks}) "
                    f"AND id NOT IN ({marks})"
                )
                blocked.update(str(row[0]) for row in conn.execute(sql, chunk + chunk) if row[0])
                continue
            if table == "memories":
                continue
            sql = f"SELECT DISTINCT {column} FROM {table} WHERE {column} IN ({marks})"
            blocked.update(str(row[0]) for row in conn.execute(sql, chunk) if row[0])
    return blocked


def _delete_originals(conn: sqlite3.Connection, ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """Delete guarded rows on purpose, then cascade by schema metadata.

    Foreign keys are enforced by us here, not by SQLite, because per-row FK checks
    over every child table make a 4k-row compaction take minutes. The predicate on
    every DELETE keeps the rule intact: only a row that is still archived, unpinned
    and unprotected can go, and it re-checks at delete time so a memory reactivated
    between selection and delete survives.
    """
    blocked = _blocked_ids(conn, ids)
    deleted: list[str] = []
    skipped: list[str] = [memory_id for memory_id in ids if memory_id in blocked]
    for memory_id in ids:
        if memory_id in blocked:
            continue
        cursor = conn.execute(
            "DELETE FROM memories WHERE id=? AND state='archived' AND pinned=0 AND protected=0",
            (memory_id,),
        )
        if cursor.rowcount:
            deleted.append(memory_id)
        else:
            skipped.append(memory_id)

    for start in range(0, len(deleted), 400):
        chunk = list(deleted[start : start + 400])
        marks = ",".join("?" for _ in chunk)
        for table, column, action in _fk_edges(conn):
            if table == "memories":
                continue
            if action == "CASCADE":
                conn.execute(f"DELETE FROM {table} WHERE {column} IN ({marks})", chunk)
            elif action == "SET NULL":
                conn.execute(f"UPDATE {table} SET {column}=NULL WHERE {column} IN ({marks})", chunk)
    return deleted, skipped


def _orphan_count(conn: sqlite3.Connection, deleted: Sequence[str]) -> int:
    """Rows still pointing at a compacted id through a CASCADE column: must be zero."""
    orphans = 0
    for start in range(0, len(deleted), 400):
        chunk = list(deleted[start : start + 400])
        marks = ",".join("?" for _ in chunk)
        for table, column, action in _fk_edges(conn):
            if table == "memories" or action != "CASCADE":
                continue
            orphans += int(
                conn.execute(f"SELECT count(*) FROM {table} WHERE {column} IN ({marks})", chunk).fetchone()[0]
            )
    return orphans


def _log(conn: sqlite3.Connection, details: dict[str, Any], *, dry_run: bool, now: datetime) -> None:
    conn.execute(
        "INSERT INTO maintenance_log(action,details,dry_run,created_at) VALUES(?,?,?,?)",
        ("archive_compaction", json.dumps(details, sort_keys=True, default=str), int(dry_run), _iso(now)),
    )


def compact(
    conn: sqlite3.Connection,
    out_dir: str | Path,
    *,
    older_than_days: int | None = None,
    limit: int | None = None,
    apply: bool = False,
    vacuum: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Collapse archived rows into summaries, originals to cold storage. Dry run by default."""
    moment = _now(now)
    _require_foreign_keys(conn)
    overview = plan(conn, older_than_days=older_than_days, limit=limit, now=moment)
    if not apply or not overview["candidates"]:
        result = {"applied": False, **{k: overview[k] for k in ("candidates", "groups", "content_chars", "bytes")}}
        if not apply:
            _log(conn, result, dry_run=True, now=moment)
            conn.commit()
        return result

    ids = overview["ids"]
    before = overview["bytes"]
    memories = _fetch_memories(conn, ids)
    related = _fetch_related(conn, ids)

    # Cascades are applied explicitly from schema metadata inside the transaction,
    # so SQLite's own per-row checks are off for the duration and restored after.
    conn.commit()
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("BEGIN IMMEDIATE")
    try:
        deleted, skipped = _delete_originals(conn, ids)
        survivors = set(deleted)
        memories = [row for row in memories if str(row["id"]) in survivors]
        related = {
            table: [row for row in rows if str(row.get("src_id") or row.get("memory_id") or "") in survivors
                    or str(row.get("dst_id") or "") in survivors]
            for table, rows in related.items()
        }

        out_dir = Path(out_dir).expanduser()
        cold_path = out_dir / f"archive-{_stamp(moment)}.jsonl.gz"
        manifest_path = out_dir / f"archive-{_stamp(moment)}.manifest.json"
        counts = _write_cold(cold_path, memories, related)
        digest = hashlib.sha256("\n".join(sorted(survivors)).encode("utf-8")).hexdigest()[:16]

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in memories:
            grouped.setdefault(_group_key(row), []).append(row)
        summary_ids = _insert_summaries(conn, grouped, manifest_path, digest, moment)

        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "created_at": _iso(moment),
            "database": str(conn.execute("PRAGMA database_list").fetchone()[2]),
            "cold_file": cold_path.name,
            "sha256": _file_sha256(cold_path),
            "bytes": cold_path.stat().st_size,
            "rows": counts,
            "groups": {group: len(rows) for group, rows in grouped.items()},
            "summary_ids": summary_ids,
            "skipped_restricted": len(skipped),
            "db_bytes_before": before["bytes"],
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))

        after = _db_bytes(conn)
        details = {
            "applied": True,
            "candidates": len(ids),
            "deleted": len(deleted),
            "skipped_restricted": len(skipped),
            "summaries": len(summary_ids),
            "cold_file": str(cold_path),
            "manifest": str(manifest_path),
            "sha256": manifest["sha256"],
            "rows": counts,
            "bytes_before": before["bytes"],
            "bytes_after_delete": after["bytes"],
        }
        _log(conn, details, dry_run=False, now=moment)
        orphans = _orphan_count(conn, deleted)
        if orphans:
            raise RuntimeError(f"refusing to commit: {orphans} rows would be orphaned")
        conn.commit()
    except Exception:
        conn.rollback()
        conn.execute("PRAGMA foreign_keys=ON")
        raise
    conn.execute("PRAGMA foreign_keys=ON")

    if vacuum:
        conn.isolation_level = None
        conn.execute("VACUUM")
        conn.isolation_level = ""
        details["bytes_after_vacuum"] = _db_bytes(conn)["bytes"]
        # The row counts changed after the log entry; append the reclaimed number.
        _log(conn, {"vacuum": True, "bytes_after_vacuum": details["bytes_after_vacuum"]}, dry_run=False, now=moment)
        conn.commit()
    return details


# ---------------------------------------------------------------- restore


def _iter_cold(cold_path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(cold_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def restore(
    conn: sqlite3.Connection,
    manifest_path: str | Path,
    *,
    remove_summaries: bool = True,
) -> dict[str, Any]:
    """Put a compacted archive back exactly as exported, then drop its summaries."""
    manifest_path = Path(manifest_path).expanduser()
    manifest = json.loads(manifest_path.read_text())
    cold_path = manifest_path.parent / str(manifest["cold_file"])
    if not cold_path.exists():
        raise FileNotFoundError(f"cold file missing: {cold_path}")
    actual = _file_sha256(cold_path)
    if actual != manifest.get("sha256"):
        raise ValueError(f"cold file sha256 mismatch: manifest {manifest.get('sha256')} != file {actual}")

    _require_foreign_keys(conn)
    restored: dict[str, int] = {}
    conn.execute("BEGIN IMMEDIATE")
    try:
        ordered = {"memories": [], **{table: [] for table in EXPORT_TABLES}}
        for entry in _iter_cold(cold_path):
            table = str(entry.get("table"))
            if table in ordered:
                ordered[table].append(entry["row"])
        # Memories first: every other table references them. ``supersedes_id`` is a
        # self-reference, so it is restored in a second pass; otherwise a chain whose
        # rows land out of order fails the foreign key.
        deferred_supersedes: list[tuple[str, str]] = []
        for table in ("memories", *EXPORT_TABLES):
            rows = ordered[table]
            if not rows:
                restored[table] = 0
                continue
            target = set(_columns(conn, table))
            inserted = 0
            for row in rows:
                usable = {k: v for k, v in row.items() if k in target}
                if table == "memories" and usable.get("supersedes_id"):
                    deferred_supersedes.append((str(usable["id"]), str(usable.pop("supersedes_id"))))
                # Primary keys are kept so a repeated restore is idempotent:
                # INSERT OR IGNORE then dedupes instead of duplicating history.
                if not usable:
                    continue
                keys = ",".join(usable)
                marks = ",".join("?" for _ in usable)
                cursor = conn.execute(f"INSERT OR IGNORE INTO {table}({keys}) VALUES({marks})", list(usable.values()))
                inserted += max(0, cursor.rowcount)
            restored[table] = inserted

        for memory_id, supersedes_id in deferred_supersedes:
            conn.execute(
                "UPDATE memories SET supersedes_id=? WHERE id=? AND supersedes_id IS NULL "
                "AND EXISTS (SELECT 1 FROM memories WHERE id=?)",
                (supersedes_id, memory_id, supersedes_id),
            )

        removed = 0
        if remove_summaries:
            for summary_id in manifest.get("summary_ids", []):
                cursor = conn.execute(
                    "DELETE FROM memories WHERE id=? AND state='archived' AND source_type=?",
                    (summary_id, SUMMARY_SOURCE_TYPE),
                )
                removed += max(0, cursor.rowcount)
        _log(
            conn,
            {"restore": manifest_path.name, "restored": restored, "summaries_removed": removed},
            dry_run=False,
            now=_now(),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"restored": restored, "summaries_removed": removed, "manifest": str(manifest_path)}


# ---------------------------------------------------------------- CLI


def _connect(db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = "compact"
    if argv and argv[0] == "restore":
        command = "restore"
        argv = argv[1:]
    parser = argparse.ArgumentParser(prog="python -m cortex.compaction")
    parser.add_argument("manifest", nargs="?", help="manifest path (restore only)")
    parser.add_argument("--db", required=True)
    parser.add_argument("--out-dir", default=None, help="cold storage directory (default: <db dir>/cold)")
    parser.add_argument("--older-than-days", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--apply", action="store_true", help="actually compact (default is a dry run)")
    parser.add_argument("--vacuum", action="store_true", help="VACUUM after compaction to reclaim pages")
    args = parser.parse_args(argv)

    conn = _connect(args.db)
    try:
        if command == "restore":
            if not args.manifest:
                parser.error("restore needs a manifest path")
            print(json.dumps(restore(conn, args.manifest), indent=1))
            return 0
        out_dir = args.out_dir or str(Path(args.db).expanduser().resolve().parent / "cold")
        result = compact(
            conn,
            out_dir,
            older_than_days=args.older_than_days,
            limit=args.limit,
            apply=args.apply,
            vacuum=args.vacuum,
        )
        print(json.dumps(result, indent=1, default=str))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
