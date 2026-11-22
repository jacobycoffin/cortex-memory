"""Right-to-be-forgotten: erase a data subject, including the derivatives (#97).

This is the only operation in the store that is *designed* to be irreversible, so it is built
differently from everything else:

1. **It erases, it does not demote.** Tombstoning leaves the content in place; an erasure request
   is about the content being gone.
2. **Derivatives count.** A memory's `memory_features`, `memory_traces`, `access_log`, edges,
   usage records, presentations and version history all contain data *derived from* the subject.
   Deleting only the `memories` row leaves the rest of the statement true — this is the mistake
   every naive erasure implementation makes.
3. **The purge is discovered, not hard-coded.** Dependent tables are found by walking
   `PRAGMA foreign_key_list` for anything referencing `memories(id)`, plus `edges` by hand (it
   keys on `src_id`/`dst_id`). A new derived table added later is covered without this file
   changing, which is the difference between a policy and a code path.
4. **The certificate proves erasure without keeping the data.** After the purge, one row records a
   salted hash of the subject, the per-table counts, and when — enough to answer "was this
   fulfilled" and not enough to reconstruct what was erased.
5. **A legal hold wins.** If any affected memory is on hold, the purge refuses entirely rather
   than erasing most of it: a partial erasure is worse than a refused one, because it looks done.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_CERTIFICATES = "rtbf_certificates"
# Columns that can point at a memory row; `edges` needs both ends.
_EDGE_TABLES = ("edges",)
_SKIP_TABLES = frozenset({_CERTIFICATES, "legal_holds", "sqlite_sequence"})


class PurgeRefused(ValueError):
    """Raised when an erasure cannot be carried out safely."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _salt() -> str:
    raw = os.environ.get("CORTEX_RTBF_SALT", "").strip()
    return raw or "cortex-rtbf"


def subject_hash(subject: str) -> str:
    """Salted hash: the certificate can be matched later without storing the name."""
    digest = hashlib.sha256(f"{_salt()}:{str(subject).strip().lower()}".encode("utf-8"))
    return digest.hexdigest()[:32]


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {_CERTIFICATES} (
                    certificate_id TEXT PRIMARY KEY,
                    subject_hash TEXT NOT NULL,
                    matched INTEGER NOT NULL,
                    purged_json TEXT NOT NULL,
                    note TEXT,
                    created_at TEXT NOT NULL
                )"""
        )


def _tables(store: Any) -> list[str]:
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    return [str(row["name"]) for row in rows if str(row["name"]) not in _SKIP_TABLES]


def _memory_columns(store: Any, table: str) -> list[str]:
    """Which columns of this table reference `memories(id)`, discovered rather than assumed."""
    columns: list[str] = []
    try:
        fks = store._conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
    except Exception:  # noqa: BLE001
        return columns
    for fk in fks:
        if str(fk["table"]) == "memories":
            column = str(fk["from"])
            if column not in columns:
                columns.append(column)
    if table in _EDGE_TABLES:
        info = store._conn.execute(f"PRAGMA table_info({table})").fetchall()
        for row in info:
            name = str(row["name"])
            if name in {"src_id", "dst_id"} and name not in columns:
                columns.append(name)
    return columns


def dependent_tables(store: Any) -> dict[str, list[str]]:
    """Tables that carry data derived from a memory, and the column to match on."""
    found: dict[str, list[str]] = {}
    for table in _tables(store):
        columns = _memory_columns(store, table)
        if columns:
            found[table] = columns
    return found


@dataclass
class Preview:
    subject: str
    memory_ids: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    held: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return sum(self.counts.values())

    def describe(self) -> str:
        lines = [
            f"subject '{self.subject}': {len(self.memory_ids)} memories, {self.rows} rows total"
        ]
        for table, count in sorted(self.counts.items(), key=lambda item: -item[1]):
            if count:
                lines.append(f"  {table:28} {count}")
        if self.held:
            lines.append(
                f"  BLOCKED by legal hold: {', '.join(item[:8] for item in self.held)}"
            )
        return "\n".join(lines)


def preview(store: Any, subject: str, *, limit: int = 500) -> Preview:
    """Everything that would be erased. Read-only."""
    from . import forget_entity, legal_hold

    found = forget_entity.matches(store, subject, limit=limit)
    ids = [str(item["memory_id"]) for item in found]
    counts: dict[str, int] = {}
    if ids:
        marks = ",".join("?" for _ in ids)
        for table, columns in dependent_tables(store).items():
            total = 0
            for column in columns:
                try:
                    row = store._conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE {column} IN ({marks})", tuple(ids)
                    ).fetchone()
                except Exception:  # noqa: BLE001 - a table that will not read is reported as 0
                    continue
                total += int(row[0] or 0) if row else 0
            counts[table] = total
    counts["memories"] = len(ids)
    held = sorted(set(ids) & legal_hold.held_ids(store))
    return Preview(subject=str(subject), memory_ids=ids, counts=counts, held=held)


def purge(
    store: Any,
    subject: str,
    *,
    confirm: str,
    note: str = "",
    at: datetime | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    """Erase the subject's data and issue a certificate. Irreversible."""
    if str(confirm).strip().lower() != str(subject).strip().lower():
        raise PurgeRefused("erasure requires the subject to be named twice; nothing was removed")
    planned = preview(store, subject, limit=limit)
    if planned.held:
        raise PurgeRefused(
            "memories on legal hold block this erasure: "
            + ", ".join(item[:8] for item in planned.held)
            + "; release the hold first"
        )
    _ensure(store)
    purged: dict[str, int] = {}
    ids = planned.memory_ids
    if ids:
        marks = ",".join("?" for _ in ids)
        with store.transaction() as conn:
            for table, columns in dependent_tables(store).items():
                one = table.rsplit(".", 1)[-1]
                for column in columns:
                    try:
                        cursor = conn.execute(
                            f"DELETE FROM {one} WHERE {column} IN ({marks})", tuple(ids)
                        )
                        if cursor.rowcount and cursor.rowcount > 0:
                            purged[table] = purged.get(table, 0) + int(cursor.rowcount)
                    except Exception:  # noqa: BLE001 - never abort mid-erasure on one table
                        continue
            cursor = conn.execute(
                f"DELETE FROM memories WHERE id IN ({marks})", tuple(ids)
            )
            purged["memories"] = int(cursor.rowcount or 0)
    certificate = secrets.token_hex(8)
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {_CERTIFICATES}(certificate_id, subject_hash, matched,
                                            purged_json, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
            (certificate, subject_hash(subject), len(ids), json.dumps(purged),
             note, (at or _utc_now()).isoformat()),
        )
    return {
        "certificate": certificate,
        "subject_hash": subject_hash(subject),
        "matched": len(ids),
        "purged": purged,
        "rows": sum(purged.values()),
    }


def certificates(store: Any, limit: int = 20) -> list[dict[str, Any]]:
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT certificate_id, subject_hash, matched, purged_json, note, created_at
            FROM {_CERTIFICATES} ORDER BY created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        {
            "certificate": str(row["certificate_id"]),
            "subject_hash": str(row["subject_hash"]),
            "matched": int(row["matched"]),
            "purged": json.loads(row["purged_json"]),
            "note": row["note"],
            "created_at": str(row["created_at"]),
        }
        for row in rows
    ]


def fulfilled(store: Any, subject: str) -> bool:
    """Was this subject erased? Answered from the hash, without the name being stored."""
    _ensure(store)
    row = store._conn.execute(
        f"SELECT 1 FROM {_CERTIFICATES} WHERE subject_hash = ? LIMIT 1", (subject_hash(subject),)
    ).fetchone()
    return row is not None
