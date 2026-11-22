"""Where the bytes actually are, and what pruning would really reclaim (#102).

The intuitive model — "memories take space, pruning memories frees it" — is wrong in this store by
two orders of magnitude. Measured on the live database:

    file total            840 MB
    memory content         ~7 MB   (all 8,371 memories, every kind)
    memory_features       ~250 MB  (the table plus its two indexes)

So a report that only counted content bytes would tell you to prune memories and free 7 MB. This
module reports the truth in two layers:

- **per-kind** cost of the *content*, which is what a person thinks of as the memory;
- **per-table** cost from `dbstat`, which is where the bytes are.

And it draws the line the retention features keep tripping over: **archiving, cooling and
tombstoning free nothing.** They change addressability, not size. Only a hard delete removes rows,
and even then the derived feature rows are what actually shrink the file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

CONTENT_OVERHEAD_BYTES = 320  # row + indexes + FTS entry, measured on this schema


def file_bytes(store: Any) -> int:
    try:
        path = getattr(store, "path", None) or getattr(store, "_path", None)
        if path:
            return int(os.path.getsize(str(path)))
    except OSError:
        pass
    try:
        page_count = store._conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = store._conn.execute("PRAGMA page_size").fetchone()[0]
        return int(page_count) * int(page_size)
    except Exception:  # noqa: BLE001
        return 0


def _dbstat(store: Any) -> dict[str, int]:
    """Bytes per table and index, when SQLite was built with dbstat."""
    try:
        rows = store._conn.execute(
            "SELECT name, SUM(pgsize) AS bytes FROM dbstat GROUP BY name"
        ).fetchall()
    except Exception:  # noqa: BLE001 - dbstat is optional in some builds
        return {}
    return {str(row["name"]): int(row["bytes"] or 0) for row in rows}


def _index_owner(store: Any) -> dict[str, str]:
    try:
        rows = store._conn.execute(
            "SELECT name, tbl_name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    return {str(row["name"]): str(row["tbl_name"]) for row in rows}


@dataclass
class TableCost:
    table: str
    bytes: int

    def describe(self, total: int) -> str:
        share = (self.bytes / total) if total else 0.0
        return f"{self.table:28} {human(self.bytes):>10}  {share:5.1%}"


@dataclass
class KindCost:
    kind: str
    memories: int
    content_bytes: int

    @property
    def average(self) -> int:
        return int(self.content_bytes / self.memories) if self.memories else 0


@dataclass
class Reclaim:
    archiveable: int = 0
    archive_bytes: int = 0
    deletable: int = 0
    delete_bytes: int = 0
    notes: list[str] = field(default_factory=list)


def human(value: int | float) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


def tables(store: Any, *, limit: int = 10) -> list[TableCost]:
    raw = _dbstat(store)
    if not raw:
        return []
    owners = _index_owner(store)
    merged: dict[str, int] = {}
    for name, size in raw.items():
        # Roll each index into the table it belongs to: "memory_features is 250MB" is the useful
        # statement, not "idx_memory_features_feature is 88MB" three times over.
        table = owners.get(name, name)
        merged[table] = merged.get(table, 0) + size
    ordered = sorted(merged.items(), key=lambda item: -item[1])
    return [TableCost(table=name, bytes=size) for name, size in ordered[:limit]]


def by_kind(store: Any) -> list[KindCost]:
    rows = store._conn.execute(
        """SELECT COALESCE(kind, '') AS kind, COUNT(*) AS n,
                  COALESCE(SUM(LENGTH(content)), 0) AS bytes
           FROM memories GROUP BY COALESCE(kind, '') ORDER BY bytes DESC"""
    ).fetchall()
    return [
        KindCost(kind=str(row["kind"] or "unknown"), memories=int(row["n"]),
                 content_bytes=int(row["bytes"]))
        for row in rows
    ]


def reclaimable(store: Any, *, at: Any = None) -> Reclaim:
    """What the retention features would free — and what they would not."""
    from . import triage

    result = Reclaim()
    triaged = triage.candidates(store, limit=100_000, at=at)
    if triaged:
        result.archiveable = len(triaged)
        result.archive_bytes = sum(
            len(str(item.get("content") or "")) + CONTENT_OVERHEAD_BYTES for item in triaged
        )
    row = store._conn.execute(
        """SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(content)), 0) AS bytes
           FROM memories WHERE state IN ('archived','tombstoned','suppressed')"""
    ).fetchone()
    result.deletable = int(row["n"] or 0)
    result.delete_bytes = int(row["bytes"] or 0) + result.deletable * CONTENT_OVERHEAD_BYTES
    result.notes.append(
        "archiving and tombstoning free no bytes: they change addressability, not size"
    )
    result.notes.append(
        "only a hard delete removes rows, and the derived feature rows are what actually shrink "
        "the file"
    )
    return result


def report(store: Any, *, at: Any = None, limit: int = 8) -> dict[str, Any]:
    total = file_bytes(store)
    kinds = by_kind(store)
    return {
        "file_bytes": total,
        "file_human": human(total),
        "content_bytes": sum(kind.content_bytes for kind in kinds),
        "tables": [(cost.table, cost.bytes, round(cost.bytes / total, 3) if total else 0.0)
                   for cost in tables(store, limit=limit)],
        "kinds": [
            {
                "kind": kind.kind,
                "memories": kind.memories,
                "bytes": kind.content_bytes,
                "average": kind.average,
                "share": round(kind.content_bytes / total, 4) if total else 0.0,
            }
            for kind in kinds
        ],
        "reclaim": reclaimable(store, at=at),
    }
