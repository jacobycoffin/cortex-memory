"""Archive compaction: reclaim the derivative rows retired memories no longer need (#98).

Measured on the live store before writing this:

    retired memories (archived/tombstoned/suppressed)   4,530
    their rows in memory_features                     660,502 of 1,203,031  (55%)
    memory_presentations                                4,530 of     8,371
    access_log                                         12,340 of    49,013
    document_chunks                                     4,163 of     7,191

A retired memory still carries its feature vectors, presentation rows, chunk embeddings and
context outcomes — the machinery that exists so it can compete for recall, none of which matters
once it has been retired. That is where the bytes are, and it is what compaction reclaims.

Three rules:

1. **Content is never touched.** This is the counterpart to the RTBF purge: compaction changes
   nothing a person would miss. The memory stays, its content stays, `restore()` still works.
2. **Audit history is never touched.** `lifecycle_events`, `memory_versions`, `deletions`,
   `prune_receipts` and `rtbf_certificates` are the record of what happened; deleting them to save
   space would trade the ability to answer questions for bytes.
3. **Use evidence is opt-in.** `access_log` and `usage_records` are history, not cache — they are
   kept by default and `include_history=True` removes them, because someone tuning retention might
   want the space and should have to say so.

Bytes only return to the filesystem after a `VACUUM`, which needs exclusive access, so it is a
separate opt-in step rather than something a compaction silently does to a live database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RETIRED_STATES = ("archived", "tombstoned", "suppressed")

# Never compacted: these answer "what happened", and space is not worth losing the answer.
KEEP = frozenset(
    {
        "memories",
        "lifecycle_events",
        "memory_versions",
        "deletions",
        "prune_receipts",
        "rtbf_certificates",
        "legal_holds",
        "suppressions",
        "triage_batches",
        "triage_kept",
        "forget_previews",
        "core_members",
        "prospective_items",
    }
)
# History rather than cache: kept unless the caller asks otherwise.
HISTORY = frozenset({"access_log", "usage_records", "access_history"})
# Structural for recall: these are not caches, they are the rows candidate generation reads.
# Compacting them made a restored memory present-but-invisible — the worst of both states — and
# the failing test that caught it is `test_a_restored_memory_is_whole_again`.
STRUCTURAL = frozenset(
    {
        "memory_recall_memberships",
        "memory_context_terms",
        "memory_neighborhood_memberships",
        # The graph is content, not cache: a restored memory's associations cannot be rebuilt
        # from its text, so compacting edges would quietly make restoring lossy.
        "edges",
        "edge_evidence",
        "memory_dependencies",
    }
)


@dataclass
class Plan:
    retired: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    bytes: int = 0

    @property
    def rows(self) -> int:
        return sum(self.counts.values())

    def describe(self) -> str:
        lines = [
            f"{self.retired} retired memories hold {self.rows} derivative rows "
            f"(~{self.bytes / (1024 * 1024):.1f} MB of table space)"
        ]
        for table, count in sorted(self.counts.items(), key=lambda item: -item[1])[:12]:
            lines.append(f"  {table:30} {count}")
        return "\n".join(lines)


def _tables(store: Any) -> dict[str, str]:
    """Derived tables that reference a memory, minus the ones we must keep."""
    from . import rtbf

    found: dict[str, str] = {}
    for table, columns in rtbf.dependent_tables(store).items():
        if table in KEEP or table in HISTORY or table in STRUCTURAL:
            continue
        found[table] = columns[0]
    return found


def _table_bytes(store: Any) -> dict[str, int]:
    try:
        rows = store._conn.execute(
            "SELECT name, SUM(pgsize) AS bytes FROM dbstat GROUP BY name"
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    return {str(row["name"]): int(row["bytes"] or 0) for row in rows}


def _retired_filter(include_history: bool) -> str:
    marks = ",".join("?" for _ in RETIRED_STATES)
    clause = (
        f"memory_id IN (SELECT id FROM memories WHERE state IN ({marks}))"
    )
    return clause


def plan(store: Any, *, include_history: bool = False) -> Plan:
    """What compaction would remove. Read-only."""
    marks = ",".join("?" for _ in RETIRED_STATES)
    retired = int(
        store._conn.execute(
            f"SELECT COUNT(*) FROM memories WHERE state IN ({marks})", RETIRED_STATES
        ).fetchone()[0]
        or 0
    )
    tables = _tables(store)
    if include_history:
        tables.update({name: "memory_id" for name in HISTORY})
    sizes = _table_bytes(store)
    result = Plan(retired=retired)
    for table, column in tables.items():
        total = 0
        for col in rtbf_columns(store, table):
            try:
                counts = store._conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {col} IN "
                    f"(SELECT id FROM memories WHERE state IN ({marks}))",
                    RETIRED_STATES,
                ).fetchone()
                total += int(counts[0] or 0) if counts else 0
            except Exception:  # noqa: BLE001 - a table that will not read is skipped
                continue
        if not total:
            continue
        result.counts[table] = total
        rows_total = int(
            store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] or 0
        )
        # Attribute table bytes proportionally: the honest estimate without per-row sizes.
        if rows_total and table in sizes:
            result.bytes += int(sizes[table] * (total / rows_total))
    return result


def rtbf_columns(store: Any, table: str) -> list[str]:
    from . import rtbf

    return rtbf.dependent_tables(store).get(table, [])


def compact(store: Any, *, include_history: bool = False, at: Any = None) -> dict[str, int]:
    """Delete derivative rows of retired memories. Content and audit history are untouched."""
    marks = ",".join("?" for _ in RETIRED_STATES)
    tables = _tables(store)
    if include_history:
        tables.update({name: "memory_id" for name in HISTORY})
    removed: dict[str, int] = {}
    with store.transaction() as conn:
        for table in tables:
            for column in rtbf_columns(store, table):
                try:
                    cursor = conn.execute(
                        f"DELETE FROM {table} WHERE {column} IN "
                        f"(SELECT id FROM memories WHERE state IN ({marks}))",
                        RETIRED_STATES,
                    )
                    if cursor.rowcount and cursor.rowcount > 0:
                        removed[table] = removed.get(table, 0) + int(cursor.rowcount)
                except Exception:  # noqa: BLE001 - one table must not abort a compaction
                    continue
    if removed:
        from . import prune_receipts

        prune_receipts.record(
            store, "compaction", "derivatives", [],
            summary=", ".join(f"{table} {count}" for table, count in sorted(removed.items())),
        )
    return removed


def vacuum(store: Any) -> bool:
    """Return the freed pages to the filesystem. Needs exclusive access; opt-in for that reason."""
    try:
        store._conn.execute("VACUUM")
    except Exception:  # noqa: BLE001 - a busy database simply does not compact today
        return False
    return True


def report(store: Any) -> dict[str, Any]:
    from . import storage_cost

    planned = plan(store)
    return {
        "retired": planned.retired,
        "rows": planned.rows,
        "bytes": planned.bytes,
        "file_bytes": storage_cost.file_bytes(store),
        "tables": sorted(planned.counts.items(), key=lambda item: -item[1])[:12],
    }
