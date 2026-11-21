#!/usr/bin/env python3
"""Forecast how fast a Cortex database is growing.

A memory store that quietly doubles every few months is discovered when the
disk fills, not before. This reports where the bytes actually are, how many rows
each growing table gains per day, and what the file will weigh in 30/90/365
days at that rate. ``--sweep-trigger-mb`` turns the report into a check: exit 1
when the 90-day projection crosses a threshold, so a timer can trigger
maintenance before the disk is full.

    python3 scripts/storage_forecast.py --db ~/.hermes/cortex/cortex.db
    python3 scripts/storage_forecast.py --db cortex.db --json
    python3 scripts/storage_forecast.py --db cortex.db --sweep-trigger-mb 2048

The database is opened read-only. Exit codes: 0 report produced (and, with a
trigger, projection stays below it), 1 sweep trigger crossed, 2 unreadable DB.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

GROWTH_TABLES = ("memories", "edges", "access_log", "episodes")
DEFAULT_WINDOW_DAYS = 30
HORIZONS = (30, 90, 365)
SWEEP_HORIZON_DAYS = 90

ASSUMPTION = (
    "linear growth at the mean insert rate of the last {window} days, with each "
    "table's average bytes-per-row (table plus its indexes, measured today) held "
    "constant; deletions, consolidation and vacuuming are not modelled"
)


def default_db_path() -> Path:
    """Match ``cli.py``/``mcp_server.py`` resolution order."""
    explicit = os.environ.get("CORTEX_DB", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HERMES_HOME", "").strip()
    if home:
        return Path(home).expanduser() / "cortex" / "cortex.db"
    return Path.home() / ".cortex" / "cortex.db"


def connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in table_names(conn):
        try:
            counts[name] = int(conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0])
        except sqlite3.DatabaseError:
            # A virtual table that needs an extension it cannot load reports
            # nothing; it must not abort the whole forecast.
            counts[name] = 0
    return counts


def has_dbstat(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT count(*) FROM dbstat").fetchone()
        return True
    except sqlite3.DatabaseError:
        return False


def sizes_dbstat(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """Bytes per table from ``dbstat``, with the table's indexes charged to it."""
    pages: dict[str, int] = {}
    for row in conn.execute("SELECT name, sum(pgsize) AS b FROM dbstat GROUP BY name"):
        pages[row["name"]] = int(row["b"] or 0)

    index_owner: dict[str, str] = {}
    for row in conn.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index'"):
        index_owner[row["name"]] = row["tbl_name"]

    sizes: dict[str, dict[str, int]] = {}
    for name in table_names(conn):
        table_bytes = pages.get(name, 0)
        index_bytes = sum(
            bytes_ for idx, bytes_ in pages.items() if index_owner.get(idx) == name
        )
        sizes[name] = {"table_bytes": table_bytes, "index_bytes": index_bytes,
                       "bytes": table_bytes + index_bytes}
    return sizes


def _sample_payload(conn: sqlite3.Connection, table: str, limit: int = 2000) -> float:
    """Mean logical payload bytes per row, from a bounded sample."""
    try:
        rows = conn.execute(f'SELECT * FROM "{table}" LIMIT {int(limit)}').fetchall()
    except sqlite3.DatabaseError:
        return 0.0
    if not rows:
        return 0.0
    total = 0
    for row in rows:
        for value in row:
            if isinstance(value, str):
                total += len(value.encode("utf-8", "replace"))
            elif isinstance(value, (bytes, bytearray)):
                total += len(value)
            elif value is not None:
                total += 8  # integer/real stored inline
    return total / len(rows)


def sizes_page_estimate(
    conn: sqlite3.Connection, counts: dict[str, int], total_bytes: int
) -> dict[str, dict[str, int]]:
    """Fallback when ``dbstat`` is unavailable.

    ``dbstat`` is the only way to attribute individual b-tree pages, so without
    it each table's logical payload is sampled and the file's real byte total is
    split between tables in proportion to that payload. Index-heavy tables are
    under-counted by this method and the report says so.
    """
    payload: dict[str, float] = {}
    for name, count in counts.items():
        if count <= 0:
            payload[name] = 0.0
            continue
        payload[name] = _sample_payload(conn, name) * count

    grand = sum(payload.values())
    sizes: dict[str, dict[str, int]] = {}
    for name in counts:
        share = payload[name] / grand if grand else 0.0
        sizes[name] = {"table_bytes": int(round(total_bytes * share)),
                       "index_bytes": 0, "bytes": int(round(total_bytes * share))}
    return sizes


# --------------------------------------------------------------------------- #
# Growth
# --------------------------------------------------------------------------- #


def _day(value: str) -> date:
    """Cortex writes ISO-8601 with an offset, but accept space-separated too."""
    text = (value or "")[:10]
    return date(int(text[0:4]), int(text[5:7]), int(text[8:10]))


def insert_rate(
    conn: sqlite3.Connection,
    table: str,
    today: date,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> dict[str, float]:
    """Rows added per day over the window, from the table's ``created_at``.

    A table younger than the window is averaged over its own lifetime instead,
    so a two-day-old table does not look ten times slower than it is. A table
    that does not exist reports no growth.
    """
    try:
        first_row = conn.execute(
            f'SELECT min(substr(created_at,1,10)) FROM "{table}"'
        ).fetchone()
    except sqlite3.OperationalError:
        return {"rows_in_window": 0, "rows_per_day": 0.0, "days": float(window_days)}
    first = (first_row[0] or "")[:10] if first_row else ""
    if not first:
        return {"rows_in_window": 0, "rows_per_day": 0.0, "days": float(window_days)}

    start = today - timedelta(days=window_days - 1)
    first_date = _day(first)
    if first_date > start:
        start = first_date

    rows = int(
        conn.execute(
            f'SELECT count(*) FROM "{table}" '
            "WHERE substr(created_at,1,10) >= ? AND substr(created_at,1,10) <= ?",
            (start.isoformat(), today.isoformat()),
        ).fetchone()[0]
    )
    days = max(1, (today - start).days + 1)
    return {"rows_in_window": rows, "rows_per_day": rows / days, "days": float(days)}


def project_bytes(current_bytes: float, bytes_per_day: float, days: int) -> float:
    """Linear projection: today's size plus ``days`` of growth."""
    return current_bytes + bytes_per_day * days


def build_report(
    db_path: Path,
    method: str = "auto",
    window_days: int = DEFAULT_WINDOW_DAYS,
    now: datetime | None = None,
) -> dict:
    conn = connect_readonly(db_path)
    try:
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
        total_bytes = page_size * page_count
        try:
            free_bytes = int(conn.execute("PRAGMA freelist_count").fetchone()[0]) * page_size
        except sqlite3.DatabaseError:  # pragma: no cover - defensive
            free_bytes = 0

        counts = row_counts(conn)

        use_dbstat = method == "dbstat" or (method == "auto" and has_dbstat(conn))
        if use_dbstat:
            if not has_dbstat(conn):
                conn.close()
                raise RuntimeError("dbstat virtual table is not available in this SQLite build")
            sizes = sizes_dbstat(conn)
            size_method = "dbstat"
            note = "per-table b-tree pages plus that table's index pages"
        else:
            sizes = sizes_page_estimate(conn, counts, total_bytes)
            size_method = "page_count"
            reason = (
                "requested with --size-method pages"
                if method == "pages"
                else "dbstat unavailable in this SQLite build"
            )
            note = (
                f"{reason}: sampled row payloads split the file's real page total "
                "across tables, so index-heavy tables read low"
            )

        clock = now or datetime.now(timezone.utc)
        today = clock.astimezone(timezone.utc).date()

        growth = []
        for table in GROWTH_TABLES:
            if table not in counts:
                continue
            rate = insert_rate(conn, table, today, window_days)
            rows = counts[table]
            table_bytes = sizes.get(table, {}).get("bytes", 0)
            bytes_per_row = (table_bytes / rows) if rows else 0.0
            growth.append(
                {
                    "table": table,
                    "rows": rows,
                    "rows_in_window": int(rate["rows_in_window"]),
                    "window_days": int(rate["days"]),
                    "rows_per_day": rate["rows_per_day"],
                    "bytes": table_bytes,
                    "bytes_per_row": bytes_per_row,
                    "bytes_per_day": rate["rows_per_day"] * bytes_per_row,
                }
            )
    finally:
        conn.close()

    bytes_per_day = sum(g["bytes_per_day"] for g in growth)
    projections = [
        {"horizon_days": days,
         "bytes": int(round(project_bytes(total_bytes, bytes_per_day, days))),
         "mb": round(project_bytes(total_bytes, bytes_per_day, days) / (1024 * 1024), 2)}
        for days in HORIZONS
    ]

    tables = [
        {
            "name": name,
            "rows": counts.get(name, 0),
            "bytes": sizes.get(name, {}).get("bytes", 0),
            "table_bytes": sizes.get(name, {}).get("table_bytes", 0),
            "index_bytes": sizes.get(name, {}).get("index_bytes", 0),
            "share": round(sizes.get(name, {}).get("bytes", 0) / total_bytes, 4) if total_bytes else 0.0,
        }
        for name in sorted(counts, key=lambda n: sizes.get(n, {}).get("bytes", 0), reverse=True)
    ]
    attributed = sum(t["bytes"] for t in tables)

    return {
        "db": str(db_path),
        "generated_at": clock.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "page_size": page_size,
        "page_count": page_count,
        "file_bytes": total_bytes,
        "file_mb": round(total_bytes / (1024 * 1024), 2),
        "free_bytes": free_bytes,
        "size_method": size_method,
        "size_method_note": note,
        "tables": tables,
        "unattributed_bytes": max(0, total_bytes - attributed),
        "window_days": window_days,
        "growth": growth,
        "bytes_per_day": bytes_per_day,
        "mb_per_day": round(bytes_per_day / (1024 * 1024), 3),
        "assumption": ASSUMPTION.format(window=window_days),
        "projections": projections,
    }


def apply_trigger(report: dict, threshold_mb: float | None) -> bool:
    """True when the 90-day projection is at or past the threshold."""
    report["sweep_trigger_mb"] = threshold_mb
    if threshold_mb is None:
        report["sweep_breached"] = False
        return False
    horizon = next(p for p in report["projections"] if p["horizon_days"] == SWEEP_HORIZON_DAYS)
    breached = horizon["mb"] >= threshold_mb
    report["sweep_breached"] = breached
    return breached


def _mb(value: float) -> str:
    return f"{value / (1024 * 1024):,.1f} MB"


def print_text(report: dict) -> None:
    print(f"db        {report['db']}")
    print(
        f"size      {_mb(report['file_bytes'])} "
        f"({report['page_count']:,} pages x {report['page_size']:,} B, "
        f"{_mb(report['free_bytes'])} free)"
    )
    print(f"method    {report['size_method']}: {report['size_method_note']}")
    print()
    print(f"{'table':34} {'rows':>9} {'bytes':>12} {'share':>7}")
    for table in report["tables"][:15]:
        print(
            f"{table['name'][:34]:34} {table['rows']:>9,} {_mb(table['bytes']):>12} "
            f"{table['share'] * 100:>6.1f}%"
        )
    if len(report["tables"]) > 15:
        print(f"{'... %d more tables' % (len(report['tables']) - 15):34}")
    print(f"{'unattributed (freelist etc.)':34} {'':>9} {_mb(report['unattributed_bytes']):>12}")
    print()
    print(f"growth (last {report['window_days']} days, from created_at)")
    if not report["growth"]:
        print("  none of the tracked tables exist in this database")
    for g in report["growth"]:
        print(
            f"  {g['table']:12} {g['rows_in_window']:>7,} rows over {g['window_days']:>2} d  "
            f"{g['rows_per_day']:>7.2f} rows/day x {g['bytes_per_row']:>9,.0f} B/row  "
            f"= {g['bytes_per_day'] / 1024:>9,.0f} KiB/day"
        )
    print(f"  total growth: {_mb(report['bytes_per_day'])}/day")
    print()
    print(f"projection (assumes {report['assumption']})")
    for p in report["projections"]:
        print(f"  {p['horizon_days']:>4} days  {_mb(p['bytes'])}")
    threshold = report.get("sweep_trigger_mb")
    if threshold is not None:
        horizon = next(p for p in report["projections"] if p["horizon_days"] == SWEEP_HORIZON_DAYS)
        state = "CROSSED" if report["sweep_breached"] else "not crossed"
        print(
            f"sweep     trigger {threshold:,.0f} MB at {SWEEP_HORIZON_DAYS} days: "
            f"{state} ({horizon['mb']:,.1f} MB)"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report per-table size, insert rate and size projection for a Cortex database."
    )
    parser.add_argument("--db", type=Path, default=default_db_path(), help="path to cortex.db")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--sweep-trigger-mb",
        type=float,
        default=None,
        metavar="MB",
        help=f"exit 1 when the {SWEEP_HORIZON_DAYS}-day projection reaches this many MiB",
    )
    parser.add_argument(
        "--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
        help=f"insert-rate window in days (default {DEFAULT_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--size-method", choices=("auto", "dbstat", "pages"), default="auto",
        help="byte attribution method; auto uses dbstat when the build has it",
    )
    args = parser.parse_args(argv)

    if args.window_days < 1:
        print("storage_forecast: --window-days must be positive", file=sys.stderr)
        return 2
    if not args.db.exists():
        print(f"storage_forecast: no such database: {args.db}", file=sys.stderr)
        return 2

    try:
        report = build_report(args.db, method=args.size_method, window_days=args.window_days)
    except (sqlite3.DatabaseError, RuntimeError) as exc:
        print(f"storage_forecast: cannot read {args.db}: {exc}", file=sys.stderr)
        return 2

    breached = apply_trigger(report, args.sweep_trigger_mb)

    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    else:
        print_text(report)
    return 1 if breached else 0


if __name__ == "__main__":
    raise SystemExit(main())
