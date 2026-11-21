#!/usr/bin/env python3
"""Report what a retention sweep would do, using the policy in docs/RETENTION.md.

Read-only. This is the executable form of the policy: same windows, same protections,
same regret rule. Running it answers "what would we archive today, and what would we
refuse to touch" without changing a row.

    python3 scripts/retention_check.py --db ~/.hermes/cortex/cortex.db
    python3 scripts/retention_check.py --db ... --json

Exit codes: 0 always (this is a report, not a gate), 2 unreadable database.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# (cool_days, archive_days or None for never)
WINDOWS = {
    "episode": (30, 90),
    "semantic": (90, 180),
    "operational": (90, 180),
    "decision": (365, None),
    "preference": (365, None),
    "identity": (365, None),
    "procedure": (365, None),
    "prospective": (365, None),
}
DEFAULT_WINDOW = (90, 180)
REGRET_WINDOW_DAYS = 30


def last_used(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute(
        """SELECT memory_id, MAX(created_at) FROM access_log
           WHERE event = 'used' GROUP BY memory_id"""
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def protected_ids(conn: sqlite3.Connection) -> dict[str, str]:
    """id -> the reason it is untouchable."""
    reasons: dict[str, str] = {}
    for mid, in conn.execute("SELECT id FROM memories WHERE pinned = 1"):
        reasons[mid] = "pinned"
    for mid, in conn.execute(
        """SELECT DISTINCT memory_id FROM access_log
           WHERE event IN ('helpful', 'confirmed', 'corrected', 'wrong', 'useful')"""
    ):
        reasons.setdefault(mid, "has feedback")
    # An edge alone is not evidence: 3,704 of 3,818 active memories sit on at least one
    # edge, so "edge endpoint" protects the entire store. Require corroboration.
    for (mid,) in conn.execute(
        "SELECT DISTINCT src_id FROM edges WHERE src_id IS NOT NULL AND evidence_count >= 2"
    ):
        reasons.setdefault(mid, "corroborated edge")
    for (mid,) in conn.execute(
        "SELECT DISTINCT dst_id FROM edges WHERE dst_id IS NOT NULL AND evidence_count >= 2"
    ):
        reasons.setdefault(mid, "corroborated edge")
    try:
        for (mid,) in conn.execute(
            "SELECT memory_id FROM prospective_items WHERE status IN ('open', 'due')"
        ):
            reasons.setdefault(mid, "open commitment")
    except sqlite3.Error:
        pass
    return reasons


def classify(conn: sqlite3.Connection) -> dict:
    used = last_used(conn)
    protected = protected_ids(conn)
    now = conn.execute("SELECT datetime('now')").fetchone()[0]

    out = {
        "now": now,
        "by_kind": {},
        "total_active": 0,
        "would_cool": 0,
        "would_archive": 0,
        "retained": 0,
        "protected": {},
    }
    rows = conn.execute(
        "SELECT id, kind, created_at FROM memories WHERE state = 'active'"
    ).fetchall()
    out["total_active"] = len(rows)

    for mid, kind, created_at in rows:
        cool, archive = WINDOWS.get(kind, DEFAULT_WINDOW)
        bucket = out["by_kind"].setdefault(
            kind, {"active": 0, "cool": 0, "archive": 0, "held": 0}
        )
        bucket["active"] += 1

        if mid in protected:
            bucket["held"] += 1
            out["protected"][protected[mid]] = out["protected"].get(protected[mid], 0) + 1
            out["retained"] += 1
            continue

        marker = used.get(mid) or created_at
        age = conn.execute(
            "SELECT julianday('now') - julianday(?)", (marker,)
        ).fetchone()[0] or 0.0

        if archive is not None and age >= archive:
            bucket["archive"] += 1
            out["would_archive"] += 1
        elif age >= cool:
            bucket["cool"] += 1
            out["would_cool"] += 1
        else:
            out["retained"] += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path.home() / ".hermes/cortex/cortex.db")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--apply", action="store_true",
                        help="Refused: this script is a report. Use the pruning runner.")
    args = parser.parse_args()

    if args.apply:
        print("retention_check: refusing to mutate; this is the read-only policy report.",
              file=sys.stderr)
        raise SystemExit(2)
    if not args.db.exists():
        print(f"retention_check: no such database: {args.db}", file=sys.stderr)
        raise SystemExit(2)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        report = classify(conn)
    except sqlite3.Error as exc:
        print(f"retention_check: {exc}", file=sys.stderr)
        raise SystemExit(2)
    finally:
        conn.close()

    if args.json:
        print(json.dumps(report, indent=1))
        return

    print(f"policy in force: windows {WINDOWS}, regret {REGRET_WINDOW_DAYS}d")
    print(f"active {report['total_active']}: "
          f"{report['would_cool']} would cool, {report['would_archive']} would archive, "
          f"{report['retained']} retained")
    print(f"{'kind':14} {'active':>7} {'cool':>6} {'archive':>8} {'held':>6}")
    for kind, b in sorted(report["by_kind"].items(), key=lambda kv: -kv[1]["active"]):
        print(f"{kind:14} {b['active']:>7} {b['cool']:>6} {b['archive']:>8} {b['held']:>6}")
    if report["protected"]:
        held = ", ".join(f"{k}: {v}" for k, v in sorted(report["protected"].items()))
        print(f"protected by rule: {held}")


if __name__ == "__main__":
    main()
