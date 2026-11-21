#!/usr/bin/env python3
"""Show why a recall picked what it picked.

    python3 scripts/explain_recall.py --task-id 8f2c1d04
    python3 scripts/explain_recall.py --last 3          # the three most recent turns

Reads the additive `recall_score_breakdown` table written by the provider. Read-only.

Exit codes: 0 printed, 1 nothing recorded for that turn, 2 unreadable database.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

TABLE = "recall_score_breakdown"


def load(db: Path, task_id: str | None, last: int) -> list[tuple]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        if task_id:
            rows = conn.execute(
                f"SELECT task_id, rank, memory_id, score, components_json, created_at "
                f"FROM {TABLE} WHERE task_id = ? ORDER BY rank",
                (task_id,),
            ).fetchall()
        else:
            ids = [
                r[0]
                for r in conn.execute(
                    f"SELECT DISTINCT task_id FROM {TABLE} ORDER BY created_at DESC LIMIT ?",
                    (int(last),),
                ).fetchall()
            ]
            rows = []
            for tid in ids:
                rows.extend(
                    conn.execute(
                        f"SELECT task_id, rank, memory_id, score, components_json, created_at "
                        f"FROM {TABLE} WHERE task_id = ? ORDER BY rank",
                        (tid,),
                    ).fetchall()
                )
        return rows
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path.home() / ".hermes/cortex/cortex.db")
    parser.add_argument("--task-id")
    parser.add_argument("--last", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"explain_recall: no such database: {args.db}", file=sys.stderr)
        raise SystemExit(2)
    try:
        rows = load(args.db, args.task_id, args.last)
    except sqlite3.Error as exc:
        print(f"explain_recall: {exc}", file=sys.stderr)
        raise SystemExit(2)

    if not rows:
        print("explain_recall: nothing recorded for that turn", file=sys.stderr)
        raise SystemExit(1)

    if args.json:
        payload = [
            {
                "task_id": t,
                "rank": rank,
                "memory_id": mid,
                "score": score,
                "components": json.loads(comp),
                "at": at,
            }
            for t, rank, mid, score, comp, at in rows
        ]
        print(json.dumps(payload, indent=1))
        return

    current = None
    for task_id, rank, memory_id, score, components_json, created_at in rows:
        if task_id != current:
            current = task_id
            print(f"\nturn {task_id[:8]}  ({created_at})")
        try:
            components = json.loads(components_json)
        except ValueError:
            components = {}
        top = ", ".join(
            f"{k}={v:g}"
            for k, v in sorted(components.items(), key=lambda kv: -abs(float(kv[1])))[:5]
        )
        print(f"  {rank}. M:{memory_id[:8]} score={score:.4f}  {top or '(no components)'}")


if __name__ == "__main__":
    main()
