#!/usr/bin/env python3
"""Report near-duplicate memories and what collapsing them would save.

Read-only. It never changes a memory: collapsing is a destructive store operation and the
plan is produced for a human to approve.

    python3 scripts/redundancy_report.py --db ~/.hermes/cortex/cortex.db
    python3 scripts/redundancy_report.py --db <copy.db> --top 20 --json plan.json

Exit codes: 0 reported (even when nothing found), 2 unreadable database.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

QUERY = """
SELECT id, content, importance, used_count, created_at, pinned, state, source_ref
FROM memories
WHERE state IN ('active', 'cold') AND length(trim(content)) > 0
"""


def _load_package() -> None:
    spec = importlib.util.spec_from_file_location(
        "cortex", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the Cortex package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cortex"] = module
    spec.loader.exec_module(module)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path.home() / ".hermes/cortex/cortex.db")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", dest="json_out", type=Path)
    parser.add_argument("--full", action="store_true", help="print every planned collapse")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"redundancy_report: no such database: {args.db}", file=sys.stderr)
        raise SystemExit(2)

    _load_package()
    from cortex.redundancy import memory_from_row, plan  # noqa: PLC0415

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(QUERY)]
    except sqlite3.Error as exc:
        print(f"redundancy_report: {exc}", file=sys.stderr)
        raise SystemExit(2)
    finally:
        conn.close()

    memories = [memory_from_row(r) for r in rows]
    plans = plan(memories)
    dropped = sum(len(p.drop) for p in plans)
    saved_tokens = sum(p.tokens_saved for p in plans)

    print(f"memories scanned {len(memories)}")
    print(f"clusters {len(plans)}  collapsing would retire {dropped} copies  "
          f"saving ~{saved_tokens} tokens of stored content")

    limit = len(plans) if args.full else args.top
    for entry in plans[:limit]:
        print(f"\n  keep {entry.keep.id[:8]} (used {entry.keep.used_count}x): "
              f"{entry.keep.content[:80]!r}")
        for memory in entry.drop:
            print(f"   drop {memory.id[:8]} (used {memory.used_count}x, "
                  f"saves ~{len(memory.content)//4} tokens): {memory.content[:70]!r}")

    if args.json_out:
        payload = [
            {
                "keep": {"id": p.keep.id, "content": p.keep.content},
                "drop": [{"id": m.id, "content": m.content} for m in p.drop],
                "tokens_saved": p.tokens_saved,
            }
            for p in plans
        ]
        args.json_out.write_text(json.dumps(payload, indent=1))
        print(f"\nplan written to {args.json_out}")


if __name__ == "__main__":
    main()
