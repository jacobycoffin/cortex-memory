#!/usr/bin/env python3
"""Read or verify a stored recall block.

    python3 scripts/recall_replay.py --db <copy.db> --task-id 8f2c1d04
    python3 scripts/recall_replay.py --db <copy.db> --session s1 --limit 3
    python3 scripts/recall_replay.py --db <copy.db> --prune-days 30

Read-only unless `--prune-days` is given, which deletes expired replay rows.
Exit codes: 0 ok, 1 nothing found, 2 unreadable database.
"""

from __future__ import annotations

import argparse
import importlib.util
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))


def _load_package() -> None:
    spec = importlib.util.spec_from_file_location(
        "cortex", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the Cortex package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cortex"] = module
    spec.loader.exec_module(module)


class _Store:
    """Minimal shim: the replay module only needs a connection."""

    def __init__(self, path: Path, *, writable: bool = False) -> None:
        if writable:
            self._conn = sqlite3.connect(str(path))
        else:
            self._conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path.home() / ".hermes/cortex/cortex.db")
    parser.add_argument("--task-id")
    parser.add_argument("--session")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--prune-days", type=int)
    parser.add_argument("--block", action="store_true", help="print the block text itself")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"recall_replay: no such database: {args.db}", file=sys.stderr)
        raise SystemExit(2)

    _load_package()
    from cortex.recall_replay import latest, load, prune  # noqa: PLC0415

    writable = args.prune_days is not None
    store = _Store(args.db, writable=writable)
    try:
        if args.prune_days is not None:
            print(f"pruned {prune(store, days=args.prune_days)} replay row(s)")
            return
        if args.task_id:
            prefix = args.task_id
            rows = [entry for entry in (load(store, prefix),) if entry]
            if not rows:
                # Allow a prefix, since task ids are uuid4 and humans copy 8 characters.
                conn = store._conn
                try:
                    match = conn.execute(
                        "SELECT task_id FROM recall_replay WHERE task_id LIKE ? LIMIT 1",
                        (prefix + "%",),
                    ).fetchone()
                except sqlite3.Error:
                    match = None
                if match:
                    entry = load(store, match[0])
                    rows = [entry] if entry else []
        else:
            rows = latest(store, session_id=args.session, limit=args.limit)
    finally:
        store._conn.close()

    if not rows:
        print("recall_replay: nothing recorded for that turn", file=sys.stderr)
        raise SystemExit(1)

    for entry in rows:
        print(f"\nturn {entry['task_id'][:8]}  session {entry['session_id']}  "
              f"{entry['created_at']}  {entry['tokens']} tokens  sha {entry['sha256'][:12]}")
        print(f"  memories: {', '.join(m[:8] for m in entry['memory_ids']) or 'none'}")
        if args.block:
            print(entry["block"])


if __name__ == "__main__":
    main()
