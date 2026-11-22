"""Suppress, unsuppress, delete, or report on memories (#91)

    python3 scripts/suppression.py --db DB --report
    python3 scripts/suppression.py --db DB --suppress <id> [--days 7] [--reason "..."]
    python3 scripts/suppression.py --db DB --unsuppress <id>
    python3 scripts/suppression.py --db DB --delete <id> --reason "..." [--force]

Suppression is reversible and keeps the content. Deletion destroys the memory, removes its edges
and leaves an audit row - it demands the id twice (`--delete` and the resolved id must match)
and refuses protected memories unless `--force`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import suppression as sup  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--suppress", help="memory id or prefix")
    parser.add_argument("--days", type=float, default=0, help="0 means until further notice")
    parser.add_argument("--reason", default="user request")
    parser.add_argument("--unsuppress", help="memory id or prefix")
    parser.add_argument("--delete", help="memory id or prefix (destructive)")
    parser.add_argument("--force", action="store_true", help="allow deleting a protected memory")
    parser.add_argument("--expire", action="store_true", help="lift every expired suppression")
    parser.add_argument("--list", action="store_true", help="current suppressions")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.expire:
            lifted = sup.expire(store)
            print(f"lifted {len(lifted)} expired suppressions")
            return 0
        if args.list:
            rows = sup.suppressions(store)
            if not rows:
                print("no suppressions")
            for row in rows:
                print(
                    f"{row['memory_id'][:8]} until={row['until'] or 'manual'} "
                    f"{row['reason']}: {' '.join(str(row['content'] or '').split())[:90]}"
                )
            return 0
        if args.suppress:
            resolved = store.resolve_id(args.suppress) or args.suppress
            until = (
                (datetime.now(timezone.utc) + timedelta(days=args.days)).isoformat()
                if args.days
                else None
            )
            ok = sup.suppress(store, resolved, reason=args.reason, until=until)
            print("suppressed" if ok else "no such memory")
            return 0 if ok else 1
        if args.unsuppress:
            resolved = store.resolve_id(args.unsuppress) or args.unsuppress
            ok = sup.unsuppress(store, resolved)
            print("suppression lifted" if ok else "that memory was not suppressed")
            return 0 if ok else 1
        if args.delete:
            resolved = store.resolve_id(args.delete) or args.delete
            try:
                result = sup.delete(
                    store, resolved, confirm=resolved, reason=args.reason,
                    allow_protected=args.force,
                )
            except sup.DeleteRefused as error:
                print(f"refused: {error}")
                return 1
            print(
                f"deleted {result['deleted'][:8]} ({result['kind']}), "
                f"{result['edges_removed']} edges removed, audit {result['content_hash']}"
            )
            return 0
        report = sup.report(store)
        for line in report.lines():
            print(line)
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
