#!/usr/bin/env python3
"""Read the abstention ledger: what would the policy have withheld, and would it have been right?

    python3 scripts/abstention_ledger_report.py --db ~/.hermes/cortex/cortex.db
    python3 scripts/abstention_ledger_report.py --days 7 --list 20

Read-only. "Wrong" means the withheld memory was later recorded as `used`, which is the only
outcome signal the store keeps.
"""

from __future__ import annotations

import argparse
import importlib.util
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


class _ReadOnlyStore:
    """Minimal shim: the ledger only needs a connection."""

    def __init__(self, path: Path) -> None:
        import sqlite3

        self._conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path.home() / ".hermes/cortex/cortex.db")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--list", type=int, default=0, help="list this many recent shadow abstentions")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"abstention_ledger_report: no such database: {args.db}", file=sys.stderr)
        raise SystemExit(2)

    _load_package()
    from cortex.abstention_ledger import decisions, summary  # noqa: PLC0415

    store = _ReadOnlyStore(args.db)
    try:
        stats = summary(store, days=args.days)
        rows = decisions(store, limit=args.list, days=args.days) if args.list else []
    finally:
        store._conn.close()

    print(f"window: last {args.days} days")
    print(f"turns recorded      {stats['turns']}")
    print(f"would have abstained {stats['would_abstain']}")
    print(f"actually abstained   {stats['abstained']}")
    print(f"would-abstain precision {stats['would_abstain_precision']:.3f} "
          f"({stats['would_abstain_then_used']} of the withheld memories were later used)")

    if stats["turns"] and not stats["would_abstain"]:
        print("\nno shadow abstentions in this window: every top score was above the threshold")

    for task_id, threshold, top_score, would, abstained, created_at in rows:
        print(f"  {created_at}  {task_id[:8]}  top {top_score:.3f} < {threshold:.2f}"
              f"  would={int(would)} real={int(abstained)}")


if __name__ == "__main__":
    main()
