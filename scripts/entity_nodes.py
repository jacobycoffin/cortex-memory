"""Entity nodes: a page per entity (#49)

    python3 scripts/entity_nodes.py --db DB                 # counts by kind
    python3 scripts/entity_nodes.py --db DB --sync 2000      # give every known entity a node
    python3 scripts/entity_nodes.py --db DB --page CT116
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import entity_nodes  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--sync", type=int, default=0)
    parser.add_argument("--page", default="")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.sync:
            print("sync:", entity_nodes.sync(store, limit=args.sync))
        if args.page:
            page = entity_nodes.page(store, args.page)
            if page is None:
                print(f"no node for {args.page}")
                return 1
            print(f"{page['canonical_name']} [{page['kind']}] id={page['entity_id']}")
            if page["aliases"]:
                print("  also known as:", ", ".join(page["aliases"][:8]))
            print(f"  {page['fact_count']} fact(s) shown, first {page['first_seen'][:10]}, "
                  f"last {page['last_seen'][:10]}")
            for fact in page["facts"][:5]:
                print("   -", str(fact["content"])[:90])
            if page["neighbours"]:
                print("  neighbours:", ", ".join(page["neighbours"]))
            return 0
        described = entity_nodes.report(store)
        print(f"{described['nodes']} node(s):", described["by_kind"])
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
