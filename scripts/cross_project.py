"""Cross-project bridges (#78)

    python3 scripts/cross_project.py --db DB                     # namespaces in the store
    python3 scripts/cross_project.py --db DB --query "..." --project cortex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import cross_project  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--query", default="")
    parser.add_argument("--project", default="")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        described = cross_project.report(store, active_project=args.project)
        print(
            f"{described['memories']} memories: {described['scoped']} scoped, "
            f"{described['unscoped']} unscoped; bar {described['min_similarity']}, "
            f"cap {described['max_bridges']}, enabled={described['enabled']}"
        )
        for project, count in described["projects"].items():
            print(f"  {project:<16} {count}")
        if args.query:
            found = cross_project.find(store, args.query, active_project=args.project or None)
            print(f"{len(found)} bridge(s)")
            for bridge in found:
                print("  " + bridge.describe())
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
