"""Dense regions of the graph and their names (#74)

    python3 scripts/topic_nodes.py --db DB                 # report
    python3 scripts/topic_nodes.py --db DB --labels 6      # name the largest regions
    python3 scripts/topic_nodes.py --db DB --create        # persist topic rows
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401

from cortex import topic_nodes  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--labels", type=int, default=0)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.create:
            regions = topic_nodes.dense_regions(store)
            for region in regions:
                topic_nodes.create(store, getattr(region, "members", []))
            print(f"created {len(regions)} topic(s)")
            return 0
        if args.labels:
            for region in topic_nodes.dense_regions(store)[: int(args.labels)]:
                members = getattr(region, "members", [])
                print(f"  size {len(members):5d}  {topic_nodes.label(store, members)[:70]}")
            return 0
        described = topic_nodes.report(store)
        print(" ".join(f"{key}={value}" for key, value in described.items() if not isinstance(value, dict)))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
