"""Two speeds: drain a little while working, sweep when quiet (#209)

    python3 scripts/consolidation_queue.py --db DB                   # depth + limits
    python3 scripts/consolidation_queue.py --db DB --enqueue id1,id2 --reason written
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401

from cortex import consolidation_queue as queue  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--enqueue", default="")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.enqueue:
            ids = [part.strip() for part in args.enqueue.split(",") if part.strip()]
            print(f"queued {queue.enqueue(store, ids, reason=args.reason)} item(s)")
        described = queue.report(store)
        print(f"queued {described['queued']}, done {described['done']}, "
              f"parked {described['parked']}; limits {described['limits']}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
