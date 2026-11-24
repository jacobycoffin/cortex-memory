"""A memory's track record (#71)

    python3 scripts/outcomes.py --db DB                        # report
    python3 scripts/outcomes.py --db DB --memory <id>          # one record
    python3 scripts/outcomes.py --db DB --harmful              # flagged memories
    python3 scripts/outcomes.py --db DB --record <id> <task-ref> success|failure|neutral
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import outcomes  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--memory", default="")
    parser.add_argument("--harmful", action="store_true")
    parser.add_argument("--record", nargs=3, metavar=("MEMORY", "TASK", "OUTCOME"), default=None)
    parser.add_argument("--detail", default="")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.record:
            memory_id, task_ref, verdict = args.record
            print(outcomes.record(store, memory_id, task_ref, verdict, detail=args.detail))
            return 0
        if args.harmful:
            flagged = outcomes.harmful(store)
            print(f"{len(flagged)} memory(ies) flagged")
            for record in flagged:
                print("  -", record.describe())
            return 0
        if args.memory:
            print(outcomes.track_record(store, args.memory).describe())
            return 0
        described = outcomes.report(store)
        print(f"{described['outcomes']} outcome(s) across {described['memories']} memories: "
              f"{described['successes']} success, {described['failures']} failure; "
              f"weighting {'on' if described['weighting'] else 'off'}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
