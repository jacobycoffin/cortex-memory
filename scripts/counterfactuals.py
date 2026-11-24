"""What a memory was chosen over (#77)

    python3 scripts/counterfactuals.py --db DB                 # report
    python3 scripts/counterfactuals.py --db DB --record        # keep the pairs
    python3 scripts/counterfactuals.py --db DB --memory <id>   # what it beat
    python3 scripts/counterfactuals.py --db DB --narrowest 5   # closest calls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401

from cortex import counterfactuals  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--memory", default="")
    parser.add_argument("--narrowest", type=int, default=0)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.record:
            print(counterfactuals.record(store).describe())
            return 0
        if args.memory:
            for row in counterfactuals.alternatives(store, args.memory):
                print(f"  beat {row['rejected_id'][:8]} by {row['score_gap']:+.4f}")
            return 0
        if args.narrowest:
            for row in counterfactuals.narrowest(store, limit=args.narrowest):
                print(f"  {row['chosen_id'][:8]} over {row['rejected_id'][:8]} "
                      f"by {row['score_gap']:+.4f}")
            return 0
        described = counterfactuals.report(store)
        print(f"{described['pairs']} pair(s) across {described['traces']} trace(s); "
              f"mean gap {described['mean_gap']:.4f}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
