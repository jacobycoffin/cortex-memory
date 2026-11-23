"""Calibration per kind (#53)

    python3 scripts/calibration.py --db DB
    python3 scripts/calibration.py --db DB --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import calibration  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        described = calibration.report(store, limit=args.limit)
        print(f"{described['judged']} of {described['total']} facts have a decided outcome "
              f"({described['judged_share'] * 100:.1f}%)")
        for kind in described["kinds"]:
            if not kind["judged"]:
                print(f"  {kind['kind']:<14} {kind['judged']:5d} judged  (no evidence)")
                continue
            ceiling = "n/a" if kind["ceiling"] is None else f"{kind['ceiling']:.2f}"
            print(f"  {kind['kind']:<14} {kind['judged']:5d} judged  accuracy "
                  f"{kind['observed_accuracy']:.2f}  Brier {kind['brier']:.3f}  "
                  f"stated {kind['mean_confidence']:.2f}  ceiling {ceiling}"
                  + ("  OVERCONFIDENT" if kind["overconfident"] else ""))
        if args.apply:
            print("applied:", calibration.apply(store, limit=args.limit))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
