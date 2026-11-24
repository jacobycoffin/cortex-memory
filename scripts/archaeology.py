"""What a memory actually taught the agent (#81)

    python3 scripts/archaeology.py --db DB                    # window report
    python3 scripts/archaeology.py --db DB --memory <id>      # one memory's effects
    python3 scripts/archaeology.py --db DB --influential 10
    python3 scripts/archaeology.py --db DB --useless 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import archaeology  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--days", type=int, default=archaeology.DEFAULT_DAYS)
    parser.add_argument("--memory", default="")
    parser.add_argument("--influential", type=int, default=0)
    parser.add_argument("--useless", type=int, default=0)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.memory:
            payload = archaeology.digest(store, args.memory, days=args.days)
            print(payload["content"][:100])
            print(" ", payload["sentence"])
            return 0
        if args.influential:
            for item in archaeology.most_influential(store, days=args.days,
                                                    limit=args.influential):
                print(f"  {item['score']:4d}  {item['sentence']}")
            return 0
        if args.useless:
            for item in archaeology.silently_useless(store, days=args.days, limit=args.useless):
                print(f"  offered {item['injected']:4d}, used {item['used']}  {item['memory_id'][:8]}")
            return 0
        described = archaeology.report(store, days=args.days)
        print(f"last {described['days']} days: {described['memories']} memories touched, "
              f"{described['injected']} offered, {described['used']} used "
              f"({described['use_share'] * 100:.1f}%)")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
