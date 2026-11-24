"""What a scoring change would have done, replayed on history (#202)

    python3 scripts/shadow_rank.py --db DB                       # every built-in policy
    python3 scripts/shadow_rank.py --db DB --policy no_usage_boost
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401

from cortex import shadow_rank  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--policy", default="")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.policy:
            chosen = next((item for item in shadow_rank.policies() if item.name == args.policy),
                          None)
            if chosen is None:
                print(f"unknown policy {args.policy}; known: "
                      f"{[item.name for item in shadow_rank.policies()]}")
                return 2
            found = shadow_rank.replay(store, chosen, limit=args.limit)
            print(found.describe())
            for example in found.examples[:2]:
                print(f"  recorded {example['recorded']} -> shadow {example['shadow']}")
            return 0
        described = shadow_rank.report(store, limit=args.limit)
        print(f"{described['traces_replayed']} trace(s) replayed")
        for item in described["policies"]:
            print(f"  {item['sentence']}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
