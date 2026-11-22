"""Learned rejection rules (#35)

    python3 scripts/reject_learn.py --db DB                 # report
    python3 scripts/reject_learn.py --db DB --check "text"  # would a rule catch this?
    python3 scripts/reject_learn.py --db DB --forget RULE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import reject_learn  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--check", help="content to test against the learned rules")
    parser.add_argument("--forget", help="rule id to remove")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.forget:
            print("forgot" if reject_learn.forget_rule(store, args.forget) else "no such rule")
            return 0
        if args.check:
            match = reject_learn.check(store, args.check)
            print(f"caught: {match.describe()}" if match else "not covered by any rule")
            return 1 if match else 0
        described = reject_learn.report(store)
        print(f"{described['rules']} rules, {described['hits']} hits")
        for code, item in described["by_reason"].items():
            print(f"  {code:<16} rules={item['rules']} hits={item['hits']}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
