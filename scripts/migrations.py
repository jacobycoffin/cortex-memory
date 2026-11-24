"""Schema migrations (#65)

    python3 scripts/migrations.py --db DB             # status
    python3 scripts/migrations.py --db DB --apply NAME --backup PATH
    python3 scripts/migrations.py --db DB --reverse NAME
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401

from cortex import migrations  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--apply", default="")
    parser.add_argument("--reverse", default="")
    parser.add_argument("--backup", default="")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.apply:
            print(migrations.apply(store, args.apply, backup_path=args.backup))
            return 0
        if args.reverse:
            print(migrations.reverse(store, args.reverse))
            return 0
        print(migrations.status(store))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
