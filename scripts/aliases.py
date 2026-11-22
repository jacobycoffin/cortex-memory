"""The entity alias table (#39)

    python3 scripts/aliases.py --db DB                       # report
    python3 scripts/aliases.py --db DB --learn               # learn the names already in use
    python3 scripts/aliases.py --db DB --add CT116 "t3 box"  # add one by hand
    python3 scripts/aliases.py --db DB --remove "t3 box" [--entity CT116]
    python3 scripts/aliases.py --db DB --merge t3 CT116      # fold one entity into another
    python3 scripts/aliases.py --db DB --resolve "t3 box"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import aliases  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--add", nargs=2, metavar=("ENTITY", "ALIAS"))
    parser.add_argument("--remove", metavar="ALIAS")
    parser.add_argument("--entity", help="restrict --remove to one entity")
    parser.add_argument("--merge", nargs=2, metavar=("SOURCE", "TARGET"))
    parser.add_argument("--resolve", metavar="NAME")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.learn:
            print("learned:", aliases.from_memories(store))
        if args.add:
            entity, alias = args.add
            added = aliases.add(store, entity, alias)
            print(f"{'added' if added else 'ignored'}: {alias} -> {entity}")
        if args.remove:
            print(f"removed {aliases.remove(store, args.remove, entity_id=args.entity)} row(s)")
        if args.merge:
            source, target = args.merge
            print("merged:", aliases.merge(store, source, target))
        if args.resolve:
            print(f"{args.resolve} -> {aliases.resolve(store, args.resolve) or 'unknown'}")
        if any((args.learn, args.add, args.remove, args.merge, args.resolve)):
            return 0
        described = aliases.report(store)
        print(f"{described['entities']} entities, {described['aliases']} aliases")
        if described["ambiguous"]:
            print("ambiguous: " + ", ".join(described["ambiguous"]))
        for item in described["top"]:
            print(f"  {item['entity_id']:<20} aliases={item['aliases']} hits={item['hits']}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
