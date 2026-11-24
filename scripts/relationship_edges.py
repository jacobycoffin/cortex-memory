"""Typed relations between entities (#62)

    python3 scripts/relationship_edges.py --db DB                     # report
    python3 scripts/relationship_edges.py --db DB --of Jacoby
    python3 scripts/relationship_edges.py --db DB --propose "Jacoby owns the r630 host."
    python3 scripts/relationship_edges.py --db DB --link Jacoby r630 owns --by jacoby
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import relationship_edges  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--of", default="")
    parser.add_argument("--propose", default="")
    parser.add_argument("--link", nargs=3, metavar=("SOURCE", "TARGET", "RELATION"), default=None)
    parser.add_argument("--by", default="")
    parser.add_argument("--evidence", default="")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.link:
            source, target, relation = args.link
            print(relationship_edges.link(store, source, target, relation,
                                          evidence=args.evidence, asserted_by=args.by))
            return 0
        if args.propose:
            for proposal in relationship_edges.propose(store, args.propose):
                print("  proposal:", proposal["sentence"])
            return 0
        if args.of:
            relations = relationship_edges.relations_of(store, args.of)
            print(f"{len(relations)} relation(s) for {args.of}")
            for relation in relations:
                print("  -", relation.describe())
            return 0
        described = relationship_edges.report(store)
        print(f"{described['relations']} relation(s), {described['open']} open")
        for relation, count in sorted(described["by_relation"].items()):
            print(f"  {relation:<12} {count}")
        print("vocabulary:", ", ".join(described["vocabulary"]))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
