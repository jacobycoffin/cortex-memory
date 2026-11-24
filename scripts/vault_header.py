"""Take the vault header out of stored content (#223)

    python3 scripts/vault_header.py --db DB              # report
    python3 scripts/vault_header.py --db DB --dry-run    # what a sweep would do
    python3 scripts/vault_header.py --db DB --apply      # rewrite, keeping the original

Dry run is the default. Applying rewrites `content` and stores the header plus the original text in
`vault_headers`, so `vault_header.restore()` puts any row back.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401

from cortex import vault_header  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--scan", type=int, default=5000)
    parser.add_argument("--mode", choices=(vault_header.DELIMITED, vault_header.PREFIX),
                        default=vault_header.DELIMITED)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.report:
            described = vault_header.report(store, scan=args.scan)
            print(f"{described['with_header']} of {described['active']} active memories "
                  f"({described['share']:.1%}) carry a vault header")
            print(f"  {described['delimited']} delimited (rewritable), "
                  f"{described['ambiguous']} ambiguous")
            print(f"  note: {described['note']}")
            return 0
        found = vault_header.strip(store, mode=args.mode, apply=args.apply, scan=args.scan)
        print(found.describe())
        for sample in found.samples:
            print(f"  before: {sample['before'][:78]}")
            print(f"  after : {sample['after'][:78]}")
        if not args.apply:
            print("dry run: nothing was written; pass --apply to rewrite")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
