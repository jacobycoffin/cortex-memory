"""Export or restore core memory (#131)

    python3 scripts/core_backup.py --db "$HOME/.hermes/cortex/cortex.db" --out core-memory.md
    python3 scripts/core_backup.py --db ... --restore core-memory.md [--apply]

Export writes a Markdown file you can read, diff and keep in a vault. Restore matches
memories by content, so a backup from one database rebuilds the profile in another.
Restore is a dry run unless `--apply` is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex.core_backup import export, restore  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--out", help="write the backup here (default: stdout)")
    parser.add_argument("--max-chars", type=int, default=0, help="0 means no cap (default)")
    parser.add_argument("--restore", help="restore from this file")
    parser.add_argument("--apply", action="store_true", help="actually restore")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.restore:
            text = Path(args.restore).read_text()
            if not args.apply:
                # Parse-only count so a dry run says something useful without writing.
                from cortex.core_backup import parse

                print(f"entries in backup: {len(parse(text))}")
                print("dry run - pass --apply to restore")
                return 0
            result = restore(store, text)
            print(result.summary)
            if result.missing:
                print("missing from this store (not invented):")
                for item in result.missing[:10]:
                    print(f"  {item}")
            return 0

        backup = export(store, max_chars=args.max_chars)
        if args.out:
            Path(args.out).expanduser().write_text(backup.text)
            print(f"wrote {args.out} ({len(backup.entries)} entries, {len(backup.text)} chars)")
        else:
            print(backup.text)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
