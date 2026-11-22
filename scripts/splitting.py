"""Split a turn into the facts it carries, before anything is written (#34)

    python3 scripts/splitting.py "text carrying several facts"
    python3 scripts/splitting.py --file turn.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import splitting  # noqa: E402
from cortex.extraction import extract_candidates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*")
    parser.add_argument("--file")
    parser.add_argument("--candidates", action="store_true", help="also show what extraction keeps")
    args = parser.parse_args()

    text = Path(args.file).read_text() if args.file else " ".join(args.text)
    described = splitting.describe(text)
    print(f"{described['count']} fact(s) (cap {splitting.MAX_FACTS})")
    for index, fact in enumerate(described["facts"], start=1):
        print(f"  {index}. {fact}")
    if args.candidates:
        kept = extract_candidates(text)
        print(f"extraction keeps {len(kept)}:")
        for candidate in kept:
            print(f"  - [{candidate.kind}] {candidate.content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
