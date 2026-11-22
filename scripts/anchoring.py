"""Anchor relative time in a piece of text (#40)

    python3 scripts/anchoring.py "the deploy broke last Tuesday"
    python3 scripts/anchoring.py --at 2026-11-18 "on call yesterday"
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import anchoring  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*")
    parser.add_argument("--at", help="reference date (YYYY-MM-DD); defaults to today")
    args = parser.parse_args()

    text = " ".join(args.text)
    if not text:
        print("pass some text")
        return 2
    moment = date.fromisoformat(args.at) if args.at else None
    result = anchoring.anchor(text, at=moment)
    print(result.text)
    print("resolved:", result.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
