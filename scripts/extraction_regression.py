"""Run the frozen extraction set, and diff it against a saved baseline (#47)

    python3 scripts/extraction_regression.py                  # run the frozen set
    python3 scripts/extraction_regression.py --only pref-terse
    python3 scripts/extraction_regression.py --save-baseline /tmp/base.json --model gpt-x
    python3 scripts/extraction_regression.py --compare /tmp/base.json --model gpt-y
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import extraction_regression  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="")
    parser.add_argument("--only", default="", help="comma-separated case ids")
    parser.add_argument("--save-baseline", default="")
    parser.add_argument("--compare", default="")
    parser.add_argument("--model", default="deterministic")
    args = parser.parse_args()

    only = [name.strip() for name in args.only.split(",") if name.strip()]
    path = Path(args.fixture) if args.fixture else None
    result = extraction_regression.suite(path=path, only=only)
    print(f"{result['passed']}/{result['cases']} cases pass ({result['failed']} failing)")
    for failure in result["failures"]:
        print(f"  FAIL {failure['id']}: {failure['reason']}")
        for missing in failure["missing"]:
            print(f"    missing: {missing}")
        for forbidden in failure["forbidden"]:
            print(f"    forbidden: {forbidden}")

    current = extraction_regression.baseline(result, model=args.model)
    if args.save_baseline:
        Path(args.save_baseline).write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"baseline saved to {args.save_baseline}")
    if args.compare:
        previous = json.loads(Path(args.compare).read_text())
        compared = extraction_regression.compare(previous, current)
        print(f"model {compared['model_before'] or '?'} -> {compared['model_after'] or '?'}: "
              f"{'changed' if compared['changed'] else 'identical'}")
        for case_id, facts in sorted(compared["appeared"].items()):
            for fact in facts:
                print(f"  + {case_id}: {fact}")
        for case_id, facts in sorted(compared["vanished"].items()):
            for fact in facts:
                print(f"  - {case_id}: {fact}")
        if compared["changed"]:
            return 1
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
