"""Are the labels independent of the signal being measured? (#207)

    python3 scripts/label_independence.py --set FROZEN.jsonl
    python3 scripts/label_independence.py --set first.jsonl --set second.jsonl
    python3 scripts/label_independence.py --set FROZEN.jsonl --blind-out /tmp/blind.jsonl --seed x

Read-only except `--blind-out`, which writes a shuffled case order for a second labeller. The label
files themselves live outside this repository on purpose: real queries and real memory ids must never
be committed, so this script prints counts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401

from cortex import label_independence as labels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", action="append", default=[],
                        help="a JSONL label file; repeat for a second labeller")
    parser.add_argument("--blind-out", default="", help="write a shuffled case order here")
    parser.add_argument("--seed", default="")
    parser.add_argument("--show", type=int, default=0, help="print N case ids per disagreement")
    parser.add_argument("--report", default="",
                        help="a frozen-eval report JSON: select the adversarial slice")
    args = parser.parse_args()

    if not args.set:
        print("nothing to read: pass --set <file> (repeat for a second labeller)")
        return 2

    sets = [labels.read(path) for path in args.set]
    described = labels.summary(sets)
    cov = described["coverage"]
    print(f"labeller(s): {', '.join(cov['labellers']) or 'none'}")
    print(f"cases: {cov['cases']}, independently labelled: {cov['independent']} "
          f"({cov['independent_share']:.1%}), single-labelled: {cov['single']}")
    if cov["note"]:
        print(f"  note: {cov['note']}")
    found = described["agreement"]
    if found.get("mean_jaccard") is None:
        print(f"agreement: undefined - {found['note']}")
    else:
        print(f"agreement: mean Jaccard {found['mean_jaccard']}, exact {found['exact']} "
              f"of {found['cases']}, zero-overlap {found['zero_overlap']}")
    for row in described.get("disagreements", []):
        extra = f" (+{row['only_first']} / +{row['only_second']})" if args.show else ""
        print(f"  {row['jaccard']:.2f}  {row['case_id']}{extra}")

    if args.report:
        payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
        sliced = labels.from_report(payload.get("cases") or [])
        print(f"adversarial slice: {sliced['near_misses']} near miss(es) of "
              f"{sliced['misses']} miss(es) in {sliced['cases']} case(s)")
        if sliced["note"]:
            print(f"  limit: {sliced['note']}")

    if args.blind_out:
        order = labels.blind_order(
            {case_id for item in sets for case_id in item.cases}, seed=args.seed
        )
        Path(args.blind_out).write_text(
            "\n".join(json.dumps({"case_id": case_id}) for case_id in order) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(order)} case id(s) in blind order to {args.blind_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
