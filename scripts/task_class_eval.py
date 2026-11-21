"""Measure task classification against a labelled set (#218).

    python3 scripts/task_class_eval.py                 # new classifier
    python3 scripts/task_class_eval.py --legacy        # the first-match-wins rules
    python3 scripts/task_class_eval.py --confusions    # what each one gets wrong

The labelled set is synthetic (written to exercise the vocabulary collisions), so it lives
in the repo — unlike the frozen recall set, which contains real queries and memory ids.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex.task_class import GENERAL, classify  # noqa: E402
from cortex.tooling import _TASK_RULES  # noqa: E402


def legacy(query: str) -> str:
    for name, pattern in _TASK_RULES:
        if pattern.search(query):
            return name
    return GENERAL


def load(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set",
        default=str(ROOT / "evalsets" / "task-class-v1.jsonl"),
        help="labelled JSONL with query + task_type",
    )
    parser.add_argument("--legacy", action="store_true", help="score the old first-match rules")
    parser.add_argument("--confusions", action="store_true", help="print every miss")
    args = parser.parse_args()

    cases = load(Path(args.set))
    correct = 0
    confusions: Counter[tuple[str, str]] = Counter()
    misses: list[str] = []
    for case in cases:
        query, expected = case["query"], case["task_type"]
        predicted = legacy(query) if args.legacy else classify(query).task_type
        if predicted == expected:
            correct += 1
        else:
            confusions[(expected, predicted)] += 1
            misses.append(f"  {query!r}: expected {expected}, got {predicted}")

    total = len(cases)
    print(f"{'legacy' if args.legacy else 'scored'}: {correct}/{total} correct ({correct / total:.1%})")
    for (expected, predicted), count in confusions.most_common(6):
        print(f"  {expected} -> {predicted}: {count}")
    if args.confusions:
        print("\n".join(misses))
    return 0 if correct == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
