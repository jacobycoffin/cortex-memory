"""Labels you can trust are labels somebody else made (#207)

A retrieval eval is only as honest as its labels, and there is one way the labels go quietly wrong:
**the same person, or the same process, produces them and also produces the signal being measured.**
This project has that exact shape — the frozen set's relevant ids come from `access_log.event='used'`,
which is the event the use-boost ranks on — so a gain on that set can partly be a measurement of
itself.

Two things make labels independent in the way that matters:

- **a different labeller**, recorded per case, so "agreement" can be computed by someone who did not
  write the first file, and so a single-labelled case is visible as unverified rather than assumed
  correct;
- **a blind order.** A second labeller who sees the first file's case order has been anchored by it;
  the cases are shuffled with a seed before they are handed over, and the same seed reproduces the
  same order.

Three rules this module refuses to break:

- **one labeller is not 100% agreement.** With a single label set, agreement is *undefined*; reporting
  it as perfect would turn "nobody checked" into a green number;
- **a case both labellers tagged with nothing in common is a disagreement**, not a Jaccard divide-by-
  zero to be skipped;
- **a case labelled by only one side is counted as uncovered**, never silently dropped from the totals
  — otherwise coverage improves every time a labeller skips a hard case.

The adversarial slice is the complement of a random one: cases where the retriever was *confident* and
*wrong*. Those are rare, and they are where ranking changes actually show up, so they are worth
selecting deliberately instead of hoping a random sample contains them.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

UNATTRIBUTED = "unattributed"


@dataclass
class LabelSet:
    labeller: str = UNATTRIBUTED
    cases: dict[str, set[str]] = field(default_factory=dict)
    path: str = ""

    def __len__(self) -> int:
        return len(self.cases)


def read(path: str | Path, *, labeller: str = "") -> LabelSet:
    """Read a JSONL label file. The labeller comes from the file, or from the caller."""
    found = LabelSet(path=str(path))
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except Exception:  # noqa: BLE001 - a missing file is a gap, not a crash
        return found
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        case_id = str(row.get("case_id") or "")
        if not case_id:
            continue
        relevant = row.get("relevant_memory_ids") or row.get("relevant") or []
        found.cases[case_id] = {str(item) for item in relevant}
        if not found.labeller or found.labeller == UNATTRIBUTED:
            found.labeller = str(row.get("labeller") or labeller or UNATTRIBUTED)
    if found.labeller == UNATTRIBUTED and labeller:
        found.labeller = str(labeller)
    return found


def overlap(first: Iterable[str], second: Iterable[str]) -> float:
    """Jaccard. Two empty answers agree; one empty answer does not."""
    left, right = set(first), set(second)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def agreement(first: LabelSet, second: LabelSet) -> dict[str, Any]:
    """How well two labellers agree, on the cases they both labelled."""
    shared = sorted(set(first.cases) & set(second.cases))
    if not shared:
        return {
            "cases": 0, "mean_jaccard": None, "exact": 0,
            "note": "no case was labelled by both, so agreement is undefined rather than perfect",
            "undecided": sorted(set(first.cases) ^ set(second.cases))[:0],
        }
    scores = {case_id: overlap(first.cases[case_id], second.cases[case_id]) for case_id in shared}
    exact = sum(1 for value in scores.values() if value == 1.0)
    return {
        "cases": len(shared),
        "mean_jaccard": round(sum(scores.values()) / len(scores), 4),
        "exact": exact,
        "exact_share": round(exact / len(shared), 4),
        "zero_overlap": sum(1 for value in scores.values() if value == 0.0),
        "scores": scores,
        "note": "",
    }


def coverage(sets: Sequence[LabelSet]) -> dict[str, Any]:
    """Which cases carry more than one independent label, and which carry none at all."""
    seen: dict[str, set[str]] = {}
    for index, found in enumerate(sets):
        name = found.labeller or f"set-{index}"
        for case_id in found.cases:
            seen.setdefault(case_id, set()).add(name)
    total = len(seen)
    independent = sum(1 for names in seen.values() if len(names) > 1)
    single = sum(1 for names in seen.values() if len(names) == 1)
    return {
        "cases": total,
        "independent": independent,
        "single": single,
        "independent_share": round(independent / total, 4) if total else 0.0,
        "labellers": sorted({name for names in seen.values() for name in names}),
        "note": (
            "a single-labelled case is unverified, not correct"
            if single and not independent else ""
        ),
    }


def disagreements(first: LabelSet, second: LabelSet, *, limit: int = 20
                  ) -> list[dict[str, Any]]:
    """The cases the two labellers read differently, worst first."""
    shared = sorted(set(first.cases) & set(second.cases))
    rows = []
    for case_id in shared:
        left, right = first.cases[case_id], second.cases[case_id]
        score = overlap(left, right)
        if score == 1.0:
            continue
        rows.append({
            "case_id": case_id,
            "jaccard": round(score, 4),
            "only_first": sorted(left - right),
            "only_second": sorted(right - left),
        })
    rows.sort(key=lambda row: (row["jaccard"], row["case_id"]))
    return rows[: int(limit)]


def blind_order(case_ids: Iterable[str], *, seed: str = "") -> list[str]:
    """A reproducible shuffle, so a second labeller cannot infer the first one's ordering."""
    ids = [str(item) for item in case_ids]
    digest = hashlib.sha256(("|".join(sorted(ids)) + str(seed)).encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    shuffled = list(ids)
    rng.shuffle(shuffled)
    return shuffled


def adversarial(judgements: Iterable[dict[str, Any]], *, min_gap: float = 0.05,
                limit: int = 30) -> list[dict[str, Any]]:
    """Cases where the retriever was confident and wrong.

    `judgements` are per-case observations: `{case_id, gap, correct}` where `gap` is the score margin
    between the pick and the runner-up and `correct` says whether the pick was relevant. A high gap
    with an incorrect pick is the case a ranking change actually has to fix; a low gap is a coin toss
    and proves nothing either way.
    """
    picked = []
    for row in judgements:
        try:
            gap = float(row.get("gap") or 0.0)
        except (TypeError, ValueError):
            continue
        if bool(row.get("correct")):
            continue
        if gap < float(min_gap):
            continue
        picked.append({
            "case_id": str(row.get("case_id") or ""),
            "gap": round(gap, 6),
            "note": "confident and wrong: the case a ranking change has to answer for",
        })
    picked.sort(key=lambda row: (-row["gap"], row["case_id"]))
    return picked[: int(limit)]


def from_report(cases: Iterable[dict[str, Any]], *, top_k: int = 6) -> dict[str, Any]:
    """Select the adversarial slice from a frozen-eval report, and say what it cannot select.

    A report row carries `first_relevant_rank` (rank of the best relevant memory *that was returned*)
    and `hit_at_k`. It does **not** carry the score of a memory that was not returned, so a miss reads
    as `rank=None` whether the correct memory came 7th or 700th. That makes "confident and wrong"
    undecidable after the fact: the adversarial slice needs the margin recorded at run time, not
    reconstructed from the returned list.
    """
    rows = [dict(row) for row in cases]
    misses = [row for row in rows if not row.get("hit_at_k")]
    with_rank = [row for row in misses if row.get("first_relevant_rank") is not None]
    orderable = [row for row in with_rank
                 if top_k < int(row["first_relevant_rank"]) <= 2 * int(top_k)]
    orderable.sort(key=lambda row: int(row["first_relevant_rank"]))
    return {
        "cases": len(rows),
        "hits": sum(1 for row in rows if row.get("hit_at_k")),
        "misses": len(misses),
        "misses_with_a_recorded_rank": len(with_rank),
        "near_misses": len(orderable),
        "selected": [str(row.get("case_id") or row.get("case_index") or "") for row in orderable],
        "note": (
            "a miss with no recorded rank cannot be told apart from a near miss"
            if misses and not with_rank else ""
        ),
    }


def summary(sets: Sequence[LabelSet]) -> dict[str, Any]:
    """Coverage plus the pairwise agreement of the first two independent labellers."""
    described: dict[str, Any] = {"coverage": coverage(sets)}
    if len(sets) < 2:
        described["agreement"] = {
            "cases": 0, "mean_jaccard": None,
            "note": "one label set: agreement is undefined, not perfect",
        }
        return described
    described["agreement"] = agreement(sets[0], sets[1])
    described["disagreements"] = disagreements(sets[0], sets[1], limit=5)
    return described
