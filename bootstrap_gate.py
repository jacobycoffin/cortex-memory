"""A policy change ships on a confidence interval, or it does not ship (#210).

Averages lie about ranking changes. Two policies differ by +0.004 hit@6 and a table calls it an
improvement, when the difference is two cases out of two hundred and the same evaluation would report
-0.002 tomorrow. Worse, the same cases are used for both policies, so the two numbers are **paired** —
treating them as independent samples throws away the only thing that makes the comparison sensitive:
the per-case variation they share.

So the gate is a **paired bootstrap** over the per-case differences:

- **paired, not pooled.** Each case contributes one delta; resampling is over cases, keeping each
  case's two numbers together.
- **an interval, not a point.** 2,000 resamples by default, and the verdict is about whether the
  interval excludes zero — not whether the mean is positive.
- **a refusal is a result.** Fewer than `MIN_CASES` paired cases is `undecided`, not a small win; a
  change that cannot be told apart from noise is `undecided`, not "probably fine".
- **reproducible.** The resampling seed is derived from the data, so two runs on the same numbers give
  the same interval — otherwise a gate becomes a lottery somebody re-runs until it passes.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

MIN_CASES = 30
RESAMPLES = 2000
ALPHA = 0.05

BETTER = "better"
WORSE = "worse"
SAME = "undecided"


def _seed_for(pairs: Sequence[tuple[float, float]]) -> int:
    """A seed that depends only on the data, so the interval is a property of the measurement."""
    digest = hashlib.sha256()
    for left, right in pairs:
        digest.update(f"{left:.9f},{right:.9f};".encode())
    return int.from_bytes(digest.digest()[:8], "big")


@dataclass
class Verdict:
    decision: str = SAME
    cases: int = 0
    mean_delta: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    resamples: int = 0
    alpha: float = ALPHA
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def decided(self) -> bool:
        return self.decision in {BETTER, WORSE}

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "cases": self.cases,
            "mean_delta": round(self.mean_delta, 6),
            "lower": round(self.lower, 6),
            "upper": round(self.upper, 6),
            "resamples": self.resamples,
            "alpha": self.alpha,
            "reason": self.reason,
            "notes": self.notes,
            "sentence": self.describe(),
        }

    def describe(self) -> str:
        span = f"[{self.lower:+.4f}, {self.upper:+.4f}]"
        if not self.cases:
            return f"undecided: nothing paired ({self.reason})"
        if self.decision == BETTER:
            return f"better by {self.mean_delta:+.4f}, 95% CI {span} over {self.cases} case(s)"
        if self.decision == WORSE:
            return f"worse by {self.mean_delta:+.4f}, 95% CI {span} over {self.cases} case(s)"
        return f"undecided: {self.mean_delta:+.4f}, 95% CI {span} over {self.cases} case(s)"


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def paired_gate(baseline: Iterable[float], candidate: Iterable[float], *,
                resamples: int = RESAMPLES, alpha: float = ALPHA,
                min_cases: int = MIN_CASES) -> Verdict:
    """Is `candidate` better than `baseline`, case by case, beyond noise?"""
    left = [float(value) for value in baseline]
    right = [float(value) for value in candidate]
    pairs = list(zip(left, right))
    if len(left) != len(right):
        return Verdict(decision=SAME, cases=min(len(left), len(right)),
                       reason="the two measurements must cover the same cases")
    if not pairs:
        return Verdict(decision=SAME, cases=0, reason="no paired cases")
    deltas = [b - a for a, b in pairs]
    mean = sum(deltas) / len(deltas)
    if len(pairs) < int(min_cases):
        return Verdict(decision=SAME, cases=len(pairs), mean_delta=mean,
                       reason=f"{len(pairs)} paired case(s), need {int(min_cases)} to decide")
    if all(delta == 0.0 for delta in deltas):
        return Verdict(decision=SAME, cases=len(pairs), mean_delta=0.0,
                       reason="every case is identical, so the policies are the same")

    rng = random.Random(_seed_for(pairs))
    count = len(deltas)
    means: list[float] = []
    for _ in range(max(100, int(resamples))):
        total = 0.0
        for _ in range(count):
            total += deltas[rng.randrange(count)]
        means.append(total / count)
    lower = _percentile(means, alpha / 2)
    upper = _percentile(means, 1 - (alpha / 2))
    verdict = Verdict(decision=SAME, cases=count, mean_delta=mean, lower=lower, upper=upper,
                      resamples=max(100, int(resamples)), alpha=alpha)
    if lower > 0:
        verdict.decision = BETTER
        verdict.reason = "the whole interval is above zero"
    elif upper < 0:
        verdict.decision = WORSE
        verdict.reason = "the whole interval is below zero"
    else:
        verdict.reason = "the interval spans zero, so this change is not distinguishable from noise"
    if count < max(2 * int(min_cases), 60):
        verdict.notes.append(
            f"small sample ({count} cases): the interval is wide and will stay wide"
        )
    return verdict


def from_hits(baseline: Iterable[bool], candidate: Iterable[bool], **kwargs: Any) -> Verdict:
    """Convenience for binary per-case metrics such as hit@k."""
    return paired_gate([float(bool(value)) for value in baseline],
                       [float(bool(value)) for value in candidate], **kwargs)


def report(named: dict[str, tuple[Sequence[float], Sequence[float]]], **kwargs: Any
           ) -> dict[str, Any]:
    """Gate several candidate policies against one baseline in one call."""
    verdicts = {name: paired_gate(left, right, **kwargs).as_dict()
                for name, (left, right) in named.items()}
    decided = [name for name, item in verdicts.items() if item["decision"] in {BETTER, WORSE}]
    return {
        "policies": verdicts,
        "decided": decided,
        "undecided": [name for name in verdicts if name not in decided],
        "note": "a change that cannot be told apart from noise is not shipped",
    }
