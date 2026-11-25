"""Consolidation A/B (#119): did last night make retrieval better or worse?

A night of consolidation changes the store, and nothing measured whether the change helped. This compares the same
frozen query set against two states - before the night and after it - and returns a verdict per metric.

Five rules, each from a way an A/B lies:

1. **The worst metric decides.** An improvement in `mrr` never cancels a regression in `hit_at_k`; a night that helps
   one query class while breaking another is a regression with a good average.
2. **Tolerance, not noise.** A delta inside the tolerance is `flat`, not "improved by 0.004".
3. **The same cases on both sides.** If the two runs scored different numbers of cases, the comparison is **refused** -
   a metric computed over different questions is not a comparison.
4. **A missing metric is `unmeasurable`, not zero.** Absence of evidence never reads as a number.
5. **Nothing private leaves this module.** The frozen set is private: this reads a set *path*, refuses one inside the
   repository, and its output contains metric names and numbers - never a query, never a memory id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# Mirrors the release gate in scripts/frozen_evalset.py: the same numbers decide "changed" in both places, so a night
# cannot pass an A/B that the gate would fail.
TOLERANCES = {
    "hit_at_k": 0.01,
    "mean_recall_at_k": 0.01,
    "mrr": 0.02,
    "zero_result_rate": 0.01,
}
HIGHER_IS_BETTER = ("hit_at_k", "mean_recall_at_k", "mrr")
LOWER_IS_BETTER = ("zero_result_rate",)

IMPROVED = "improved"
REGRESSED = "regressed"
FLAT = "flat"
UNMEASURABLE = "unmeasurable"


@dataclass(frozen=True)
class Delta:
    metric: str
    before: float
    after: float
    verdict: str
    tolerance: float

    @property
    def change(self) -> float:
        return self.after - self.before

    def render(self) -> str:
        return (f"{self.metric} {self.before:.4f} -> {self.after:.4f} "
                f"({self.change:+.4f}, tolerance {self.tolerance}) {self.verdict}")

    def as_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "before": self.before, "after": self.after,
                "change": round(self.change, 6), "tolerance": self.tolerance, "verdict": self.verdict}


@dataclass
class Comparison:
    deltas: list[Delta] = field(default_factory=list)
    verdict: str = FLAT
    cases: int = 0
    reason: str = ""

    @property
    def regressions(self) -> list[Delta]:
        return [delta for delta in self.deltas if delta.verdict == REGRESSED]

    def render(self) -> list[str]:
        if self.verdict == UNMEASURABLE:
            return [f"unmeasurable: {self.reason}"]
        lines = [f"{self.cases} case(s): {self.verdict}"]
        for delta in self.deltas:
            lines.append(f"  - {delta.render()}")
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "cases": self.cases, "reason": self.reason,
                "deltas": [delta.as_dict() for delta in self.deltas]}


def _metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare(before: Mapping[str, Any], after: Mapping[str, Any], *,
            cases_before: int | None = None, cases_after: int | None = None) -> Comparison:
    """Compare two metric summaries of the same query set. Reads nothing, writes nothing."""
    if cases_before is not None and cases_after is not None and cases_before != cases_after:
        return Comparison(verdict=UNMEASURABLE,
                          reason=f"different case counts ({cases_before} vs {cases_after}): not the same comparison")
    deltas: list[Delta] = []
    for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        old = _metric(before.get(metric))
        new = _metric(after.get(metric))
        tolerance = TOLERANCES[metric]
        if old is None or new is None:
            deltas.append(Delta(metric, old if old is not None else 0.0, new if new is not None else 0.0,
                                UNMEASURABLE, tolerance))
            continue
        change = new - old
        if metric in LOWER_IS_BETTER:
            change = -change  # normalise so "bigger is better" holds for the verdict
        if change > tolerance:
            verdict = IMPROVED
        elif change < -tolerance:
            verdict = REGRESSED
        else:
            verdict = FLAT
        deltas.append(Delta(metric, old, new, verdict, tolerance))
    # The worst check decides: one regression outweighs any number of improvements.
    if any(delta.verdict == REGRESSED for delta in deltas):
        overall = REGRESSED
    elif any(delta.verdict == IMPROVED for delta in deltas):
        overall = IMPROVED
    elif all(delta.verdict == UNMEASURABLE for delta in deltas):
        overall = UNMEASURABLE
    else:
        overall = FLAT
    reason = ""
    if overall == UNMEASURABLE:
        reason = "no metric had a value on both sides"
    return Comparison(deltas=deltas, verdict=overall,
                      cases=cases_before if cases_before is not None else 0, reason=reason)


def assert_set_is_private(set_dir: str | Path, *, repo_root: str | Path) -> None:
    """Refuse a frozen set that lives inside the repository.

    The set holds real queries and real memory ids. A set path under version control is a set that will be committed,
    so this raises rather than warning.
    """
    resolved = Path(set_dir).resolve()
    root = Path(repo_root).resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError(f"the frozen set must live outside the repository, not in {resolved}")


def run_pair(measure, *, before_store: str | Path, after_store: str | Path,
             set_dir: str | Path, repo_root: str | Path) -> Comparison:
    """Measure both states with the caller's evaluator and compare them.

    `measure(store_path, set_dir)` returns a summary dict with the metric names above plus `cases`. The evaluator is
    injected because the frozen set is private and must not be reachable from a test.
    """
    assert_set_is_private(set_dir, repo_root=repo_root)
    before = measure(before_store, set_dir)
    after = measure(after_store, set_dir)
    return compare(before, after,
                   cases_before=int(before.get("cases") or 0),
                   cases_after=int(after.get("cases") or 0))
