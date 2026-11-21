"""A/B assign ranking policies to sessions, and compare them fairly (backlog item #25).

The point of the item is that **retrieval metrics are not the outcome**. `hit@6` can improve
while answers get worse, because what matters is whether the injected block lets the model
say the right thing, not whether the right memory reached slot 4.

Two halves:

- `assign()` — a live session deterministically gets one policy, so real traffic can be split
  without a database. Deterministic by session id, so a session never flips mid-conversation
  (a policy switch mid-thread would make the logging useless).
- `compare()` — offline, the *same* cases run under both policies and are scored pairwise.
  Paired comparison is far more sensitive than two independent averages: with 240 cases a
  3-point difference is invisible unpaired and obvious paired.

Judging is pluggable. `coverage` is deterministic and free (does the injected context carry
the fact the case needs); `model` asks a model to answer from the block and grades the
answer against the gold fact — that one costs tokens and is opt-in.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

# Policies are option dicts handed to the retriever, so an experiment never has to patch
# code: {"name": {"option": value}}. Missing keys mean "use the shipped default".
POLICIES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "no_entity_anchor": {"entity_anchor": False},
    "no_usage_boost": {"usage_boost": False},
    "no_rerank": {"rerank": False},
    "no_density": {"density_budget": False},
}


def assign(session_id: str, *, policies: Sequence[str] | None = None) -> str:
    """Pick a policy for a session, stably.

    Stable across turns and processes: a session that flipped policy halfway would poison
    the comparison, and Python's `hash()` is salted per process so it cannot be used.
    """
    names = list(POLICIES if policies is None else policies)
    if not names:
        raise ValueError("no policies to assign")
    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % len(names)
    return names[index]


@dataclass
class CaseOutcome:
    case_id: str
    # Per policy: the judge's score for this case (0.0 = no answer, 1.0 = answered).
    scores: dict[str, float] = field(default_factory=dict)
    # Per policy: the answer or context the judge looked at, kept for inspection.
    evidence: dict[str, str] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class Comparison:
    challenger: str
    baseline: str
    wins: int
    losses: int
    ties: int
    mean_delta: float
    ci_low: float
    ci_high: float
    per_case: list[CaseOutcome]

    @property
    def decisive(self) -> bool:
        """A CI that excludes zero is the only result worth acting on."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def summary(self) -> str:
        verdict = (
            "challenger better" if self.ci_low > 0
            else "challenger worse" if self.ci_high < 0
            else "no difference"
        )
        return (
            f"{self.challenger} vs {self.baseline}: "
            f"{self.wins}W/{self.losses}L/{self.ties}T, "
            f"mean delta {self.mean_delta:+.3f} "
            f"(95% CI {self.ci_low:+.3f}..{self.ci_high:+.3f}) -> {verdict}"
        )


def _bootstrap_ci(deltas: Sequence[float], *, resamples: int = 2000, seed: int = 7) -> tuple[float, float]:
    """Percentile bootstrap of the mean. Deterministic: seeded, no global RNG state."""
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low = means[int(0.025 * resamples)]
    high = means[min(resamples - 1, int(0.975 * resamples))]
    return low, high


def compare(
    run: Callable[[str, Mapping[str, Any]], tuple[float, str, float]],
    cases: Iterable[tuple[str, Any]],
    *,
    baseline: str = "baseline",
    challenger: str,
    policies: Mapping[str, Mapping[str, Any]] | None = None,
) -> Comparison:
    """Run every case under both policies and compare them pairwise.

    `run(policy_name, case)` returns `(score, evidence, latency_ms)`. Both policies see the
    same case, so the comparison is paired and the variance of case difficulty cancels.
    """
    # Validation only applies when the caller supplied a table: `run` receives the policy
    # name and may resolve it itself (tests do exactly that).
    if policies is not None and (challenger not in policies or baseline not in policies):
        raise KeyError(f"unknown policy: {challenger!r} / {baseline!r}")
    outcomes: list[CaseOutcome] = []
    for case_id, case in cases:
        outcome = CaseOutcome(case_id=str(case_id))
        for name in (baseline, challenger):
            score, evidence, latency = run(name, case)
            outcome.scores[name] = float(score)
            outcome.evidence[name] = evidence
            outcome.latency_ms[name] = float(latency)
        outcomes.append(outcome)
    deltas = [o.scores[challenger] - o.scores[baseline] for o in outcomes]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = len(deltas) - wins - losses
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    low, high = _bootstrap_ci(deltas)
    return Comparison(challenger, baseline, wins, losses, ties, mean_delta, low, high, outcomes)


# --- judges ---------------------------------------------------------------


def coverage_judge(gold_text: str | Sequence[str]) -> Callable[[str], float]:
    """Deterministic: did the rendered context carry a fact this case needs?

    Any of the case's relevant memories counts — judging on only the first one measures
    whether that particular memory was recalled, which is a retrieval metric wearing a
    different coat.

    This is a *proxy*, not answer quality: it says the model had what it needed. It is free,
    repeatable and fast, which is why it is the default — but it cannot tell whether the
    model used it, so treat it as an upper bound on answer quality.
    """
    needles = [gold_text] if isinstance(gold_text, str) else list(gold_text)
    needles = [" ".join(n.lower().split()) for n in needles if n.strip()]

    def judge(context: str) -> float:
        haystack = " ".join(context.lower().split())
        return 1.0 if any(needle in haystack for needle in needles) else 0.0

    return judge


def model_judge(
    question: str,
    gold_text: str | Sequence[str],
    *,
    complete: Callable[[str], str],
) -> Callable[[str], float]:
    """Ask a model to answer from the block, then grade the answer against the gold fact.

    `complete(prompt) -> str` is injected so the harness has no provider dependency of its
    own; the caller decides which model, and whether spending tokens is acceptable.
    """
    needles = [gold_text] if isinstance(gold_text, str) else list(gold_text)
    needles = [" ".join(n.lower().split()) for n in needles if n.strip()]

    def judge(context: str) -> float:
        prompt = (
            "Answer the question using only this context. If the context does not contain "
            "the answer, say NOT IN CONTEXT.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        answer = " ".join(complete(prompt).lower().split())
        return 1.0 if any(needle in answer for needle in needles) else 0.0

    return judge
