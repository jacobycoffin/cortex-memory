"""When the night is short, do the work that matters first (#117).

A nightly cycle has more work available than budget: replay wants fifty units, consolidation wants the unclassified
backlog, the digest wants twenty minutes. `sleep_budget` answers *may I start another unit*; this answers **which**
unit. The difference matters only when the budget runs out — and that is exactly when a scheduler that just walks its
list in a fixed order looks like it is working while doing the least valuable third of the job.

The rule is value per millisecond, and five constraints keep it honest:

- **cost is measured or marked unmeasured.** A kind whose unit cost nobody measured cannot be ranked — it is placed
  last, labelled `unmeasured`, instead of being given a made-up number. (A constant in a module is a guess about a
  distribution; a guess wearing a decimal point is worse than an admission.)
- **value is a declared number, not a feeling.** Each kind carries `value` in (0, 1] chosen by whoever owns the work,
  and it is printed with every decision so a wrong value can be argued with.
- **a unit that does not fit is skipped, never truncated.** Half a replay unit is a half-written trace; the budget
  says no and the plan says so.
- **the plan names the owner of each kind** — `replay_schedule`, `merge_diff`, … — so the scheduler can never invent
  work it has no way to hand off.
- **the order is deterministic** (ratio, then value, then kind), so two runs on the same inputs produce the same night.

Read-only: `plan()` computes an order; it consumes nothing and runs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

UNMEASURED = "unmeasured"


@dataclass(frozen=True)
class Kind:
    """One class of nightly work. `unit_cost_ms` is None when nobody measured it."""

    name: str
    owner: str
    value: float
    unit_cost_ms: float | None
    available: int

    def __post_init__(self) -> None:
        if not (0.0 < self.value <= 1.0):
            raise ValueError(f"value for {self.name} must be in (0, 1], got {self.value}")
        if self.available < 0:
            raise ValueError(f"available for {self.name} must be >= 0")

    @property
    def measured(self) -> bool:
        return self.unit_cost_ms is not None and self.unit_cost_ms > 0

    def ratio(self) -> float:
        """Value per millisecond. Unmeasured kinds have no ratio and are never ranked by a fake one."""
        cost = self.unit_cost_ms
        if cost is None or cost <= 0:
            return 0.0
        return float(self.value) / float(cost)


@dataclass
class Task:
    kind: str
    owner: str
    unit_cost_ms: float | None
    value: float
    reason: str = ""

    def render(self) -> str:
        cost = f"{self.unit_cost_ms:.0f}ms" if self.unit_cost_ms else UNMEASURED
        return f"{self.kind} x1 ({cost}, value {self.value:.2f}) via {self.owner}"


@dataclass
class Plan:
    budget_ms: float
    order: list[Task] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    unmeasured: list[str] = field(default_factory=list)
    spent_ms: float = 0.0

    @property
    def remaining_ms(self) -> float:
        return max(0.0, self.budget_ms - self.spent_ms)

    def render(self) -> list[str]:
        lines = [f"{len(self.order)} kind(s) in {self.spent_ms:.0f}ms of {self.budget_ms:.0f}ms"]
        lines.extend(f"- {task.render()}" for task in self.order)
        if self.unmeasured:
            lines.append(f"unmeasured (last, because nobody knows what they cost): {', '.join(self.unmeasured)}")
        if self.skipped:
            lines.append("skipped: " + ", ".join(f"{kind} ({reason})" for kind, reason in sorted(self.skipped.items())))
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {"budget_ms": self.budget_ms, "spent_ms": round(self.spent_ms, 1),
                "order": [{"kind": t.kind, "owner": t.owner, "value": t.value,
                           "unit_cost_ms": t.unit_cost_ms, "reason": t.reason} for t in self.order],
                "unmeasured": self.unmeasured, "skipped": self.skipped}


def plan(kinds: Iterable[Kind], *, budget_ms: float) -> Plan:
    """Order work by value per millisecond and take what fits. Nothing is truncated."""
    budget = max(0.0, float(budget_ms))
    out = Plan(budget_ms=budget)
    pool = list(kinds)

    measured = sorted(
        (kind for kind in pool if kind.measured and kind.available > 0),
        key=lambda kind: (-kind.ratio(), -kind.value, kind.name),
    )
    for kind in measured:
        if out.remaining_ms <= 0:
            out.skipped[kind.name] = "budget exhausted"
            continue
        fits = int(out.remaining_ms // float(kind.unit_cost_ms or 1.0))
        if fits <= 0:
            out.skipped[kind.name] = f"one unit costs {kind.unit_cost_ms:.0f}ms, {out.remaining_ms:.0f}ms left"
            continue
        units = min(fits, kind.available)
        out.order.append(Task(kind=kind.name, owner=kind.owner, unit_cost_ms=kind.unit_cost_ms,
                              value=kind.value,
                              reason=f"{units} of {kind.available} unit(s) fit"))
        out.spent_ms += units * float(kind.unit_cost_ms or 0.0)
        if units < kind.available:
            out.skipped[kind.name] = f"{kind.available - units} unit(s) did not fit"

    # Nothing measured, nothing ranked: a kind whose cost nobody knows goes last, in name order, and says so.
    for kind in sorted((k for k in pool if not k.measured and k.available > 0), key=lambda k: (-k.value, k.name)):
        out.unmeasured.append(kind.name)
        out.order.append(Task(kind=kind.name, owner=kind.owner, unit_cost_ms=None, value=kind.value,
                              reason="cost unmeasured: scheduled last, ranked by nothing"))
    for kind in pool:
        if kind.available <= 0:
            out.skipped.setdefault(kind.name, "nothing available")
    return out


def measured_from(kind: str, owner: str, value: float, available: int,
                  costs_ms: Iterable[float]) -> Kind:
    """Build a kind from observed costs. The median, not the mean: one 12-second outlier must not set the price."""
    samples = sorted(float(cost) for cost in costs_ms if cost and cost > 0)
    if not samples:
        return Kind(name=kind, owner=owner, value=value, unit_cost_ms=None, available=available)
    return Kind(name=kind, owner=owner, value=value, unit_cost_ms=samples[len(samples) // 2],
                available=available)
