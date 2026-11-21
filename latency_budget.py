"""A hard latency ceiling: return the best recall so far instead of finishing (#22).

Recall has a soft SLO (p50 ≤150ms, p95 ≤250ms, max ≤900ms) but nothing enforcing it. On a
cold cache with a large candidate pool a single query can run long, and a turn waits for it
— the agent is late for something the user never sees.

A budget is a deadline, not a target. When it expires the expensive optional phases stop
being started (semantic search, graph expansion), and the selection loop keeps whatever it
has already chosen. **Best-so-far always beats late-and-complete** for an interactive agent:
a slightly worse memory block now is worth more than a perfect one after the user has moved
on.

Off unless configured. Skipping work changes results, and a deterministic retriever should
not silently become time-dependent — so a deployment opts in, and the tests here prove the
degradation path is safe rather than assuming it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

DEFAULT_BUDGET_MS = 0.0  # disabled
ENV = "CORTEX_LATENCY_BUDGET_MS"


@dataclass
class Budget:
    """A monotonic deadline in milliseconds. `ms <= 0` means no budget."""

    ms: float = DEFAULT_BUDGET_MS
    _start: float = 0.0

    def __post_init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def enabled(self) -> bool:
        return self.ms and self.ms > 0

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

    def remaining_ms(self) -> float:
        if not self.enabled:
            return float("inf")
        return self.ms - self.elapsed_ms()

    def expired(self) -> bool:
        return bool(self.enabled and self.remaining_ms() <= 0.0)


def budget_from(configured: float | None = None) -> Budget:
    """Resolution order: explicit argument, environment, then disabled."""
    if configured is not None:
        return Budget(float(configured))
    raw = os.environ.get(ENV, "").strip()
    if raw:
        try:
            return Budget(float(raw))
        except ValueError:
            return Budget(DEFAULT_BUDGET_MS)
    return Budget(DEFAULT_BUDGET_MS)
