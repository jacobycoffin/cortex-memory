"""Dormancy: memories that only ever surface as noise get demoted first (#94).

The store already records the difference between "was shown" and "was used":
`retrieved_count`/`injected_count` against `used_count`. A memory with a high retrieval count and
zero uses has been offered many times and never once helped — that is the strongest signal of
noise the schema can produce, and it should shorten the wait, not lengthen it.

Two rules keep it honest:

1. **A minimum sample.** One retrieval with no use is ordinary; five is a pattern. `MIN_RETRIEVED`
   defaults to 4 and the ratio only counts above it.
2. **Never-walk kinds and protections still win.** A dormant *procedure* is still the procedure.
   Dormancy shortens a window inside the ladder's policy; it never overrides the things the
   ladder refuses to touch.

This is the mirror image of the regret guard: regret widens a window because the ladder was
wrong, dormancy narrows one because the memory has already been tested.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Measured on the live store (3,828 active/cold) before choosing: 4 offers at 0.85 flagged
# 1,077 memories — 28% of the store, which is not "noise", it is "most of memory". 12 offers at
# 0.95 flags 153 (4%), and those are the ones offered a dozen times and never once used.
MIN_RETRIEVED = 12
DORMANT_RATIO = 0.95
# A dormant memory's waits are cut to this fraction of normal — never below MIN_DAYS.
DORMANT_SCALE = 0.5
MIN_DAYS = 7


def _min_retrieved() -> int:
    raw = os.environ.get("CORTEX_DORMANT_MIN_RETRIEVED", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return MIN_RETRIEVED
    return value if value >= 1 else MIN_RETRIEVED


def enabled() -> bool:
    return os.environ.get("CORTEX_DORMANCY", "1").strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Dormancy:
    retrieved: int
    used: int
    ratio: float

    @property
    def dormant(self) -> bool:
        return self.ratio >= DORMANT_RATIO


def assess(row: dict[str, Any]) -> Dormancy:
    """How noisy has this memory been? Ratio of offers to uses, above the minimum sample."""
    try:
        retrieved = int(row.get("retrieved_count") or 0) + int(row.get("injected_count") or 0)
    except (TypeError, ValueError):
        retrieved = 0
    try:
        used = int(row.get("used_count") or 0)
    except (TypeError, ValueError):
        used = 0
    if retrieved < _min_retrieved():
        return Dormancy(retrieved=retrieved, used=used, ratio=0.0)
    ratio = (retrieved - used) / retrieved if retrieved else 0.0
    return Dormancy(retrieved=retrieved, used=used, ratio=max(0.0, min(1.0, ratio)))


def scale(row: dict[str, Any]) -> float:
    """Multiplier applied to this memory's ladder windows: 1.0 normally, DORMANT_SCALE if noisy."""
    if not enabled():
        return 1.0
    return DORMANT_SCALE if assess(row).dormant else 1.0


def apply_to(dwell: tuple[int, int, int], row: dict[str, Any]) -> tuple[int, int, int]:
    multiplier = scale(row)
    if multiplier == 1.0:
        return dwell
    return tuple(max(MIN_DAYS, int(round(day * multiplier))) for day in dwell)  # type: ignore[return-value]


def report(store: Any, *, limit: int = 15) -> dict[str, Any]:
    """Which memories are the noisiest, and what that would do to their windows."""
    rows = store._conn.execute(
        """SELECT id, kind, content, state, retrieved_count, injected_count, used_count,
                  last_used_at, created_at
           FROM memories WHERE state IN ('active','cold')"""
    ).fetchall()
    assessed = []
    for raw in rows:
        row = dict(raw)
        result = assess(row)
        if result.dormant:
            assessed.append(
                {
                    "memory_id": str(row["id"]),
                    "kind": row["kind"],
                    "retrieved": result.retrieved,
                    "used": result.used,
                    "ratio": round(result.ratio, 3),
                    "content": " ".join(str(row.get("content") or "").split())[:90],
                }
            )
    assessed.sort(key=lambda item: (-item["ratio"], -item["retrieved"], item["memory_id"]))
    return {
        "enabled": enabled(),
        "min_retrieved": _min_retrieved(),
        "dormant_ratio": DORMANT_RATIO,
        "scale": DORMANT_SCALE,
        "count": len(assessed),
        "worst": assessed[:limit],
    }
