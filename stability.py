"""Recency versus stability: a fact restated ten times is not beaten by one fresh sentence.

Retrieval is biased toward whatever was written most recently. That is usually right — a
changed preference should win — but it means a single off-hand remark can bury something the
user has confirmed repeatedly over months, which is exactly the memory that makes an agent
feel like it knows you.

This blends two signals that are kept deliberately separate:

- **corroboration** — how many independent supports a memory has (`confirmed_count`,
  `validated_count`, `helpful_count`). Saturating: five supports reach full weight, the
  twentieth adds almost nothing, so a chatty loop cannot manufacture stability.
- **recency of the last support** — decays over a long half-life, so stability fades
  rather than vanishing.

A memory corroborated only once is deliberately held to half weight: one statement is a
claim, not a pattern. The resulting boost is small (max `0.02`) and can never reorder
memories whose relevance genuinely differs — it settles near-ties in favour of the fact
that has actually held up over time.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Iterable

MAX_BOOST = 0.02
SATURATION_AT = 5.0
HALF_LIFE_DAYS = 180.0
SINGLE_SUPPORT_FACTOR = 0.5


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def corroboration(memory: dict[str, Any]) -> int:
    """Independent supports, ignoring a memory's own retrieval counters."""
    total = 0
    for field in ("confirmed_count", "validated_count", "helpful_count"):
        try:
            total += max(0, int(memory.get(field) or 0))
        except (TypeError, ValueError):
            continue
    return total


def last_support(memory: dict[str, Any]) -> datetime | None:
    stamps = [
        _parse(memory.get(field))
        for field in ("last_helpful_at", "last_used_at", "observed_at", "updated_at", "created_at")
    ]
    present = [s for s in stamps if s is not None]
    return max(present) if present else None


def stability(memory: dict[str, Any], *, now: datetime | None = None) -> float:
    """0..1 stability for one memory."""
    now = now or datetime.now(timezone.utc)
    supports = corroboration(memory)
    if supports <= 0:
        return 0.0

    saturation = min(1.0, math.log1p(supports) / math.log1p(SATURATION_AT))
    if supports < 2:
        saturation *= SINGLE_SUPPORT_FACTOR

    stamp = last_support(memory)
    if stamp is None:
        return 0.0
    age_days = max(0.0, (now - stamp).total_seconds() / 86400.0)
    decay = 0.5 ** (age_days / HALF_LIFE_DAYS)
    return max(0.0, min(1.0, saturation * decay))


def apply_stability_blend(
    results: Iterable[Any], *, now: datetime | None = None
) -> tuple[list[Any], int]:
    """Add the (small) stability boost. Returns (results, count)."""
    results = list(results)
    now = now or datetime.now(timezone.utc)
    out: list[Any] = []
    boosted = 0
    for result in results:
        memory = getattr(result, "memory", {}) or {}
        value = stability(memory, now=now)
        if value <= 0.0:
            out.append(result)
            continue
        components = dict(getattr(result, "components", {}) or {})
        components["stability"] = round(value, 6)
        out.append(
            replace(result, score=float(result.score) + MAX_BOOST * value, components=components)
        )
        boosted += 1
    return out, boosted
