"""Optional Ebbinghaus-style decay for episodic memories (#93).

The ladder forgets by *elapsed quiet time* with a hard step. Ebbinghaus forgetting is smooth:
retention decays exponentially with time, and each successful recall flattens the curve rather
than resetting the clock.

This is **off by default** and applies only to episodic kinds, because that is the only place the
model is defensible: a working note about a Tuesday genuinely does fade, while a preference or a
procedure does not become 40% less true next month. For those kinds the decay factor is exactly
1.0 and the whole module is a no-op.

    strength = exp(-Δt / stability),  stability = (half_life/ln2) × (1 + ln(1 + reinforcements))

The stability term is the important half: an episode recalled three times decays slower than one
never recalled, which is the difference between a forgetting curve and a countdown timer.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any

DECAY_KINDS = frozenset({"episode", "episodic"})
HALF_LIFE_DAYS = 14.0

# Half-life by volatility class (#54). The classes exist because one curve cannot serve both a
# rota entry and a port number: an ephemeral fact should lose confidence in days, a seasonal one
# over a couple of months, and a durable one effectively **plateaus** — 540 days means a stable fact
# is still ~90% at the end of a year, which is the point.
HALF_LIFE_BY_VOLATILITY: dict[str, float] = {
    "ephemeral": 7.0,
    "seasonal": 60.0,
    "durable": 540.0,
}
# Volatility classes allowed to decay even when the kind is not episodic. Durable is included so it
# can plateau explicitly rather than by accident.
DECAYING_VOLATILITY = frozenset({"ephemeral", "seasonal", "durable"})
FLOOR = 0.05
# How much a fully-decayed episode is penalised in scoring. Small on purpose: decay should reorder
# near-ties, not override relevance.
WEIGHT = 0.12


def enabled() -> bool:
    return os.environ.get("CORTEX_DECAY", "").strip().lower() in {"1", "true", "yes", "on"}


def _half_life(volatility: str | None = None) -> float:
    """The half-life for one memory: an explicit override, else its class, else the default."""
    raw = os.environ.get("CORTEX_DECAY_HALF_LIFE_DAYS", "").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if value > 0:
        return value
    name = str(volatility or "").strip().lower()
    return HALF_LIFE_BY_VOLATILITY.get(name, HALF_LIFE_DAYS)


def _utc(text: Any) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _last_rehearsal(row: dict[str, Any]) -> datetime | None:
    for key in ("last_used_at", "last_helpful_at", "last_confirmed_at", "last_retrieved_at",
                "created_at"):
        stamp = _utc(row.get(key))
        if stamp:
            return stamp
    return None


def reinforcements(row: dict[str, Any]) -> int:
    total = 0
    for key in ("used_count", "confirmed_count", "validated_count", "success_count"):
        try:
            total += int(row.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return total


def stability_days(row: dict[str, Any]) -> float:
    base = _half_life(row.get("volatility")) / math.log(2.0)
    return base * (1.0 + math.log(1.0 + reinforcements(row)))


def applies(row: dict[str, Any]) -> bool:
    if not enabled():
        return False
    if str(row.get("kind") or "").strip().lower() in DECAY_KINDS:
        return True
    # A memory that knows how fast it goes stale (#31) decays on that schedule instead of on its
    # kind, which is the whole point of #54.
    return str(row.get("volatility") or "").strip().lower() in DECAYING_VOLATILITY


def strength(row: dict[str, Any], *, at: datetime | None = None) -> float:
    """Retention in [FLOOR, 1.0]. Exactly 1.0 when decay does not apply to this memory."""
    if not applies(row):
        return 1.0
    rehearsal = _last_rehearsal(row)
    if rehearsal is None:
        return 1.0
    now = at or datetime.now(timezone.utc)
    elapsed_days = max(0.0, (now - rehearsal).total_seconds() / 86400.0)
    stability = stability_days(row)
    if stability <= 0:
        return 1.0
    return max(FLOOR, min(1.0, math.exp(-elapsed_days / stability)))


def penalty(row: dict[str, Any], *, at: datetime | None = None) -> float:
    """How much scoring should lose for this memory's decay: 0.0 when nothing applies."""
    if not applies(row):
        return 0.0
    return 1.0 - strength(row, at=at)


def score_weight(row: dict[str, Any], *, at: datetime | None = None) -> float:
    """The multiplier a score is reduced by, i.e. `1 - WEIGHT × penalty`."""
    return 1.0 - WEIGHT * penalty(row, at=at)


def describe(store: Any, *, at: datetime | None = None, sample: int = 10) -> dict[str, Any]:
    """What decay would do to this store right now, without changing anything."""
    now = at or datetime.now(timezone.utc)
    rows = store._conn.execute(
        """SELECT id, kind, content, created_at, last_used_at, last_helpful_at,
                  used_count, confirmed_count, validated_count, success_count
           FROM memories WHERE state IN ('active','cold')"""
    ).fetchall()
    strengths = []
    faded = 0
    for raw in rows:
        row = dict(raw)
        if not applies(row):
            continue
        value = strength(row, at=now)
        strengths.append(value)
        if value < 0.5:
            faded += 1
    strengths.sort()
    return {
        "enabled": enabled(),
        "kinds": sorted(DECAY_KINDS),
        "half_life_days": _half_life(),
        "episodes": len(strengths),
        "faded_below_half": faded,
        "median_strength": strengths[len(strengths) // 2] if strengths else None,
        "weakest": strengths[:sample],
    }
