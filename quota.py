"""A soft cap on active memories that triggers a sweep instead of refusing writes (#101).

A hard quota on a memory system is the wrong shape: the moment it fills, the assistant silently
stops learning, and the failure surfaces weeks later as "why doesn't it remember anything new".
A quota guard instead treats the cap as a **signal to tidy up**: past the warn band it walks the
forget ladder and archives the noisiest triage material, then lets the write through.

The write is never refused. That is the whole design: `remember` always succeeds, and the cost of
the write is paid by the memories that were already not earning their place.

The guard is cheap when it should be — a single `COUNT` — and only sweeps when the store is inside
the top of its band, so the common path costs one indexed count rather than a sweep.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

DEFAULT_LIMIT = 6000
WARN_AT = 0.85


def limit() -> int:
    raw = os.environ.get("CORTEX_MAX_ACTIVE_MEMORIES", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_LIMIT
    return value if value > 0 else DEFAULT_LIMIT


@dataclass
class Status:
    active: int
    limit: int
    band: str

    @property
    def pct(self) -> float:
        return self.active / self.limit if self.limit else 0.0

    def describe(self) -> str:
        return (
            f"{self.active}/{self.limit} active memories ({self.pct:.0%}) — {self.band}"
        )


def status(store: Any) -> Status:
    active = int(
        store._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE state IN ('active','cold')"
        ).fetchone()[0]
        or 0
    )
    cap = limit()
    ratio = active / cap if cap else 0.0
    band = "ok" if ratio < WARN_AT else ("warn" if ratio < 1.0 else "full")
    return Status(active=active, limit=cap, band=band)


@dataclass
class Sweep:
    before: Status
    after: Status
    walked: dict[str, int] = field(default_factory=dict)
    archived: int = 0
    skipped: str = ""

    @property
    def reclaimed(self) -> int:
        return max(0, self.before.active - self.after.active)

    @property
    def summary(self) -> str:
        if self.skipped:
            return f"quota {self.before.describe()}: no sweep ({self.skipped})"
        moves = ", ".join(f"{key} {value}" for key, value in sorted(self.walked.items()))
        return (
            f"quota {self.before.describe()} -> {self.after.active} after sweeping"
            f"{(' (' + moves + ')') if moves else ''}, archived {self.archived}"
        )


def ensure(
    store: Any,
    *,
    apply: bool = True,
    at: datetime | None = None,
    max_triage: int = 50,
) -> Sweep:
    """Sweep if the store is in the warn band or above. Never refuses anything."""
    before = status(store)
    if before.band == "ok":
        return Sweep(before=before, after=before, skipped="inside quota")

    from . import forget_ladder, triage

    walked: dict[str, int] = {}
    archived = 0
    if apply:
        # Order matters: triage archives never-used memories outright, which actually lowers the
        # count. The ladder only moves one rung (active -> cold), and cold still counts, so
        # running it first would hide the candidates triage needs (they must still be `active`).
        try:
            built = triage.batch(store, size=max_triage, at=at)
            archived = len(triage.apply(store, built, at=at)) if built.items else 0
        except Exception:  # noqa: BLE001 - a triage failure must not block the write
            archived = 0
        planned, applied = forget_ladder.sweep(store, at=at, apply_steps=True)
        walked = applied
    after = status(store)
    return Sweep(before=before, after=after, walked=walked, archived=archived)


def check(store: Any) -> dict[str, Any]:
    """Report the guard's view without changing anything (for dashboards and tests)."""
    current = status(store)
    return {
        "active": current.active,
        "limit": current.limit,
        "pct": round(current.pct, 3),
        "band": current.band,
        "warn_at": WARN_AT,
        "would_sweep": current.band != "ok",
    }
