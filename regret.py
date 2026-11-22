"""Regret guard: a memory that came back widens the window it was forgotten through (#90).

The forget ladder can be wrong. When it is, the evidence is specific and already in the store:
a memory that was walked down the ladder and later came **back up** — restored by hand, or
genuinely used again after being demoted. That is a measured regret, not a hunch, and it should
change future behaviour for that kind of memory rather than being quietly absorbed.

The guard is deliberately one-directional: regrets only ever *widen* a window, never narrow it,
because every failure mode of this system is asymmetric. Forgetting something that mattered is
unrecoverable in the moment; keeping something a bit longer costs nothing.

Scaling: `factor = 1 + 0.25 × revivals`, capped at 3×. The cap matters — a store with a handful
of churny memories must not widen episodes to infinity and stop the ladder entirely, which would
turn a guard into a bug. The window is measured over `REGRET_WINDOW_DAYS`, so a mistake from
eight months ago is not still inflating today's policy.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

REGRET_WINDOW_DAYS = 180
STEP = 0.25
CAP = 3.0
_DOWNGRADES = ("cold", "archived", "tombstoned")


def enabled() -> bool:
    raw = os.environ.get("CORTEX_FORGET_REGRET", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _window_days() -> int:
    raw = os.environ.get("CORTEX_REGRET_WINDOW_DAYS", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return REGRET_WINDOW_DAYS
    return value if value > 0 else REGRET_WINDOW_DAYS


def revivals(store: Any, *, days: int | None = None) -> dict[str, int]:
    """How often each kind was walked down the ladder and then came back."""
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=days or _window_days())
    ).isoformat()
    try:
        rows = store._conn.execute(
            f"""SELECT COALESCE(m.kind, '') AS kind, COUNT(*) AS n
                FROM lifecycle_events e
                JOIN memories m ON m.id = e.memory_id
                WHERE e.to_state = 'active'
                  AND e.from_state IN ({','.join('?' for _ in _DOWNGRADES)})
                  AND e.created_at >= ?
                GROUP BY COALESCE(m.kind, '')""",
            (*_DOWNGRADES, cutoff),
        ).fetchall()
    except Exception:  # noqa: BLE001 - a store without lifecycle history has no regrets
        return {}
    return {str(row["kind"]): int(row["n"]) for row in rows}


def factor(store: Any, kind: str, *, revivals_by_kind: dict[str, int] | None = None) -> float:
    """The multiplier for this kind's windows. 1.0 when there is nothing to regret."""
    if not enabled():
        return 1.0
    counts = revivals_by_kind if revivals_by_kind is not None else revivals(store)
    count = counts.get(str(kind or "").strip().lower(), 0)
    if count <= 0:
        return 1.0
    return min(CAP, 1.0 + STEP * count)


def windows(
    dwell: tuple[int, int, int],
    multiplier: float,
    *,
    floors: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    """Scale a dwell tuple, never shrinking a window below the declared policy."""
    base = floors or dwell
    scaled = tuple(int(round(day * multiplier)) for day in dwell)
    return tuple(max(base[index], scaled[index]) for index in range(3))  # type: ignore[return-value]


def describe(store: Any, kind: str) -> dict[str, Any]:
    counts = revivals(store)
    return {
        "kind": kind,
        "revivals": counts.get(str(kind or "").strip().lower(), 0),
        "factor": factor(store, kind, revivals_by_kind=counts),
        "window_days": _window_days(),
        "step": STEP,
        "cap": CAP,
        "enabled": enabled(),
        "all": counts,
    }
