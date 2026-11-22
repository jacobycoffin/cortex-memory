"""Unused links fade; the evidence behind them does not (#72).

An edge that nobody has crossed in a year is not evidence that the two memories are unrelated — it
is evidence that *nobody asked*. Two mistakes follow from treating those as the same thing:

- **keep every edge at full weight** and a 14,000-edge legacy graph outvotes everything that was
  connected last week;
- **delete stale edges** and the provenance goes with them, so "why were these ever linked?" becomes
  unanswerable, and the answer would have to be guessed from scratch.

So this module fades *reversibly*. The edge's weight is reduced on a half-life, but the **base
weight is recorded first**, `edge_evidence` rows are never touched, and `restore()` puts the original
weight back exactly. A fade that cannot be undone is a deletion with extra steps.

Decay keys on `last_reinforced_at` — the last time the link was confirmed — not on creation, so a
frequently used association stays strong indefinitely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

HALF_LIFE_DAYS = 120.0
FLOOR_RATIO = 0.25  # an edge never fades below a quarter of its base weight
TABLE = "edge_decay"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(text: Any) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    src_id TEXT NOT NULL,
                    dst_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    base_weight REAL NOT NULL,
                    faded_to REAL NOT NULL,
                    applied_at TEXT NOT NULL,
                    PRIMARY KEY (src_id, dst_id, relation)
                )"""
        )


def factor(idle_days: float, *, half_life: float = HALF_LIFE_DAYS) -> float:
    """How much of its weight an edge keeps after this long without reinforcement."""
    if idle_days <= 0:
        return 1.0
    if half_life <= 0:
        return FLOOR_RATIO
    return max(FLOOR_RATIO, 0.5 ** (idle_days / half_life))


@dataclass
class Faded:
    src_id: str
    dst_id: str
    relation: str
    base_weight: float
    weight: float
    idle_days: float

    def describe(self) -> str:
        return (
            f"{self.src_id[:8]}->{self.dst_id[:8]} {self.relation} "
            f"{self.base_weight:.2f}->{self.weight:.2f} (idle {self.idle_days:.0f}d)"
        )


def _edges(store: Any, limit: int = 5000) -> list[dict[str, Any]]:
    rows = store._conn.execute(
        """SELECT src_id, dst_id, relation, weight, created_at, last_reinforced_at
           FROM edges ORDER BY last_reinforced_at ASC LIMIT ?""",
        (int(limit),),
    ).fetchall()
    return [dict(row) for row in rows]


def planned(store: Any, *, at: datetime | None = None, limit: int = 5000) -> list[Faded]:
    """What a fade would do right now. Read-only."""
    moment = at or _now()
    out: list[Faded] = []
    for edge in _edges(store, limit=limit):
        anchor = _utc(edge.get("last_reinforced_at")) or _utc(edge.get("created_at"))
        if anchor is None:
            continue
        idle = max(0.0, (moment - anchor).total_seconds() / 86400.0)
        scale = factor(idle)
        if scale >= 0.999:
            continue
        base = float(edge.get("weight") or 0.0)
        out.append(
            Faded(
                str(edge["src_id"]), str(edge["dst_id"]), str(edge["relation"]),
                round(base, 4), round(base * scale, 4), round(idle, 1),
            )
        )
    return out


def apply(
    store: Any, *, at: datetime | None = None, limit: int = 5000, dry_run: bool = False
) -> dict[str, Any]:
    """Fade stale edges, recording the base weight so the fade is reversible."""
    moment = at or _now()
    faded = planned(store, at=moment, limit=limit)
    if dry_run or not faded:
        return {"faded": 0, "planned": len(faded), "dry_run": bool(dry_run)}
    if not _ensure_safe(store):
        return {"faded": 0, "planned": len(faded), "dry_run": False}
    _ensure(store)
    applied = 0
    with store.transaction() as conn:
        # A second run must recompute from the recorded base: decaying a decayed weight would
        # compound the fade every sweep, which is a deletion on an instalment plan.
        existing = {
            (str(row["src_id"]), str(row["dst_id"]), str(row["relation"])): float(row["base_weight"])
            for row in conn.execute(
                f"SELECT src_id, dst_id, relation, base_weight FROM {TABLE}"
            ).fetchall()
        }
        for edge in faded:
            base = existing.get((edge.src_id, edge.dst_id, edge.relation), edge.base_weight)
            decayed = max(base * FLOOR_RATIO, round(base * (edge.weight / edge.base_weight),
                                                    6)) if edge.base_weight else edge.weight
            edge.weight = decayed
            edge.base_weight = base
            conn.execute(
                f"""INSERT INTO {TABLE}
                    (src_id, dst_id, relation, base_weight, faded_to, applied_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(src_id, dst_id, relation) DO UPDATE SET
                        faded_to = excluded.faded_to,
                        applied_at = excluded.applied_at""",
                (edge.src_id, edge.dst_id, edge.relation, edge.base_weight, edge.weight,
                 moment.isoformat()),
            )
            conn.execute(
                """UPDATE edges SET weight = ? WHERE src_id = ? AND dst_id = ? AND relation = ?""",
                (edge.weight, edge.src_id, edge.dst_id, edge.relation),
            )
            applied += 1
    return {"faded": applied, "planned": len(faded), "dry_run": False}


def _ensure_safe(store: Any) -> bool:
    try:
        store._conn.execute("SELECT 1 FROM edges LIMIT 1").fetchone()
    except Exception:  # noqa: BLE001 - no edges table means nothing to fade
        return False
    return True


def restore(store: Any, *, limit: int = 5000) -> int:
    """Put every faded edge back to its recorded base weight. Evidence was never touched."""
    try:
        _ensure(store)
        rows = store._conn.execute(
            f"""SELECT src_id, dst_id, relation, base_weight FROM {TABLE} LIMIT ?""",
            (int(limit),),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return 0
    restored = 0
    with store.transaction() as conn:
        for row in rows:
            conn.execute(
                """UPDATE edges SET weight = ? WHERE src_id = ? AND dst_id = ? AND relation = ?""",
                (float(row["base_weight"]), str(row["src_id"]), str(row["dst_id"]),
                 str(row["relation"])),
            )
            restored += 1
        conn.execute(f"DELETE FROM {TABLE}")
    return restored


def evidence_count(store: Any) -> int:
    try:
        return int(store._conn.execute("SELECT COUNT(*) FROM edge_evidence").fetchone()[0] or 0)
    except Exception:  # noqa: BLE001
        return 0


def report(store: Any, *, at: datetime | None = None, limit: int = 200_000) -> dict[str, Any]:
    faded = planned(store, at=at, limit=limit)
    total = 0
    try:
        total = int(store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] or 0)
    except Exception:  # noqa: BLE001
        pass
    return {
        "edges": total,
        "would_fade": len(faded),
        "evidence_rows": evidence_count(store),
        "half_life_days": HALF_LIFE_DAYS,
        "floor_ratio": FLOOR_RATIO,
        "steepest": [item.describe() for item in sorted(faded, key=lambda i: i.weight)[:5]],
    }
