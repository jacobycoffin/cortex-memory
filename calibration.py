"""Calibration per kind: a 0.9 is not worth the same everywhere (#53).

A single confidence scale across kinds assumes that "my name is Jacoby" and "the deploy takes four
minutes" are equally verifiable from the same amount of evidence. They are not: identity facts come
from the person themselves and are checked by everyday use, operational facts arrive inferred and are
checked only when something breaks — so the same number means different things, and a store that
reports one Brier score is averaging two different questions.

This module measures calibration **per kind**, from outcomes the store already recorded:

| outcome | source |
|---|---|
| wrong | superseded by a `USER_CORRECTION`, or lost a dispute |
| right | used at least `MIN_USES` times and never corrected |

Everything else is **unjudged** and excluded — counting an unjudged fact as a success is how a
calibration study talks itself into a good number. Because unjudged dominates, coverage is reported
first, and a kind with too few judged facts gets no ceiling at all rather than a ceiling from noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Below this many judged facts, a kind's observed accuracy is not evidence of anything.
MIN_JUDGED = 20
# And below this many *wrong* ones, it is worse than that: with no negatives a kind scores perfect
# accuracy by construction and hands out a 1.00 ceiling nobody earned. The live store has 84
# corrected facts in 8,371, so this guard is the difference between a measurement and a tautology.
MIN_WRONG = 5
MIN_USES = 1


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _columns(store: Any) -> set[str]:
    rows = _rows(store, "PRAGMA table_info(memories)")
    return {str(row["name"]) for row in rows}


def outcomes(store: Any, *, limit: int = 20000) -> dict[str, dict[str, int]]:
    """How many memories of each kind are decided right, wrong, or not decided at all."""
    columns = _columns(store)
    wrong_ids: set[str] = set()
    # a correction that replaced this fact is the strongest evidence it was wrong
    for row in _rows(
        store,
        "SELECT superseded_id FROM self_corrections WHERE superseded_id IS NOT NULL "
        "AND superseded_id <> ''",
    ):
        wrong_ids.add(str(row["superseded_id"]))
    # losing a dispute counts too
    for row in _rows(
        store, "SELECT left_id, right_id, winner_id FROM disputes WHERE winner_id IS NOT NULL"
    ):
        for key in ("left_id", "right_id"):
            if str(row[key]) != str(row["winner_id"]):
                wrong_ids.add(str(row[key]))

    used: set[str] = set()
    for row in _rows(
        store,
        "SELECT DISTINCT memory_id FROM access_log WHERE event = 'used' LIMIT ?",
        (int(limit),),
    ):
        used.add(str(row["memory_id"]))

    selection = "id, kind, confidence"
    if "state" in columns:
        selection += ", state"
    rows = _rows(
        store,
        f"SELECT {selection} FROM memories WHERE confidence IS NOT NULL LIMIT ?",
        (int(limit),),
    )
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        kind = str(row.get("kind") or "unknown")
        bucket = counts.setdefault(kind, {"right": 0, "wrong": 0, "unjudged": 0, "total": 0})
        bucket["total"] += 1
        memory_id = str(row["id"])
        if memory_id in wrong_ids:
            bucket["wrong"] += 1
        elif memory_id in used:
            bucket["right"] += 1
        else:
            bucket["unjudged"] += 1
    return counts


@dataclass
class KindCalibration:
    kind: str
    judged: int = 0
    right: int = 0
    wrong: int = 0
    brier: float = 0.0
    mean_confidence: float = 0.0
    observed_accuracy: float = 0.0
    ceiling: float | None = None
    note: str = ""

    @property
    def overconfident(self) -> bool:
        return self.ceiling is not None and self.mean_confidence > self.ceiling + 1e-9

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "judged": self.judged,
            "right": self.right,
            "wrong": self.wrong,
            "brier": round(self.brier, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "observed_accuracy": round(self.observed_accuracy, 4),
            "ceiling": None if self.ceiling is None else round(self.ceiling, 4),
            "overconfident": self.overconfident,
            "note": self.note,
        }

    def describe(self) -> str:
        if self.ceiling is None:
            return f"{self.kind}: {self.judged} judged - not enough to calibrate"
        return (f"{self.kind}: {self.judged} judged, accuracy {self.observed_accuracy:.2f}, "
                f"Brier {self.brier:.3f}, stated {self.mean_confidence:.2f} -> "
                f"ceiling {self.ceiling:.2f}")


def measure(store: Any, *, limit: int = 20000) -> list[KindCalibration]:
    """Brier score and observed accuracy per kind, plus the ceiling that accuracy implies.

    The ceiling is the observed accuracy itself: if facts of this kind that the store calls 0.90 are
    right 0.65 of the time, then 0.65 is the highest number this kind has earned.
    """
    counts = outcomes(store, limit=limit)
    rows = _rows(
        store,
        "SELECT id, kind, confidence FROM memories WHERE confidence IS NOT NULL LIMIT ?",
        (int(limit),),
    )
    confidences: dict[str, list[float]] = {}
    for row in rows:
        confidences.setdefault(str(row.get("kind") or "unknown"), []).append(
            float(row.get("confidence") or 0.0)
        )
    judged_ids: dict[str, list[tuple[float, int]]] = {}
    wrong_ids: set[str] = set()
    for row in _rows(store, "SELECT superseded_id FROM self_corrections "
                            "WHERE superseded_id IS NOT NULL AND superseded_id <> ''"):
        wrong_ids.add(str(row["superseded_id"]))
    for row in _rows(store, "SELECT left_id, right_id, winner_id FROM disputes "
                            "WHERE winner_id IS NOT NULL"):
        for key in ("left_id", "right_id"):
            if str(row[key]) != str(row["winner_id"]):
                wrong_ids.add(str(row[key]))
    used = {
        str(row["memory_id"])
        for row in _rows(store, "SELECT DISTINCT memory_id FROM access_log WHERE event = 'used'")
    }
    for row in rows:
        memory_id = str(row["id"])
        kind = str(row.get("kind") or "unknown")
        confidence = float(row.get("confidence") or 0.0)
        if memory_id in wrong_ids:
            judged_ids.setdefault(kind, []).append((confidence, 0))
        elif memory_id in used:
            judged_ids.setdefault(kind, []).append((confidence, 1))

    results: list[KindCalibration] = []
    for kind, bucket in sorted(counts.items()):
        judged = judged_ids.get(kind, [])
        calibration = KindCalibration(kind=kind)
        values = confidences.get(kind, [])
        calibration.mean_confidence = (sum(values) / len(values)) if values else 0.0
        calibration.judged = len(judged)
        calibration.right = sum(1 for _, outcome in judged if outcome == 1)
        calibration.wrong = calibration.judged - calibration.right
        if judged:
            calibration.brier = sum((confidence - outcome) ** 2 for confidence, outcome in judged) / len(judged)
            calibration.observed_accuracy = calibration.right / len(judged)
        if calibration.judged >= MIN_JUDGED and calibration.wrong >= MIN_WRONG:
            calibration.ceiling = calibration.observed_accuracy
            if calibration.overconfident:
                calibration.note = "stated confidence exceeds what this kind has earned"
            else:
                calibration.note = "stated confidence is within what this kind has earned"
        elif calibration.judged < MIN_JUDGED:
            calibration.note = (f"only {calibration.judged} judged fact(s); "
                               f"{bucket.get('unjudged', 0)} unjudged")
        else:
            calibration.note = (f"{calibration.wrong} wrong of {calibration.judged} judged - "
                               f"too few negatives to calibrate against")
        results.append(calibration)
    return sorted(results, key=lambda item: -item.judged)


def report(store: Any, *, limit: int = 20000) -> dict[str, Any]:
    measured = measure(store, limit=limit)
    judged = sum(item.judged for item in measured)
    total = sum(
        bucket.get("total", 0) for bucket in outcomes(store, limit=limit).values()
    )
    return {
        "kinds": [item.as_dict() for item in measured],
        "judged": judged,
        "total": total,
        "judged_share": round(judged / total, 4) if total else 0.0,
        "note": "unjudged facts are excluded: counting them as successes is how a calibration "
                "study talks itself into a good number",
    }


def apply(store: Any, *, limit: int = 20000, at: Any = None) -> dict[str, Any]:
    """Clamp each kind's stated confidence to the ceiling its own outcomes earned.

    Only memories **above** their kind's ceiling move, and only downward: calibration can say a fact
    is overclaimed, never that it is underclaimed.
    """
    from datetime import datetime, timezone

    _ = (at or datetime.now(timezone.utc)).isoformat()  # the clamp is not time-dependent
    changed = 0
    by_kind: dict[str, int] = {}
    for calibration in measure(store, limit=limit):
        if calibration.ceiling is None:
            continue
        ceiling = float(calibration.ceiling)
        rows = _rows(
            store,
            "SELECT id FROM memories WHERE kind = ? AND confidence > ? LIMIT ?",
            (calibration.kind, ceiling, int(limit)),
        )
        if not rows:
            continue
        with store.transaction() as conn:
            for row in rows:
                conn.execute(
                    "UPDATE memories SET confidence = ? WHERE id = ?", (ceiling, str(row["id"]))
                )
                changed += 1
        by_kind[calibration.kind] = len(rows)
    return {"changed": changed, "by_kind": by_kind}
