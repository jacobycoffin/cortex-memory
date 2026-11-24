"""One trust number per memory, composed from parts that are already recorded (#187)

The store keeps the ingredients and never bakes them: a source-authority ceiling (`source_authority`), a
confidence, a currentness, evidence counts on edges, confirmations, corrections, harmful reports, and a
retirement date. A reader asking "how much should I trust this?" gets eight columns and no answer.

So this composes them, under four rules that decide whether the number is honest:

- **source authority is a ceiling, not a multiplier** (#41). A document cannot be trusted more than a
  document, no matter how many times it was retrieved; `final = min(composite, ceiling)`.
- **an unknown part is excluded from the denominator, not counted as zero.** A memory that has never been
  retrieved has *no* usage evidence — scoring that as "unused therefore untrustworthy" would punish every
  new memory and reward every old one.
- **penalties subtract, they do not multiply.** One harmful report should dent a belief, not delete it;
  a belief is not a probability that collapses on contact with a counterexample.
- **the score explains itself.** Every contribution is printed with its weight, so the number can be
  argued with — which is the only way a single number is allowed to exist here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .source_authority import authority_for

# Weights over the *known* parts. They are stated here rather than buried in an expression so that a
# reader can disagree with one number instead of reverse-engineering a formula.
WEIGHTS = {
    "confidence": 0.30,
    "currentness": 0.15,
    "evidence": 0.25,
    "use": 0.15,
    "confirmations": 0.15,
}
# Evidence saturates: twenty edges are not ten times better than two.
EVIDENCE_SATURATION = 4.0
USE_SATURATION = 10.0
CONFIRMATION_SATURATION = 3.0
# Below this p10-p90 span the number is nearly the same for every memory, which is not a measurement.
MIN_USEFUL_SPREAD = 0.20
HARMFUL_PENALTY = 0.25
CORRECTION_BONUS = 0.10
STALE_PENALTY = 0.10


@dataclass
class Score:
    memory_id: str = ""
    score: float = 0.0
    ceiling: float = 0.0
    capped: bool = False
    parts: dict[str, Any] = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "score": round(self.score, 3),
                "ceiling": round(self.ceiling, 3), "capped": self.capped, "parts": self.parts,
                "unknown": self.unknown, "notes": self.notes}


def _saturate(value: float, ceiling: float) -> float:
    return min(1.0, max(0.0, float(value)) / float(ceiling)) if ceiling else 0.0


def score(row: dict[str, Any], *, evidence: int | None = None, stale: bool = False) -> Score:
    """Compose the parts that exist, cap by authority, subtract penalties."""
    found = Score(memory_id=str(row.get("id") or ""))
    found.ceiling = authority_for(row)
    parts: dict[str, Any] = {}

    confidence = row.get("confidence")
    if confidence is None:
        found.unknown.append("confidence")
    else:
        parts["confidence"] = float(confidence)

    currentness = row.get("currentness_confidence")
    if currentness is None:
        found.unknown.append("currentness")
    else:
        parts["currentness"] = float(currentness)

    if evidence is None:
        found.unknown.append("evidence")
    else:
        parts["evidence"] = _saturate(evidence, EVIDENCE_SATURATION)

    used = row.get("used_count")
    if used is None:
        found.unknown.append("use")
    else:
        parts["use"] = _saturate(int(used or 0), USE_SATURATION)

    confirmed = row.get("confirmed_count")
    if confirmed is None:
        found.unknown.append("confirmations")
    else:
        parts["confirmations"] = _saturate(int(confirmed or 0), CONFIRMATION_SATURATION)

    weight_total = sum(WEIGHTS[name] for name in parts)
    if not weight_total:
        found.notes.append("nothing is known about this memory, so it has no score rather than a low one")
        found.parts = {}
        return found
    composite = sum(WEIGHTS[name] * float(value) for name, value in parts.items()) / weight_total

    harmful = int(row.get("harmful_count") or 0)
    if harmful:
        composite -= HARMFUL_PENALTY * min(1.0, harmful / 2.0)
        found.notes.append(f"harmful reports: -{HARMFUL_PENALTY * min(1.0, harmful / 2.0):.2f}")
    if int(row.get("correction_count") or 0):
        composite += CORRECTION_BONUS
        found.notes.append(f"was corrected: +{CORRECTION_BONUS:.2f}")
    if stale:
        composite -= STALE_PENALTY
        found.notes.append(f"unconfirmed past its window: -{STALE_PENALTY:.2f}")

    composite = max(0.0, min(1.0, composite))
    if composite > found.ceiling:
        found.capped = True
        found.notes.append(f"capped at the source authority ({found.ceiling:.2f}): "
                           "a document cannot be trusted more than a document")
    found.score = min(composite, found.ceiling)
    found.parts = {name: round(float(value), 3) for name, value in parts.items()}
    found.parts["_weights"] = {name: WEIGHTS[name] for name in parts}
    return found


def _evidence_counts(store: Any) -> dict[str, int]:
    """One grouped query instead of a query per memory."""
    try:
        rows = store._conn.execute(
            "SELECT src_id AS id, SUM(evidence_count) AS total FROM edges GROUP BY src_id "
            "UNION ALL SELECT dst_id AS id, SUM(evidence_count) AS total FROM edges GROUP BY dst_id"
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    totals: dict[str, int] = {}
    for row in rows:
        key = str(row["id"])
        totals[key] = totals.get(key, 0) + int(row["total"] or 0)
    return totals


def rank(store: Any, *, limit: int = 10, lowest: bool = True) -> list[dict[str, Any]]:
    """The memories most worth reviewing, with the reason they scored where they did."""
    try:
        rows = [dict(row) for row in store._conn.execute(
            "SELECT * FROM memories WHERE state <> 'archived' AND valid_to IS NULL"
        ).fetchall()]
    except Exception:  # noqa: BLE001
        return []
    evidence = _evidence_counts(store)
    scored = []
    for row in rows:
        memory_id = str(row.get("id") or "")
        found = score(row, evidence=evidence.get(memory_id, 0))
        if not found.parts:
            continue
        scored.append({
            "memory_id": memory_id,
            "score": round(found.score, 3),
            "ceiling": round(found.ceiling, 3),
            "kind": str(row.get("kind") or ""),
            "source_category": str(row.get("source_category") or ""),
            "content": str(row.get("content") or "")[:80],
            "unknown": found.unknown,
        })
    scored.sort(key=lambda item: (item["score"], item["memory_id"]), reverse=not lowest)
    return scored[: max(1, int(limit))]


def distribution(store: Any, *, buckets: int = 5) -> dict[str, Any]:
    """Where the store sits, so a single memory's score has a reference class."""
    try:
        rows = [dict(row) for row in store._conn.execute(
            "SELECT * FROM memories WHERE state <> 'archived' AND valid_to IS NULL"
        ).fetchall()]
    except Exception:  # noqa: BLE001
        return {"count": 0, "buckets": {}, "note": "no memories could be read"}
    evidence = _evidence_counts(store)
    scores = []
    for row in rows:
        found = score(row, evidence=evidence.get(str(row.get("id") or ""), 0))
        if found.parts:
            scores.append(found.score)
    if not scores:
        return {"count": 0, "buckets": {}, "note": "nothing had enough information to score"}
    scores.sort()
    width = 1.0 / max(1, int(buckets))
    counted: dict[str, int] = {}
    for value in scores:
        index = min(int(buckets) - 1, int(value / width))
        label = f"{index * width:.1f}-{(index + 1) * width:.1f}"
        counted[label] = counted.get(label, 0) + 1
    middle = scores[len(scores) // 2]
    p10 = scores[int(len(scores) * 0.1)]
    p90 = scores[min(len(scores) - 1, int(len(scores) * 0.9))]
    spread = round(p90 - p10, 3)
    note = "a score only means something against this distribution"
    if spread < MIN_USEFUL_SPREAD:
        note = (f"the score barely discriminates: p10-p90 span only {spread}. Confidence and "
                "currentness are near-constant across this store, so almost all of the range would "
                "have to come from evidence and use - and a number that is nearly the same for "
                "everything is not a measurement (#147, #176)")
    return {
        "count": len(scores),
        "buckets": dict(sorted(counted.items())),
        "median": round(middle, 3),
        "p10": round(p10, 3),
        "p90": round(p90, 3),
        "spread": spread,
        "note": note,
    }


def render(store: Any, *, limit: int = 8) -> list[str]:
    described = distribution(store)
    lines = [f"trust: {described['count']} scored, median {described.get('median')}, "
             f"p10 {described.get('p10')}, p90 {described.get('p90')}"]
    if described["buckets"]:
        lines.append("  distribution: " + ", ".join(f"{name} {count}"
                                                    for name, count in described["buckets"].items()))
    for item in rank(store, limit=limit):
        lines.append(f"  · {item['score']:.2f} {item['kind']}/{item['source_category']} "
                     f"{item['content'][:60]}")
    if described.get("note"):
        lines.append(f"  {described['note']}")
    return lines
