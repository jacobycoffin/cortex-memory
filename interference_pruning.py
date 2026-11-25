"""Interference pruning: settle the duplicates, queue the contradictions (#108).

Two memories can interfere in exactly two ways. They can say the **same thing** — in which case one of them is noise,
and the store can decide which. Or they can say **different things about the same subject** — in which case only a
person can decide, and a batch job that "resolves" it is guessing with other people's facts.

This module is the nightly planner over both, and its whole design is that distinction:

- **duplicates may be settled in batch; contradictions may not.** An exact duplicate has one right answer. A
  contradiction has two claims and no evidence in the store that says which is true, so it is *queued* with its reason
  and left alone.
- **only exact duplicates are automatic.** Anything that merely looks similar goes to a person: fuzzy merging is how a
  store loses the sentence that was slightly different and slightly important. The similarity band exists so those
  near-misses are *listed* (`near_duplicate`) instead of being invisible — measured on this store, exact duplicates are
  largely prevented at write time (`add_memory` returns the existing row), so the band is where the real work is.
- **a merge that would lose something is never automatic.** `merge_diff.compute()` reports lost facts and structural
  losses; if either is non-empty the pair is queued, not merged.
- **dry run by default.** Applying takes an explicit `apply=True` plus consent, the same discipline as every other
  write path in this store.
- **bounded, and the skips are counted.** A night that considered 3,000 rows must say what it did not look at.

The heavy lifting is not re-implemented here: duplicates are found by exact normalised content, the consequences of
each merge come from `merge_diff.compute`, and contradictions come from `contradiction_cluster`. This module decides
**what may happen without a human** and what may not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

MEMORY_TABLE = "memories"
ACTIVE_STATES = ("active", "cold")

DEFAULT_LIMIT = 25
AUTO = "auto_merge"
DECIDE = "needs_decision"
SKIP = "skip"

# Above this Jaccard overlap two rows read as the same claim to a person, so they are worth showing; below it they are
# different facts that happen to share words. The band is a reporting threshold, never an auto-merge rule.
NEAR_DUPLICATE = 0.85


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def _normalise(text: Any) -> str:
    """Whitespace and case are formatting, not meaning: two rows that differ only there are the same claim."""
    return " ".join(str(text or "").split()).strip().lower()


@dataclass
class Item:
    action: str
    survivor_id: str
    absorbed_id: str
    reason: str
    diff: Any = None
    lost_facts: list[str] = field(default_factory=list)
    structural_losses: list[dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        return f"{self.action}: keep {self.survivor_id[:8]}, absorb {self.absorbed_id[:8]} — {self.reason}"

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "survivor_id": self.survivor_id, "absorbed_id": self.absorbed_id,
                "reason": self.reason, "lost_facts": self.lost_facts,
                "structural_losses": self.structural_losses}


@dataclass
class Plan:
    items: list[Item] = field(default_factory=list)
    skips: dict[str, int] = field(default_factory=dict)
    considered: int = 0
    duplicate_groups: int = 0
    readable: bool = True
    reason: str = ""

    def by_action(self, action: str) -> list[Item]:
        return [item for item in self.items if item.action == action]

    @property
    def auto_mergeable(self) -> list[Item]:
        return self.by_action(AUTO)

    @property
    def needs_decision(self) -> list[Item]:
        return self.by_action(DECIDE)

    def as_dict(self) -> dict[str, Any]:
        return {"readable": self.readable, "reason": self.reason, "considered": self.considered,
                "duplicate_groups": self.duplicate_groups, "skips": self.skips,
                "auto_merge": [item.as_dict() for item in self.auto_mergeable],
                "near_duplicates": [item.as_dict() for item in self.items if item.action == DECIDE and
                                    item.reason.startswith("near duplicate")],
                "needs_decision": [item.as_dict() for item in self.needs_decision]}

    def render(self) -> list[str]:
        if not self.readable:
            return [self.reason]
        lines = [f"{self.considered} active row(s), {self.duplicate_groups} duplicate group(s): "
                 f"{len(self.auto_mergeable)} mergeable, {len(self.needs_decision)} need a decision"]
        lines.extend(f"- {item.render()}" for item in self.items[:12])
        if len(self.items) > 12:
            lines.append(f"… and {len(self.items) - 12} more")
        if self.skips:
            lines.append("skipped: " + ", ".join(f"{reason} {count}" for reason, count in sorted(self.skips.items())))
        return lines


def _tokens(text: Any) -> set[str]:
    return {word for word in "".join(ch if ch.isalnum() else " " for ch in str(text or "").lower()).split()
            if len(word) > 2}


def similarity(left: Any, right: Any) -> float:
    """Jaccard overlap of word sets. Cheap, explainable, and stable enough to report."""
    first, second = _tokens(left), _tokens(right)
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def _survivor_key(row: dict[str, Any]) -> tuple:
    """The row that stays: most used, then most confirmed, then oldest, then the id as a tie-break."""
    return (
        -(int(row.get("used_count") or 0)),
        int(row.get("confirmed_count") or 0),
        str(row.get("created_at") or ""),
        str(row.get("id")),
    )


def plan(store: Any, *, limit: int = DEFAULT_LIMIT, include_contradictions: bool = True) -> Plan:
    """What a night could settle on its own, and what it must leave alone. Writes nothing."""
    rows = _rows(
        store,
        f"SELECT id, kind, content, state, used_count, confirmed_count, created_at FROM {MEMORY_TABLE} "
        f"WHERE state IN ({','.join('?' * len(ACTIVE_STATES))})",
        ACTIVE_STATES,
    )
    if rows is None:
        return Plan(readable=False, reason="the store could not be read")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("kind") or ""), _normalise(row.get("content"))), []).append(row)

    out = Plan(considered=len(rows))
    budget = max(0, int(limit))
    for (_kind, _text), members in sorted(groups.items(), key=lambda pair: str(pair[0][1])):
        if len(members) < 2:
            continue
        out.duplicate_groups += 1
        ordered = sorted(members, key=_survivor_key)
        survivor = ordered[0]
        for absorbed in ordered[1:]:
            if len(out.items) >= budget:
                out.skips["over budget"] = out.skips.get("over budget", 0) + 1
                continue
            from . import merge_diff

            diff = merge_diff.compute(store, str(survivor["id"]), str(absorbed["id"]))
            lost = list(getattr(diff, "lost_facts", []) or [])
            structural = list(getattr(diff, "structural_losses", []) or [])
            if lost or structural:
                out.items.append(Item(
                    action=DECIDE, survivor_id=str(survivor["id"]), absorbed_id=str(absorbed["id"]),
                    reason=f"the merge would lose {len(lost)} fact(s) and {len(structural)} structural item(s)",
                    diff=diff, lost_facts=lost, structural_losses=structural,
                ))
            else:
                out.items.append(Item(
                    action=AUTO, survivor_id=str(survivor["id"]), absorbed_id=str(absorbed["id"]),
                    reason="same claim, nothing lost by keeping the more-used row", diff=diff,
                ))

    # Near-misses: not identical, but close enough that a person should look. Never auto-merged, and only reported
    # when the budget allows — the point is to surface them, not to resolve them.
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_kind.setdefault(str(row.get("kind") or ""), []).append(row)
    for kind, members in by_kind.items():
        for index, left in enumerate(members):
            for right in members[index + 1:]:
                if len(out.items) >= budget * 2:
                    out.skips["over budget (near duplicates)"] = out.skips.get("over budget (near duplicates)", 0) + 1
                    continue
                score = similarity(left.get("content"), right.get("content"))
                if score < NEAR_DUPLICATE or score >= 1.0:
                    continue
                ordered = sorted((left, right), key=_survivor_key)
                out.items.append(Item(
                    action=DECIDE, survivor_id=str(ordered[0]["id"]), absorbed_id=str(ordered[1]["id"]),
                    reason=f"near duplicate ({score:.2f} overlap), not identical: a person must confirm they say "
                           f"the same thing",
                ))

    if include_contradictions:
        try:
            from . import contradiction_cluster

            for candidate in contradiction_cluster.candidates(store, limit=max(1, budget)):
                if len(out.items) >= budget * 2:
                    out.skips["over budget (contradictions)"] = out.skips.get("over budget (contradictions)", 0) + 1
                    continue
                out.items.append(Item(
                    action=DECIDE,
                    survivor_id=str(candidate.get("left_id") or ""),
                    absorbed_id=str(candidate.get("right_id") or ""),
                    reason=f"contradiction ({candidate.get('reason') or 'incompatible claims'}): only a person "
                           f"can say which side is true",
                ))
        except Exception:  # noqa: BLE001
            out.skips["contradiction scan unavailable"] = out.skips.get("contradiction scan unavailable", 0) + 1
    return out


def apply(store: Any, plan: Plan, *, apply: bool = False, consented_by: str = "") -> dict[str, Any]:
    """Merge what the plan marked automatic. Refuses without consent, and never touches a decision item."""
    if not apply:
        return {"ok": True, "applied": 0, "dry_run": True,
                "would_merge": len(plan.auto_mergeable), "queued": len(plan.needs_decision)}
    if not str(consented_by or "").strip():
        return {"ok": False, "reason": "applying merges needs consent: name who approved them"}
    from . import merge_diff

    applied = 0
    failures: list[str] = []
    for item in plan.auto_mergeable:
        try:
            result = merge_diff.apply(store, item.diff, consent=True, consented_by=consented_by)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{item.absorbed_id[:8]}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        if isinstance(result, dict) and result.get("ok") is False:
            failures.append(f"{item.absorbed_id[:8]}: {result.get('reason', 'refused')}")
            continue
        applied += 1
    return {"ok": not failures, "applied": applied, "dry_run": False,
            "queued": len(plan.needs_decision), "failures": failures,
            "at": _now()}
