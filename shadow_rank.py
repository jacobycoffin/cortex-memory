"""What a scoring change would have done, measured on history (#202).

A ranking change is the easiest thing in this system to ship on a hunch and the hardest to justify
afterwards: by the time a policy is live, the counterfactual is gone. So this replays what the store
already recorded — every recall keeps its candidate list, each candidate's score, and the component
each boost stage contributed — and asks what the *other* policy would have selected instead.

Three properties make that trustworthy:

- **exact, not simulated.** A boost's contribution is recorded per candidate (`use_boost`,
  `entity_boost`, `session_boost`, `association_boost` — the keys `boost_caps.COMPONENT_KEYS`
  defines), so removing one is subtraction on a number that exists, not a re-run with different
  settings and a hope that nothing else moved.
- **read-only.** Nothing is written to memories, edges or scores. This answers a question; it does not
  change the answer to it.
- **an absent component is reported as absent.** A policy that touches nothing must not look like a
  policy that was measured and found harmless: `touched` counts candidates that actually carried the
  component, and `inert` says so out loud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# In step with boost_caps.COMPONENT_KEYS: the stage name and the component it records.
COMPONENTS = {
    "usage": "use_boost",
    "entity": "entity_boost",
    "session": "session_boost",
    "association": "association_boost",
}
DEFAULT_TOP_K = 6


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 - a replay that cannot read reports nothing
        return []


@dataclass(frozen=True)
class Policy:
    name: str
    remove: tuple[str, ...] = ()
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "remove": list(self.remove), "note": self.note}


def policies() -> list[Policy]:
    """The candidate policies worth asking about, one family per question."""
    return [
        Policy(name="no_usage_boost", remove=("usage",),
               note="drop the record-of-use term"),
        Policy(name="no_entity_boost", remove=("entity",),
               note="drop the entity-anchor term"),
        Policy(name="no_session_boost", remove=("session",),
               note="drop the same-session term"),
        Policy(name="no_association_boost", remove=("association",),
               note="drop the co-retrieval term"),
        Policy(name="no_boost_at_all", remove=tuple(COMPONENTS),
               note="drop every history-boost family"),
    ]


def _candidates(trace: dict[str, Any]) -> list[dict[str, Any]]:
    items = trace.get("candidate_memories") or []
    return [dict(item) for item in items if isinstance(item, dict) and item.get("memory_id")]


def _score(candidate: dict[str, Any], remove: Iterable[str]) -> float:
    score = float(candidate.get("score") or 0.0)
    components = candidate.get("components") or {}
    if not isinstance(components, dict):
        return score
    for family in remove:
        key = COMPONENTS.get(family)
        if not key:
            continue
        score -= float(components.get(key) or 0.0)
    return score


def _selected(candidates: list[dict[str, Any]], top_k: int) -> tuple[str, ...]:
    """The ids a policy would surface, ordered by (adjusted) score then id.

    The id tie-break is deliberate: without it, two equal scores would come back in whatever order
    the dict happened to have, and a replay would report a change that no ranker made.
    """
    ordered = sorted(candidates, key=lambda item: (-float(item["_adjusted"]), str(item["memory_id"])))
    return tuple(str(item["memory_id"]) for item in ordered[:top_k])


@dataclass
class Replay:
    policy: str = ""
    traces: int = 0
    compared: int = 0
    changed: int = 0
    top1_changed: int = 0
    touched: int = 0
    inert: bool = False
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def changed_share(self) -> float:
        return round(self.changed / self.compared, 4) if self.compared else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "traces": self.traces,
            "compared": self.compared,
            "changed": self.changed,
            "changed_share": self.changed_share,
            "top1_changed": self.top1_changed,
            "touched": self.touched,
            "inert": self.inert,
            "examples": self.examples,
            "sentence": self.describe(),
        }

    def describe(self) -> str:
        if self.inert:
            return (f"{self.policy}: inert - no candidate carried the component(s) it removes, "
                    f"so nothing was measured")
        return (f"{self.policy}: {self.changed}/{self.compared} trace(s) would select differently "
                f"({self.changed_share:.1%}), top-1 changes in {self.top1_changed}, "
                f"{self.touched} candidate(s) carried the term")


def replay(store: Any, policy: Policy, *, limit: int = 200,
           top_k: int = DEFAULT_TOP_K) -> Replay:
    """What this policy would have selected across recorded traces."""
    traces = []
    try:
        traces = store.memory_traces(limit=int(limit))
    except Exception:  # noqa: BLE001
        traces = []
    found = Replay(policy=policy.name, traces=len(traces))
    for trace in traces:
        candidates = _candidates(trace)
        if len(candidates) < 2:
            continue
        for candidate in candidates:
            candidate["_adjusted"] = _score(candidate, policy.remove)
        components = [item.get("components") or {} for item in candidates]
        found.touched += sum(
            1 for payload in components
            if isinstance(payload, dict)
            and any(float(payload.get(COMPONENTS[family]) or 0.0) > 0.0
                    for family in policy.remove if family in COMPONENTS)
        )
        recorded = tuple(
            str(item["memory_id"]) for item in sorted(
                candidates, key=lambda item: (int(item.get("rank") or 0), str(item["memory_id"]))
            )
            if item.get("selected")
        )
        if not recorded:
            continue
        shadow = _selected(candidates, len(recorded) if len(recorded) > 1 else top_k)
        found.compared += 1
        if shadow != recorded:
            found.changed += 1
            if record_top(recorded) != record_top(shadow):
                found.top1_changed += 1
            if len(found.examples) < 3:
                found.examples.append({
                    "trace_id": str(trace.get("trace_id") or ""),
                    "recorded": list(recorded[:3]),
                    "shadow": list(shadow[:3]),
                })
    if not found.touched:
        found.inert = True
    return found


def record_top(selection: Iterable[str]) -> str:
    for memory_id in selection:
        return str(memory_id)
    return ""


def report(store: Any, *, limit: int = 200, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Every built-in policy, replayed over the same traces."""
    replays = [replay(store, policy, limit=limit, top_k=top_k) for policy in policies()]
    return {
        "policies": [item.as_dict() for item in replays],
        "traces_replayed": replays[0].traces if replays else 0,
        "note": "read-only: a replay answers a question, it does not change the answer to it",
    }
