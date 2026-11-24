"""Context as a retrieval key (#83).

Two memories can both match a query and mean different things, because one is about *this* project and
one is about another. Today the store scores them on text alone and hopes the right one wins; the
project it already inferred at capture time (`scope_json`, #30 / `project_scope`) never enters the
ranking.

This module makes context a key rather than a filter:

- **tags are read, never invented.** They come from what the row already carries: the project in
  `scope_json`, the scope's who/where/when, the kind, and the source family. A memory with no scope is
  `unknown`, not `wrong`.
- **absence is neutral.** A memory that says nothing about the project is neither boosted nor
  penalised — punishing silence would systematically bury everything written before scope capture
  existed, which is most of a real store.
- **a named mismatch is a penalty, and only for families where it means something.** Claiming project
  A when the turn is about project B is evidence against; a different `kind` is not.
- **the boost is capped.** ±0.15, inside the store's existing total boost budget (0.25), because a tag
  is thinner evidence than the text it sits next to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

# Families where naming a different value is evidence against the memory. `kind` and `source` are
# deliberately absent: a note of the wrong kind is still about the right thing.
DECISIVE = ("project", "who", "where")
FAMILIES = ("project", "who", "where", "when", "kind", "source")
MAX_BOOST = 0.15
WEIGHT = {"project": 0.5, "who": 0.2, "where": 0.2, "when": 0.1}


def _load(blob: Any) -> dict[str, Any]:
    if isinstance(blob, dict):
        return blob
    try:
        loaded = json.loads(blob or "{}")
    except Exception:  # noqa: BLE001 - a malformed scope is a missing scope, not a crash
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _clean(value: Any) -> str:
    return str(value or "").strip().lower()


@dataclass
class Tags:
    project: str = "unknown"
    who: tuple[str, ...] = ()
    where: tuple[str, ...] = ()
    when: str = ""
    kind: str = ""
    source: str = ""

    def known(self, family: str) -> bool:
        value = getattr(self, family, "")
        return bool(value) and _clean(value) != "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "who": list(self.who),
            "where": list(self.where),
            "when": self.when,
            "kind": self.kind,
            "source": self.source,
        }


def tags_for(store: Any, memory_id: str) -> Tags:
    try:
        row = store._conn.execute(
            "SELECT scope_json, kind, source_type FROM memories WHERE id = ?", (str(memory_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001
        row = None
    if row is None:
        return Tags()
    scope = _load(row["scope_json"])
    who = scope.get("who") or []
    where = scope.get("where") or []
    return Tags(
        project=_clean(scope.get("project")) or "unknown",
        who=tuple(_clean(item) for item in who) if isinstance(who, list) else (),
        where=tuple(_clean(item) for item in where) if isinstance(where, list) else (),
        when=_clean(scope.get("when")),
        kind=_clean(row["kind"]),
        source=_clean(row["source_type"]),
    )


@dataclass
class Context:
    project: str = ""
    who: tuple[str, ...] = ()
    where: tuple[str, ...] = ()
    when: str = ""
    kind: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"project": self.project, "who": list(self.who), "where": list(self.where),
                "when": self.when, "kind": self.kind}

    @property
    def named(self) -> list[str]:
        named = []
        for family in FAMILIES:
            value = getattr(self, family, "")
            if isinstance(value, str):
                named.append(family) if _clean(value) else None
            elif value:
                named.append(family)
        return named


def context(*, project: str = "", who: Iterable[str] = (), where: Iterable[str] = (),
            when: str = "", kind: str = "") -> Context:
    return Context(
        project=_clean(project),
        who=tuple(_clean(item) for item in who),
        where=tuple(_clean(item) for item in where),
        when=_clean(when),
        kind=_clean(kind),
    )


@dataclass
class Match:
    memory_id: str = ""
    boost: float = 0.0
    matched: list[str] = field(default_factory=list)
    penalised: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)

    @property
    def multiplier(self) -> float:
        return round(1.0 + self.boost, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "boost": round(self.boost, 4),
            "multiplier": self.multiplier,
            "matched": self.matched,
            "penalised": self.penalised,
            "ignored": self.ignored,
            "sentence": self.describe(),
        }

    def describe(self) -> str:
        if self.matched and self.penalised:
            return (f"matches {', '.join(self.matched)}, against {', '.join(self.penalised)} "
                    f"-> x{self.multiplier}")
        if self.matched:
            return f"matches {', '.join(self.matched)} -> x{self.multiplier}"
        if self.penalised:
            return f"against {', '.join(self.penalised)} -> x{self.multiplier}"
        return "no context in common — neutral"


def compatibility(tags: Tags, at: Context, *, memory_id: str = "") -> Match:
    """How well a memory's tags fit the current context."""
    match = Match(memory_id=memory_id)
    total = 0.0
    for family in FAMILIES:
        if family in {"who", "where"}:
            # These are tuple families. Cleaning one with `_clean` produced the string
            # "('jacoby',)" and turned every match into a mismatch, so they are compared as sets.
            raw = getattr(at, family, ()) or ()
            wanted_values = {_clean(item) for item in
                             (raw if not isinstance(raw, str) else (raw,))} - {""}
            if not wanted_values:
                continue
            mine = {_clean(item) for item in (getattr(tags, family, ()) or ())} - {""}
            if not mine:
                match.ignored.append(family)
                continue
            if mine & wanted_values:
                total += WEIGHT.get(family, 0.1)
                match.matched.append(family)
            elif family in DECISIVE:
                total -= WEIGHT.get(family, 0.1)
                match.penalised.append(family)
            continue
        wanted = _clean(getattr(at, family, ""))
        if not wanted:
            continue
        mine_scalar = _clean(getattr(tags, family, ""))
        if not mine_scalar or mine_scalar == "unknown":
            # Silence is not disagreement: most of a real store predates scope capture.
            match.ignored.append(family)
            continue
        if mine_scalar == wanted:
            total += WEIGHT.get(family, 0.1)
            match.matched.append(family)
        elif family in DECISIVE:
            total -= WEIGHT.get(family, 0.1)
            match.penalised.append(family)
        else:
            match.ignored.append(family)
    # A cap on both ends: tags break near-ties, they never overturn a clearly better text match.
    match.boost = max(-MAX_BOOST, min(MAX_BOOST, total))
    return match


def boost(store: Any, memory_id: str, at: Context) -> float:
    """The multiplier for one memory in one context, or 1.0 when the context says nothing."""
    if not at.named:
        return 1.0
    return compatibility(tags_for(store, memory_id), at, memory_id=str(memory_id)).multiplier


def rank(store: Any, memory_ids: Iterable[str], at: Context) -> list[dict[str, Any]]:
    """The same ids, ordered by context compatibility, with the reason attached."""
    matches = [compatibility(tags_for(store, memory_id), at, memory_id=str(memory_id))
               for memory_id in memory_ids]
    matches.sort(key=lambda item: (-item.boost, item.memory_id))
    return [item.as_dict() for item in matches]


def applied(store: Any, memory_ids: Iterable[str], at: Context) -> list[tuple[str, float]]:
    """(id, multiplier) pairs a ranker would apply — the shape a scoring hook needs."""
    return [(str(memory_id), boost(store, memory_id, at)) for memory_id in memory_ids]


def report(store: Any, *, sample: int = 500) -> dict[str, Any]:
    rows: list[Any] = []
    try:
        rows = store._conn.execute(
            "SELECT id, scope_json FROM memories WHERE state = 'active' ORDER BY updated_at DESC "
            "LIMIT ?", (int(sample),)
        ).fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    counted = {"project": 0, "who": 0, "where": 0, "when": 0, "none": 0}
    for row in rows:
        scope = _load(row["scope_json"])
        found = False
        for family in ("project", "who", "where", "when"):
            value = scope.get(family)
            if value and (not isinstance(value, list) or value):
                counted[family] += 1
                found = True
        if not found:
            counted["none"] += 1
    total = len(rows)
    return {
        "sampled": total,
        "scoped": {family: counted[family] for family in ("project", "who", "where", "when")},
        "unscoped": counted["none"],
        "coverage": round(sum(counted[family] for family in ("project", "who", "where", "when")) /
                          max(1, total), 4),
        "caps": {"max_boost": MAX_BOOST, "decisive": list(DECISIVE)},
        "note": "silence is not disagreement — an unscoped memory is neutral, never penalised",
    }
