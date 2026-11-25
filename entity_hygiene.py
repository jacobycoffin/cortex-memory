"""How healthy the record is for each thing you talk about (#116).

A store can be large and still have entities it tells you nothing useful about: one glancing mention, a fact whose
counterpart was retired five months ago, two statements that disagree and both still say `active`. Those are the
records that fail when they are asked about — and no per-memory score finds them, because the problem is the
**entity**, not any one row.

The score is deliberately boring and fully explained:

    hygiene = 1.0 - 0.35*contradictory - 0.25*thin - 0.15*stale - 0.10*unexercised

with each term a named fact about the entity, reported next to the number so nobody has to reverse-engineer it:

- **contradictory**: two *active* memories about the same subject and predicate assert different objects. A retired
  side is not a contradiction, it is history — counting it would punish the store for having been corrected.
- **thin**: exactly one memory. One mention is not a record; it is a memory that happened to name something.
- **stale**: the newest mention is older than `stale_days`.
- **unexercised**: nothing about it has ever been retrieved. Common and not always wrong, so it costs the least.

Four rules, each a way this score would mislead:

- **the score never travels without its parts.** A bare 0.55 invites tuning the constant instead of fixing the record.
- **thin is reported apart from bad.** A new entity is thin for a week and that is correct; the verdict names `thin`
  and the caller decides whether it is a problem.
- **only active memories count.** Archived, cold, quarantined and tombstoned rows are decisions about the past.
- **an entity with no memories is not scored.** It does not exist in the store, and inventing a zero for it would put
  fiction in the report.

Read-only, bounded, worst first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

MEMORY_TABLE = "memories"
ACTIVE_STATES = ("active", "cold")

STALE_DAYS = 180
MAX_ENTITIES = 25

W_CONTRADICTION = 0.35
W_THIN = 0.25
W_STALE = 0.15
W_UNEXERCISED = 0.10

VERDICTS = ("contradictory", "thin", "stale", "unexercised", "healthy")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def _ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value))
    except Exception:  # noqa: BLE001
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


@dataclass
class Hygiene:
    entity: str
    memories: int
    hygiene: float
    verdict: str
    newest: str = ""
    contradictions: int = 0
    unexercised: int = 0

    def parts(self) -> dict[str, float]:
        return {
            "contradiction": round(W_CONTRADICTION * (1.0 if self.contradictions else 0.0), 3),
            "thin": round(W_THIN * (1.0 if self.memories <= 1 else 0.0), 3),
            "stale": round(W_STALE * (1.0 if self.verdict == "stale" else 0.0), 3),
            "unexercised": round(W_UNEXERCISED * (1.0 if self.memories and self.unexercised == self.memories else 0.0), 3),
        }

    def render(self) -> str:
        why = ", ".join(f"{key} -{value}" for key, value in self.parts().items() if value)
        return (f"{self.entity[:56]}: {self.hygiene:.2f} {self.verdict} "
                f"({self.memories} memory(s){', ' + why if why else ''})")

    def as_dict(self) -> dict[str, Any]:
        return {"entity": self.entity, "memories": self.memories, "hygiene": round(self.hygiene, 4),
                "verdict": self.verdict, "newest": self.newest, "contradictions": self.contradictions,
                "unexercised": self.unexercised, "parts": self.parts()}


@dataclass
class Report:
    readable: bool = True
    reason: str = ""
    considered: int = 0
    entities: list[Hygiene] = field(default_factory=list)
    hidden: int = 0
    verdicts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"readable": self.readable, "reason": self.reason, "considered": self.considered,
                "hidden": self.hidden, "verdicts": self.verdicts,
                "entities": [entity.as_dict() for entity in self.entities]}

    def render(self, limit: int = MAX_ENTITIES) -> list[str]:
        if not self.readable:
            return [self.reason]
        if not self.considered:
            return ["No entity in the store carries a memory yet."]
        head = f"{self.considered} entit(ies): " + ", ".join(
            f"{count} {verdict}" for verdict, count in sorted(self.verdicts.items()))
        lines = [head]
        lines.extend(f"- {entity.render()}" for entity in self.entities[:limit])
        if self.hidden:
            lines.append(f"… and {self.hidden} more (report(limit=…))")
        return lines


def _contradictions(store: Any) -> dict[str, int]:
    """Active memories about one entity that assert different objects for the same subject and predicate."""
    rows = _rows(
        store,
        f"""SELECT value AS entity, COUNT(DISTINCT object_value) AS objects
            FROM {MEMORY_TABLE}, json_each({MEMORY_TABLE}.entities_json)
            WHERE json_valid({MEMORY_TABLE}.entities_json)
              AND state IN ({','.join('?' * len(ACTIVE_STATES))})
              AND subject IS NOT NULL AND subject <> '' AND predicate IS NOT NULL AND predicate <> ''
              AND object_value IS NOT NULL AND object_value <> ''
            GROUP BY value, subject, predicate
            HAVING COUNT(DISTINCT object_value) > 1""",
        ACTIVE_STATES,
    )
    counts: dict[str, int] = {}
    for row in rows or []:
        key = str(row["entity"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def report(store: Any, *, stale_days: int = STALE_DAYS, now: datetime | None = None,
           limit: int = MAX_ENTITIES) -> Report:
    moment = now or _now()
    rows = _rows(
        store,
        f"""SELECT value AS entity, COUNT(*) AS memories,
                   MAX(COALESCE(observed_at, created_at)) AS newest,
                   SUM(CASE WHEN COALESCE(used_count, 0) = 0 THEN 1 ELSE 0 END) AS unexercised
            FROM {MEMORY_TABLE}, json_each({MEMORY_TABLE}.entities_json)
            WHERE json_valid({MEMORY_TABLE}.entities_json)
              AND state IN ({','.join('?' * len(ACTIVE_STATES))})
            GROUP BY value""",
        ACTIVE_STATES,
    )
    if rows is None:
        return Report(readable=False, reason="the store could not be read")
    conflicts = _contradictions(store)

    scored: list[Hygiene] = []
    verdicts: dict[str, int] = {}
    for row in rows:
        name = str(row.get("entity") or "").strip()
        count = int(row.get("memories") or 0)
        if not name or count <= 0:
            continue  # an entity with no memories is not scored; inventing a zero puts fiction in the report
        newest_ts = _ts(row.get("newest"))
        stale = newest_ts is not None and (moment - newest_ts) > timedelta(days=max(1, int(stale_days)))
        contradictions = int(conflicts.get(name, 0))
        thin = count <= 1
        unexercised = int(row.get("unexercised") or 0)

        hygiene = 1.0
        if contradictions:
            hygiene -= W_CONTRADICTION
        if thin:
            hygiene -= W_THIN
        if stale:
            hygiene -= W_STALE
        if unexercised and unexercised == count:
            hygiene -= W_UNEXERCISED
        hygiene = round(max(0.0, min(1.0, hygiene)), 4)

        if contradictions:
            verdict = "contradictory"
        elif thin:
            verdict = "thin"
        elif stale:
            verdict = "stale"
        elif unexercised == count:
            verdict = "unexercised"
        else:
            verdict = "healthy"
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        scored.append(Hygiene(entity=name, memories=count, hygiene=hygiene, verdict=verdict,
                              newest=str(row.get("newest") or ""), contradictions=contradictions,
                              unexercised=unexercised))

    scored.sort(key=lambda item: (item.hygiene, -item.memories, item.entity))
    out = Report(considered=len(scored), verdicts=verdicts)
    out.entities = scored[: max(0, int(limit))]
    out.hidden = max(0, len(scored) - len(out.entities))
    return out
