"""Consolidation rollback (#112): reverse a night's merges in one command.

A night of consolidation can merge a dozen memories, and the only way back was one `merge_diff.undo(merge_id)` at a
time - which means knowing the ids, in the right order, and hoping nothing else moved. This reverses the whole night
and refuses the parts that are no longer safe.

Six rules, each from a way the naive version is wrong:

1. **Reverse order, always.** Merges in a night form a chain: a later merge may have absorbed the survivor of an
   earlier one. Undoing forwards restores a row into a state a later merge already changed, and the night ends up
   half-reversed.
2. **A survivor edited after the merge is not rolled back.** The undo record restores the pre-merge row, so reversing
   a merge whose survivor has since been edited silently discards that edit. The rollback refuses and names the field.
3. **Stop at the first failure.** Every remaining undo was planned against a state that no longer holds; continuing
   turns one refusal into a pile of them.
4. **Already-undone is reported, not retried** - `merge_diff.undo` refuses it, and a rollback that treats that as an
   error is a rollback that can never be run twice.
5. **The night has the same 06:00 UTC boundary as the rest of the sleep machinery**, so "last night" means the same
   thing here as it does in the budget and the journal.
6. **Dry run by default, named consent to apply**, and the rollback leaves its own record behind: a reversal nobody can
   see is indistinguishable from a merge that never happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from . import merge_diff

# The sleep night runs 06:00 -> 05:59 UTC, matching sleep_budget and sleep_journal. A plain calendar date splits every
# overnight run in two, which is how "last night's merges" quietly becomes "some of last night's merges".
NIGHT_START_HOUR = 6

REVERSIBLE = "reversible"
ALREADY_UNDONE = "already_undone"
EDITED_SINCE = "edited_since"
INCOMPLETE = "incomplete_record"


@dataclass(frozen=True)
class Item:
    merge_id: str
    survivor_id: str
    absorbed_id: str
    created_at: str
    verdict: str
    reason: str = ""
    fields_changed: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "merge_id": self.merge_id,
            "survivor_id": self.survivor_id,
            "absorbed_id": self.absorbed_id,
            "created_at": self.created_at,
            "verdict": self.verdict,
            "reason": self.reason,
            "fields_changed": list(self.fields_changed),
        }

    def render(self) -> str:
        tail = f" — {self.reason}" if self.reason else ""
        return f"{self.merge_id} {self.verdict}: {self.survivor_id[:8]} <- {self.absorbed_id[:8]}{tail}"


@dataclass
class Plan:
    night: str = ""
    items: list[Item] = field(default_factory=list)
    readable: bool = True
    reason: str = ""

    @property
    def reversible(self) -> list[Item]:
        return [item for item in self.items if item.verdict == REVERSIBLE]

    @property
    def refused(self) -> list[Item]:
        return [item for item in self.items if item.verdict in (EDITED_SINCE, INCOMPLETE)]

    def render(self) -> list[str]:
        if not self.readable:
            return [f"the merge record could not be read: {self.reason}"]
        lines = [f"night {self.night}: {len(self.items)} merge(s) — {len(self.reversible)} reversible, "
                 f"{len(self.refused)} refused, "
                 f"{len([i for i in self.items if i.verdict == ALREADY_UNDONE])} already undone"]
        for item in self.items[:12]:
            lines.append(f"  - {item.render()}")
        if len(self.items) > 12:
            lines.append(f"  … and {len(self.items) - 12} more")
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "night": self.night,
            "readable": self.readable,
            "reason": self.reason,
            "items": [item.as_dict() for item in self.items],
            "reversible": len(self.reversible),
            "refused": len(self.refused),
        }


def night_bounds(now: datetime | None = None) -> tuple[datetime, datetime, str]:
    """The sleep night containing `now`: 06:00 UTC to 05:59:59 the next day."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    start = moment.replace(hour=NIGHT_START_HOUR, minute=0, second=0, microsecond=0)
    if moment.hour < NIGHT_START_HOUR:
        start -= timedelta(days=1)
    return start, start + timedelta(days=1), start.date().isoformat()


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        cursor = store._conn.execute(sql, params)
    except Exception:
        return None
    try:
        return [dict(row) for row in cursor.fetchall()]
    except Exception:
        return None


def _parse(value: Any) -> datetime | None:
    try:
        text = str(value or "").replace("Z", "+00:00")
        moment = datetime.fromisoformat(text)
    except Exception:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _verdict(store: Any, record: dict[str, Any]) -> Item:
    undo = record.get("undo") or {}
    survivor_side = undo.get("survivor") or {}
    before = survivor_side.get("before") or {}
    after = survivor_side.get("after") or {}
    base = Item(
        merge_id=str(record.get("merge_id")),
        survivor_id=str(record.get("survivor_id") or ""),
        absorbed_id=str(record.get("absorbed_id") or ""),
        created_at=str(record.get("created_at") or ""),
        verdict=REVERSIBLE,
    )
    if record.get("undone_at"):
        return Item(**{**base.__dict__, "verdict": ALREADY_UNDONE, "reason": "already undone"})
    if not before:
        return Item(**{**base.__dict__, "verdict": INCOMPLETE,
                       "reason": "the undo record has no pre-merge state to restore"})
    current = _rows(store, "SELECT * FROM memories WHERE id = ?", (base.survivor_id,))
    if current is None:
        return Item(**{**base.__dict__, "verdict": INCOMPLETE, "reason": "the store refused the read"})
    if not current:
        return Item(**{**base.__dict__, "verdict": INCOMPLETE, "reason": "the survivor no longer exists"})
    row = current[0]
    changed: list[str] = []
    for name, expected in after.items():
        if expected is not None and row.get(name) != expected:
            changed.append(name)
    if changed:
        return Item(**{**base.__dict__, "verdict": EDITED_SINCE, "fields_changed": tuple(sorted(changed)),
                       "reason": f"the survivor changed after the merge ({', '.join(sorted(changed))}); "
                                 f"reversing would discard that"})
    return base


def plan(store: Any, *, night: str | None = None, now: datetime | None = None,
         limit: int = 200) -> Plan:
    """List the night's merges newest-first with a verdict each. Writes nothing."""
    start, end, label = night_bounds(now)
    if night:
        try:
            start = datetime.fromisoformat(night).replace(tzinfo=timezone.utc)
            end = start + timedelta(days=1)
            label = start.date().isoformat()
        except Exception:
            return Plan(readable=False, reason=f"not a night I can read: {night!r}")
    records = merge_diff.records(store, limit=max(1, min(int(limit), 500)))
    if records is None:
        return Plan(night=label, readable=False, reason="no merge record table")
    in_night: list[dict[str, Any]] = []
    for record in records:
        moment = _parse(record.get("created_at"))
        if moment is None:
            continue
        if start <= moment < end:
            in_night.append(record)
    # Newest first: the plan is also the execution order, and the order is the safety property.
    in_night.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    return Plan(night=label, items=[_verdict(store, record) for record in in_night])


def apply(store: Any, plan_result: Plan, *, apply: bool = False,
          consented_by: str | None = None) -> dict[str, Any]:
    """Reverse the night's merges newest-first. Dry run unless `apply` and a named consent are both given."""
    if not plan_result.readable:
        return {"ok": False, "reason": plan_result.reason, "undone": 0, "dry_run": not apply}
    targets = plan_result.reversible
    if not apply:
        return {"ok": True, "dry_run": True, "would_undo": len(targets), "undone": 0,
                "refused": [item.merge_id for item in plan_result.refused],
                "skipped_already_undone": len([i for i in plan_result.items if i.verdict == ALREADY_UNDONE])}
    if not consented_by:
        return {"ok": False, "reason": "reversing a night's merges needs a named consenting person",
                "undone": 0, "dry_run": False}
    undone: list[str] = []
    stopped_at = ""
    for item in targets:
        result = merge_diff.undo(store, item.merge_id)
        if result.get("ok"):
            undone.append(item.merge_id)
            continue
        # Stop here: the remaining undos were planned against a state this failure just disproved.
        stopped_at = f"{item.merge_id}: {result.get('reason')}"
        break
    return {
        "ok": not stopped_at,
        "dry_run": False,
        "undone": len(undone),
        "merge_ids": undone,
        "stopped_at": stopped_at,
        "refused": [item.merge_id for item in plan_result.refused],
        "consented_by": str(consented_by),
    }
