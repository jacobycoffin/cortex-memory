"""What changed about this thing, in order (#50).

"Everything about CT116" is a pile. A timeline is the same pile sorted by when it was learned, with
the changes marked — and the marking is the whole value: *"the dashboard port was 8100 in May, 8200
in November"* is a fact about the world; *"port 8100"* and *"port 8200"* stored side by side is a
contradiction with no date attached.

Two rules:

- **the date is when the store learned it**, not when the event happened. A memory recorded today
  about last year belongs in today's position, because that is when the store's belief changed.
  Anchored phrases (#40) carry the event date in the text, so both are readable.
- **a change is a supersession pair**, not a similarity. Two memories about the same entity that
  disagree are only a *change* when one explicitly supersedes the other (#48/#130); otherwise they
  are a dispute, and the timeline says so instead of guessing which is newer.

Read-only. A timeline that mutates the store is a report that cannot be run twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

DEFAULT_STATES = ("active", "cold", "archived", "suppressed")
MAX_EVENTS = 200


def _utc(text: Any) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass
class Event:
    when: datetime | None
    memory_id: str
    kind: str
    state: str
    content: str
    change: str = ""

    def describe(self) -> str:
        stamp = self.when.date().isoformat() if self.when else "unknown"
        mark = f"  <{self.change}>" if self.change else ""
        return f"{stamp}  {str(self.content)[:80]}{mark}"

    @property
    def sort_key(self) -> tuple[int, str]:
        # Undated rows sort last rather than at the epoch: "unknown" is not "1970".
        return (1, "") if self.when is None else (0, self.when.isoformat())


def _names(store: Any, entity: str) -> list[str]:
    """Every name this entity answers to, so the timeline is not one spelling wide (#39)."""
    names = {str(entity).strip()}
    try:
        from . import aliases

        for name in aliases.expand(store, [entity]):
            names.add(str(name))
    except Exception:  # noqa: BLE001 - the alias table is optional here
        pass
    try:
        from . import aliases

        for alias in aliases.aliases_for(store, entity):
            names.add(alias.alias)
    except Exception:  # noqa: BLE001
        pass
    return sorted(name for name in names if name)


def memories(store: Any, entity: str, *, states: Iterable[str] = DEFAULT_STATES,
             limit: int = MAX_EVENTS) -> list[dict[str, Any]]:
    """Every memory attached to the entity or any of its aliases. Read-only."""
    names = _names(store, entity)
    if not names:
        return []
    placeholders = ",".join("?" for _ in names)
    state_list = list(states) or list(DEFAULT_STATES)
    state_placeholders = ",".join("?" for _ in state_list)
    try:
        rows = store._conn.execute(
            f"""SELECT DISTINCT m.id, m.content, m.kind, m.state, m.created_at, m.observed_at,
                       m.supersedes_id
                FROM memories m, json_each(m.entities_json)
                WHERE json_valid(m.entities_json)
                  AND value IN ({placeholders})
                  AND m.state IN ({state_placeholders})
                LIMIT ?""",
            (*names, *state_list, int(limit)),
        ).fetchall()
    except Exception:  # noqa: BLE001 - a timeline never raises into a chat path
        return []
    return [dict(row) for row in rows]


def timeline(
    store: Any,
    entity: str,
    *,
    states: Iterable[str] = DEFAULT_STATES,
    limit: int = MAX_EVENTS,
) -> list[Event]:
    """The entity's facts in the order the store learned them, with changes marked."""
    rows = memories(store, entity, states=states, limit=limit)
    if not rows:
        return []

    by_id = {str(row["id"]): row for row in rows}
    superseded_by: dict[str, list[str]] = {}
    for row in rows:
        source = str(row.get("supersedes_id") or "")
        if source and source in by_id:
            superseded_by.setdefault(source, []).append(str(row["id"]))

    events: list[Event] = []
    for row in rows:
        memory_id = str(row["id"])
        change = ""
        if memory_id in superseded_by:
            change = f"replaced by {superseded_by[memory_id][0][:8]}"
        elif str(row.get("supersedes_id") or "") in by_id:
            change = f"replaces {str(row['supersedes_id'])[:8]}"
        events.append(
            Event(
                when=_utc(row.get("observed_at")) or _utc(row.get("created_at")),
                memory_id=memory_id,
                kind=str(row.get("kind") or "unknown"),
                state=str(row.get("state") or "active"),
                content=str(row.get("content") or ""),
                change=change,
            )
        )
    return sorted(events, key=lambda event: event.sort_key)


def changes(store: Any, entity: str, *, limit: int = MAX_EVENTS) -> list[dict[str, Any]]:
    """Only the supersession pairs: what the store used to believe, and what it believes now."""
    rows = memories(store, entity, limit=limit)
    by_id = {str(row["id"]): row for row in rows}
    pairs: list[dict[str, Any]] = []
    for row in rows:
        old_id = str(row.get("supersedes_id") or "")
        if not old_id or old_id not in by_id:
            continue
        old = by_id[old_id]
        pairs.append(
            {
                "old_id": old_id,
                "new_id": str(row["id"]),
                "was": str(old.get("content") or ""),
                "now": str(row.get("content") or ""),
                "when": _utc(row.get("observed_at")) or _utc(row.get("created_at")),
            }
        )
    return sorted(pairs, key=lambda pair: pair["when"] or datetime.min.replace(tzinfo=timezone.utc))


def summarise(store: Any, entity: str, *, limit: int = MAX_EVENTS) -> dict[str, Any]:
    events = timeline(store, entity, limit=limit)
    pairs = changes(store, entity, limit=limit)
    if not events:
        return {
            "entity": str(entity),
            "events": 0,
            "changes": 0,
            "first": None,
            "last": None,
            "kinds": {},
            "states": {},
        }
    kinds: dict[str, int] = {}
    states: dict[str, int] = {}
    for event in events:
        kinds[event.kind] = kinds.get(event.kind, 0) + 1
        states[event.state] = states.get(event.state, 0) + 1
    dated = [event.when for event in events if event.when]
    return {
        "entity": str(entity),
        "events": len(events),
        "changes": len(pairs),
        "first": min(dated).date().isoformat() if dated else None,
        "last": max(dated).date().isoformat() if dated else None,
        "kinds": dict(sorted(kinds.items(), key=lambda item: -item[1])),
        "states": dict(sorted(states.items(), key=lambda item: -item[1])),
    }


def render(store: Any, entity: str, *, limit: int = 12) -> str:
    described = summarise(store, entity)
    if not described["events"]:
        return f"no memories for {entity}"
    head = (
        f"{entity}: {described['events']} memories"
        f" ({described['first']} to {described['last']}),"
        f" {described['changes']} change(s)"
    )
    lines = [head]
    for event in timeline(store, entity)[:limit]:
        lines.append("  " + event.describe())
    for pair in changes(store, entity)[:3]:
        when = pair["when"].date().isoformat() if pair["when"] else "unknown"
        lines.append(f"  change {when}: {pair['was'][:50]!r} -> {pair['now'][:50]!r}")
    return "\n".join(lines)
