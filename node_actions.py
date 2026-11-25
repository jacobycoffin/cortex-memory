"""The actions a node or an edge offers, decided in one place (#167).

Every surface that shows a memory wants to let you act on it — pin it, mute it, correct it, retire it,
quarantine an edge — and every surface that implements that itself grows a slightly different idea of
what "mute" means, whether it can be undone, and whether it should have asked first. This module is the
one catalogue: **the actions are data**, and a view renders them instead of deciding them.

Five rules, each a way an action layer goes wrong:

- **an action is idempotent and says `already`.** Pin twice and the second call reports `already` rather
  than flipping the pin off. A toggle on replay is how a retried request silently reverses a decision.
- **nothing here deletes, and what looks destructive is a demotion.** `retire` sets `state='archived'`,
  `mute` sets `state='cold'`, `quarantine` sets `state='quarantine'` with its reason recorded. All three
  have an inverse. The store's own rule (`demote, never delete`) is enforced by the catalogue containing
  no delete at all — a missing action cannot be called by mistake.
- **a protected memory refuses, with the reason.** A legal hold or `protected=1` outranks every action;
  the refusal names which protection stopped it. Stealth-refusing by silently doing nothing teaches the
  caller that the button is broken.
- **an unknown target is a named refusal.** `unknown memory` / `unknown edge` / `unknown action`, so a
  view can distinguish "you cannot do that to this" from "that no longer exists".
- **the catalogue declares what it needs before it runs**: `needs_confirmation` for anything a person
  would not want a stray tap to do, and `requires_argument` for actions that cannot be applied without
  new text (a correction). A UI can render the difference instead of guessing it.

Read-only actions (`explain`) are in the catalogue too: they carry `writes=False`, so a catalogue built
by this module is a complete description of what a node can do, not only what changes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from . import correction_memory, legal_hold

MEMORY_TABLE = "memories"
EDGE_TABLE = "edges"

# The states this module will move a memory into or out of. Nothing else is written to `state` here:
# a vocabulary that grows at every call site is how a store ends up with six spellings of "hidden".
ACTIVE = "active"
COLD = "cold"
ARCHIVED = "archived"
QUARANTINED = "quarantine"

# Weight moves for an edge. Stated as a constant with a name so the live measurement can quote it, and
# clamped in both directions: a reinforce loop must not reach weight 2.0 and re-rank the graph forever.
EDGE_STEP = 0.05
WEIGHT_CEILING = 1.0
WEIGHT_FLOOR = 0.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Rows, or [] when the store cannot answer — the caller distinguishes the two by what it asked."""
    try:
        cursor = store._conn.execute(sql, params)
    except Exception:  # noqa: BLE001
        return []
    try:
        return [dict(row) for row in cursor.fetchall()]
    except Exception:  # noqa: BLE001
        return []


@dataclass(frozen=True)
class Action:
    """One thing a node or an edge can do. Data, so a surface can render it without knowing it."""

    key: str
    target: str  # "memory" or "edge"
    label: str
    effect: str  # what happens to the row, in the row's own vocabulary
    why: str  # what it is for
    writes: bool = True
    needs_confirmation: bool = False
    requires_argument: bool = False
    reversible: bool = True
    inverse: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "target": self.target,
            "label": self.label,
            "effect": self.effect,
            "why": self.why,
            "writes": self.writes,
            "needs_confirmation": self.needs_confirmation,
            "requires_argument": self.requires_argument,
            "reversible": self.reversible,
            "inverse": self.inverse,
        }


# The catalogue. Ordered by how often a person reaches for it, not alphabetically: a menu nobody
# scrolls should lead with pin and mute.
CATALOGUE: tuple[Action, ...] = (
    Action(
        key="pin", target="memory", label="Pin", effect="pinned = 1",
        why="keep it in the prompt regardless of retrieval", writes=True, inverse="unpin",
    ),
    Action(
        key="unpin", target="memory", label="Unpin", effect="pinned = 0",
        why="stop spending prompt budget on it", writes=True, inverse="pin",
    ),
    Action(
        key="mute", target="memory", label="Mute", effect="state = 'cold'",
        why="stop it being recalled without losing the row", writes=True, inverse="unmute",
    ),
    Action(
        key="unmute", target="memory", label="Unmute", effect="state = 'active'",
        why="let it be recalled again", writes=True, inverse="mute",
    ),
    Action(
        key="retire", target="memory", label="Retire", effect="state = 'archived'",
        why="it was true and is not any more", writes=True,
        needs_confirmation=True, inverse="restore",
    ),
    Action(
        key="restore", target="memory", label="Restore", effect="state = 'active'",
        why="bring a retired or muted memory back", writes=True, inverse="retire",
    ),
    Action(
        key="quarantine", target="memory", label="Quarantine", effect="state = 'quarantine' + reason",
        why="suspect, and not to be shown until somebody looks", writes=True,
        needs_confirmation=True, inverse="restore",
    ),
    Action(
        key="hold", target="memory", label="Legal hold", effect="a row in the hold table",
        why="this must not be touched, altered or released by any automated path", writes=True,
        needs_confirmation=True, inverse="release_hold",
    ),
    Action(
        key="release_hold", target="memory", label="Release hold", effect="the hold is released",
        why="the matter it was held for is closed", writes=True, needs_confirmation=True,
    ),
    Action(
        key="correct", target="memory", label="Correct", effect="a new memory that supersedes it",
        why="what it says is wrong, and the correction is the record", writes=True,
        needs_confirmation=True, requires_argument=True,
    ),
    Action(
        key="explain", target="memory", label="Explain", effect="nothing",
        why="show why this memory is held, ranked and trusted", writes=False,
    ),
    Action(
        key="reinforce", target="edge", label="Reinforce", effect=f"weight += {EDGE_STEP}",
        why="the pair held up again", writes=True, inverse="weaken",
    ),
    Action(
        key="weaken", target="edge", label="Weaken", effect=f"weight -= {EDGE_STEP}",
        why="the pair did not hold up", writes=True, inverse="reinforce",
    ),
    Action(
        key="quarantine_edge", target="edge", label="Quarantine edge", effect="the edge is quarantined",
        why="the relation was asserted and is not supported", writes=True,
        needs_confirmation=True, inverse="reinforce",
    ),
    Action(
        key="explain_edge", target="edge", label="Explain edge", effect="nothing",
        why="show the evidence behind the relation", writes=False,
    ),
)

BY_KEY: dict[str, Action] = {action.key: action for action in CATALOGUE}


def actions_for(target: str) -> list[Action]:
    """The catalogue for one kind of thing, in display order."""
    wanted = str(target or "").strip().lower()
    return [action for action in CATALOGUE if action.target == wanted]


def render_catalogue(target: str = "memory", *, limit: int = 20) -> list[str]:
    lines: list[str] = []
    for action in actions_for(target)[:limit]:
        flags = []
        if action.needs_confirmation:
            flags.append("asks first")
        if action.requires_argument:
            flags.append("needs text")
        if not action.writes:
            flags.append("read-only")
        if not action.reversible:
            flags.append("not reversible")
        lines.append(f"{action.key}: {action.effect}" + (f" ({', '.join(flags)})" if flags else ""))
    return lines


@dataclass
class Result:
    """What an action did, or why it did not. One shape for every action, so a caller has one branch."""

    action: str = ""
    target: str = ""
    target_id: str = ""
    status: str = ""  # applied | already | refused | unknown_target | unknown_action | needs_argument
    reason: str = ""
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {"applied", "already"}

    @property
    def changed(self) -> bool:
        return self.status == "applied"

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "target_id": self.target_id,
            "status": self.status,
            "reason": self.reason,
            "changed": self.changed,
            "before": self.before,
            "after": self.after,
        }


def render(result: Result) -> list[str]:
    if result.status == "applied":
        return [f"{result.action} on {result.target} {result.target_id[:8]}: {result.reason}"]
    if result.status == "already":
        return [f"{result.action} on {result.target} {result.target_id[:8]}: already — {result.reason}"]
    return [f"{result.action} on {result.target} {result.target_id[:8]}: {result.status} — {result.reason}"]


def _memory_row(store: Any, memory_id: str) -> dict[str, Any] | None:
    rows = _rows(
        store,
        f"SELECT id, state, pinned, protected, quarantine_reason, content FROM {MEMORY_TABLE} WHERE id = ?",
        (str(memory_id),),
    )
    return rows[0] if rows else None


def _edge_row(store: Any, src_id: str, dst_id: str, relation: str) -> dict[str, Any] | None:
    rows = _rows(
        store,
        f"SELECT src_id, dst_id, relation, weight FROM {EDGE_TABLE} "
        "WHERE ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?)) AND relation = ?",
        (str(src_id), str(dst_id), str(dst_id), str(src_id), str(relation)),
    )
    return rows[0] if rows else None


def _protected(store: Any, memory_id: str, row: dict[str, Any]) -> str:
    """The protection standing in the way, or '' when nothing does."""
    try:
        if legal_hold.is_held(store, memory_id):
            return "a legal hold is in force"
    except Exception:  # noqa: BLE001
        return "the hold table could not be read, so the hold state is unknown"
    if int(row.get("protected") or 0):
        return "the memory is marked protected"
    return ""


def _snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in ("state", "pinned", "protected", "quarantine_reason", "weight")}


def available(store: Any, target_id: str, *, target: str = "memory", relation: str = "",
              dst_id: str = "") -> list[dict[str, Any]]:
    """Each action and whether it is available right now, with the reason it is not.

    A view needs the reason as much as the verdict: a disabled button with no explanation is the same
    dead end as a refusal that says nothing.
    """
    out: list[dict[str, Any]] = []
    if str(target) == "edge":
        row = _edge_row(store, target_id, dst_id, relation) if dst_id and relation else None
        for action in actions_for("edge"):
            entry = action.as_dict()
            entry["available"] = row is not None
            entry["reason"] = "" if row is not None else "unknown edge"
            out.append(entry)
        return out
    row = _memory_row(store, target_id)
    for action in actions_for("memory"):
        entry = action.as_dict()
        if row is None:
            entry["available"] = False
            entry["reason"] = "unknown memory"
            out.append(entry)
            continue
        blocked = _protected(store, str(target_id), row) if action.writes else ""
        entry["available"] = not blocked
        entry["reason"] = blocked
        out.append(entry)
    return out


def apply(  # noqa: PLR0911 - one return per outcome is the point of Result
    store: Any,
    target_id: str,
    action_key: str,
    *,
    target: str = "memory",
    relation: str = "",
    dst_id: str = "",
    argument: str = "",
    consent: bool = False,
    reason: str = "",
    at: datetime | None = None,
) -> Result:
    """Run one action. Never raises: every outcome is a Result a caller can render."""
    action = BY_KEY.get(str(action_key))
    result = Result(action=str(action_key), target=str(target), target_id=str(target_id))
    if action is None:
        result.status = "unknown_action"
        result.reason = f"no action {action_key!r}; see the catalogue"
        return result
    if action.target != str(target):
        result.status = "unknown_action"
        result.reason = f"{action_key!r} acts on a {action.target}, not a {target}"
        return result
    if action.requires_argument and not str(argument).strip():
        result.status = "needs_argument"
        result.reason = f"{action_key} needs the text it will record"
        return result
    if action.needs_confirmation and not consent:
        result.status = "refused"
        result.reason = f"{action_key} asks first; pass consent to run it"
        return result

    if action.target == "edge":
        return _apply_edge(store, action, result, target_id, dst_id, relation, at)
    return _apply_memory(store, action, result, target_id, argument, reason, at)


def _apply_memory(store: Any, action: Action, result: Result, memory_id: str,
                  argument: str, reason: str, at: datetime | None) -> Result:
    row = _memory_row(store, memory_id)
    if row is None:
        result.status = "unknown_target"
        result.reason = "unknown memory"
        return result
    result.before = _snapshot(row)

    if not action.writes:
        result.status = "already"
        result.reason = "read-only action; nothing was written"
        return result

    # `hold` and `release_hold` are the protection itself, so it cannot block them: checking it here
    # made "hold this" refuse with "a legal hold is in force" and made releasing one impossible.
    if action.key not in {"hold", "release_hold"}:
        blocked = _protected(store, memory_id, row)
        if blocked:
            result.status = "refused"
            result.reason = blocked
            return result

    state = str(row.get("state") or ACTIVE)
    moment = (at or _now()).isoformat()

    def write(sql: str, params: tuple) -> None:
        with store.transaction() as conn:
            conn.execute(sql, params)

    if action.key == "pin":
        if int(row.get("pinned") or 0) == 1:
            result.status, result.reason = "already", "already pinned"
        else:
            write(f"UPDATE {MEMORY_TABLE} SET pinned = 1, updated_at = ? WHERE id = ?", (moment, memory_id))
            result.status, result.reason = "applied", "pinned"
    elif action.key == "unpin":
        if int(row.get("pinned") or 0) == 0:
            result.status, result.reason = "already", "not pinned"
        else:
            write(f"UPDATE {MEMORY_TABLE} SET pinned = 0, updated_at = ? WHERE id = ?", (moment, memory_id))
            result.status, result.reason = "applied", "unpinned"
    elif action.key in {"mute", "unmute", "retire", "quarantine", "restore"}:
        wanted = {"mute": COLD, "unmute": ACTIVE, "retire": ARCHIVED,
                  "quarantine": QUARANTINED, "restore": ACTIVE}[action.key]
        if state == wanted:
            result.status, result.reason = "already", f"already {wanted}"
        else:
            if action.key == "quarantine":
                write(f"UPDATE {MEMORY_TABLE} SET state = ?, quarantine_reason = ?, updated_at = ? WHERE id = ?",
                      (wanted, reason or "quarantined by an operator", moment, memory_id))
            else:
                write(f"UPDATE {MEMORY_TABLE} SET state = ?, updated_at = ? WHERE id = ?",
                      (wanted, moment, memory_id))
            result.status = "applied"
            result.reason = f"{state} -> {wanted}"
    elif action.key == "hold":
        if legal_hold.is_held(store, memory_id):
            result.status, result.reason = "already", "already held"
            return result
        held = legal_hold.place(store, [memory_id], reason=reason or "held by an operator", at=at)
        if held.placed:
            result.status, result.reason = "applied", "hold placed"
        elif held.refused:
            result.status, result.reason = "already", "already held or the id could not be resolved"
        else:  # pragma: no cover - place() either places or refuses
            result.status, result.reason = "refused", "the hold table did not accept the row"
    elif action.key == "release_hold":
        released = legal_hold.release(store, memory_id, reason=reason or "released by an operator", at=at)
        result.status = "applied" if released else "already"
        result.reason = "hold released" if released else "no hold to release"
    elif action.key == "correct":
        record = correction_memory.record(store, supersedes=memory_id, content=argument, now=at)
        result.status = "applied" if record.ok else "refused"
        result.reason = (f"correction {record.correction_id[:8]} supersedes this memory"
                         if record.ok else (record.reason or "the correction was refused"))
        if record.ok:
            result.after = {"supersedes": memory_id, "correction_id": record.correction_id}
            return result
    else:  # pragma: no cover - the catalogue and this branch are checked by test_every_action_is_handled
        result.status = "unknown_action"
        result.reason = "the catalogue offers an action this module cannot apply"
        return result

    after = _memory_row(store, memory_id)
    result.after = _snapshot(after or {})
    return result


def _apply_edge(store: Any, action: Action, result: Result, src_id: str, dst_id: str,
                relation: str, at: datetime | None) -> Result:
    if not dst_id or not relation:
        result.status = "unknown_target"
        result.reason = "an edge needs dst_id and relation as well as src_id"
        return result
    row = _edge_row(store, src_id, dst_id, relation)
    if row is None:
        result.status = "unknown_target"
        result.reason = f"unknown edge {relation}"
        return result
    result.before = _snapshot(row)
    if not action.writes:
        result.status, result.reason = "already", "read-only action; nothing was written"
        return result

    weight = float(row.get("weight") or 0.0)
    if action.key in {"reinforce", "weaken"}:
        delta = EDGE_STEP if action.key == "reinforce" else -EDGE_STEP
        wanted = round(min(WEIGHT_CEILING, max(WEIGHT_FLOOR, weight + delta)), 4)
        if wanted == round(weight, 4):
            result.status = "already"
            result.reason = f"weight already at {wanted} ({'ceiling' if delta > 0 else 'floor'})"
        else:
            with store.transaction() as conn:
                conn.execute(
                    f"UPDATE {EDGE_TABLE} SET weight = ? WHERE src_id = ? AND dst_id = ? AND relation = ?",
                    (wanted, str(row["src_id"]), str(row["dst_id"]), str(row["relation"])),
                )
            result.status = "applied"
            result.reason = f"weight {weight} -> {wanted}"
    elif action.key == "quarantine_edge":
        if not _has_edge_quarantine(store):
            result.status = "refused"
            result.reason = "this store has no edge_quarantine table, so an edge cannot be quarantined"
            return result
        with store.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO edge_quarantine (src_id, dst_id, relation, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(row["src_id"]), str(row["dst_id"]), str(row["relation"]),
                 "quarantined by an operator", (at or _now()).isoformat()),
            )
        result.status, result.reason = "applied", "edge quarantined"
    else:  # pragma: no cover - guarded by test_every_action_is_handled
        result.status = "unknown_action"
        result.reason = "the catalogue offers an edge action this module cannot apply"
        return result

    result.after = _snapshot(_edge_row(store, src_id, dst_id, relation) or {})
    return result


def _has_edge_quarantine(store: Any) -> bool:
    return bool(_rows(store, "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'edge_quarantine'"))


def survey(store: Any, *, limit: int = 5) -> dict[str, Any]:
    """How many live rows each memory action would change, so a catalogue can be argued with."""
    rows = _rows(store, f"SELECT state, pinned, protected, COUNT(*) AS n FROM {MEMORY_TABLE} GROUP BY 1, 2, 3")
    counts = {"pinned": 0, "mutable": 0, "cold": 0, "archived": 0, "quarantined": 0, "protected": 0}
    for row in rows:
        n = int(row.get("n") or 0)
        state = str(row.get("state") or ACTIVE)
        counts["pinned"] += n if int(row.get("pinned") or 0) else 0
        counts["protected"] += n if int(row.get("protected") or 0) else 0
        if state == COLD:
            counts["cold"] += n
        elif state == ARCHIVED:
            counts["archived"] += n
        elif state == QUARANTINED:
            counts["quarantined"] += n
        if state in {ACTIVE, COLD}:
            counts["mutable"] += n
    count = _rows(store, f"SELECT COUNT(*) AS n FROM {MEMORY_TABLE}")
    counts["memories"] = int(count[0]["n"]) if count else 0
    counts["actions"] = len(CATALOGUE)
    counts["writes"] = sum(1 for action in CATALOGUE if action.writes)
    return counts
