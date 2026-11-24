"""Inline memory controls on a phone, built so a fat thumb cannot corrupt the store (#146)

A memory shown in Telegram gets a row of buttons: pin, correct, forget, explain, mute. Each of those is a
`callback_data` string that comes back to the bot — untrusted input, arriving twice, on a device where the
finger is bigger than the target.

Five rules, each one a failure this has to survive:

- **a callback can arrive twice.** Double-tap, retry, network replay. Every action is **idempotent**: pin
  twice is still pinned, forget twice is still one demotion. A control that toggles on replay is a control
  that un-pins what he just pinned.
- **a prefix is not an id.** `callback_data` is capped at **64 bytes** by Telegram, so a full UUID plus an
  action name does not fit; the button carries a short prefix and the server resolves it. Live, **8-char
  prefixes collide twice across 3,843 memories and 10-char prefixes collide zero times** — so the width is
  measured, not guessed. The resolver still **refuses an ambiguous prefix** rather than taking the first
  match, because collisions will come back as the store grows.
- **nothing destructive happens on one tap.** `forget` demotes and needs `confirm=True`; the reversible
  actions (pin, mute) apply immediately.
- **forget demotes, it never deletes.** The row keeps its content and gets a retirement date; a phone
  button is not a licence to erase history.
- **the label states the action, the data carries the id.** A button labelled with a UUID is unreadable;
  a button whose data contains a sentence is over the byte cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

PREFIX = 10
MAX_CALLBACK_BYTES = 64
SEPARATOR = ":"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


@dataclass(frozen=True)
class Action:
    name: str
    label: str
    destructive: bool = False
    needs_confirmation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "label": self.label, "destructive": self.destructive,
                "needs_confirmation": self.needs_confirmation}


ACTIONS: tuple[Action, ...] = (
    Action("explain", "Why?"),
    Action("pin", "Pin"),
    Action("unpin", "Unpin"),
    Action("mute", "Mute"),
    Action("unmute", "Unmute"),
    Action("correct", "Correct", needs_confirmation=True),
    Action("forget", "Forget", destructive=True, needs_confirmation=True),
)


def action_for(name: str) -> Action | None:
    for action in ACTIONS:
        if action.name == str(name):
            return action
    return None


def callback(action: str, memory_id: str) -> str:
    """The `callback_data` string. Refuses rather than truncating an id silently."""
    if action_for(action) is None:
        raise ValueError(f"unknown action {action!r}")
    prefix = str(memory_id)[:PREFIX]
    if len(prefix) < PREFIX:
        raise ValueError(f"memory id {memory_id!r} is shorter than the {PREFIX}-char prefix")
    data = f"{action}{SEPARATOR}{prefix}"
    if len(data.encode("utf-8")) > MAX_CALLBACK_BYTES:
        raise ValueError(f"callback_data would be {len(data)} bytes; Telegram allows "
                         f"{MAX_CALLBACK_BYTES}")
    return data


def parse(data: str) -> tuple[str, str]:
    """Split a callback back into (action, prefix). Raises on anything malformed."""
    text = str(data or "")
    if SEPARATOR not in text:
        raise ValueError("callback_data has no separator")
    action, prefix = text.split(SEPARATOR, 1)
    if action_for(action) is None:
        raise ValueError(f"unknown action {action!r}")
    if not prefix or len(prefix) > PREFIX:
        raise ValueError(f"prefix {prefix!r} is not {PREFIX} characters")
    return action, prefix


def resolve(store: Any, prefix: str) -> dict[str, Any]:
    """Find the one memory a prefix means. Ambiguity is reported, never resolved by luck."""
    rows = _rows(store, "SELECT id FROM memories WHERE id LIKE ?", (f"{prefix}%",))
    ids = [str(row["id"]) for row in rows]
    if not ids:
        return {"status": "missing", "id": "", "candidates": []}
    if len(ids) > 1:
        return {"status": "ambiguous", "id": "", "candidates": ids[:5]}
    return {"status": "unique", "id": ids[0], "candidates": ids}


@dataclass
class Result:
    ok: bool = False
    action: str = ""
    memory_id: str = ""
    status: str = ""
    changed: bool = False
    needs_confirmation: bool = False
    message: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "action": self.action, "memory_id": self.memory_id,
                "status": self.status, "changed": self.changed,
                "needs_confirmation": self.needs_confirmation, "message": self.message,
                "fields": self.fields}


def handle(store: Any, data: str, *, confirm: bool = False, now: datetime | None = None) -> Result:
    """Apply one button press. Idempotent: pressing twice does not undo the first press."""
    try:
        action_name, prefix = parse(data)
    except ValueError as error:
        return Result(status="malformed", message=str(error))
    action = action_for(action_name)
    assert action is not None  # parse() already checked
    found = resolve(store, prefix)
    if found["status"] != "unique":
        return Result(action=action_name, status=found["status"],
                      message=f"cannot act: {found['status']} prefix ({len(found['candidates'])} "
                              "candidate(s))")
    memory_id = found["id"]
    if action.needs_confirmation and not confirm:
        return Result(action=action_name, memory_id=memory_id, status="needs_confirmation",
                      needs_confirmation=True,
                      message=f"{action.label} needs a confirmation tap")
    rows = _rows(store, "SELECT * FROM memories WHERE id = ?", (memory_id,))
    if not rows:
        return Result(action=action_name, memory_id=memory_id, status="missing",
                      message="the memory disappeared between the button and the press")
    row = rows[0]
    moment = (now or _now()).isoformat()
    try:
        with store.transaction() as conn:
            if action_name == "pin":
                if row.get("pinned"):
                    return Result(ok=True, action=action_name, memory_id=memory_id,
                                  status="already", message="already pinned")
                conn.execute("UPDATE memories SET pinned = 1 WHERE id = ?", (memory_id,))
            elif action_name == "unpin":
                if not row.get("pinned"):
                    return Result(ok=True, action=action_name, memory_id=memory_id,
                                  status="already", message="not pinned")
                conn.execute("UPDATE memories SET pinned = 0 WHERE id = ?", (memory_id,))
            elif action_name == "mute":
                if str(row.get("state") or "") == "muted":
                    return Result(ok=True, action=action_name, memory_id=memory_id,
                                  status="already", message="already muted")
                conn.execute("UPDATE memories SET state = 'muted' WHERE id = ?", (memory_id,))
            elif action_name == "unmute":
                if str(row.get("state") or "") != "muted":
                    return Result(ok=True, action=action_name, memory_id=memory_id,
                                  status="already", message="not muted")
                conn.execute("UPDATE memories SET state = 'active' WHERE id = ?", (memory_id,))
            elif action_name == "forget":
                if row.get("valid_to"):
                    return Result(ok=True, action=action_name, memory_id=memory_id,
                                  status="already", message="already forgotten")
                conn.execute("UPDATE memories SET valid_to = ?, state = 'superseded' WHERE id = ?",
                             (moment, memory_id))
            elif action_name == "correct":
                return Result(action=action_name, memory_id=memory_id, status="needs_text",
                              message="a correction needs the corrected sentence; "
                                      "use correction_memory.record()")
            else:  # explain is read-only
                return Result(ok=True, action=action_name, memory_id=memory_id, status="read_only",
                              message="explanation requested")
        return Result(ok=True, action=action_name, memory_id=memory_id, status="applied",
                      changed=True, message=f"{action.label.lower()} applied")
    except Exception as error:  # noqa: BLE001
        return Result(action=action_name, memory_id=memory_id, status="failed",
                      message=f"the store refused: {error}")


def keyboard(memory_id: str, *, include_forget: bool = False) -> list[list[dict[str, str]]]:
    """The inline keyboard, two rows, labels readable and data inside the byte cap."""
    first = [action for action in ACTIONS if action.name in ("explain", "pin", "mute")]
    second = [action for action in ACTIONS
              if action.name in ("correct", "forget") and (include_forget or action.name != "forget")]
    rows = [[{"text": action.label, "callback_data": callback(action.name, memory_id)}
             for action in first]]
    if second:
        rows.append([{"text": action.label, "callback_data": callback(action.name, memory_id)}
                     for action in second])
    return rows


def coverage(store: Any) -> dict[str, Any]:
    """Does a short prefix still identify a memory uniquely at this store size?"""
    rows = _rows(store, f"SELECT SUBSTR(id, 1, {PREFIX}) AS prefix, COUNT(*) AS count "
                        "FROM memories GROUP BY prefix HAVING count > 1")
    total = _rows(store, "SELECT COUNT(*) AS count FROM memories")
    return {
        "memories": int(total[0]["count"] or 0) if total else 0,
        "prefix": PREFIX,
        "collisions": len(rows),
        "note": ("a collision is refused, not guessed; widen the prefix when this stops being zero"
                 if rows else "every prefix identifies exactly one memory"),
    }


def render(store: Any) -> list[str]:
    described = coverage(store)
    lines = [f"telegram controls: {described['memories']} memor(ies), {PREFIX}-char prefix, "
             f"{described['collisions']} collision(s)"]
    lines.append("  actions: " + ", ".join(action.name for action in ACTIONS))
    lines.append(f"  {described['note']}")
    return lines
