"""One store, several devices, no cross-talk (#156)

The same memory is reachable from the phone, the desktop, the watch and whatever else is wired in. Most of
it should be: a preference is a preference wherever it is read. Some of it must not be: "this device is on
192.168.1.44" is true of one phone and false of every other device, and a namespace that lets it leak makes
every device slightly wrong in a way nobody can see.

The scope already exists in the schema — `scope_json` holds `{"project": …}` today — so this extends it
with a `device` key rather than adding a parallel mechanism.

Five rules, and the first two are the ones that decide whether this is usable:

- **unscoped means shared, not device-less.** 3,540 of 3,843 live memories carry `{}`. If an empty scope
  meant "belongs to no device", a phone would open to an empty store. Absence is the *widest* scope, not
  the narrowest.
- **a device scope narrows, never widens.** A phone asking for another phone's memory gets nothing, even
  if it asks politely; there is no `include_all` that a client can set.
- **merging scope must not clobber.** `scope_json` already carries `project` and `system`; writing a device
  must merge, not replace, or scoping a memory silently de-projects it.
- **an unknown device is not a wildcard.** A blank or unrecognised name is refused rather than treated as
  "show me everything".
- **the split is reported.** A namespace nobody can count is a namespace nobody trusts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

DEVICE_KEY = "device"
# Names are lowercase, hyphenated, and deliberately boring: they end up inside a JSON column and a URL.
_MAX_NAME = 32


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def normalise(device: Any) -> str:
    """A device name, or empty when it is unusable. Never a wildcard."""
    text = "".join(char if (char.isalnum() or char in "-_") else "-"
                   for char in str(device or "").strip().lower())
    text = "-".join(part for part in text.split("-") if part)
    return text[:_MAX_NAME]


def scope_of(row: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(row.get("scope_json") or "{}")
    except Exception:  # noqa: BLE001
        return {}
    return value if isinstance(value, dict) else {}


def device_of(row: dict[str, Any]) -> str:
    """Normalised on read as well as on write: five live rows already hold names like
    `COBY-CLX gaming-pc`, and an un-normalised comparison makes them invisible to their own device."""
    return normalise(scope_of(row).get(DEVICE_KEY) or "")


def set_device(store: Any, memory_id: str, device: Any) -> dict[str, Any]:
    """Merge a device into a memory's scope without touching the keys already there."""
    name = normalise(device)
    if not name:
        return {"ok": False, "reason": "a device name is required; an empty one is not a wildcard"}
    rows = _rows(store, "SELECT id, scope_json FROM memories WHERE id = ?", (str(memory_id),))
    if not rows:
        return {"ok": False, "reason": "no such memory"}
    scope = scope_of(rows[0])
    scope[DEVICE_KEY] = name
    try:
        with store.transaction() as conn:
            conn.execute("UPDATE memories SET scope_json = ? WHERE id = ?",
                         (json.dumps(scope, sort_keys=True), str(memory_id)))
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "reason": f"the store refused the scope: {error}"}
    return {"ok": True, "device": name, "scope": scope}


def clear_device(store: Any, memory_id: str) -> dict[str, Any]:
    rows = _rows(store, "SELECT id, scope_json FROM memories WHERE id = ?", (str(memory_id),))
    if not rows:
        return {"ok": False, "reason": "no such memory"}
    scope = scope_of(rows[0])
    if DEVICE_KEY not in scope:
        return {"ok": True, "changed": False, "scope": scope}
    scope.pop(DEVICE_KEY, None)
    with store.transaction() as conn:
        conn.execute("UPDATE memories SET scope_json = ? WHERE id = ?",
                     (json.dumps(scope, sort_keys=True), str(memory_id)))
    return {"ok": True, "changed": True, "scope": scope}


def visible(store: Any, device: Any, *, limit: int = 200, include_shared: bool = True
            ) -> list[dict[str, Any]]:
    """What one device may see: its own memories, plus everything unscoped when asked."""
    name = normalise(device)
    if not name:
        return []
    rows = _rows(store, "SELECT id, content, scope_json FROM memories WHERE state <> 'archived' "
                        "ORDER BY created_at DESC LIMIT ?", (max(1, int(limit)),))
    kept: list[dict[str, Any]] = []
    for row in rows:
        scoped = device_of(row)
        if scoped == name:
            kept.append({"id": row["id"], "content": row["content"], "scope": "own"})
        elif not scoped and include_shared:
            kept.append({"id": row["id"], "content": row["content"], "scope": "shared"})
    return kept


def deny(store: Any, device: Any, memory_id: str) -> bool:
    """True when a device may not read this memory. Narrowing only, never widening."""
    name = normalise(device)
    rows = _rows(store, "SELECT id, scope_json FROM memories WHERE id = ?", (str(memory_id),))
    if not rows:
        return True
    scoped = device_of(rows[0])
    if not scoped:
        return False
    return scoped != name


@dataclass
class Split:
    memories: int = 0
    by_device: dict[str, int] = field(default_factory=dict)
    shared: int = 0
    unknown_scopes: int = 0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"memories": self.memories, "by_device": self.by_device, "shared": self.shared,
                "unknown_scopes": self.unknown_scopes, "note": self.note}


def split(store: Any, *, scan: int = 5000) -> Split:
    """How the store divides, so a namespace can be counted instead of believed."""
    rows = _rows(store, "SELECT scope_json FROM memories WHERE state <> 'archived' LIMIT ?",
                 (max(1, int(scan)),))
    found = Split(memories=len(rows))
    for row in rows:
        scope = scope_of(row)
        name = normalise(scope.get(DEVICE_KEY) or "")
        if name:
            found.by_device[name] = found.by_device.get(name, 0) + 1
        else:
            found.shared += 1
        if scope and not isinstance(scope, dict):
            found.unknown_scopes += 1
    if not found.by_device:
        found.note = ("no memory carries a device scope, so every memory is visible to every device; "
                      "that is the current behaviour, not a namespace")
    return found


def render(store: Any) -> list[str]:
    described = split(store)
    lines = [f"device namespaces: {described.memories} memor(ies), "
             f"{described.shared} shared, {len(described.by_device)} device(s)"]
    if described.by_device:
        lines.append("  by device: " + ", ".join(f"{name} {count}"
                                                 for name, count in sorted(described.by_device.items())))
    if described.note:
        lines.append(f"  {described.note}")
    return lines
