"""The thirty seconds after an action, when it can still be taken back (#186).

`node_actions` applies a change and reports what it did. This module is what makes that change **reversible
by the person who made it**, without them having to know which option undoes which: an action opens a
window, the window carries exactly the state needed to put the row back, and `undo()` uses it once.

Five rules, each a way an undo button goes wrong:

- **the window is stated, and it is short.** Every token prints when it expires (`undo until 12:04:31`).
  An undo button with no stated deadline is either a lie or an invitation to treat a destructive action as
  free — so the default is thirty seconds and a caller can lower it, never silently extend it.
- **undo restores the exact previous value of the columns that changed**, from the token, not from a
  guess about what the value probably was. A restore that sets `state='active'` on a memory that was
  `cold` before has not undone anything; it has imposed a third state.
- **a token is single-use and bound to its row.** Using it twice reports `already`, and a token for one
  memory cannot be replayed against another: the target id is compared, not trusted.
- **an expired token refuses and says what to do instead.** "Too late" alone teaches nothing; the refusal
  names the durable path (a correction, or the reverse action) so the caller is not stuck.
- **an undo that can no longer restore exactly refuses rather than approximating.** If the row changed
  again since the action, the token is stale: undoing it would overwrite somebody else's change. That is
  a refusal, and it names the conflict.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

TOKEN_TABLE = "undo_tokens"

# Thirty seconds: long enough for "wait, no", short enough that it cannot be mistaken for a general
# recovery tool. The number is printed with every token rather than assumed by the caller.
DEFAULT_WINDOW_SECONDS = 30

# The columns an undo can put back. Deliberately small: a token that captured the whole row would restore
# a memory that has since been corrected, which is a worse outcome than refusing.
RESTORABLE = ("state", "pinned", "weight", "quarantine_reason")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def _table_for(target: str) -> str:
    return "edges" if str(target) == "edge" else "memories"


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TOKEN_TABLE} (
                    token_id TEXT PRIMARY KEY,
                    target TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    restore_json TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    used_at TEXT,
                    at_open_json TEXT NOT NULL
                )"""
        )


@dataclass
class Token:
    token_id: str = ""
    target: str = "memory"
    target_id: str = ""
    action: str = ""
    restore: dict[str, Any] = field(default_factory=dict)
    opened_at: str = ""
    expires_at: str = ""
    used_at: str = ""
    at_open: dict[str, Any] = field(default_factory=dict)

    def expires(self) -> datetime | None:
        try:
            return datetime.fromisoformat(self.expires_at)
        except Exception:  # noqa: BLE001
            return None

    def seconds_left(self, *, at: datetime | None = None) -> float:
        moment = at or _now()
        expiry = self.expires()
        return 0.0 if expiry is None else max(0.0, (expiry - moment).total_seconds())

    def statement(self, *, at: datetime | None = None) -> str:
        if self.used_at:
            return f"undo {self.token_id}: already used at {self.used_at[11:19]}"
        left = self.seconds_left(at=at)
        if left <= 0:
            return f"undo {self.token_id}: window closed at {self.expires_at[11:19]}"
        return f"undo until {self.expires_at[11:19]} ({int(left)}s left)"

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "target": self.target,
            "target_id": self.target_id,
            "action": self.action,
            "restore": self.restore,
            "opened_at": self.opened_at,
            "expires_at": self.expires_at,
            "used_at": self.used_at,
            "statement": self.statement(),
        }


def open_window(
    store: Any,
    *,
    target: str,
    target_id: str,
    action: str,
    restore: dict[str, Any],
    seconds: int = DEFAULT_WINDOW_SECONDS,
    at: datetime | None = None,
) -> Token:
    """Record how to take one action back, with an expiry. Writes one row, atomically."""
    moment = at or _now()
    window = max(0, int(seconds))
    token = Token(
        target=str(target), target_id=str(target_id), action=str(action),
        restore={key: value for key, value in (restore or {}).items() if key in RESTORABLE},
        opened_at=moment.isoformat(),
        expires_at=(moment + timedelta(seconds=window)).isoformat(),
    )
    if not token.restore:
        return token  # nothing restorable: no token, and the caller renders the statement as empty
    token.token_id = hashlib.sha256(
        json.dumps([token.target, token.target_id, token.action, token.opened_at, token.restore],
                   sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    token.at_open = _current(store, token.target, token.target_id)
    try:
        _ensure(store)
    except Exception:  # noqa: BLE001
        return Token()  # no table, no token: a window the caller cannot use is worse than none
    with store.transaction() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {TOKEN_TABLE} (token_id, target, target_id, action, restore_json, "
            "opened_at, expires_at, used_at, at_open_json) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)",
            (token.token_id, token.target, token.target_id, token.action,
             json.dumps(token.restore, sort_keys=True, default=str),
             token.opened_at, token.expires_at, json.dumps(token.at_open, sort_keys=True, default=str)),
        )
    return token


def _current(store: Any, target: str, target_id: str) -> dict[str, Any]:
    table = _table_for(target)
    if table == "edges":
        rows = _rows(store, "SELECT weight FROM edges WHERE src_id = ? OR dst_id = ? LIMIT 1", (target_id, target_id))
    else:
        rows = _rows(store, "SELECT state, pinned, quarantine_reason FROM memories WHERE id = ?", (target_id,))
    return rows[0] if rows else {}


@dataclass
class Result:
    ok: bool = False
    status: str = ""  # undone | already | expired | unknown | stale | refused
    reason: str = ""
    token_id: str = ""
    restored: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "status": self.status, "reason": self.reason,
                "token_id": self.token_id, "restored": self.restored}

    def render(self) -> list[str]:
        head = "undone" if self.ok else self.status
        return [f"{head}: {self.reason}"]


def undo(store: Any, token_id: str, *, at: datetime | None = None) -> Result:
    """Take one action back, once, inside its window. Never raises."""
    result = Result(token_id=str(token_id))
    try:
        _ensure(store)
    except Exception:  # noqa: BLE001 - an unusable store is a refusal, not a traceback in a view
        result.status, result.reason = "refused", "the token table could not be read or created"
        return result
    rows = _rows(store, f"SELECT * FROM {TOKEN_TABLE} WHERE token_id = ?", (str(token_id),))
    if rows is None:
        result.status, result.reason = "refused", "the token table could not be read"
        return result
    if not rows:
        result.status, result.reason = "unknown", f"no undo token {token_id}"
        return result
    row = rows[0]
    moment = at or _now()
    if row.get("used_at"):
        result.status, result.reason = "already", f"this token was used at {str(row['used_at'])[11:19]}"
        return result
    try:
        expires = datetime.fromisoformat(str(row.get("expires_at")))
    except Exception:  # noqa: BLE001
        result.status, result.reason = "refused", "the token has no readable expiry"
        return result
    if moment > expires:
        result.status = "expired"
        result.reason = (f"the window closed at {expires.isoformat()[11:19]}; undo is no longer exact - "
                         "use the reverse action, or a correction if the value itself was wrong")
        return result

    restore = json.loads(str(row.get("restore_json") or "{}"))
    at_open = json.loads(str(row.get("at_open_json") or "{}"))
    target, target_id = str(row.get("target")), str(row.get("target_id"))
    now = _current(store, target, target_id)
    if not now:
        result.status, result.reason = "stale", "the row the token describes is gone"
        return result
    for column, opened_value in at_open.items():
        if column in restore:
            continue
        if str(now.get(column)) != str(opened_value):
            result.status = "stale"
            result.reason = (f"{column} changed since this action ({opened_value!r} -> {now.get(column)!r}); "
                             "undoing it would overwrite that change")
            return result

    table = _table_for(target)
    assignments = ", ".join(f"{column} = ?" for column in restore)
    params = list(restore.values())
    where = ("src_id = ? OR dst_id = ?" if table == "edges" else "id = ?")
    params.extend([target_id, target_id] if table == "edges" else [target_id])
    try:
        with store.transaction() as conn:
            conn.execute(f"UPDATE {table} SET {assignments} WHERE {where}", tuple(params))
            conn.execute(f"UPDATE {TOKEN_TABLE} SET used_at = ? WHERE token_id = ?", (moment.isoformat(), token_id))
    except Exception as error:  # noqa: BLE001
        result.status, result.reason = "refused", f"the store refused the undo: {type(error).__name__}"
        return result
    result.ok = True
    result.status = "undone"
    result.restored = restore
    result.reason = "restored " + ", ".join(f"{column} = {value!r}" for column, value in restore.items())
    return result


def pending(store: Any, *, at: datetime | None = None) -> list[Token]:
    """The tokens whose window is still open - what an "undo" affordance can honestly offer."""
    try:
        _ensure(store)
    except Exception:  # noqa: BLE001 - no readable token table means no undo to offer
        return []
    rows = _rows(store, f"SELECT * FROM {TOKEN_TABLE} WHERE used_at IS NULL ORDER BY opened_at DESC")
    out: list[Token] = []
    for row in rows or []:
        token = Token(
            token_id=str(row.get("token_id")), target=str(row.get("target")),
            target_id=str(row.get("target_id")), action=str(row.get("action")),
            restore=json.loads(str(row.get("restore_json") or "{}")),
            opened_at=str(row.get("opened_at")), expires_at=str(row.get("expires_at")),
            used_at=str(row.get("used_at") or ""),
        )
        if token.seconds_left(at=at) > 0:
            out.append(token)
    return out


def render(tokens: list[Token], *, at: datetime | None = None) -> list[str]:
    return [f"{token.action} on {token.target} {token.target_id[:8]}: {token.statement(at=at)}"
            for token in tokens]
