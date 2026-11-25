"""The right arguments for a recurring command (#134).

Some invocations are learned the hard way and then forgotten: `pytest tests/ -q -n 4` is the sweet spot while `-n 8`
buys nothing; `systemd-inhibit --mode=block` works locally and is refused over SSH; a CLI wants `--json` before the
positional. Each is a small fact about a command's *shape*, and each gets rediscovered the next time because nothing
wrote it down.

Six rules, each a way this table becomes a liability:

- **failures are worth keeping.** An argv that fails is as valuable as one that works — more, if it fails slowly.
- **the invocation is stored as data** (a command plus an argv list), never as prose, so it can be replayed verbatim
  instead of re-parsed by a human or a model.
- **a later success does not erase an earlier failure.** Same argv, both outcomes, means the result depends on
  context; that is reported as `mixed` with counts rather than resolved by whoever wrote last.
- **recall never hands back a known-bad argv as the recommendation.** If every recorded attempt failed, the answer is
  "this failed, here is why", not a suggestion.
- **no secret ever reaches the table.** An argument that looks like a key, token or bearer credential is refused at
  the door, and the refusal names the *pattern*, never the value — a flag database is exactly where credentials leak.
- **recording is idempotent**: the same command, argv and outcome increments a count instead of adding a row.

Read-only for recall; `remember()` writes only to this table.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

ARGUMENT_TABLE = "command_arguments"

SUCCESS = "success"
FAILURE = "failure"
OUTCOMES = (SUCCESS, FAILURE)

# Anything that looks like a credential is refused before it is stored. The patterns are deliberately broad: a false
# refusal costs one retry, a leaked key costs a rotation.
SECRET_PATTERNS = (
    ("bearer token", re.compile(r"bearer\s+\S+", re.I)),
    ("provider key", re.compile(r"\b(sk|pk|ghp|gho|glpat|xox[baprs])[-_][A-Za-z0-9_-]{12,}")),
    ("named secret flag", re.compile(r"--?(api[-_]?key|token|secret|password|passwd|auth)\b", re.I)),
    ("long opaque value", re.compile(r"^[A-Za-z0-9+/_\-]{40,}={0,2}$")),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {ARGUMENT_TABLE} (
                    command TEXT NOT NULL,
                    argv TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    times INTEGER NOT NULL DEFAULT 1,
                    note TEXT NOT NULL DEFAULT '',
                    first_at TEXT NOT NULL,
                    last_at TEXT NOT NULL,
                    PRIMARY KEY (command, argv, outcome)
                )"""
        )


def secret_in(argv: Iterable[str]) -> str:
    """The name of the pattern that matched, or an empty string. Never the value."""
    for argument in argv:
        text = str(argument)
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                return name
    return ""


@dataclass
class Attempt:
    argv: list[str]
    outcome: str
    times: int
    note: str = ""
    last_at: str = ""

    def render(self) -> str:
        body = " ".join(self.argv) or "(no arguments)"
        suffix = f" x{self.times}" if self.times > 1 else ""
        return f"{body} — {self.outcome}{suffix}" + (f" ({self.note})" if self.note else "")

    def as_dict(self) -> dict[str, Any]:
        return {"argv": self.argv, "outcome": self.outcome, "times": self.times,
                "note": self.note, "last_at": self.last_at}


@dataclass
class Recall:
    command: str
    known_good: list[Attempt] = field(default_factory=list)
    known_bad: list[Attempt] = field(default_factory=list)
    mixed: list[Attempt] = field(default_factory=list)
    readable: bool = True
    reason: str = ""

    @property
    def best(self) -> Attempt | None:
        return self.known_good[0] if self.known_good else None

    def statement(self) -> str:
        if not self.readable:
            return self.reason
        if self.best is not None:
            return f"for {self.command}: {' '.join(self.best.argv)}"
        if self.mixed:
            return (f"for {self.command}: {' '.join(self.mixed[0].argv)} — works in some contexts and fails in "
                    f"others ({self.mixed[0].times} record(s)); check the context before reusing it")
        if self.known_bad:
            return (f"for {self.command}: every recorded attempt failed, most recently "
                    f"{' '.join(self.known_bad[0].argv)} ({self.known_bad[0].note or 'no reason recorded'})")
        return f"nothing recorded for {self.command}"

    def render(self) -> list[str]:
        if not self.readable:
            return [self.reason]
        lines = [self.statement()]
        if self.known_good:
            lines.append("works:")
            lines.extend(f"  + {attempt.render()}" for attempt in self.known_good[:3])
        if self.known_bad:
            lines.append("fails:")
            lines.extend(f"  - {attempt.render()}" for attempt in self.known_bad[:3])
        return lines


def remember(store: Any, command: str, argv: Sequence[str], *, outcome: str = SUCCESS,
             note: str = "") -> dict[str, Any]:
    """Record one invocation. Refuses secrets and unknown outcomes."""
    name = str(command or "").strip()
    if not name:
        return {"ok": False, "reason": "a command name is required"}
    if outcome not in OUTCOMES:
        return {"ok": False, "reason": f"outcome must be one of {', '.join(OUTCOMES)}"}
    parts = [str(part) for part in argv]
    if " " in name:
        return {"ok": False, "reason": "the command is a program name; put flags in argv, not in the command"}
    pattern = secret_in(parts)
    if pattern:
        return {"ok": False, "reason": f"refused: an argument looks like a {pattern}; "
                                      f"the value is not stored and was not echoed"}
    _ensure(store)
    key = json.dumps(parts)
    stamp = _now()
    existing = _rows(store, f"SELECT times FROM {ARGUMENT_TABLE} WHERE command = ? AND argv = ? AND outcome = ?",
                     (name, key, outcome))
    if existing:
        with store.transaction() as conn:
            conn.execute(
                f"UPDATE {ARGUMENT_TABLE} SET times = times + 1, last_at = ?, note = COALESCE(NULLIF(?, ''), note) "
                "WHERE command = ? AND argv = ? AND outcome = ?",
                (stamp, str(note or ""), name, key, outcome),
            )
        return {"ok": True, "command": name, "argv": parts, "outcome": outcome,
                "times": int(existing[0]["times"]) + 1, "deduplicated": True}
    with store.transaction() as conn:
        conn.execute(
            f"INSERT INTO {ARGUMENT_TABLE} (command, argv, outcome, times, note, first_at, last_at) "
            "VALUES (?, ?, ?, 1, ?, ?, ?)",
            (name, key, outcome, str(note or ""), stamp, stamp),
        )
    return {"ok": True, "command": name, "argv": parts, "outcome": outcome, "times": 1, "deduplicated": False}


def _table_exists(store: Any) -> bool | None:
    """True/False when the store answered, None when it could not be asked at all."""
    rows = _rows(store, "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (ARGUMENT_TABLE,))
    if rows is None:
        return None
    return bool(rows)


def recall(store: Any, command: str) -> Recall:
    """What is known about this command's arguments. Never suggests an argv that always failed."""
    name = str(command or "").strip()
    # Reading never creates schema: a store that has never recorded an argument has no table, and that is "nothing
    # known", not "the store is broken" — the two must not look the same.
    exists = _table_exists(store)
    if exists is None:
        return Recall(command=name, readable=False, reason="the argument table could not be read")
    if not exists:
        return Recall(command=name)
    rows = _rows(store, f"SELECT argv, outcome, times, note, last_at FROM {ARGUMENT_TABLE} WHERE command = ? "
                        "ORDER BY last_at DESC", (name,))
    if rows is None:
        return Recall(command=name, readable=False, reason="the argument table could not be read")
    by_argv: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["argv"])
        entry = by_argv.setdefault(key, {"argv": json.loads(key), "success": 0, "failure": 0, "note": "",
                                         "last_at": ""})
        entry[str(row["outcome"])] = int(row["times"])
        if not entry["note"] and row["note"]:
            entry["note"] = str(row["note"])
        entry["last_at"] = max(entry["last_at"], str(row["last_at"]))

    out = Recall(command=name)
    for entry in by_argv.values():
        if entry["success"] and entry["failure"]:
            out.mixed.append(Attempt(argv=entry["argv"], outcome="mixed",
                                     times=entry["success"] + entry["failure"], note=entry["note"],
                                     last_at=entry["last_at"]))
        elif entry["success"]:
            out.known_good.append(Attempt(argv=entry["argv"], outcome=SUCCESS, times=entry["success"],
                                          note=entry["note"], last_at=entry["last_at"]))
        else:
            out.known_bad.append(Attempt(argv=entry["argv"], outcome=FAILURE, times=entry["failure"],
                                         note=entry["note"], last_at=entry["last_at"]))
    out.known_good.sort(key=lambda attempt: (-attempt.times, attempt.last_at), reverse=False)
    out.known_bad.sort(key=lambda attempt: -attempt.times)
    out.mixed.sort(key=lambda attempt: -attempt.times)
    return out


def known_commands(store: Any) -> list[str]:
    rows = _rows(store, f"SELECT command, SUM(times) AS uses FROM {ARGUMENT_TABLE} "
                        "GROUP BY command ORDER BY uses DESC, command")
    return [str(row["command"]) for row in rows or []]
