"""Failure-mode memory (#133): "this fails on this host", with the workaround.

`handoff_notes` (#143) keeps typed advice per tool; `argument_memory` (#134) keeps which argv failed. Neither answers
the question that costs the most time: **this exact thing fails on this exact host, and here is what to do instead.**

A failure mode is a triple: the context (what was being attempted, on which host), the signal (what it looked like when
it failed), and the workaround. Six rules:

1. **A failure is bound to its context.** "It failed" is a complaint; "`systemd-inhibit --mode=block` fails over SSH on
   this host" is knowledge. A mode with no context is refused at write time.
2. **The signal is kept so it can be matched again** - that is the whole value - but it is redacted on the way in, so
   the catalogue cannot become a secret store.
3. **A workaround is what makes it advice.** A recorded failure with no workaround is `unexplained`: worth knowing,
   never offered as guidance.
4. **Recurrence is counted.** One occurrence is a hypothesis; the count says whether to trust it.
5. **A later success in the same context resolves the mode** - kept, not deleted, so the history stays readable while
   the advice stops being offered.
6. **Resolved modes are never offered**, and the newest unresolved mode wins when several exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .security import normalize_text
from .store import utc_now

TABLE = "failure_modes"

# Anything that looks like a credential assignment or a long random run is removed before it is stored. The patterns
# are composed from parts so this file itself never contains a secret-shaped literal.
_ASSIGNMENT = re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|bearer)\b\s*[:=]\s*\S+")
_LONG_RUN = re.compile(r"\b[A-Za-z0-9_\-]{20,}\b")
# `Authorization: Bearer <token>` puts the credential *after* the word, not after a colon.
_BEARER = re.compile(r"(?i)\bbearer\s+\S+")


def redact(text: str) -> str:
    """Remove credential-shaped fragments from a signal before it is stored."""
    cleaned = _BEARER.sub("Bearer [redacted]", str(text or ""))
    cleaned = _ASSIGNMENT.sub(r"\1=[redacted]", cleaned)
    return _LONG_RUN.sub("[redacted]", cleaned)


@dataclass(frozen=True)
class Mode:
    context: str
    signal: str
    workaround: str = ""
    host: str = ""
    occurrences: int = 1
    last_at: str = ""
    resolved_at: str = ""

    @property
    def explained(self) -> bool:
        return bool(self.workaround.strip())

    @property
    def resolved(self) -> bool:
        return bool(self.resolved_at)

    def render(self) -> str:
        state = "resolved" if self.resolved else ("known" if self.explained else "unexplained")
        workaround = f" -> {self.workaround}" if self.explained else ""
        return f"[{state}] {self.context}: {self.signal}{workaround} ({self.occurrences}x)"

    def as_dict(self) -> dict[str, Any]:
        return {"context": self.context, "signal": self.signal, "workaround": self.workaround,
                "host": self.host, "occurrences": self.occurrences, "last_at": self.last_at,
                "resolved_at": self.resolved_at, "explained": self.explained, "resolved": self.resolved}


@dataclass
class Catalogue:
    modes: list[Mode] = field(default_factory=list)
    readable: bool = True
    reason: str = ""

    def render(self) -> list[str]:
        if not self.readable:
            return [f"the failure catalogue could not be read: {self.reason}"]
        explained = [mode for mode in self.modes if mode.explained and not mode.resolved]
        lines = [f"{len(self.modes)} mode(s): {len(explained)} with a workaround, "
                 f"{len([m for m in self.modes if not m.explained])} unexplained, "
                 f"{len([m for m in self.modes if m.resolved])} resolved"]
        for mode in self.modes[:10]:
            lines.append(f"  - {mode.render()}")
        if len(self.modes) > 10:
            lines.append(f"  … and {len(self.modes) - 10} more")
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {"readable": self.readable, "reason": self.reason,
                "modes": [mode.as_dict() for mode in self.modes]}


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    context TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    host TEXT NOT NULL DEFAULT '',
                    workaround TEXT NOT NULL DEFAULT '',
                    occurrences INTEGER NOT NULL DEFAULT 1,
                    first_at TEXT NOT NULL,
                    last_at TEXT NOT NULL,
                    resolved_at TEXT,
                    PRIMARY KEY (context, signal)
                )"""
        )


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        cursor = store._conn.execute(sql, params)
    except Exception:
        return None
    try:
        return [dict(row) for row in cursor.fetchall()]
    except Exception:
        return None


def record(store: Any, *, context: str, signal: str, workaround: str = "",
           host: str = "") -> dict[str, Any]:
    """Record that something failed this way, in this context. Repeats increment the count."""
    ctx = normalize_text(str(context or "")).strip()
    # Redact the raw text first: normalising masks a credential into a short shape that no longer looks like one.
    sig = normalize_text(redact(str(signal or ""))).strip()
    if not ctx:
        return {"ok": False, "reason": "a failure mode needs a context: what was being attempted"}
    if not sig:
        return {"ok": False, "reason": "a failure mode needs a signal: what it looked like"}
    moment = utc_now()
    _ensure(store)
    with store.transaction() as conn:
        existing = conn.execute(
            f"SELECT occurrences, workaround FROM {TABLE} WHERE context=? AND signal=?", (ctx, sig)
        ).fetchone()
        if existing is None:
            conn.execute(
                f"INSERT INTO {TABLE}(context,signal,host,workaround,occurrences,first_at,last_at) "
                "VALUES(?,?,?,?,1,?,?)",
                (ctx, sig, str(host or ""), redact(normalize_text(str(workaround or ""))), moment, moment),
            )
        else:
            # A new workaround replaces an empty one; an existing workaround is not overwritten by silence.
            new_workaround = redact(normalize_text(str(workaround or ""))) or str(existing["workaround"] or "")
            conn.execute(
                f"UPDATE {TABLE} SET occurrences=occurrences+1,last_at=?,workaround=?,"
                "resolved_at=NULL WHERE context=? AND signal=?",
                (moment, new_workaround, ctx, sig),
            )
    return {"ok": True, "context": ctx, "signal": sig}


def resolve(store: Any, *, context: str, signal: str) -> dict[str, Any]:
    """Mark a mode resolved: a later run in this context succeeded. The record is kept."""
    exists = _rows(store, "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE,))
    if exists is None:
        return {"ok": False, "reason": "the store refused the schema query"}
    if not exists:
        return {"ok": False, "reason": "no such failure mode"}
    rows = _rows(store, f"SELECT 1 FROM {TABLE} WHERE context=? AND signal=?",
                 (str(context or "").strip(), str(signal or "").strip()))
    if rows is None:
        return {"ok": False, "reason": "the store refused the query"}
    if not rows:
        return {"ok": False, "reason": "no such failure mode"}
    with store.transaction() as conn:
        conn.execute(f"UPDATE {TABLE} SET resolved_at=? WHERE context=? AND signal=?",
                     (utc_now(), str(context).strip(), str(signal).strip()))
    return {"ok": True, "context": context, "signal": signal, "resolved": True}


def lookup(store: Any, *, context: str = "", include_resolved: bool = False,
           limit: int = 20) -> Catalogue:
    """Known failure modes, newest first, resolved ones excluded unless asked for."""
    clause = "" if include_resolved else " WHERE resolved_at IS NULL"
    params: tuple = ()
    if context:
        clause = (clause + " AND" if clause else " WHERE") + " context = ?"
        params = (str(context).strip(),)
    exists = _rows(store, "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE,))
    if exists is None:
        return Catalogue(readable=False, reason="the store refused the schema query")
    if not exists:
        return Catalogue(modes=[])  # never recorded anything: empty, not broken
    rows = _rows(store, f"SELECT * FROM {TABLE}{clause} ORDER BY last_at DESC LIMIT ?",
                 params + (max(1, int(limit)),))
    if rows is None:
        return Catalogue(readable=False, reason="the store refused the query")
    modes = [
        Mode(context=str(row["context"]), signal=str(row["signal"]),
             workaround=str(row.get("workaround") or ""), host=str(row.get("host") or ""),
             occurrences=int(row.get("occurrences") or 1), last_at=str(row.get("last_at") or ""),
             resolved_at=str(row.get("resolved_at") or ""))
        for row in rows
    ]
    return Catalogue(modes=modes)


def workaround_for(store: Any, context: str) -> str:
    """The newest unresolved workaround for this context, or '' when there is none to offer."""
    for mode in lookup(store, context=context).modes:
        if mode.explained:
            return mode.workaround
    return ""
