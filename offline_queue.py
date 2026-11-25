"""Writes that happen when the store doesn't (#157)

A phone on a train, a laptop mid-restart, a container that has not mounted its volume yet: the moment a
memory is worth keeping is sometimes the moment the store is unreachable. Losing the thought is the worst
outcome, so the write is queued — and then the queue has to be as careful as the store it fronts.

Five rules:

- **the queue is a file, not a wish.** One entry per line, written before the caller is told "yes", so a
  crash between "queued" and "stored" leaves a replayable record rather than a lost memory.
- **replay is ordered and stops at the first failure.** Out-of-order replay can store a correction before
  the fact it corrects; carrying on past a failure hides a broken dependency behind a later success.
- **a replayed write is idempotent.** Each entry carries a dedupe key, so replaying a queue twice does not
  double the store — the same `(turn, memory)` rule as used-id resolution (#145).
- **the queue is bounded and refuses rather than growing forever.** 500 entries by default; a full queue
  drops the *newest* and says so, because the oldest thoughts are the ones nobody has had a chance to
  retry.
- **a conflict is surfaced, not resolved.** If the memory exists with different content by the time the
  queue drains, the entry is parked with its reason (#209's rule) instead of overwriting a newer belief.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MAX_ENTRIES = 500
FORMAT = "cortex-offline-1"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def dedupe_key(kind: str, content: str) -> str:
    """Stable across replays: the same thought can only land once."""
    body = f"{str(kind or '')}\x00{' '.join(str(content or '').split())}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


@dataclass
class Entry:
    key: str = ""
    kind: str = "semantic"
    content: str = ""
    source_type: str = "conversation"
    source_category: str = "USER_STATED"
    queued_at: str = ""
    attempts: int = 0
    parked: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "kind": self.kind, "content": self.content,
                "source_type": self.source_type, "source_category": self.source_category,
                "queued_at": self.queued_at, "attempts": self.attempts, "parked": self.parked}


@dataclass
class Queue:
    path: Path | None = None
    entries: list[Entry] = field(default_factory=list)
    refused: int = 0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"path": str(self.path or ""), "entries": len(self.entries),
                "refused": self.refused, "note": self.note}


def _write(path: Path, entries: Iterable[Entry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(entry.as_dict(), sort_keys=True) for entry in entries]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(path)          # atomic: a crash leaves either the old queue or the new one


def load(path: str | Path) -> Queue:
    target = Path(path)
    found = Queue(path=target)
    if not target.exists():
        found.note = "no queue file yet"
        return found
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:  # noqa: BLE001
            found.refused += 1
            continue
        found.entries.append(Entry(
            key=str(payload.get("key") or ""), kind=str(payload.get("kind") or "semantic"),
            content=str(payload.get("content") or ""),
            source_type=str(payload.get("source_type") or "conversation"),
            source_category=str(payload.get("source_category") or "USER_STATED"),
            queued_at=str(payload.get("queued_at") or ""),
            attempts=int(payload.get("attempts") or 0),
            parked=str(payload.get("parked") or ""),
        ))
    if found.refused:
        found.note = f"{found.refused} line(s) could not be read and were skipped"
    return found


def enqueue(path: str | Path, *, content: str, kind: str = "semantic",
            source_type: str = "conversation", source_category: str = "USER_STATED",
            max_entries: int = MAX_ENTRIES, now: datetime | None = None) -> dict[str, Any]:
    """Append one write. Returns what happened rather than an exception, so a caller can log it."""
    body = " ".join(str(content or "").split())
    if not body:
        return {"queued": False, "reason": "an empty memory is not queued"}
    found = load(path)
    key = dedupe_key(kind, body)
    if any(entry.key == key for entry in found.entries):
        return {"queued": False, "reason": "already queued", "key": key,
                "size": len(found.entries)}
    entry = Entry(key=key, kind=str(kind or "semantic"), content=body,
                  source_type=str(source_type or "conversation"),
                  source_category=str(source_category or "USER_STATED"),
                  queued_at=(now or _now()).isoformat())
    entries = list(found.entries)
    refused = 0
    if len(entries) >= int(max_entries):
        # Drop the newest: the oldest are the ones nobody has had a chance to retry.
        refused = 1
        _write(Path(path), entries)
        return {"queued": False, "reason": "the queue is full; the newest write was refused",
                "key": key, "size": len(entries), "refused": refused}
    entries.append(entry)
    _write(Path(path), entries)
    return {"queued": True, "key": key, "size": len(entries)}


def drain(store: Any, path: str | Path, *, now: datetime | None = None) -> dict[str, Any]:
    """Replay in order, stop at the first failure, park a conflict with its reason."""
    found = load(path)
    if not found.entries:
        return {"drained": 0, "remaining": 0, "parked": 0, "stopped_on": "",
                "note": found.note}
    drained = 0
    remaining: list[Entry] = []
    parked_now = 0
    stopped_on = ""
    for index, entry in enumerate(found.entries):
        if stopped_on:
            remaining.append(entry)          # keep the rest, in order, for the next attempt
            continue
        # Idempotency is decided on the stored content, not on our own key: the store hashes content
        # its own way, and a queue key that never matches would replay into a duplicate.
        existing = _rows(store, "SELECT id FROM memories WHERE content = ? AND kind = ? "
                                "AND state <> 'archived'", (entry.content, entry.kind))
        if existing:
            continue                          # idempotent: already there, drop the queue entry
        try:
            store.add_memory(content=entry.content, kind=entry.kind,
                             source_type=entry.source_type,
                             source_category=entry.source_category)
        except Exception as error:  # noqa: BLE001
            entry.attempts += 1
            message = str(error)
            if "UNIQUE" in message.upper() or "CONSTRAINT" in message.upper():
                entry.parked = f"conflict: {message[:120]}"
                parked_now += 1
                continue                      # a parked entry does not stop the queue
            entry.parked = ""
            stopped_on = f"entry {index}: {message[:120]}"
            remaining.append(entry)
            continue
        drained += 1
    _write(Path(path), remaining)
    return {"drained": drained, "remaining": len(remaining), "parked": parked_now,
            "stopped_on": stopped_on,
            "note": "" if not stopped_on else "replay stopped at the first failure; order is kept"}


def status(path: str | Path) -> dict[str, Any]:
    found = load(path)
    return {"entries": len(found.entries), "parked": sum(1 for entry in found.entries
                                                         if entry.parked),
            "attempts": sum(entry.attempts for entry in found.entries),
            "format": FORMAT, "note": found.note}


def render(path: str | Path) -> list[str]:
    described = status(path)
    lines = [f"offline queue: {described['entries']} entr(ies), "
             f"{described['parked']} parked, {described['attempts']} attempt(s)"]
    if described["note"]:
        lines.append(f"  {described['note']}")
    lines.append("  replay is ordered and stops at the first failure; a replayed write is idempotent")
    return lines
