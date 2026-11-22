"""Write receipts in chat, and the inline edits they invite (#37).

Memories used to appear silently. The user found out what the assistant had stored only by asking,
and by then a wrong fact had shaped several answers. A receipt fixes that with one line:

    remembered 2 things — 1) "The dashboard runs on port 8100" (a1b2c3) 2) "Jacoby prefers short
    answers" (d4e5f6) · say fix a1b2 <words> or forget a1b2

The **short id is the point**. A receipt without a handle is only an announcement; with one, the
user can correct the store in the next message, which is the cheapest repair path there is.

Three rules keep receipts from becoming noise:

- **one line, capped** — at most three entries and 240 characters; a receipt long enough to scroll
  past is a receipt nobody reads;
- **silence when nothing was written** — a receipt that appears unprompted trains people to ignore
  receipts;
- **`forget` suppresses, it does not delete** — the natural reading of "forget that" is "stop using
  it", not "destroy the audit trail". A deletion stays a separate, confirmed operation.

`apply_edits()` interprets the next message, so the correction loop is closed rather than merely
advertised. It only ever acts on ids it printed, and it reports what it did.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

MAX_ENTRIES = 3
MAX_LINE_CHARS = 240
SHORT_ID_CHARS = 6

_EDIT = re.compile(
    r"\b(?:fix|correct|update|edit)\s+(?P<id>[0-9a-f]{4,32})\s*(?:[:=]|\s)\s*(?P<text>.+?)(?:\n|$)",
    re.IGNORECASE,
)
_FORGET = re.compile(r"\b(?:forget|drop|remove|ignore)\s+(?P<id>[0-9a-f]{4,32})\b", re.IGNORECASE)


def short_id(memory_id: str) -> str:
    return str(memory_id).replace("-", "")[:SHORT_ID_CHARS]


def resolve(store: Any, token: str) -> str | None:
    """Expand a short id to a full memory id, or None when it is ambiguous or unknown."""
    needle = str(token).replace("-", "").lower()
    if not needle:
        return None
    try:
        rows = store._conn.execute(
            "SELECT id FROM memories WHERE replace(id, '-', '') LIKE ? LIMIT 5",
            (needle + "%",),
        ).fetchall()
    except Exception:  # noqa: BLE001 - a receipt must never raise in a reply path
        return None
    ids = [str(row["id"]) for row in rows]
    return ids[0] if len(ids) == 1 else None


def _excerpt(content: str, limit: int = 60) -> str:
    text = " ".join(str(content or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


@dataclass
class Receipt:
    lines: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        if not self.lines:
            return ""
        return " · ".join(self.lines)

    def __bool__(self) -> bool:
        return bool(self.lines)


def build(
    store: Any,
    memory_ids: Iterable[str],
    *,
    verb: str = "remembered",
    max_entries: int = MAX_ENTRIES,
    max_chars: int = MAX_LINE_CHARS,
) -> Receipt:
    """One compact line naming what was written, with a handle for each entry."""
    entries: list[str] = []
    ids: list[str] = []
    for memory_id in list(memory_ids)[:max_entries]:
        row = store.get_memory(memory_id)
        if not row:
            continue
        entries.append(f'{_excerpt(row.get("content"))} ({short_id(memory_id)})')
        ids.append(str(memory_id))
    if not entries:
        return Receipt()

    count = len(ids)
    head = f"{verb} {count} thing{'s' if count != 1 else ''} — "
    hint = " · say fix <id> <words> or forget <id>"
    body = " ".join(f"{index}) {entry}" for index, entry in enumerate(entries, start=1))
    text = head + body
    if len(text) + len(hint) > max_chars:
        text = text[: max(20, max_chars - len(hint) - 1)].rstrip() + "…"
    return Receipt(lines=[text + hint], ids=ids)


@dataclass
class Edit:
    action: str
    memory_id: str
    text: str = ""
    applied: bool = False
    note: str = ""

    def describe(self) -> str:
        if not self.applied:
            return f"{self.action} {short_id(self.memory_id)}: {self.note}"
        return f"{self.action} {short_id(self.memory_id)}: {self.note}"


_CLAUSE = re.compile(r"\s+(?:and|but|then|,|;)\s+|\n+")


def parse(text: str) -> list[Edit]:
    """Read the corrections a user typed. Pure: no store access, no side effects.

    Parsed clause by clause, because "fix a1b2 the port is 8200 and forget d4e5f6" is two
    instructions — a fix whose text swallows the rest of the line would silently drop the second.
    """
    edits: list[Edit] = []
    for clause in _CLAUSE.split(str(text or "")):
        piece = clause.strip()
        if not piece:
            continue
        fixed = _EDIT.match(piece)
        if fixed:
            edits.append(Edit("fix", fixed.group("id"), fixed.group("text").strip()))
            continue
        forgotten = _FORGET.search(piece)
        if forgotten:
            edits.append(Edit("forget", forgotten.group("id")))
    return edits


def apply_edits(
    store: Any, text: str, *, session_id: str | None = None, scope: dict[str, Any] | None = None
) -> list[Edit]:
    """Apply `fix` and `forget` instructions found in a message. Never deletes content."""
    from . import suppression

    results: list[Edit] = []
    for edit in parse(text):
        memory_id = resolve(store, edit.memory_id)
        if not memory_id:
            edit.note = "no single memory matches that id"
            results.append(edit)
            continue
        edit.memory_id = memory_id
        if edit.action == "forget":
            edit.applied = bool(
                suppression.suppress(store, memory_id, reason="user asked in chat")
            )
            edit.note = "excluded from recall (content kept)" if edit.applied else "not suppressed"
            results.append(edit)
            continue
        try:
            store.add_memory(
                content=edit.text,
                kind="correction",
                source_type="user_turn",
                source_category="USER_CORRECTION",
                session_id=session_id,
                supersedes_id=memory_id,
                scope=scope or {},
            )
            edit.applied = True
            edit.note = "corrected; the wrong version no longer answers"
        except Exception as error:  # noqa: BLE001 - a failed edit is reported, not raised
            edit.note = f"could not correct: {type(error).__name__}"
        results.append(edit)
    return results


def summarise(edits: list[Edit]) -> str:
    """The reply line for what the edits did, or empty when there was nothing to do."""
    if not edits:
        return ""
    applied = [edit for edit in edits if edit.applied]
    failed = [edit for edit in edits if not edit.applied]
    parts: list[str] = []
    if applied:
        parts.append(f"{len(applied)} correction{'s' if len(applied) != 1 else ''} applied")
    if failed:
        parts.append(f"{len(failed)} not applied ({failed[0].note})")
    return "; ".join(parts)
