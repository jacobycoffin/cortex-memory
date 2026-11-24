"""The surface a watch or a voice assistant can actually show (#149)

A watch is 40 mm of glass and a voice assistant has one sentence. Both are fed by a system that produces
thousands of memories, four dashboards and a graph — so the surface's whole job is **subtraction**, and its
failure mode is not "too little", it is "a wall of text on a wrist".

Five rules:

- **one thing at a time.** A watch gets a verdict, not a dashboard. If three things need him, it shows the
  one that matters and a count of the rest.
- **the character budget is a hard cap, not a target.** #147's voice summary measured a median of 277 and a
  maximum of 280 against a 280 cap; the same discipline applies here at a smaller size.
- **no ids, no paths, no markup.** A watch renders plain text at a glance; `7b61d06a-fef8` and
  `~/src/.worktrees` are noise that costs a whole line.
- **nothing to say is a valid answer.** "Nothing needs you" is a real sentence; a blank screen is
  indistinguishable from a broken connection.
- **never lie about freshness.** If the store has not been written to in a week, the glance says so rather
  than presenting week-old state as current.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

GLANCE_CHARS = 90
ANSWER_CHARS = 140
MAX_LINES = 3
_ID = re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){0,4}\b", re.I)
_PATH = re.compile(r"(?:~|/root|/home|/Users)/[^\s,;)]*")
_MARKUP = re.compile(r"[*_`#>]+")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _parse(stamp: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def plain(text: Any) -> str:
    """Strip the three things a watch cannot use: ids, paths and markup."""
    body = str(text or "")
    body = _ID.sub("", body)
    body = _PATH.sub("", body)
    body = _MARKUP.sub("", body)
    return " ".join(body.split())


def fit(text: str, limit: int) -> str:
    """Trim to the cap on a word boundary. Never mid-word, never with a trailing space."""
    body = plain(text)
    if len(body) <= limit:
        return body
    cut = body[: max(1, limit - 1)]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:") + "…"


@dataclass
class Glance:
    line: str = ""
    detail: str = ""
    count: int = 0
    fresh: bool = True
    age_hours: float | None = None
    lines: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"line": self.line, "detail": self.detail, "count": self.count,
                "fresh": self.fresh, "age_hours": self.age_hours, "lines": list(self.lines)}


def glance(store: Any, *, now: datetime | None = None) -> Glance:
    """The one thing worth showing, plus how stale the picture is."""
    moment = now or _now()
    found = Glance()
    try:
        from .staleness_prompts import due  # noqa: PLC0415 - optional helper

        questions = due(store, limit=1, now=moment)
    except Exception:  # noqa: BLE001
        questions = None
    pending = len(questions.items) if questions is not None else 0
    corrections = _rows(store, "SELECT COUNT(*) AS count FROM memories "
                               "WHERE source_category = 'USER_CORRECTION' AND valid_to IS NULL")
    standing = int(corrections[0]["count"] or 0) if corrections else 0
    last = _rows(store, "SELECT MAX(created_at) AS last FROM memories WHERE state <> 'archived'")
    written = _parse(last[0]["last"]) if last else None
    if written is not None:
        found.age_hours = round((moment - written).total_seconds() / 3600.0, 1)
        found.fresh = found.age_hours <= 72
    if pending:
        found.line = "Something to confirm"
        found.detail = questions.items[0]["prompt"] if questions and questions.items else ""
        found.count = pending
    elif standing:
        found.line = f"{standing} correction(s) standing"
        found.detail = "nothing needs you"
    elif not found.fresh and found.age_hours is not None:
        found.line = "Nothing new to say"
        found.detail = f"last write {found.age_hours:.0f}h ago"
    else:
        found.line = "Nothing needs you"
        found.detail = "all quiet"
    found.lines = [fit(found.line, GLANCE_CHARS), fit(found.detail, GLANCE_CHARS)][:MAX_LINES]
    return found


def answer(store: Any, text: str, *, now: datetime | None = None) -> str:
    """One sentence, no ids, no paths — what a voice assistant would read aloud."""
    body = fit(text, ANSWER_CHARS)
    return body or "Nothing to say about that"


def search_line(store: Any, query: str, *, limit: int = 1) -> str:
    """The best single answer to a question, or an honest nothing.

    "Nothing to say" and "memory is unreachable" are different sentences: a watch that reports a broken
    retriever as silence is lying by omission, which is the same `unknown != 0` rule as everywhere else.
    """
    try:
        from .retrieval import MemoryRetriever  # noqa: PLC0415 - optional dependency

        results = MemoryRetriever(store).search(query, limit=limit) or []
    except Exception:  # noqa: BLE001
        return "Can't reach memory right now"
    if not results:
        return "Nothing to say about that"
    first = results[0]
    content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else "")
    return fit(str(content or ""), ANSWER_CHARS) or "Nothing to say about that"


def render(store: Any, *, now: datetime | None = None) -> list[str]:
    described = glance(store, now=now)
    return described.lines


def budget(store: Any, *, now: datetime | None = None) -> dict[str, Any]:
    """Measure the surface against its cap, the way the voice summary is measured."""
    described = glance(store, now=now)
    lengths = [len(line) for line in described.lines]
    return {
        "lines": len(described.lines),
        "max_line": max(lengths) if lengths else 0,
        "cap": GLANCE_CHARS,
        "over": max(lengths) > GLANCE_CHARS if lengths else False,
        "fresh": described.fresh,
        "age_hours": described.age_hours,
    }
