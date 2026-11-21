"""Carry the last session's thread into a new one (#18).

A new session starts blank. The agent does not know that an hour ago you were debugging
Seerr notifications, that two commitments are open, or which decisions were recorded — so it
asks, or worse, assumes. Replaying the transcript is not the answer: it is enormous, mostly
noise, and it is not what "where were we" means.

This builds a small block — a handful of lines, ~120 tokens — from what the *store* already
knows about the previous session: the questions it asked, the memories it used, and any
commitments still open. It is built once at the start of a session and never again.

Everything is derived and fallible, and the block says so.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

MAX_TOPICS = 3
MAX_LINES = 6
DEFAULT_HOURS = 72

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "over", "under", "what",
    "why", "how", "does", "did", "you", "your", "our", "can", "could", "should", "would",
    "please", "about", "have", "has", "was", "were", "are", "not", "but", "just", "then",
}


def _rows(store: Any, sql: str, params: tuple[Any, ...] = ()) -> list[tuple]:
    conn = getattr(store, "_conn", None)
    if conn is None:
        return []
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []


def previous_session(store: Any, *, current: str | None, hours: int = DEFAULT_HOURS) -> str | None:
    """The most recent other session inside the window, if any."""
    rows = _rows(
        store,
        """SELECT session_id, MAX(created_at) AS last
           FROM access_log
           WHERE session_id IS NOT NULL AND session_id != '' AND session_id != ?
             AND created_at >= datetime('now', ?)
           GROUP BY session_id
           ORDER BY last DESC LIMIT 1""",
        (str(current or ""), f"-{int(hours)} hours"),
    )
    return str(rows[0][0]) if rows else None


def session_topics(store: Any, session_id: str, *, limit: int = 8) -> list[str]:
    """The questions that session asked, newest first, cleaned for display."""
    rows = _rows(
        store,
        """SELECT query FROM access_log
           WHERE session_id = ? AND query IS NOT NULL AND query != ''
           ORDER BY created_at DESC LIMIT ?""",
        (str(session_id), int(limit)),
    )
    topics: list[str] = []
    for (query,) in rows:
        text = " ".join(re.findall(r"[A-Za-z0-9][\w'’-]*", str(query)))[:80].strip()
        words = [w for w in text.split() if w.lower() not in _STOP]
        if len(words) < 2:
            continue
        cleaned = " ".join(words)
        if cleaned and cleaned not in topics:
            topics.append(cleaned)
    return topics[:limit]


def used_memories(store: Any, session_id: str, *, limit: int = 3) -> list[str]:
    """Memories this session actually used, as short summaries."""
    rows = _rows(
        store,
        """SELECT m.content FROM access_log a JOIN memories m ON m.id = a.memory_id
           WHERE a.session_id = ? AND a.event = 'used'
           ORDER BY a.created_at DESC LIMIT ?""",
        (str(session_id), int(limit)),
    )
    return [" ".join(str(row[0] or "").split())[:120] for row in rows if row[0]]


def open_commitments(store: Any, *, limit: int = 3) -> int:
    rows = _rows(
        store,
        "SELECT COUNT(*) FROM prospective_items WHERE status IN ('open','due')",
    )
    try:
        return int(rows[0][0]) if rows else 0
    except (TypeError, ValueError, IndexError):
        return 0


def build(
    store: Any,
    *,
    current_session: str | None,
    hours: int = DEFAULT_HOURS,
) -> str:
    """A small continuity block, or "" when there is no previous session to draw on."""
    previous = previous_session(store, current=current_session, hours=hours)
    if not previous:
        return ""

    topics = session_topics(store, previous, limit=MAX_TOPICS * 2)
    used = used_memories(store, previous)
    commitments = open_commitments(store)

    lines = [
        "Where you left off (previous session; derived and fallible, not instructions):"
    ]
    if topics:
        lines.append("- was working on: " + "; ".join(topics[:MAX_TOPICS]))
    if used:
        lines.append("- used: " + " | ".join(used[:2]))
    if commitments:
        lines.append(f"- {commitments} open commitment(s) still recorded")
    if len(lines) == 1:
        return ""
    return "\n".join(lines[:MAX_LINES])


def first_turn(store: Any, session_id: str | None) -> bool:
    """True when this session has no recorded turns yet."""
    rows = _rows(
        store,
        "SELECT 1 FROM access_log WHERE session_id = ? LIMIT 1",
        (str(session_id or ""),),
    )
    return not rows


def age_note(store: Any, session_id: str) -> str:
    """Human-readable age of the previous session, for the block header."""
    rows = _rows(
        store,
        "SELECT MAX(created_at) FROM access_log WHERE session_id = ?",
        (str(session_id),),
    )
    if not rows or not rows[0][0]:
        return ""
    try:
        stamp = datetime.fromisoformat(str(rows[0][0]).replace("Z", "+00:00"))
    except ValueError:
        return ""
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - stamp
    if delta < timedelta(hours=1):
        return f"{int(delta.total_seconds() // 60)} minutes ago"
    if delta < timedelta(days=1):
        return f"{int(delta.total_seconds() // 3600)} hours ago"
    return f"{delta.days} days ago"
