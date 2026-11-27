"""The two questions a person asks about a memory store

*"Is it working?"* and *"why did it recall that?"* are the questions the dashboard, the audit tables and
the trace dump do not answer. The dashboard answers *what is in here*; `traces` answers *what happened*;
neither answers a person who wants one plain sentence. So this module renders two small screens and
nothing else:

- **`cortex-memory status`** - four counts and a timestamp, in words.
- **`cortex-memory why <id>`** - one memory, what it is, when it was last recalled, whether it was ever
  used in an answer, and what the last recorded score breakdown says it ranked on.

Five rules held throughout:

- **a count with no data says `not measurable yet`, never `0`.** A missing table and an empty table are
  different facts: one means the instrument never ran, the other means it ran and found nothing. Printing
  zero for both is how an unfed mechanism reads as a healthy store (see #169's rule, applied to counts).
- **no id soup and no jargon dump.** A reader gets one short id, plain words for the kind, the source and
  every ranking signal, and the numbers that go with them. `semantic_fusion` is not a word anybody asked to read.
- **read-only by construction.** The database is opened `mode=ro`, so no command here can create a schema,
  run a migration, or write a row - and pointing `--db` at a wrong path cannot manufacture an empty store
  that reads as a working one, which is the failure that makes a real brain look blank.
- **a missing table is a missing measurement, not a crash.** Every query is guarded; an unreadable table
  costs that line, never the screen.
- **short.** Status is at most a dozen lines; `why` is a handful of sentences.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

WINDOW_DAYS = 7
NOT_MEASURABLE = "not measurable yet"
GRADED_ANSWERS = ("helpful", "harmful", "validated", "corrected")
RECALL_EVENTS = ("retrieved", "selected", "injected")

KIND_WORDS = {
    "semantic": "a fact",
    "preference": "a preference",
    "procedure": "a procedure",
    "episode": "something that happened",
    "identity": "who someone is",
    "operational": "how your system is set up",
    "decision": "a decision",
    "constraint": "a rule",
    "prospective": "something to do later",
    "schema": "a shared pattern",
    "tool_workflow": "a tool workflow",
    "tool_preference": "which tool works best",
}

SOURCE_WORDS = {
    "user_turn": "from a conversation",
    "assistant_turn": "from a conversation",
    "builtin_memory": "from your agent's memory file",
    "vault_markdown": "from your notes",
    "cli": "added at the command line",
    "explicit_tool": "added by a tool",
    "agent_tool_proposal": "proposed by an agent",
}

STATE_WORDS = {
    "active": "still active",
    "cold": "cooled but kept",
    "archived": "archived (kept, no longer recalled)",
    "quarantined": "quarantined",
    "tombstoned": "tombstoned",
}

COMPONENT_WORDS = {
    "semantic": "wording match",
    "semantic_similarity": "wording match",
    "lexical": "wording match",
    "keyword": "keyword match",
    "bm25": "keyword match",
    "semantic_fusion": "found by meaning",
    "recency": "recency",
    "importance": "importance",
    "trust": "trust",
    "confidence": "confidence",
    "currentness": "up-to-dateness",
    "usage": "past usefulness",
    "usage_weight": "past usefulness",
    "activation": "how often it comes up",
    "graph": "connected memories",
    "graph_expansion": "connected memories",
    "association": "connected memories",
    "entity_anchored": "named-entity match",
    "entity_boost": "named-entity match",
    "project_match": "same project",
    "scope_match": "same scope",
    "goal_match": "same goal",
    "system_match": "same system",
    "version_match": "same version",
    "precondition_match": "preconditions match",
}


def open_read_only(path: str | Path) -> sqlite3.Connection | None:
    """Open the store without the ability to write. ``None`` when there is no file yet."""

    target = Path(path).expanduser()
    if not target.is_file():
        return None
    uri = f"file:{target.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: Any, sql: str, params: tuple = ()) -> list[Any] | None:
    """Rows, or ``None`` when the query cannot run (no table, no such column, unreadable file)."""

    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return None


def _one(conn: Any, sql: str, params: tuple = ()) -> Any | None:
    rows = _rows(conn, sql, params)
    return rows[0] if rows else None


def _count(value: Any) -> str:
    return f"{int(value or 0):,}"


def _times(value: Any) -> str:
    total = int(value or 0)
    return "once" if total == 1 else f"{total:,} times"


def _parse(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _date(value: Any) -> str:
    moment = _parse(value)
    return moment.strftime("%Y-%m-%d") if moment else "unknown"


def _ago(value: Any) -> str:
    moment = _parse(value)
    if moment is None:
        return ""
    days = (datetime.now(timezone.utc) - moment).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:") + "…"


def _column(row: Any, name: str) -> Any:
    try:
        return row[name]
    except (IndexError, KeyError):
        return None


# --------------------------------------------------------------------------- status


def _memories(conn: Any) -> str:
    row = _one(
        conn,
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN state = 'active' THEN 1 ELSE 0 END) AS active FROM memories",
    )
    if row is None:
        return NOT_MEASURABLE
    total = int(row["total"] or 0)
    if total == 0:
        return "0 (the store is empty)"
    return f"{_count(total)} ({_count(row['active'])} active)"


def _recalled_last_week(conn: Any) -> str:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).isoformat()
    placeholders = ",".join("?" for _ in RECALL_EVENTS)
    row = _one(
        conn,
        f"SELECT COUNT(DISTINCT memory_id) AS memories FROM access_log "
        f"WHERE event IN ({placeholders}) AND datetime(created_at) >= datetime(?)",
        (*RECALL_EVENTS, cutoff),
    )
    if row is None or not row["memories"]:
        return NOT_MEASURABLE
    total = int(row["memories"] or 0)
    return f"{_count(total)} {'memory' if total == 1 else 'memories'}"


def _answers_graded(conn: Any) -> str:
    placeholders = ",".join("?" for _ in GRADED_ANSWERS)
    row = _one(
        conn,
        f"SELECT COUNT(DISTINCT task_id) AS graded FROM usage_records "
        f"WHERE outcome IN ({placeholders})",
        GRADED_ANSWERS,
    )
    if row is None or not row["graded"]:
        return NOT_MEASURABLE
    return _count(row["graded"])


def _procedures_graded(conn: Any) -> str:
    row = _one(
        conn,
        "SELECT COUNT(DISTINCT procedure_id) AS procedures FROM procedure_outcomes",
    )
    if row is None or not row["procedures"]:
        return NOT_MEASURABLE
    return _count(row["procedures"])


def _last_recall(conn: Any) -> str:
    row = _one(conn, "SELECT MAX(created_at) AS at FROM recall_runs")
    stamp = row["at"] if row is not None else None
    if not stamp:
        row = _one(conn, "SELECT MAX(created_at) AS at FROM access_log")
        stamp = row["at"] if row is not None else None
    if not stamp:
        return NOT_MEASURABLE
    when = _ago(stamp)
    return f"{when} ({_date(stamp)})" if when else _date(stamp)


def status_lines(conn: Any) -> list[str]:
    """The short answer to "is it working?", from stored rows only."""

    return [
        "Cortex, in plain numbers:",
        f"  memories: {_memories(conn)}",
        f"  recalled in the last {WINDOW_DAYS} days: {_recalled_last_week(conn)}",
        f"  answers graded: {_answers_graded(conn)}",
        f"  procedures graded: {_procedures_graded(conn)}",
        f"  last recall: {_last_recall(conn)}",
    ]


# --------------------------------------------------------------------------- why


def _resolve(conn: Any, query: str) -> tuple[Any | None, str]:
    value = str(query or "").strip()
    if not value:
        return None, "empty"
    rows = _rows(conn, "SELECT * FROM memories WHERE id = ?", (value,))
    if rows is None:
        return None, "unreadable"
    if rows:
        return rows[0], ""
    rows = _rows(
        conn,
        "SELECT * FROM memories WHERE id LIKE ? ORDER BY created_at LIMIT 2",
        (value + "%",),
    )
    if rows is None:
        return None, "unreadable"
    if len(rows) == 1:
        return rows[0], ""
    if len(rows) > 1:
        return None, "ambiguous"
    return None, "missing"


def _recall_history(conn: Any, memory_id: str) -> str:
    placeholders = ",".join("?" for _ in RECALL_EVENTS)
    row = _one(
        conn,
        f"SELECT COUNT(*) AS n, MAX(created_at) AS at FROM access_log "
        f"WHERE memory_id = ? AND event IN ({placeholders})",
        (memory_id, *RECALL_EVENTS),
    )
    if row is None:
        return NOT_MEASURABLE
    total = int(row["n"] or 0)
    if not total:
        return "never, since it was saved"
    return f"{_ago(row['at'])} ({_date(row['at'])}), {_times(total)} in total"


def _used_in_answer(conn: Any, row: Any) -> str:
    memory_id = str(row["id"])
    placeholders = ",".join("?" for _ in GRADED_ANSWERS)
    counted = _one(
        conn,
        f"SELECT COUNT(*) AS n, MAX(created_at) AS at FROM usage_records "
        f"WHERE memory_id = ? AND (used = 1 OR outcome IN ({placeholders}))",
        (memory_id, *GRADED_ANSWERS),
    )
    if counted is None:
        fallback = int(_column(row, "used_count") or 0)
        if not fallback:
            return NOT_MEASURABLE
        return f"yes, {_times(fallback)} (from the stored counter)"
    total = int(counted["n"] or 0)
    if not total:
        return "no, never credited in an answer"
    return f"yes, {_times(total)}, most recently on {_date(counted['at'])}"


def _component_words(key: str) -> str:
    return COMPONENT_WORDS.get(key) or key.replace("_", " ")


def _ranking(conn: Any, memory_id: str) -> list[str]:
    row = _one(
        conn,
        "SELECT rank, score, components_json, created_at FROM recall_score_breakdown "
        "WHERE memory_id = ? ORDER BY created_at DESC LIMIT 1",
        (memory_id,),
    )
    if row is None:
        return [
            "  Why it ranked: not recorded yet - Cortex keeps a score breakdown only for the top"
            " results of a recall."
        ]
    try:
        components = json.loads(row["components_json"] or "{}")
    except ValueError:
        components = {}
    if not isinstance(components, dict):
        components = {}
    parts = sorted(
        (
            (float(value), str(key))
            for key, value in components.items()
            if isinstance(value, (int, float)) and abs(float(value)) > 0.0005
        ),
        reverse=True,
    )[:4]
    score = float(row["score"] or 0.0)
    when = _ago(row["created_at"])
    headline = f"  Why it ranked: #{int(row['rank'] or 0)} (score {score:.2f}) in its most recent recall"
    headline += f", {when}" if when else ""
    if not parts:
        return [headline + "; the breakdown stored no signal values."]
    because = ", ".join(f"{_component_words(key)} {value:.2f}" for value, key in parts)
    return [headline + ", because:", f"    {because}"]


def why_lines(conn: Any, query: str) -> list[str]:
    """One memory explained in plain English, or one honest sentence about why not."""

    row, problem = _resolve(conn, query)
    if row is None:
        if problem == "ambiguous":
            return [f"{str(query).strip()!r} matches more than one memory. Give a longer prefix."]
        if problem == "unreadable":
            return ["This database has no readable memories table, so there is nothing to explain."]
        if problem == "empty":
            return ["Give a memory id, or the first few characters of one."]
        return [f"No memory matches {str(query).strip()!r}. Try `cortex-memory status` first."]

    memory_id = str(row["id"])
    kind_key = str(row["kind"] or "").replace("_", " ")
    kind = KIND_WORDS.get(str(row["kind"] or "")) or (
        f"a {kind_key} memory" if kind_key else "a memory"
    )
    source_key = str(row["source_type"] or "").replace("_", " ")
    source = SOURCE_WORDS.get(str(row["source_type"] or "")) or (
        f"from {source_key}" if source_key else "from an unrecorded source"
    )
    state = STATE_WORDS.get(str(row["state"] or "")) or (
        f"state {row['state']}" if row["state"] else "in an unrecorded state"
    )
    lines = [f"Memory {memory_id[:8]} - {kind}, {source}."]
    text = " ".join(str(row["content"] or "").split())
    if text:
        lines.append(f'  "{_trim(text, 160)}"')
    facts = [f"saved {_date(row['created_at'])}", state]
    if row["confidence"] is not None:
        facts.append(f"confidence {float(row['confidence']):.2f}")
    lines.append("  " + ", ".join(facts) + ".")
    lines.append(f"  Last recalled: {_recall_history(conn, memory_id)}.")
    lines.append(f"  Used in an answer: {_used_in_answer(conn, row)}.")
    lines.extend(_ranking(conn, memory_id))
    return lines


def render_status(conn: Any) -> str:
    return "\n".join(status_lines(conn))


def render_why(conn: Any, query: str) -> str:
    return "\n".join(why_lines(conn, query))
