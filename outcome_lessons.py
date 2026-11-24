"""Turn failures into questions, not into rewritten memories (#208).

#71 records that a memory informed a task that then failed. That tally is where most systems stop,
or worse, where they silently down-rank the memory and move on. Neither is a lesson: a procedure that
failed twice either **is wrong** or was **used in the wrong situation**, and only a person (or a later
turn with the context) can say which.

So this module produces a lesson that is deliberately shaped as a question with its evidence attached:

- **evidence first** — the tasks, the outcomes, and how the memory was used come from the store;
- **no rewriting** — the original memory is never edited, and no corrective text is invented here;
- **a correction is a separate, consented write** — `propose_correction()` requires a person's name and
  a statement, and it supersedes rather than overwrites (the pair stays auditable, as in #33/#29);
- **a lesson closes** when the correction is written, or when a person says the failure was situational.

The distinction is the whole point: "this failed twice" is a fact; "therefore do this instead" is a
judgement, and a store that makes that judgement for itself is guessing with a citation attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

LESSONS = "outcome_lessons"
MIN_FAILURES = 2
BAD_SHARE = 0.5


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 - a report that cannot read reports nothing
        return []


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {LESSONS} (
                    lesson_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'open',
                    failures INTEGER NOT NULL DEFAULT 0,
                    successes INTEGER NOT NULL DEFAULT 0,
                    tasks_json TEXT NOT NULL DEFAULT '[]',
                    correction_id TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    closed_at TEXT NOT NULL DEFAULT ''
                )"""
        )


@dataclass
class Lesson:
    lesson_id: str
    memory_id: str
    failures: int = 0
    successes: int = 0
    tasks: list[str] = field(default_factory=list)
    content: str = ""
    state: str = "open"
    note: str = ""

    @property
    def share(self) -> float:
        total = self.failures + self.successes
        return round(self.failures / total, 4) if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "memory_id": self.memory_id,
            "failures": self.failures,
            "successes": self.successes,
            "share": self.share,
            "tasks": self.tasks,
            "state": self.state,
            "note": self.note,
            "question": self.question(),
            "content": self.content,
        }

    def question(self) -> str:
        """The lesson as a question — what this module produces instead of a verdict."""
        if not self.failures:
            return "no failures recorded"
        tasks = ", ".join(self.tasks[:3]) or "the recorded task(s)"
        return (f"this informed {tasks} and {self.failures} of "
                f"{self.failures + self.successes} judged outcome(s) failed: is the memory wrong, or "
                f"was it used in the wrong situation?")


def failing(store: Any, *, min_failures: int = MIN_FAILURES,
            bad_share: float = BAD_SHARE, limit: int = 50) -> list[dict[str, Any]]:
    """Memories whose record is bad enough to ask a question about."""
    rows = _rows(
        store,
        """SELECT memory_id,
                  SUM(outcome = 'failure') AS failures,
                  SUM(outcome = 'success') AS successes
           FROM memory_outcomes GROUP BY memory_id
           HAVING failures >= ? ORDER BY failures DESC LIMIT ?""",
        (int(min_failures), int(limit)),
    )
    found: list[dict[str, Any]] = []
    for row in rows:
        failures = int(row["failures"] or 0)
        successes = int(row["successes"] or 0)
        share = failures / (failures + successes) if (failures + successes) else 1.0
        if share < bad_share:
            # Three successes and two failures is a mixed record, not a lesson: it needs more data,
            # not a rewrite. Only a genuinely bad share raises the question.
            continue
        tasks = [str(item["task_ref"]) for item in _rows(
            store,
            """SELECT task_ref FROM memory_outcomes
               WHERE memory_id = ? AND outcome = 'failure' ORDER BY created_at DESC LIMIT 5""",
            (str(row["memory_id"]),),
        )]
        content = ""
        detail = _rows(store, "SELECT content FROM memories WHERE id = ?", (str(row["memory_id"]),))
        if detail:
            content = str(detail[0]["content"])
        found.append({
            "memory_id": str(row["memory_id"]),
            "failures": failures,
            "successes": successes,
            "tasks": tasks,
            "content": content,
        })
    return found


def as_lessons(store: Any, **kwargs: Any) -> list[Lesson]:
    return [
        Lesson(lesson_id=f"{item['memory_id']}:{item['failures']}", memory_id=item["memory_id"],
               failures=item["failures"], successes=item["successes"], tasks=item["tasks"],
               content=item["content"])
        for item in failing(store, **kwargs)
    ]


def persist(store: Any, *, at: datetime | None = None, **kwargs: Any) -> int:
    """Record the open lessons, idempotently: a lesson that is already recorded is not duplicated."""
    _ensure(store)
    moment = (at or _now()).isoformat()
    written = 0
    with store.transaction() as conn:
        for lesson in as_lessons(store, **kwargs):
            existing = conn.execute(
                f"SELECT state FROM {LESSONS} WHERE memory_id = ? AND state = 'open'",
                (lesson.memory_id,),
            ).fetchone()
            if existing is not None:
                continue
            conn.execute(
                f"""INSERT OR REPLACE INTO {LESSONS}
                    (lesson_id, memory_id, state, failures, successes, tasks_json, correction_id,
                     note, created_at)
                    VALUES (?, ?, 'open', ?, ?, ?, '', '', ?)""",
                (lesson.lesson_id, lesson.memory_id, lesson.failures, lesson.successes,
                 "[]" if not lesson.tasks else str(lesson.tasks).replace("'", '"'), moment),
            )
            written += 1
    return written


class ConsentRequired(ValueError):
    """A correction changes what the store believes, so it needs a person behind it."""


def propose_correction(store: Any, memory_id: str, statement: str, *, by: str = "",
                       at: datetime | None = None) -> dict[str, Any]:
    """Write a corrective memory that supersedes the failing one — with a name attached.

    The corrective statement is **somebody's judgement**, so this refuses without one. The original
    row is superseded, never deleted or rewritten: the pair stays auditable, exactly as in #33 and #29.
    """
    if not str(by or "").strip():
        raise ConsentRequired("a correction needs a person: pass by='<name>'")
    if not str(statement or "").strip():
        raise ValueError("a correction needs a statement")
    source = _rows(store, "SELECT content, kind FROM memories WHERE id = ?", (str(memory_id),))
    if not source:
        return {"corrected": False, "reason": "no such memory"}
    moment = at or _now()
    correction_id = store.add_memory(
        content=str(statement),
        kind=str(source[0]["kind"] or "semantic"),
        source_type="correction",
        source_category="USER_CORRECTION",
        confidence=0.9,
        source_context=f"correction of {memory_id} by {by} after repeated failures",
    )[0]
    with store.transaction() as conn:
        conn.execute(
            "UPDATE memories SET state = 'superseded' WHERE id = ? AND state = 'active'",
            (memory_id,),
        )
        # `supersedes_id` lives on the winner's row, as everywhere else in this store (#33).
        conn.execute(
            "UPDATE memories SET supersedes_id = ?, updated_at = ? WHERE id = ?",
            (memory_id, moment.isoformat(), correction_id),
        )
    _ensure(store)
    with store.transaction() as conn:
        # The open lesson is the one that gets closed. Inserting a second "corrected" row beside it
        # would leave the open one dangling and the report claiming work that was already done.
        closed = conn.execute(
            f"""UPDATE {LESSONS} SET state = 'corrected', correction_id = ?, note = ?, closed_at = ?
                WHERE memory_id = ? AND state = 'open'""",
            (correction_id, f"corrected by {by}", moment.isoformat(), memory_id),
        ).rowcount
        if not closed:
            conn.execute(
                f"""INSERT INTO {LESSONS}
                    (lesson_id, memory_id, state, failures, successes, tasks_json, correction_id, note,
                     created_at, closed_at)
                    VALUES (?, ?, 'corrected', 0, 0, '[]', ?, ?, ?, ?)""",
                (f"{memory_id}:correction:{correction_id}", memory_id, correction_id,
                 f"corrected by {by}", moment.isoformat(), moment.isoformat()),
            )
    return {"corrected": True, "memory_id": memory_id, "correction_id": correction_id, "by": by}


def close(store: Any, memory_id: str, *, by: str, note: str = "",
          at: datetime | None = None) -> dict[str, Any]:
    """A person says the failure was situational — the lesson closes without a rewrite."""
    if not str(by or "").strip():
        raise ConsentRequired("closing a lesson needs a person: pass by='<name>'")
    _ensure(store)
    moment = (at or _now()).isoformat()
    with store.transaction() as conn:
        cursor = conn.execute(
            f"""UPDATE {LESSONS} SET state = 'situational', note = ?, closed_at = ?
                WHERE memory_id = ? AND state = 'open'""",
            (f"{note} ({by})".strip(), moment, str(memory_id)),
        )
        closed = cursor.rowcount
    return {"closed": int(closed), "memory_id": str(memory_id), "by": by}


def lessons(store: Any, *, state: str = "", limit: int = 50) -> list[dict[str, Any]]:
    _ensure(store)
    sql = f"SELECT * FROM {LESSONS}"
    params: tuple = ()
    if state:
        sql += " WHERE state = ?"
        params = (str(state),)
    sql += " ORDER BY created_at DESC LIMIT ?"
    return _rows(store, sql, params + (int(limit),))


def report(store: Any) -> dict[str, Any]:
    _ensure(store)
    rows = _rows(store, f"SELECT state, COUNT(*) AS count FROM {LESSONS} GROUP BY state")
    by_state = {str(row["state"]): int(row["count"] or 0) for row in rows}
    recorded = _rows(store, "SELECT COUNT(*) AS total FROM memory_outcomes")
    return {
        "lessons": sum(by_state.values()),
        "by_state": by_state,
        "open": by_state.get("open", 0),
        "outcomes_recorded": int((recorded[0] if recorded else {}).get("total") or 0),
        "thresholds": {"min_failures": MIN_FAILURES, "bad_share": BAD_SHARE},
        "note": "'this failed twice' is a fact; 'therefore do this instead' is a judgement",
    }
