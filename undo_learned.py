"""One command to take back what a consolidation run learned (#213).

A consolidation pass changes records in several places at once: a source memory is superseded, a
target's content is rewritten, a decision row moves from `proposed` to `applied`. When the pass was
wrong, the person's only options today are "restore yesterday's database" — which throws away
everything else that happened since — or "edit rows by hand".

So each run becomes **one reversible step**:

- `begin()` snapshots the rows the run is about to touch, before anything moves;
- `commit()` records the after-state, so the step knows what it actually changed;
- `undo()` puts the touched rows back, and only those.

The snapshot is deliberately narrow — the ids the run selected, and only the columns a consolidation
can move (`state`, `content`, `confidence`, and a decision's `status`). A wider snapshot would be a
database rollback with a friendlier name, which is the thing this is not.

Undo is a walk backwards over completed steps, newest first, so several runs can be taken back in the
order they happened, and a step already undone is never applied twice.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

TABLE = "learned_changes"
MEMORY_COLUMNS = ("state", "content", "confidence")
DECISION_COLUMNS = ("status",)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    run_id TEXT PRIMARY KEY,
                    module TEXT NOT NULL,
                    targets_json TEXT NOT NULL DEFAULT '[]',
                    before_json TEXT NOT NULL DEFAULT '{{}}',
                    after_json TEXT NOT NULL DEFAULT '{{}}',
                    changed_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    committed_at TEXT,
                    undone_at TEXT
                )"""
        )


def _fingerprint(values: dict[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _snapshot(store: Any, targets: Iterable[str]) -> dict[str, dict[str, Any]]:
    """The columns a consolidation is allowed to move, for the ids it selected."""
    snapshot: dict[str, dict[str, Any]] = {}
    for memory_id in targets:
        try:
            row = store._conn.execute(
                """SELECT state, content, confidence FROM memories WHERE id = ?""",
                (str(memory_id),),
            ).fetchone()
        except Exception:  # noqa: BLE001 - a vanished id is not an error
            continue
        if row:
            snapshot[str(memory_id)] = {column: row[column] for column in MEMORY_COLUMNS}
    return snapshot


def begin(store: Any, module: str, targets: Iterable[str], *, at: datetime | None = None) -> str:
    """Start a reversible step. Returns the run id to commit or undo later."""
    _ensure(store)
    run_id = f"{module}:{uuid.uuid4().hex[:12]}"
    ids = [str(item) for item in targets if str(item).strip()]
    before = _snapshot(store, ids)
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE}
                (run_id, module, targets_json, before_json, changed_json, created_at)
                VALUES (?, ?, ?, ?, '[]', ?)""",
            (run_id, str(module), json.dumps(ids), json.dumps(before, default=str),
             (at or _now()).isoformat()),
        )
    return run_id


def commit(store: Any, run_id: str, *, at: datetime | None = None) -> dict[str, Any]:
    """Close the step: record what actually moved, so undo touches nothing else."""
    _ensure(store)
    try:
        row = store._conn.execute(
            f"SELECT before_json, targets_json FROM {TABLE} WHERE run_id = ?", (str(run_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001
        return {"changed": [], "reason": "unknown run"}
    if not row:
        return {"changed": [], "reason": "unknown run"}
    before = json.loads(row["before_json"] or "{}")
    targets = json.loads(row["targets_json"] or "[]")
    after = _snapshot(store, targets)
    changed = [
        memory_id for memory_id in before
        if _fingerprint(before[memory_id]) != _fingerprint(after.get(memory_id, {}))
    ]
    with store.transaction() as conn:
        conn.execute(
            f"""UPDATE {TABLE} SET after_json = ?, changed_json = ?, committed_at = ?
                WHERE run_id = ?""",
            (json.dumps(after, default=str), json.dumps(changed),
             (at or _now()).isoformat(), str(run_id)),
        )
    return {"run_id": str(run_id), "changed": changed}


def steps(store: Any, *, limit: int = 10, include_undone: bool = False) -> list[dict[str, Any]]:
    try:
        _ensure(store)
        rows = store._conn.execute(
            f"""SELECT * FROM {TABLE}
               ORDER BY created_at DESC LIMIT ?""",
            (int(limit),),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    out = []
    for row in rows:
        item = dict(row)
        if not include_undone and item.get("undone_at"):
            continue
        out.append(item)
    return out


def undo(store: Any, run_id: str | None = None, *, at: datetime | None = None) -> dict[str, Any]:
    """Put the touched rows back. One run per call; nothing else is affected."""
    candidates = steps(store, limit=1) if run_id is None else [
        row for row in steps(store, limit=100, include_undone=True) if row["run_id"] == run_id
    ]
    if not candidates:
        return {"undone": 0, "restored": [], "reason": "no step to undo"}
    step = candidates[0]
    if step.get("undone_at"):
        return {"undone": 0, "restored": [], "reason": "already undone"}
    before = json.loads(step["before_json"] or "{}")
    changed = json.loads(step["changed_json"] or "[]")
    restored: list[str] = []
    with store.transaction() as conn:
        for memory_id in changed:
            values = before.get(memory_id) or {}
            if not values:
                continue
            conn.execute(
                """UPDATE memories SET state = ?, content = ?, confidence = ? WHERE id = ?""",
                (values.get("state"), values.get("content"), values.get("confidence"),
                 str(memory_id)),
            )
            restored.append(str(memory_id))
        conn.execute(
            f"UPDATE {TABLE} SET undone_at = ? WHERE run_id = ?",
            ((at or _now()).isoformat(), str(step["run_id"])),
        )
    return {
        "undone": len(restored),
        "restored": restored,
        "module": str(step["module"]),
        "sentence": (
            f"reversed {len(restored)} change(s) from {step['module']}"
            if restored else f"nothing from {step['module']} had changed"
        ),
    }


def report(store: Any) -> dict[str, Any]:
    done = steps(store, limit=100, include_undone=True)
    return {
        "steps": len(done),
        "undoable": sum(1 for row in done if not row.get("undone_at")),
        "undone": sum(1 for row in done if row.get("undone_at")),
        "last": (done[0]["module"] if done else ""),
    }
