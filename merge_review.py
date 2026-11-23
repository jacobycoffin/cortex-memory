"""A merge is shown side by side, needs consent, and keeps the old record (#29).

Merging two memories is the most destructive ordinary operation in the store: one row stops being
retrievable on its own, and another starts claiming to say what both said. Three rules follow, and
none of them are optional:

1. **side by side** — the person sees both texts and a word-level diff before deciding, not a summary
   of them;
2. **consent** — `apply()` refuses without an explicit `consent=True` and a named consenter. A merge
   that happens because a caller forgot to pass a flag is indistinguishable from a bug;
3. **the old record is retained** — the source row keeps its content and moves to `superseded`, and
   the merge history stores both prior texts, so `undo()` restores the pair exactly.

The diff is computed on words rather than characters: a character diff of two sentences that share
most of their vocabulary is unreadable, and the question the reader is actually asking is *which
words moved*.
"""

from __future__ import annotations

import difflib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

TABLE = "memory_merges"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    merge_id TEXT PRIMARY KEY,
                    target_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_before TEXT NOT NULL,
                    source_before TEXT NOT NULL,
                    target_state_before TEXT NOT NULL,
                    source_state_before TEXT NOT NULL,
                    merged_content TEXT NOT NULL,
                    consented_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    undone_at TEXT
                )"""
        )


def _row(store: Any, memory_id: str) -> dict[str, Any] | None:
    try:
        row = store._conn.execute(
            "SELECT id, content, state FROM memories WHERE id = ?", (str(memory_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    return dict(row) if row else None


def diff(left: str, right: str, *, limit: int = 40) -> list[str]:
    """Word-level diff: what the merge adds and what it drops, in order."""
    left_words = str(left or "").split()
    right_words = str(right or "").split()
    lines: list[str] = []
    for operation, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, left_words, right_words
    ).get_opcodes():
        if operation == "equal":
            continue
        if operation in ("delete", "replace"):
            removed = " ".join(left_words[i1:i2]).strip()
            if removed:
                lines.append(f"- {removed}")
        if operation in ("insert", "replace"):
            added = " ".join(right_words[j1:j2]).strip()
            if added:
                lines.append(f"+ {added}")
        if len(lines) >= limit:
            break
    return lines


@dataclass
class Proposal:
    target_id: str
    source_id: str
    target_text: str
    source_text: str
    merged_text: str
    diff: list[str] = field(default_factory=list)
    reason: str = ""

    def describe(self) -> str:
        lines = ["left  (kept):   " + self.target_text[:160],
                 "right (merged): " + self.source_text[:160],
                 "result:         " + self.merged_text[:160]]
        if self.diff:
            lines.append("changes: " + " | ".join(self.diff[:6]))
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "source_id": self.source_id,
            "left": self.target_text,
            "right": self.source_text,
            "merged": self.merged_text,
            "diff": self.diff,
            "reason": self.reason,
            "preview": self.describe(),
        }


def propose(store: Any, target_id: str, source_id: str, *, merged_text: str = "",
            reason: str = "") -> Proposal | None:
    """Prepare a merge: both sides, the result, and the words that move."""
    target = _row(store, target_id)
    source = _row(store, source_id)
    if target is None or source is None:
        return None
    merged = str(merged_text or "").strip() or str(target["content"])
    return Proposal(
        target_id=str(target_id),
        source_id=str(source_id),
        target_text=str(target["content"]),
        source_text=str(source["content"]),
        merged_text=merged,
        diff=diff(str(target["content"]), merged),
        reason=str(reason),
    )


@dataclass
class Applied:
    ok: bool
    merge_id: str = ""
    reason: str = ""
    diff: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.ok:
            return f"not merged: {self.reason}"
        return f"merged {self.merge_id} ({len(self.diff)} word change(s)); the old record is kept"


def apply(store: Any, proposal: Proposal, *, consent: bool = False,
          consented_by: str = "", at: datetime | None = None) -> Applied:
    """Carry out a merge, but only with consent, and keeping both old texts."""
    if not isinstance(proposal, Proposal):
        return Applied(False, reason="no proposal")
    if not consent:
        return Applied(False, reason="consent required: nothing was merged")
    if not str(consented_by).strip():
        return Applied(False, reason="consent must name who gave it")
    target = _row(store, proposal.target_id)
    source = _row(store, proposal.source_id)
    if target is None or source is None:
        return Applied(False, reason="one of the memories no longer exists")
    _ensure(store)
    merge_id = uuid.uuid4().hex[:12]
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE}
                (merge_id, target_id, source_id, target_before, source_before,
                 target_state_before, source_state_before, merged_content, consented_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (merge_id, proposal.target_id, proposal.source_id, str(target["content"]),
             str(source["content"]), str(target["state"]), str(source["state"]),
             proposal.merged_text, str(consented_by), (at or _now()).isoformat()),
        )
        conn.execute(
            "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
            (proposal.merged_text, (at or _now()).isoformat(), proposal.target_id),
        )
        # The old record is retained: the source keeps its content and stops being retrievable
        # on its own, rather than being deleted.
        conn.execute(
            "UPDATE memories SET state = 'superseded' WHERE id = ?", (proposal.source_id,)
        )
    return Applied(True, merge_id, diff=diff(str(target["content"]), proposal.merged_text))


def history(store: Any, *, limit: int = 10) -> list[dict[str, Any]]:
    try:
        _ensure(store)
        rows = store._conn.execute(
            f"SELECT * FROM {TABLE} ORDER BY created_at DESC LIMIT ?", (int(limit),)
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [dict(row) for row in rows]


def undo(store: Any, merge_id: str | None = None, *, at: datetime | None = None) -> dict[str, Any]:
    """Put both memories back exactly as they were."""
    rows = history(store, limit=50)
    if merge_id is not None:
        rows = [row for row in rows if str(row["merge_id"]) == str(merge_id)]
    rows = [row for row in rows if not row.get("undone_at")]
    if not rows:
        return {"undone": False, "reason": "no merge to undo"}
    entry = rows[0]
    with store.transaction() as conn:
        conn.execute(
            """UPDATE memories SET content = ?, state = ? WHERE id = ?""",
            (str(entry["target_before"]), str(entry["target_state_before"]),
             str(entry["target_id"])),
        )
        conn.execute(
            """UPDATE memories SET content = ?, state = ? WHERE id = ?""",
            (str(entry["source_before"]), str(entry["source_state_before"]),
             str(entry["source_id"])),
        )
        conn.execute(
            f"UPDATE {TABLE} SET undone_at = ? WHERE merge_id = ?",
            ((at or _now()).isoformat(), str(entry["merge_id"])),
        )
    return {
        "undone": True,
        "merge_id": str(entry["merge_id"]),
        "sentence": f"restored both sides of merge {entry['merge_id']}",
    }


def report(store: Any) -> dict[str, Any]:
    rows = history(store, limit=1000)
    return {
        "merges": len(rows),
        "undoable": sum(1 for row in rows if not row.get("undone_at")),
        "note": "a merge keeps both prior texts, so undo is exact rather than reconstructed",
    }
