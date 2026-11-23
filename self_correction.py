"""A user correcting themselves supersedes immediately (#33).

"Actually, no — it's port 8200." is the strongest evidence the store will ever receive: the person
who owns the fact is replacing it, in the same breath, with the reason attached. Sending that through
the ordinary proposal queue means the store keeps serving the wrong value until a judge gets around to
it — and the judge's job is to decide whether a *machine* should write, a question that does not exist
here.

Two guards keep the shortcut narrow:

- **a strong marker or nothing.** "I meant", "correction:", "scratch that", "no, it's", "not X but Y",
  "I was wrong" are corrections. A bare "actually" inside an ordinary sentence ("actually the
  dashboard is fine") is not, and a weak marker only acts when it also contradicts a stored fact.
- **the loser is superseded, never deleted.** The old row stays readable with its content intact and
  `supersedes_id` on the **winner's** row, which is where this store has always kept it. An undo
  restores the pair, because a person confident enough to correct themselves once may correct
  themselves twice.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

TABLE = "self_corrections"

# Deliberately explicit. Each of these is a claim about the previous statement, which is what
# separates a correction from a sentence that merely contains the word "actually".
STRONG = (
    re.compile(r"\bi meant\b", re.IGNORECASE),
    re.compile(r"\bcorrection\s*[:,]", re.IGNORECASE),
    re.compile(r"\bscratch that\b", re.IGNORECASE),
    re.compile(r"\bthat'?s (?:wrong|incorrect|not right)\b", re.IGNORECASE),
    re.compile(r"\bi was wrong\b", re.IGNORECASE),
    # A message that opens with "No," is replacing what was just said; the narrower forms below
    # cover the same move mid-sentence.
    re.compile(r"^\s*no\s*[,!.:—-]", re.IGNORECASE),
    re.compile(r"[.!?]\s+no\s*[,!.:—-]", re.IGNORECASE),
    re.compile(r"\bno,? (?:it'?s|its|that'?s)\b", re.IGNORECASE),
    # "not X, but Y" only when the comma is there: without it the same words read as a preference
    # ("make it feel like X instead of Y") and superseding on that retires a fact nobody corrected.
    re.compile(r"\bnot\s+\S+(?:\s+\S+){0,4},\s+but\b", re.IGNORECASE),
    re.compile(r"\bactually,?\s+no\b", re.IGNORECASE),
)
WEAK = (re.compile(r"\bactually\b", re.IGNORECASE),)

# Corrections are conversational. A long passage that happens to contain "not X but Y" is
# documentation disagreeing with itself, and treating it as a person replacing a fact retires the
# wrong row - the live store has exactly that shape in it (a qBittorrent settings write-up).
MAX_CORRECTION_CHARS = 240

def weak_enabled(environ: Any = None) -> bool:
    """The weak path ships **off**.

    Measured on the live store: of 238 user-typed memories, the detector called two corrections and
    both were ordinary sentences containing "actually" ("Give me ones you think I actually need").
    A bare "actually" is not a claim about the previous statement often enough to act on unattended.
    """
    import os

    env = environ if environ is not None else os.environ
    return str(env.get("CORTEX_WEAK_CORRECTION", "")).strip().lower() in {"1", "true", "yes", "on"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    correction_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    superseded_id TEXT,
                    statement TEXT NOT NULL,
                    marker TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reverted_at TEXT
                )"""
        )


@dataclass
class Detection:
    is_correction: bool
    strength: str = ""
    marker: str = ""
    statement: str = ""
    raw: str = ""

    def describe(self) -> str:
        if not self.is_correction:
            return "not a correction"
        return f"{self.strength} correction after {self.marker!r}: {self.statement[:100]}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_correction": self.is_correction,
            "strength": self.strength,
            "marker": self.marker,
            "statement": self.statement,
        }


def detect(message: str) -> Detection:
    """Decide whether a message replaces a fact, and which of it is the replacement."""
    text = str(message or "").strip()
    if not text or len(text) > MAX_CORRECTION_CHARS:
        return Detection(False)
    for pattern in STRONG:
        match = pattern.search(text)
        if match:
            # The correction is what follows the *strongest* marker already found; a second pass
            # over the same words only re-matches a shorter prefix of it and keeps the old value.
            body = text[match.end():].lstrip(" ,.;:-—").strip()
            return Detection(True, "strong", match.group(0).strip(), body or text, text)
    for pattern in WEAK:
        match = pattern.search(text)
        if match:
            return Detection(True, "weak", match.group(0).strip(),
                             text[match.end():].strip() or text, text)
    return Detection(False)


_TRAILING = ".,;:!?\"'()[]{}"
# A correction usually shares only the part of the claim that survived it ("dashboard"), so the bar
# is low - the tie rule below is what actually keeps it safe.
MIN_MATCH = 0.25
STOPWORDS = frozenset(
    "the a an is are was were be been being it its it's this that these those of to in on for "
    "and or but not no with from by as at if then than so just really very now port".split()
)


def _tokens(text: str) -> list[str]:
    cleaned = str(text or "").lower().translate(str.maketrans({char: " " for char in _TRAILING}))
    return [word for word in cleaned.split() if len(word) > 1 and word not in STOPWORDS]


def _candidate(store: Any, statement: str, *, limit: int = 40) -> dict[str, Any] | None:
    """The stored memory this correction most likely replaces.

    Matching is on the statement's own distinctive words rather than a retrieve call: the correction
    is usually about the *previous* value, so the shared vocabulary is exactly the part that did not
    change, and a similarity rank on that alone is a coin toss between the true target and its
    neighbours. Containment on the shorter side, as elsewhere in this store, then the longest match
    wins and ties break on the newest row.
    """
    words = sorted(set(_tokens(statement)), key=len, reverse=True)[:6]
    if not words:
        return None
    clause = " OR ".join("content LIKE ?" for _ in words)
    try:
        rows = store._conn.execute(
            f"""SELECT id, content FROM memories
                WHERE state = 'active' AND ({clause})
                ORDER BY updated_at DESC LIMIT ?""",
            tuple(f"%{word}%" for word in words) + (int(limit),),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return None
    wanted = set(_tokens(statement))
    scored: list[tuple[float, str]] = []
    for row in rows:
        have = set(_tokens(str(row["content"])))
        if not have:
            continue
        scored.append((len(wanted & have) / min(len(wanted), len(have)), str(row["id"])))
    if not scored:
        return None
    scored.sort(key=lambda item: -item[0])
    best_score, best_id = scored[0]
    if best_score < MIN_MATCH:
        return None
    if len(scored) > 1 and abs(scored[1][0] - best_score) < 1e-9:
        # Two memories match equally well: superseding the wrong one silently retires a fact
        # nobody corrected, so the correction is written with no target instead.
        return None
    return {"memory_id": best_id, "score": round(best_score, 4)}


def intercept(store: Any, message: str, *, target_id: str = "", at: datetime | None = None,
              previous: str = "", allow_weak: bool | None = None) -> dict[str, Any]:
    """Write the correction now and supersede the fact it replaces.

    `previous` is the statement being corrected when the caller has it; without one the target is
    found by searching the store, and a correction with no target is still written — the person said
    it, which is all the authorisation a user fact needs.
    """
    found = detect(message)
    if not found.is_correction:
        return {"applied": False, "reason": "not a correction"}
    if found.strength != "strong":
        if not (weak_enabled() if allow_weak is None else allow_weak):
            return {"applied": False, "reason": "weak marker ignored (CORTEX_WEAK_CORRECTION)",
                    "detection": found.as_dict()}
        target = {"memory_id": str(target_id)} if target_id else _candidate(
            store, previous or found.statement
        )
        if not target:
            return {"applied": False, "reason": "weak marker and nothing to correct",
                    "detection": found.as_dict()}
        return _supersede(store, found, target["memory_id"], at)
    target = {"memory_id": str(target_id)} if target_id else _candidate(
        store, previous or found.statement
    )
    return _supersede(store, found, (target or {}).get("memory_id", ""), at)


def _supersede(store: Any, found: Any, target_id: str, at: datetime | None) -> dict[str, Any]:
    _ensure(store)
    statement = found.statement or found.raw
    moment = at or _now()
    new_id = store.add_memory(
        content=statement,
        kind="correction",
        source_type="user_turn",
        source_category="USER_CORRECTION",
    )[0]
    with store.transaction() as conn:
        if target_id:
            conn.execute(
                "UPDATE memories SET state = 'superseded' WHERE id = ? AND state = 'active'",
                (str(target_id),),
            )
            # supersedes_id lives on the winner's row, as everywhere else in this store.
            conn.execute(
                "UPDATE memories SET supersedes_id = ?, updated_at = ? WHERE id = ?",
                (str(target_id), moment.isoformat(), str(new_id)),
            )
        conn.execute(
            f"""INSERT INTO {TABLE}
                (correction_id, memory_id, superseded_id, statement, marker, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
            (uuid.uuid4().hex[:12], str(new_id), str(target_id) or None, statement,
             found.marker, moment.isoformat()),
        )
    try:
        from . import provenance

        provenance.link(store, str(new_id), "corrected",
                        detail=f"replaced {str(target_id)[:8]}" if target_id else "no target",
                        source_id=str(target_id), at=moment)
        if target_id:
            provenance.link(store, str(target_id), "superseded",
                            detail=f"replaced by {str(new_id)[:8]}",
                            source_id=str(new_id), at=moment)
    except Exception:  # noqa: BLE001 - provenance is evidence, never a reason a correction fails
        pass
    return {
        "applied": True,
        "memory_id": str(new_id),
        "superseded_id": str(target_id) or None,
        "statement": statement,
        "strength": found.strength,
        "sentence": (f"corrected {str(target_id)[:8]}" if target_id
                     else "recorded a correction with nothing to replace"),
    }


def history(store: Any, *, limit: int = 10) -> list[dict[str, Any]]:
    try:
        _ensure(store)
        rows = store._conn.execute(
            f"SELECT * FROM {TABLE} ORDER BY created_at DESC LIMIT ?", (int(limit),)
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [dict(row) for row in rows]


def undo(store: Any, correction_id: str = "") -> dict[str, Any]:
    """Put the corrected fact back the way it was."""
    rows = [row for row in history(store, limit=50) if not row.get("reverted_at")]
    if correction_id:
        rows = [row for row in rows if str(row["correction_id"]) == str(correction_id)]
    if not rows:
        return {"undone": False, "reason": "no correction to undo"}
    entry = rows[0]
    with store.transaction() as conn:
        if entry.get("superseded_id"):
            conn.execute(
                "UPDATE memories SET state = 'active' WHERE id = ? AND state = 'superseded'",
                (str(entry["superseded_id"]),),
            )
        conn.execute(
            "UPDATE memories SET state = 'superseded' WHERE id = ?", (str(entry["memory_id"]),)
        )
        conn.execute(
            f"UPDATE {TABLE} SET reverted_at = ? WHERE correction_id = ?",
            (_now().isoformat(), str(entry["correction_id"])),
        )
    return {"undone": True, "correction_id": str(entry["correction_id"]),
            "restored_id": entry.get("superseded_id")}


def report(store: Any) -> dict[str, Any]:
    rows = history(store, limit=1000)
    return {
        "corrections": len(rows),
        "with_target": sum(1 for row in rows if row.get("superseded_id")),
        "undoable": sum(1 for row in rows if not row.get("reverted_at")),
    }
