"""Rewrite core memory for the model, not the human (backlog item #128).

Core memories are captured as *notes*: "Vault note: Duo Tabletop Path: Temple/… Section: …",
"Jacoby prefers Jarvis to proactively delegate … so the main conversation stays available;
Jarvis should keep ambiguity, approval …". A human skims that. A model spends tokens on the
scaffolding — the path, the section label, the third-person narration — and the actual
instruction is a clause in the middle.

This module rewrites an entry into the form the block wants:

- **imperative**, not narrated: "Jacoby prefers short answers" → "Prefer short answers."
- **scoped**, keeping exceptions explicit: an `except`/`unless`/`but not` clause is lifted into
  `Except: …` rather than left buried at the end of a sentence.
- **stripped** of provenance scaffolding: `Vault note:`, `Path: …`, `Section: …`, heading
  markers and the note's own title.

Deliberately rule-based and deterministic. A model rewrite would read better and cost a call
per memory per render, and the block is the one thing that has to render when no model is
available (#120). Rules also mean the transform is inspectable: every change here is a regex
someone can argue with, which is the right property for text that is prepended to every turn.

The transform never invents content. If a rule cannot fire cleanly the original sentence is
kept — a verbose true memory beats a crisp invented one.
"""

from __future__ import annotations

import os
import re
from typing import Any

MAX_VOICE_CHARS = 220

_SELF = re.compile(r"\b(?:jacoby|the user|he|she|they)\b", re.I)
_PREFERENCE = re.compile(
    r"^(?:also[:,]?\s*)?(?:jacoby|the user)\s+(?:prefers?|likes?|wants?|would like)\s+(.+)$",
    re.I,
)
_RULE = re.compile(r"^(?:jacoby|the user)\s+(?:should|must|needs? to)\s+(.+)$", re.I)
_EXCEPTION = re.compile(
    r"\s*(?:[,;]\s*)?(?:except|unless|but not|other than)\s+(.+)$", re.I
)
_PATH_TOKEN = re.compile(r"\bpath:\s*\S+", re.I)
_SECTION_TAIL = re.compile(r"\bsection:\s*(.*)$", re.I)
_NOTE_PREFIX = re.compile(r"^(?:vault\s+note|note)\s*:\s*", re.I)
_HEADING = re.compile(r"^#+\s*", re.M)
_DASH = re.compile(r"\s+[\u2014\u2013]\s+|\s+-\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*;\s*")
_NARRATION = re.compile(r"\b(?:jarvis|hermes|the assistant|it) should\b", re.I)


def enabled() -> bool:
    """Off unless asked for: this rewrites the text a model reads on every turn."""
    return os.environ.get("CORTEX_CORE_VOICE", "0") != "0"


def _strip_scaffolding(text: str) -> str:
    """Remove provenance scaffolding, keeping the sentence after it.

    Done step by step rather than with one regex, because the first attempt used
    `^vault note:.*?$` on a single-line memory and deleted the *entire* memory — the
    scaffolding and the fact were on the same line, and a lazy match ran to the end of it.
    """
    cleaned = _HEADING.sub("", text)
    cleaned = _PATH_TOKEN.sub(" ", cleaned)
    # "Section: Rules — Jacoby prefers X": the label ends at the dash, the fact follows it.
    match = _SECTION_TAIL.search(cleaned)
    if match:
        head, _, tail = match.group(1).partition("\u2014")
        if not tail:
            head, _, tail = match.group(1).partition(" - ")
        if tail and len(head.strip()) < 40:
            cleaned = cleaned[: match.start()] + " " + tail
    cleaned = _NOTE_PREFIX.sub(" ", cleaned)
    # A short title before the first dash is a label; a long one is the fact itself.
    # An earlier version took the tail unconditionally and threw away the sentence that
    # mattered ("check the vault first … — do NOT treat it as a source" kept only the tail).
    parts = _DASH.split(cleaned, maxsplit=1)
    if len(parts) > 1 and len(parts[0].strip()) < 40 and parts[1].strip():
        cleaned = f"{parts[0].strip()}: {parts[1]}" if len(parts[0].split()) <= 3 else parts[1]
    return " ".join(cleaned.split())


def _imperative(clause: str) -> str:
    """Turn one narrated clause into a directive where the pattern is unambiguous."""
    clause = clause.strip()
    if not clause:
        return ""
    match = _PREFERENCE.match(clause)
    if match:
        body = match.group(1).strip().rstrip(".")
        return f"Prefer {body}."
    match = _RULE.match(clause)
    if match:
        body = match.group(1).strip().rstrip(".")
        return f"{body[0].upper()}{body[1:]}."
    # "X so Y" / "X; the assistant should keep Z" — keep only the directive half.
    if _NARRATION.search(clause):
        head = _NARRATION.split(clause)[0].rstrip(" ,;")
        if head:
            body = head[0].upper() + head[1:] if head[0].islower() else head
            return f"{body.rstrip('.')}."
    return clause if clause.endswith((".", "!", "?")) else f"{clause}."


def to_model_voice(text: str, *, max_chars: int = MAX_VOICE_CHARS) -> str:
    """Rewrite one memory body into directive, scoped form. Never empty, never invented."""
    original = " ".join(str(text or "").split())
    if not original:
        return ""
    body = _strip_scaffolding(original)
    if not body:
        return original

    exception = ""
    match = _EXCEPTION.search(body)
    if match:
        exception = match.group(1).strip().rstrip(".")
        body = body[: match.start()].strip()

    clauses = [part for part in _CLAUSE_SPLIT.split(body) if part.strip()]
    rewritten = [piece for piece in (_imperative(clause) for clause in clauses[:3]) if piece]
    if not rewritten:
        return original

    out = " ".join(rewritten)
    if exception:
        out = f"{out.rstrip('.')}. Except: {exception}."
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "\u2026"
    return out or original


def rewrite_row(row: dict[str, Any]) -> dict[str, Any]:
    """A copy of the row with its content in model voice. Other fields are untouched."""
    rewritten = dict(row)
    rewritten["content"] = to_model_voice(str(row.get("content") or ""))
    return rewritten


def measure(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average characters saved per entry — the reason to switch this on."""
    before = sum(len(str(row.get("content") or "")) for row in rows)
    after = sum(len(to_model_voice(str(row.get("content") or ""))) for row in rows)
    saved = before - after
    return {
        "entries": float(len(rows)),
        "chars_before": float(before),
        "chars_after": float(after),
        "chars_saved": float(saved),
        "share_saved": round(saved / before, 4) if before else 0.0,
    }
