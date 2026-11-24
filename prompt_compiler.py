"""Compile the system prompt from memory, deterministically (#151)

The prompt is assembled every turn from things that live in the store: what is pinned, what he has
corrected, what is being worked on, what the standing limits are. Assembling it ad hoc means the prompt
changes shape between turns for reasons nobody wrote down — and a prompt that changes every turn destroys
prefix caching, which is the difference between a cheap turn and an expensive one.

Five rules:

- **same state, same bytes.** Sections are ordered, items are sorted, and **no timestamp is embedded** —
  a prompt that differs only by "generated at 12:00:01" cannot be cached and cannot be diffed. `digest()`
  gives a stable 12-char hash so two runs can be compared instead of eyeballed.
- **a correction is never the section that gets truncated.** It outranks everything: he said it in so many
  words, and a prompt that drops it re-litigates a settled question.
- **budget is per section and stated.** When the total exceeds the budget the lowest-priority section
  shrinks first, and what was dropped is **counted and reported**, never silently cut.
- **an empty section is omitted, not printed.** A heading with nothing under it costs tokens and reads as
  a bug.
- **every line carries a short provenance tag** so the panel (#148) and receipts (#174) can attribute it —
  eight characters, because the id is for tracing, not for reading.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

BUDGET_CHARS = 1800
TAG_CHARS = 8
# Fixed order. The first two are never truncated; the rest shrink from the bottom.
SECTIONS: tuple[tuple[str, str, bool], ...] = (
    ("pinned", "Pinned", True),
    ("corrections", "Corrections", True),
    ("working", "Working on", False),
    ("limits", "Limits", False),
    ("commitments", "Commitments", False),
)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def tag(memory_id: Any) -> str:
    return str(memory_id or "")[:TAG_CHARS]


def one_line(text: Any, limit: int = 160) -> str:
    body = " ".join(str(text or "").split())
    return body if len(body) <= limit else body[: limit - 1].rstrip() + "…"


def collect(store: Any, *, per_section: int = 6) -> dict[str, list[dict[str, Any]]]:
    """Gather each section's items. Sorted, so the same state gives the same order."""
    gathered: dict[str, list[dict[str, Any]]] = {name: [] for name, _title, _keep in SECTIONS}
    pinned = _rows(store, "SELECT id, content, kind FROM memories WHERE pinned = 1 "
                          "AND state <> 'archived' AND valid_to IS NULL "
                          "ORDER BY created_at DESC, id LIMIT ?", (per_section,))
    gathered["pinned"] = [{"id": tag(row["id"]), "text": one_line(row["content"])} for row in pinned]
    corrections = _rows(store, "SELECT id, content FROM memories "
                               "WHERE source_category = 'USER_CORRECTION' AND valid_to IS NULL "
                               "ORDER BY created_at DESC, id LIMIT ?", (per_section,))
    gathered["corrections"] = [{"id": tag(row["id"]), "text": one_line(row["content"])}
                               for row in corrections]
    working = _rows(store, "SELECT id, content FROM memories WHERE state <> 'archived' "
                           "AND valid_to IS NULL AND kind IN ('prospective', 'operational') "
                           "ORDER BY created_at DESC, id LIMIT ?", (per_section,))
    gathered["working"] = [{"id": tag(row["id"]), "text": one_line(row["content"])} for row in working]
    limits = _rows(store, "SELECT id, content FROM memories WHERE state <> 'archived' "
                          "AND valid_to IS NULL AND kind IN ('preference', 'identity') "
                          "ORDER BY created_at DESC, id LIMIT ?", (per_section,))
    gathered["limits"] = [{"id": tag(row["id"]), "text": one_line(row["content"])} for row in limits]
    commitments = _rows(store, "SELECT id, content FROM memories WHERE state <> 'archived' "
                               "AND valid_to IS NULL AND kind = 'decision' "
                               "ORDER BY created_at DESC, id LIMIT ?", (per_section,))
    gathered["commitments"] = [{"id": tag(row["id"]), "text": one_line(row["content"])}
                               for row in commitments]
    return gathered


@dataclass
class Compiled:
    text: str = ""
    digest: str = ""
    chars: int = 0
    budget: int = BUDGET_CHARS
    sections: list[dict[str, Any]] = field(default_factory=list)
    dropped: int = 0
    over_budget: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "digest": self.digest, "chars": self.chars,
                "budget": self.budget, "sections": self.sections, "dropped": self.dropped,
                "over_budget": self.over_budget}


def _render_section(title: str, items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    lines = [f"{title}:"]
    lines.extend(f"  - {item['text']} [{item['id']}]" for item in items)
    return "\n".join(lines)


def compile_prompt(store: Any, *, budget: int = BUDGET_CHARS, per_section: int = 6) -> Compiled:
    """Assemble the prompt. Truncation happens from the bottom, and is counted."""
    gathered = collect(store, per_section=per_section)
    blocks: list[tuple[str, str, str, bool, int]] = []
    for name, title, keep in SECTIONS:
        rendered = _render_section(title, gathered[name])
        blocks.append((name, title, rendered, keep, len(gathered[name])))
    total = sum(len(block[2]) + 2 for block in blocks if block[2])
    dropped = 0
    if total > budget:
        # Shrink from the bottom: drop whole sections that may be dropped, then trim items.
        for index in range(len(blocks) - 1, -1, -1):
            name, title, rendered, keep, count = blocks[index]
            if not rendered or keep or total <= budget:
                continue
            total -= len(rendered) + 2
            blocks[index] = (name, title, "", keep, count)
            dropped += count
        if total > budget:
            # Still over: trim the last non-protected section item by item.
            for index in range(len(blocks) - 1, -1, -1):
                name, title, rendered, keep, count = blocks[index]
                if not rendered or keep:
                    continue
                items = list(gathered[name])
                while items and total > budget:
                    items.pop()
                    dropped += 1
                    shortened = _render_section(title, items)
                    total = total - len(rendered) + len(shortened)
                    rendered = shortened
                blocks[index] = (name, title, rendered, keep, len(items))
                if total <= budget:
                    break
    parts = [block[2] for block in blocks if block[2]]
    text = "\n\n".join(parts)
    return Compiled(
        text=text,
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
        chars=len(text),
        budget=int(budget),
        sections=[{"name": name, "chars": len(rendered), "items": count}
                  for name, _title, rendered, _keep, count in blocks],
        dropped=dropped,
        over_budget=len(text) > int(budget),
    )


def stable(store: Any, **kwargs: Any) -> dict[str, Any]:
    """Two compilations of the same state must be byte-identical, or nothing can be cached."""
    first = compile_prompt(store, **kwargs)
    second = compile_prompt(store, **kwargs)
    return {"stable": first.digest == second.digest, "digest": first.digest,
            "chars": first.chars}


def render(store: Any, **kwargs: Any) -> list[str]:
    compiled = compile_prompt(store, **kwargs)
    lines = [f"prompt: {compiled.chars} of {compiled.budget} chars, digest {compiled.digest}"
             f"{', OVER BUDGET' if compiled.over_budget else ''}"]
    for section in compiled.sections:
        state = f"{section['items']} item(s), {section['chars']} chars" if section["chars"] else "omitted"
        lines.append(f"  · {section['name']}: {state}")
    if compiled.dropped:
        lines.append(f"  dropped {compiled.dropped} item(s) to fit the budget")
    return lines
