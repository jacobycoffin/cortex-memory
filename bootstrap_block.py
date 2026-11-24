"""The fixed preamble: who, what, and what is not allowed (#150)

Every session starts cold. A bootstrap block is the same handful of lines handed over every time, so
the assistant does not re-derive its own situation from scratch — and so a person reading the prompt can
see what it was told.

Four properties, and the block is worthless without all four:

- **a fixed shape.** Three headings, always, in the same order, even when a section has nothing. A
  heading that disappears when its section is empty silently reshapes the prompt, and nobody notices
  because nothing looks wrong.
- **deterministic.** Same store, same block, in the same order — no sampling, no recency window. A
  preamble that changes between two identical sessions cannot be reasoned about.
- **traceable.** Every line ends with the memory id it came from, so a wrong line can be found and
  fixed rather than argued about. This matters because the store's own `kind` labels are not reliable:
  of 15 live `identity` memories, at least one is a gateway-restart log and another is an instruction
  to "reply exactly: I am <model> running via <provider>".
- **bounded.** A total character budget and a per-section cap, with the truncation counted and said out
  loud rather than a block that quietly shrinks.

It composes what already exists — pinned memories for *who*, `scope` project tags for *what*, and
`constraints.context_block()` for *what is not allowed* — rather than inventing a fourth source of
truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import constraints

BUDGET = 900
WHO_ITEMS = 5
PROJECT_ITEMS = 6
WHO_CHARS = 100
PROJECT_CHARS = 40

# The heading names the **query that ran**, not the concept one wishes there were. The store cannot
# support an "identity" section honestly - of 15 live `identity` memories, one is a restart log and one
# is a "reply exactly" instruction - so the section says what it actually holds: what is pinned.
WHO = "pinned"
PROJECTS = "working on"
CONSTRAINTS = "constraints"


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 - a preamble must never break a session
        return []


def clause(text: Any, limit: int) -> str:
    body = " ".join(str(text or "").split())
    for stop in (". ", "; "):
        index = body.find(stop)
        if 0 < index <= limit:
            return body[:index]
    if len(body) <= limit:
        return body
    cut = body[:limit]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut + "…"


def who(store: Any, *, limit: int = WHO_ITEMS) -> list[dict[str, Any]]:
    """Identity, taken from what is pinned — not from a `kind` label that lies."""
    rows = _rows(
        store,
        "SELECT id, kind, content, confidence FROM memories "
        "WHERE state = 'active' AND pinned = 1 ORDER BY confidence DESC, id LIMIT ?",
        (max(1, int(limit)) * 3,),
    )
    chosen: list[dict[str, Any]] = []
    for row in rows:
        text = clause(row["content"], WHO_CHARS)
        if not text:
            continue
        chosen.append({"id": str(row["id"]), "kind": str(row["kind"] or ""), "text": text,
                       "confidence": round(float(row["confidence"] or 0.0), 2)})
        if len(chosen) >= int(limit):
            break
    return chosen


def projects(store: Any, *, limit: int = PROJECT_ITEMS) -> list[dict[str, Any]]:
    """Active projects, as the scope tags the store already carries."""
    rows = _rows(
        store,
        "SELECT scope_json FROM memories WHERE state = 'active' AND scope_json LIKE '%project%'",
    )
    counts: dict[str, int] = {}
    for row in rows:
        try:
            scope = json.loads(row["scope_json"] or "{}")
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(scope, dict):
            continue
        value = scope.get("project")
        names = value if isinstance(value, list) else [value]
        for name in names:
            text = clause(name, PROJECT_CHARS)
            if text:
                counts[text] = counts.get(text, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"project": name, "memories": count} for name, count in ranked[: int(limit)]]


def limits(store: Any) -> list[dict[str, Any]]:
    """The standing constraints, from the module that already owns them."""
    try:
        items = constraints.all(store, limit=constraints.MAX_ITEMS)
    except Exception:  # noqa: BLE001 - the preamble must appear even when a source is unreadable
        return []
    return [{"id": str(getattr(item, "memory_id", "") or ""), "text": str(item.line())}
            for item in items][: constraints.MAX_ITEMS]


@dataclass
class Block:
    lines: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    truncated: int = 0
    chars: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"lines": self.lines, "ids": self.ids, "truncated": self.truncated,
                "chars": self.chars, "text": "\n".join(self.lines)}

    def __str__(self) -> str:
        return "\n".join(self.lines)


def build(store: Any, *, budget: int = BUDGET, who_items: int = WHO_ITEMS,
          project_items: int = PROJECT_ITEMS) -> Block:
    """The fixed preamble. Always three sections, always in this order."""
    cap = max(120, int(budget))
    found = Block()
    sections: list[tuple[str, list[str]]] = []

    who_lines = [f"- {row['text']} [{row['id'][:8]}]" for row in who(store, limit=who_items)]
    sections.append((WHO, who_lines))

    project_lines = [f"- {row['project']} ({row['memories']} memories)"
                     for row in projects(store, limit=project_items)]
    sections.append((PROJECTS, project_lines))

    limit_lines = []
    for row in limits(store):
        # `Constraint.line()` already carries its own bullet; a second one reads as a typo.
        text = row["text"].strip()
        while text.startswith("- "):
            text = text[2:].lstrip()
        if text:
            marker = f" [{row['id'][:8]}]" if row["id"] else ""
            limit_lines.append(f"- {text}{marker}")
    sections.append((CONSTRAINTS, limit_lines))

    used = 0
    for name, lines in sections:
        heading = f"{name.capitalize()}:"
        if used + len(heading) + 1 > cap:
            found.truncated += len(lines)
            continue
        found.lines.append(heading)
        used += len(heading) + 1
        if not lines:
            # A missing section reshapes the prompt silently; an explicit "unknown" does not.
            line = "- (nothing recorded)"
            found.lines.append(line)
            used += len(line) + 1
            continue
        for line in lines:
            if used + len(line) + 1 > cap:
                found.truncated += 1
                continue
            found.lines.append(line)
            used += len(line) + 1
            for token in line.split("["):
                if token.endswith("]"):
                    found.ids.append(token[:-1].strip())
    found.chars = sum(len(line) + 1 for line in found.lines)
    return found


def text(store: Any, **kwargs: Any) -> str:
    return str(build(store, **kwargs))
