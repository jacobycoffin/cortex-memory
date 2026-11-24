"""What has to survive a compaction, and how you know it did (#152)

Compaction is the moment the assistant forgets on purpose. Everything that was in the context window is
replaced by a summary, and the memories that were injected, used or corrected in that conversation are not
automatically in the summary — so the thing the agent was halfway through doing quietly loses its footing.

Four rules, and the third is the one that makes this auditable rather than hopeful:

- **what survives is a decision, not an accident.** Corrections first (he said them in so many words),
  then pinned, then what was *used* in this session, then what was merely retrieved. A retrieval that was
  never used is not knowledge the agent had.
- **the post-compaction budget is smaller and stated.** Compaction exists to save room; a rebuild that
  spends the same budget has defeated the exercise. Drops are counted.
- **a lost memory is reported, never assumed present.** `verify()` takes the ids before and after and
  names what did not survive. Silence here is how an agent ends up confidently continuing a task with a
  belief that was dropped.
- **the rebuild is deterministic.** Same session state, same block, same digest — otherwise two
  compactions of the same conversation produce different knowledge.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

COMPACTION_BUDGET_CHARS = 700     # deliberately under the 1,800 normal prompt budget
PER_ITEM_CHARS = 120
PRIORITY = ("corrections", "pinned", "used", "retrieved")


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _one_line(text: Any, limit: int = PER_ITEM_CHARS) -> str:
    body = " ".join(str(text or "").split())
    return body if len(body) <= limit else body[: limit - 1].rstrip() + "…"


def session_ids(store: Any, session_id: str, *, limit: int = 200) -> dict[str, list[str]]:
    """What this session actually saw: selected (retrieved) versus resolved (used)."""
    selected: list[str] = []
    for row in _rows(store, "SELECT selected_memory_ids_json FROM memory_traces "
                            "WHERE session_id = ? ORDER BY created_at", (str(session_id),))[:limit]:
        try:
            for memory_id in json.loads(row.get("selected_memory_ids_json") or "[]"):
                if memory_id not in selected:
                    selected.append(str(memory_id))
        except Exception:  # noqa: BLE001
            continue
    used: list[str] = []
    for row in _rows(store, "SELECT memory_id FROM used_resolutions WHERE turn_id LIKE ? "
                            "ORDER BY resolved_at", (f"%{session_id}%",)):
        if str(row["memory_id"]) not in used:
            used.append(str(row["memory_id"]))
    return {"selected": selected, "used": used}


def candidates(store: Any, session_id: str) -> dict[str, list[dict[str, Any]]]:
    """Each priority class, resolved to content, in the order they must survive."""
    seen: set[str] = set()
    gathered: dict[str, list[dict[str, Any]]] = {name: [] for name in PRIORITY}

    def take(name: str, ids: Iterable[str]) -> None:
        for memory_id in ids:
            key = str(memory_id)
            if key in seen:
                continue
            rows = _rows(store, "SELECT id, content FROM memories WHERE id = ? AND state <> 'archived'",
                         (key,))
            if not rows:
                continue
            seen.add(key)
            gathered[name].append({"id": key, "text": _one_line(rows[0]["content"])})

    corrections = _rows(store, "SELECT id FROM memories WHERE source_category = 'USER_CORRECTION' "
                               "AND valid_to IS NULL ORDER BY created_at DESC")
    take("corrections", [row["id"] for row in corrections])
    pinned = _rows(store, "SELECT id FROM memories WHERE pinned = 1 AND state <> 'archived' "
                          "AND valid_to IS NULL ORDER BY created_at DESC")
    take("pinned", [row["id"] for row in pinned])
    seen_ids = session_ids(store, session_id)
    take("used", seen_ids["used"])
    take("retrieved", seen_ids["selected"])
    return gathered


@dataclass
class Rebuild:
    session_id: str = ""
    text: str = ""
    digest: str = ""
    chars: int = 0
    budget: int = COMPACTION_BUDGET_CHARS
    ids: list[str] = field(default_factory=list)
    sections: dict[str, int] = field(default_factory=dict)
    dropped: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "text": self.text, "digest": self.digest,
                "chars": self.chars, "budget": self.budget, "ids": list(self.ids),
                "sections": dict(self.sections), "dropped": self.dropped}


def rebuild(store: Any, session_id: str, *, budget: int = COMPACTION_BUDGET_CHARS) -> Rebuild:
    """A block that fits the smaller post-compaction budget, highest priority first."""
    gathered = candidates(store, session_id)
    lines: list[str] = []
    kept_ids: list[str] = []
    sections: dict[str, int] = {}
    dropped = 0
    used = 0
    for name in PRIORITY:
        items = gathered[name]
        if not items:
            continue
        title = {"corrections": "Corrections", "pinned": "Pinned",
                 "used": "Used in this session", "retrieved": "Retrieved earlier"}[name]
        kept = 0
        for item in items:
            line = f"  - {item['text']} [{item['id'][:8]}]"
            if used + len(line) + 1 > budget:
                dropped += 1
                continue
            if kept == 0:
                header = f"{title}:"
                if used + len(header) + 1 + len(line) + 1 > budget:
                    dropped += 1
                    continue
                lines.append(header)
                used += len(header) + 1
            lines.append(line)
            used += len(line) + 1
            kept += 1
            kept_ids.append(item["id"])
        if kept:
            sections[name] = kept
    text = "\n".join(lines)
    return Rebuild(session_id=str(session_id), text=text,
                   digest=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
                   chars=len(text), budget=int(budget), ids=kept_ids, sections=sections,
                   dropped=dropped)


def verify(before: Iterable[str], after: Iterable[str]) -> dict[str, Any]:
    """Did it survive? Names the losses instead of assuming the best."""
    old = [str(item) for item in before]
    new = {str(item) for item in after}
    lost = [item for item in old if item not in new]
    return {
        "before": len(old),
        "after": len(new),
        "survived": len(old) - len(lost),
        "lost": lost,
        "loss_share": round(len(lost) / len(old), 4) if old else None,
        "ok": not lost,
        "note": "an empty 'before' list proves nothing: there was nothing to lose",
    }


def render(store: Any, session_id: str, *, budget: int = COMPACTION_BUDGET_CHARS) -> list[str]:
    described = rebuild(store, session_id, budget=budget)
    lines = [f"compaction rebuild for {session_id[:12]}: {described.chars} of {described.budget} chars, "
             f"digest {described.digest}"]
    if described.sections:
        lines.append("  sections: " + ", ".join(f"{name} {count}"
                                                for name, count in described.sections.items()))
    else:
        lines.append("  nothing to rebuild: this session has no corrections, pins or memories")
    if described.dropped:
        lines.append(f"  dropped {described.dropped} item(s) to fit the post-compaction budget")
    return lines
