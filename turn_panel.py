"""What was injected this turn, in the terminal (#148)

The dashboard answers this with a browser and a 637 KB page; on a box, over SSH, in a tmux pane, the
question is the same and the answer should be a handful of lines. "What did the brain just put in front
of the model" is the fastest way to notice that a turn is running on nothing, or on the wrong three
memories.

Design rules, in the order they matter when you are reading a terminal at 2am:

- **one line per memory.** Kind, confidence, and the first clause of the text. A wrapped paragraph per
  memory turns eight memories into a screen of text nobody reads.
- **the width is the terminal's, not a constant.** `width` defaults to `COLUMNS`, clamped to something
  readable; nothing is cut mid-word.
- **plain text unless colour is asked for.** Piping this into a file, a log, or a chat message should
  produce prose, not escape codes.
- **it says what it does not know.** No trace means "no turn recorded", not an empty panel; a memory
  that is injected but never used says so when a receipt exists (#174), because injected-but-unused is
  the interesting state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_WIDTH = 100
MIN_WIDTH = 40
MAX_WIDTH = 200
CLAUSE = 70

_GREY = "\x1b[2m"
_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"
_RESET = "\x1b[0m"


def terminal_width(default: int = DEFAULT_WIDTH) -> int:
    try:
        columns = int(os.environ.get("COLUMNS") or 0)
    except (TypeError, ValueError):
        columns = 0
    if not columns:
        return default
    return max(MIN_WIDTH, min(MAX_WIDTH, columns))


def clause(text: Any, limit: int = CLAUSE) -> str:
    """The first clause of a memory: enough to recognise it, short enough for one line."""
    body = " ".join(str(text or "").split())
    for stop in (". ", "; ", " — ", ", "):
        index = body.find(stop)
        if 0 < index <= limit:
            return body[:index]
    if len(body) <= limit:
        return body
    cut = body[:limit]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut + "…"


@dataclass
class Panel:
    task_id: str = ""
    recorded: bool = False
    query: str = ""
    reason: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "recorded": self.recorded,
            "query": self.query,
            "reason": self.reason,
            "rows": self.rows,
            "note": self.note,
        }


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def build(store: Any, task_id: str = "", *, limit: int = 8) -> Panel:
    """The memories this turn injected, newest turn first when no id is given."""
    found = Panel(task_id=str(task_id))
    try:
        traces = store.memory_traces(limit=1, task_id=str(task_id)) if task_id \
            else store.memory_traces(limit=1)
    except Exception:  # noqa: BLE001
        traces = []
    if not traces:
        found.note = "no turn recorded yet" if not task_id else "that turn is not recorded"
        return found
    trace = traces[0]
    found.recorded = True
    found.task_id = str(trace.get("task_id") or task_id)
    queries = trace.get("queries") or []
    found.query = str(queries[0]) if queries else ""
    found.reason = str(trace.get("retrieval_reason") or "")
    injected: list[str] = []
    for candidate in trace.get("candidate_memories") or []:
        if not (candidate or {}).get("selected"):
            continue
        memory_id = str((candidate or {}).get("memory_id") or "")
        if memory_id and memory_id not in injected:
            injected.append(memory_id)
    for memory_id in trace.get("selected_memory_ids") or []:
        text = str(memory_id)
        if text and text not in injected:
            injected.append(text)
    used: set[str] = set()
    try:
        receipt = _rows(store, "SELECT cited_ids FROM recall_receipts WHERE turn_id = ?",
                        (found.task_id,))
        if receipt:
            used = {item for item in str(receipt[0]["cited_ids"] or "").split(",") if item}
    except Exception:  # noqa: BLE001
        used = set()
    for memory_id in injected[: max(1, int(limit))]:
        memory = _rows(store, "SELECT id, kind, confidence, content FROM memories WHERE id = ?",
                       (memory_id,))
        if not memory:
            continue
        row = memory[0]
        found.rows.append({
            "memory_id": memory_id,
            "kind": str(row["kind"] or ""),
            "confidence": round(float(row["confidence"] or 0.0), 2),
            "clause": clause(row["content"]),
            "used": memory_id in used if used else None,
        })
    if not injected:
        found.note = "this turn injected no memories"
    return found


def render(panel: Panel, *, width: int = 0, color: bool = False, limit: int = 8) -> list[str]:
    """The panel as terminal lines."""
    span = int(width) or terminal_width()
    if not panel.recorded:
        return [panel.note or "nothing to show"]
    header = f"turn {panel.task_id[:8]} — {len(panel.rows)} injected"
    if panel.query:
        header += f" · “{clause(panel.query, 40)}”"
    lines = [_fit(header, span)]
    if panel.reason:
        lines.append(_dim(_fit(f"  retrieval: {panel.reason}", span), color))
    for row in panel.rows[: int(limit)]:
        mark = "·" if row.get("used") is None else ("✓" if row["used"] else "✗")
        body = f"  {mark} {row['kind'][:10]:10} {row['confidence']:.2f}  {row['clause']}"
        painted = body
        if color:
            if row.get("used") is True:
                painted = f"{_GREEN}{body}{_RESET}"
            elif row.get("used") is False:
                painted = f"{_YELLOW}{body}{_RESET}"
            else:
                painted = f"{_GREY}{body}{_RESET}"
        lines.append(_fit(painted, span))
    hidden = len(panel.rows) - min(len(panel.rows), int(limit))
    if hidden:
        lines.append(_dim(_fit(f"  … {hidden} more injected", span), color))
    if panel.note:
        lines.append(_dim(_fit(f"  note: {panel.note}", span), color))
    return lines


def _dim(text: str, color: bool) -> str:
    return f"{_GREY}{text}{_RESET}" if color else text


def _fit(text: str, width: int) -> str:
    """Trim to the terminal width without cutting through an escape sequence's payload."""
    if len(text) <= width:
        return text
    if "\x1b" in text:
        # Colour is only added around whole lines, so the reset can be moved to the cut safely.
        plain_end = text.rfind(_RESET)
        if plain_end >= 0:
            head = text[: plain_end].rstrip()
            return head[: max(1, width - len(_RESET))] + _RESET
    cut = text[: max(1, width - 1)]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut + "…"


def text(panel: Panel, **kwargs: Any) -> str:
    return "\n".join(render(panel, **kwargs))
