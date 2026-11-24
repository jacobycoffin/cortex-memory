"""Terminal memory panel: show what Cortex injected on a turn.

The trace already records every candidate, its score, its token estimate and why it
was selected or rejected. The dashboard shows that in a browser; this shows it where
the work happens, in one screen, read-only:

    cortex-memory panel              # the most recent turn
    cortex-memory panel --limit 3    # the last three turns
    cortex-memory panel --all        # every excluded candidate, not just the top few

Nothing here retrieves, writes, or re-ranks. It renders ``store.memory_traces()``.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from typing import Any, Sequence

MAX_WIDTH = 120
MIN_WIDTH = 48
GOAL_CHARS = 180
PREVIEW_CHARS = 150
INDENT = "     "

ANSI = {
    "reset": "\x1b[0m",
    "dim": "\x1b[2m",
    "bold": "\x1b[1m",
    "cyan": "\x1b[36m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
}


def _paint(text: str, style: str, color: bool) -> str:
    if not color or not text:
        return text
    return f"{ANSI.get(style, '')}{text}{ANSI['reset']}"


def _clip(text: Any, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    if limit <= 1:
        return value[:limit]
    return value[: limit - 1].rstrip() + "…"


def _local_time(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "--:--:--"
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw[:8]
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    return parsed.strftime("%H:%M:%S %Z").strip()


def _short_id(value: Any, length: int = 8) -> str:
    return str(value or "").strip()[:length] or "--------"


def _influence_map(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in trace.get("influence") or []:
        if isinstance(item, dict) and item.get("memory_id"):
            out[str(item["memory_id"])] = item
    return out


def _candidates(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [c for c in (trace.get("candidate_memories") or []) if isinstance(c, dict)]
    selected = [c for c in candidates if c.get("selected")]
    excluded = [c for c in candidates if not c.get("selected")]
    selected.sort(key=lambda c: (int(c.get("rank") or 0), -float(c.get("score") or 0.0)))
    excluded.sort(key=lambda c: (int(c.get("rank") or 0), -float(c.get("score") or 0.0)))
    return selected, excluded


def _candidate_head(candidate: dict[str, Any], width: int, *, with_tokens: bool) -> str:
    """One line naming a candidate, degrading gracefully on narrow terminals."""

    budget = max(10, width - 1)
    rank = int(candidate.get("rank") or 0)
    score = float(candidate.get("score") or 0.0)
    memory_id = _short_id(candidate.get("memory_id"))
    kind = _clip(candidate.get("kind") or "memory", 9)
    compact = f"{rank:>2}  {memory_id}  {score:.3f}"
    full = f"{rank:>2}  {memory_id}  {kind:<9}  score {score:.3f}"
    if with_tokens:
        tokens = int(candidate.get("estimated_tokens") or 0)
        full += f"  ~{tokens:>4} tok"
    value = full if len(full) <= budget else compact
    return " " + _clip(value, budget)


def _rejection_reason(candidate: dict[str, Any], width: int) -> str:
    reason = _clip(candidate.get("reason"), max(24, width - len(INDENT)))
    if reason and reason != "unspecified decision":
        return reason
    components = candidate.get("components") or {}
    if float(components.get("context_gate", 1.0) or 0.0) < 1.0:
        return "held back by the context gate"
    return "not selected"


def _meta_line(candidate: dict[str, Any], width: int) -> str:
    meta = candidate.get("metacognition")
    if not isinstance(meta, dict) or not meta:
        return ""
    decision = str(meta.get("decision") or "").strip()
    reason = _clip(meta.get("reason"), 120)
    probability = meta.get("calibrated_probability")
    bits = []
    if decision:
        bits.append(decision)
    if isinstance(probability, (int, float)):
        bits.append(f"p={float(probability):.2f}")
    if reason:
        bits.append(reason)
    if meta.get("applied") is False:
        bits.append("(advice only, gate off)")
    return _clip(" · ".join(bits), max(20, width - len(INDENT) - len("judge  ")))


def render_memory_panel(
    traces: Sequence[dict[str, Any]],
    *,
    width: int | None = None,
    color: bool = False,
    show_all_excluded: bool = False,
    max_excluded: int = 4,
) -> str:
    """Render one text panel per trace, newest first. Pure function, no I/O."""

    if width is None:
        width = shutil.get_terminal_size((MAX_WIDTH, 24)).columns
    width = max(MIN_WIDTH, min(MAX_WIDTH, int(width)))
    rule = "─" * width

    if not traces:
        return "Cortex memory panel: no turns recorded in this database yet."

    blocks: list[str] = []
    for trace in traces:
        selected, excluded = _candidates(trace)
        title = _paint("CORTEX · MEMORY PANEL", "bold", color)
        title_plain = "CORTEX · MEMORY PANEL"
        head_tail = (
            f"turn {_local_time(trace.get('created_at'))} · trace "
            f"{_short_id(trace.get('trace_id'))} · {trace.get('recall_mode') or 'unknown'} mode"
        )
        if len(title_plain) + len(head_tail) + 2 <= width:
            gap = width - len(title_plain) - len(head_tail)
            lines = [f"{title}{' ' * gap}{_paint(head_tail, 'dim', color)}"]
        else:
            lines = [title, _paint(_clip(head_tail, width), "dim", color)]

        goal = _clip(trace.get("goal"), min(GOAL_CHARS, max(24, width - len("goal   "))))
        if goal:
            lines.append(f"{_paint('goal   ', 'dim', color)}{goal}")

        tokens = sum(int(c.get("estimated_tokens") or 0) for c in selected)
        if selected:
            summary = (
                f"injected {len(selected)} of {len(selected) + len(excluded)} candidates"
                f" · ~{tokens} tok · {trace.get('outcome') or 'pending'}"
            )
        else:
            summary = (
                f"nothing injected · {len(excluded)} candidate"
                f"{'s' if len(excluded) != 1 else ''} considered"
                f" · {trace.get('retrieval_reason') or trace.get('outcome') or 'no reason recorded'}"
            )
        lines.append(_paint(_clip(summary, width), "dim", color))
        lines.append(rule)

        influence = _influence_map(trace)
        if selected:
            lines.append(_paint("INJECTED", "green", color))
            for candidate in selected:
                memory_id = str(candidate.get("memory_id") or "")
                lines.append(_paint(_candidate_head(candidate, width, with_tokens=True), "cyan", color))
                preview = _clip(
                    candidate.get("content_preview"),
                    min(PREVIEW_CHARS, max(24, width - len(INDENT))),
                )
                lines.append(f"{INDENT}{preview or '(no preview stored)'}")
                meta = _meta_line(candidate, width)
                if meta:
                    lines.append(f"{INDENT}{_paint('judge  ', 'dim', color)}{meta}")
                used = influence.get(memory_id)
                if used:
                    if used.get("influenced"):
                        attribution = used.get("attribution_score")
                        suffix = (
                            f" ({float(attribution):.2f})"
                            if isinstance(attribution, (int, float))
                            else ""
                        )
                        lines.append(
                            f"{INDENT}{_paint('used   ', 'green' if color else 'dim', color)}"
                            f"cited in the answer{suffix}"
                        )
                    else:
                        lines.append(
                            f"{INDENT}{_paint('used   ', 'yellow' if color else 'dim', color)}"
                            "retrieved, no answer-use evidence"
                        )
        else:
            lines.append(_paint("NOTHING INJECTED", "yellow", color))
            reason = _clip(trace.get("retrieval_reason"), max(24, width - len(INDENT)))
            if reason:
                lines.append(f"{INDENT}{reason}")

        if excluded:
            shown = excluded if show_all_excluded else excluded[:max_excluded]
            lines.append("")
            lines.append(
                _paint(
                    f"EXCLUDED ({len(excluded)})",
                    "yellow" if color else "dim",
                    color,
                )
            )
            for candidate in shown:
                lines.append(
                    _paint(_candidate_head(candidate, width, with_tokens=False), "dim", color)
                )
                lines.append(f"{INDENT}{_rejection_reason(candidate, width)}")
                preview = _clip(
                    candidate.get("content_preview"),
                    min(PREVIEW_CHARS, max(24, width - len(INDENT))),
                )
                if preview:
                    lines.append(_paint(f"{INDENT}{preview}", "dim", color))
            hidden = len(excluded) - len(shown)
            if hidden > 0:
                lines.append(
                    _paint(
                        _clip(f" … {hidden} more excluded (--all to show)", width), "dim", color
                    )
                )

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
