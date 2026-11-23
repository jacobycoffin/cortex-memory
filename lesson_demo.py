"""A lesson is only learned if you can show the question it changes (#211).

Every change to memory behaviour makes a claim: *"this query now answers better"*. The claim is
worthless on its own — it is a description, and descriptions are what a confident system produces
whether or not anything improved. What makes it checkable is the **question**:

    Q: "which port does the dashboard run on?"
    before: Port 8100 (staging), Port 8100 — the staging copy surfaced alongside the real one
    after : Port 8100 — the near-miss pair now keeps only the stronger

So a demonstration is a diff of answers, drawn from real recorded turns rather than from invented
examples:

1. take a stored recall (`recall_replay` keeps the exact block and ids that were injected, #24);
2. re-run the same query against the same store, today;
3. report only the turns where the **selected set changed**, with what left and what arrived.

What it cannot tell you is whether the new answer is *right* — that requires labelled cases, which is
a different exercise (#207). It answers the narrower, honest question: **did this change anything, and
where?** A demonstration with no differing questions is a change nobody can see, and that is worth
learning before shipping rather than after.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_LIMIT = 20


def _content(store: Any, memory_id: str, *, cap: int = 70) -> str:
    try:
        row = store._conn.execute(
            "SELECT content FROM memories WHERE id = ?", (str(memory_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001
        return memory_id[:8]
    if not row:
        return memory_id[:8]
    body = " ".join(str(row["content"]).split())
    return body[: cap - 1].rstrip() + "…" if len(body) > cap else body


def _from_traces(store: Any, limit: int) -> list[dict[str, Any]]:
    """Real turns that ran: the trace records what was selected, and the run records the query."""
    try:
        rows = store._conn.execute(
            """SELECT t.task_id, t.session_id, t.selected_memory_ids_json, t.created_at,
                      COALESCE(runs.query, '') AS query,
                      COALESCE(runs.requested_limit, 6) AS requested_limit,
                      COALESCE(runs.token_budget, 700) AS token_budget
               FROM memory_traces t
               LEFT JOIN recall_runs runs ON runs.task_id = t.task_id
               WHERE t.selected_memory_ids_json NOT IN ('', '[]')
                 AND COALESCE(runs.query, '') != ''
               ORDER BY t.created_at DESC LIMIT ?""",
            (int(limit),),
        ).fetchall()
    except Exception:  # noqa: BLE001 - traces are optional too
        return []
    return _rows(rows)


def recorded(store: Any, *, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    """Recent turns that were actually injected, with the query that caused them.

    Two sources, because the store may have either: `memory_traces` records every recall that ran,
    while `recall_replay` (#24) keeps the exact injected block. Traces are enough for a before/after
    diff, so they are tried first; replay is the fallback for stores that kept only the block.
    """
    traces = _from_traces(store, limit)
    if traces:
        return traces
    try:
        rows = store._conn.execute(
            """SELECT r.task_id, r.session_id, r.memory_ids_json, r.created_at,
                      COALESCE(runs.query, '') AS query
               FROM recall_replay r
               LEFT JOIN recall_runs runs ON runs.task_id = r.task_id
               WHERE COALESCE(runs.query, '') != ''
               ORDER BY r.created_at DESC LIMIT ?""",
            (int(limit),),
        ).fetchall()
    except Exception:  # noqa: BLE001 - replay is optional
        return []
    return _rows(rows)


def _rows(rows: Any) -> list[dict[str, Any]]:
    import json

    out: list[dict[str, Any]] = []
    for row in rows:
        raw = row["memory_ids_json"] if "memory_ids_json" in row.keys() \
            else row["selected_memory_ids_json"]
        try:
            ids = json.loads(raw or "[]")
        except (TypeError, ValueError):
            ids = []
        if not str(row["query"]).strip():
            continue
        out.append(
            {
                "task_id": str(row["task_id"]),
                "session_id": str(row["session_id"] or ""),
                "query": str(row["query"]),
                "memory_ids": [str(item) for item in ids],
                "created_at": str(row["created_at"]),
                "limit": int(row["requested_limit"]) if "requested_limit" in row.keys() else 6,
                "token_budget": (int(row["token_budget"])
                                 if "token_budget" in row.keys() else 700),
            }
        )
    return out


@dataclass
class Demonstration:
    query: str
    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    arrived: list[str] = field(default_factory=list)
    left: list[str] = field(default_factory=list)
    basis: str = "drift"

    @property
    def changed(self) -> bool:
        return bool(self.arrived or self.left)

    def describe(self, store: Any) -> str:
        if not self.changed:
            return f'Q: "{self.query}" — the answer is unchanged'
        header = "Q" if self.basis == "baseline" else "Q (drift, not a demonstration)"
        lines = [f'{header}: "{self.query}"']
        for memory_id in self.left:
            lines.append(f"  left  ({memory_id[:8]}): {_content(store, memory_id)}")
        for memory_id in self.arrived:
            lines.append(f"  now   ({memory_id[:8]}): {_content(store, memory_id)}")
        return "\n".join(lines)

    def as_dict(self, store: Any) -> dict[str, Any]:
        return {
            "query": self.query,
            "changed": self.changed,
            "arrived": [{"id": mid, "content": _content(store, mid)} for mid in self.arrived],
            "left": [{"id": mid, "content": _content(store, mid)} for mid in self.left],
            "sentence": self.describe(store),
        }


def _now_selection(store: Any, query: str, *, limit: int = 6, session_id: str = "",
                   token_budget: int = 700) -> list[str]:
    """Re-run under the same limits and session as the recorded turn.

    Comparing a six-memory recall with a fresh one that ignores the session would make almost every
    turn "differ" for reasons that have nothing to do with the change under test.
    """
    try:
        from .retrieval import MemoryRetriever, RetrievalContext

        context = RetrievalContext(scope={"session_id": session_id} if session_id else {})
        results = MemoryRetriever(store, threshold=0.0).search(
            query, limit=limit, token_budget=token_budget, context=context
        )
    except Exception:  # noqa: BLE001 - a replay that cannot run is not a demonstration
        return []
    return [str(result.memory["id"]) for result in results]


def compare(store: Any, turn: dict[str, Any], *, limit: int = 6,
            baseline: Any | None = None) -> Demonstration:
    """What the same query selects today, against what was injected then.

    With `baseline` — a copy of the store from before the change — the two selections come from the
    same code and differ only by the change: that is a demonstration. Without it the comparison is
    against a **recorded** selection made by a differently configured retriever, months of new
    memories ago, so a difference is drift and nothing more. The `basis` field says which one you are
    looking at, because a tool that reports drift as causation is the thing this item exists to stop.
    """
    if baseline is not None:
        options = dict(limit=int(turn.get("limit") or limit),
                       session_id=str(turn.get("session_id") or ""),
                       token_budget=int(turn.get("token_budget") or 700))
        before = _now_selection(baseline, str(turn["query"]), **options)
        after = _now_selection(store, str(turn["query"]), **options)
        basis = "baseline"
    else:
        before = [str(item) for item in turn.get("memory_ids") or []]
        after = _now_selection(
            store,
            str(turn["query"]),
            limit=int(turn.get("limit") or limit),
            session_id=str(turn.get("session_id") or ""),
            token_budget=int(turn.get("token_budget") or 700),
        )
        basis = "drift"
    demo = Demonstration(query=str(turn["query"]), before=before, after=after, basis=basis)
    if after:
        demo.arrived = [item for item in after if item not in before]
        demo.left = [item for item in before if item not in after]
    return demo


def demonstrations(store: Any, *, limit: int = DEFAULT_LIMIT, only_changed: bool = True,
                   baseline: Any | None = None) -> list[Demonstration]:
    out: list[Demonstration] = []
    for turn in recorded(store, limit=limit):
        demo = compare(store, turn, baseline=baseline)
        if only_changed and not demo.changed:
            continue
        out.append(demo)
    return out


def report(store: Any, *, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    turns = recorded(store, limit=limit)
    demos = demonstrations(store, limit=limit, only_changed=False)
    changed = [demo for demo in demos if demo.changed]
    return {
        "turns": len(turns),
        "compared": len(demos),
        "changed": len(changed),
        "examples": [demo.describe(store) for demo in changed[:3]],
    }
