"""Link a memory to the tool call it informed (#70).

The store already records every tool execution (`tool_executions`) and every memory it selected, and
until now those two facts sat in different tables with nothing joining them. So the store could not
answer the only question that matters about a procedure: *when this memory was in front of the model,
did the call work?*

That join is this module. A link says one thing — **this memory was used to inform this call** — and it
is deliberately narrow:

- **not a claim about causality.** The memory was in context and the call happened; whether the call
  succeeded *because of* it is a separate question (#71), and conflating the two is how a store starts
  believing its own citation.
- **both sides must exist.** A link to a memory id or an execution id that is not in the store is
  refused, because a dangling link is a fiction that looks like data.
- **one link per pair.** Re-linking the same pair is a no-op, not a duplicate; a memory that informed
  forty calls has forty links, not forty copies of one.

The measured output is per tool: how many calls were informed by a memory, and what share of them
succeeded. An untested procedure looks identical to a good one until it is joined to its outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

TABLE = "memory_tool_links"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    memory_id TEXT NOT NULL,
                    execution_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (memory_id, execution_id)
                )"""
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {TABLE}_tool_idx ON {TABLE} (tool_name)"
        )


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _exists(store: Any, table: str, key: str, value: str) -> bool:
    rows = _rows(store, f"SELECT 1 AS found FROM {table} WHERE {key} = ? LIMIT 1", (str(value),))
    return bool(rows)


@dataclass
class Use:
    memory_id: str
    execution_id: str
    tool_name: str
    success: bool
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "execution_id": self.execution_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "created_at": self.created_at,
        }

    def describe(self) -> str:
        outcome = "worked" if self.success else "failed"
        return f"{self.tool_name or 'a tool'} {outcome} with {self.memory_id[:8]} in context"


def link(store: Any, memory_id: str, execution_id: str, *, tool_name: str = "",
         at: datetime | None = None) -> dict[str, Any]:
    """Record that a memory was in context for a call. Both sides must already exist."""
    memory = str(memory_id or "").strip()
    execution = str(execution_id or "").strip()
    if not memory or not execution:
        return {"linked": False, "reason": "a link needs a memory and an execution"}
    if not _exists(store, "memories", "id", memory):
        return {"linked": False, "reason": f"unknown memory {memory[:12]}"}
    if not _exists(store, "tool_executions", "execution_id", execution):
        return {"linked": False, "reason": f"unknown execution {execution[:12]}"}
    name = str(tool_name or "").strip()
    if not name:
        rows = _rows(
            store, "SELECT tool_name FROM tool_executions WHERE execution_id = ?", (execution,)
        )
        name = str(rows[0]["tool_name"]) if rows else ""
    _ensure(store)
    moment = (at or _now()).isoformat()
    with store.transaction() as conn:
        cursor = conn.execute(
            f"""INSERT OR IGNORE INTO {TABLE} (memory_id, execution_id, tool_name, created_at)
                VALUES (?, ?, ?, ?)""",
            (memory, execution, name, moment),
        )
        created = int(cursor.rowcount or 0)
    return {"linked": True, "created": bool(created), "tool_name": name}


def link_many(store: Any, memory_ids: Iterable[str], execution_id: str, *,
              at: datetime | None = None) -> dict[str, Any]:
    """A turn used several memories; link all of them to the call they informed."""
    linked = 0
    refused: list[str] = []
    for memory_id in memory_ids:
        outcome = link(store, memory_id, execution_id, at=at)
        if outcome.get("linked"):
            linked += 1
        else:
            refused.append(str(outcome.get("reason") or ""))
    return {"linked": linked, "refused": refused}


def for_memory(store: Any, memory_id: str, *, limit: int = 20) -> list[Use]:
    """Which calls this memory was in context for, and how they went."""
    _ensure(store)
    rows = _rows(
        store,
        f"""SELECT l.memory_id, l.execution_id, l.tool_name, l.created_at,
                   COALESCE(e.success, 0) AS success
            FROM {TABLE} l LEFT JOIN tool_executions e ON e.execution_id = l.execution_id
            WHERE l.memory_id = ? ORDER BY l.created_at DESC LIMIT ?""",
        (str(memory_id), int(limit)),
    )
    return [
        Use(memory_id=str(row["memory_id"]), execution_id=str(row["execution_id"]),
            tool_name=str(row["tool_name"] or ""), success=bool(row["success"]),
            created_at=str(row["created_at"] or ""))
        for row in rows
    ]


def for_tool(store: Any, tool_name: str, *, limit: int = 50) -> list[str]:
    """Memories that have informed this tool, most recently used first."""
    _ensure(store)
    rows = _rows(
        store,
        f"""SELECT memory_id, MAX(created_at) AS last FROM {TABLE}
            WHERE tool_name = ? GROUP BY memory_id ORDER BY last DESC LIMIT ?""",
        (str(tool_name), int(limit)),
    )
    return [str(row["memory_id"]) for row in rows]


def by_tool(store: Any, *, limit: int = 50) -> dict[str, dict[str, Any]]:
    """Per tool: how many calls a memory informed, and what share of them succeeded."""
    _ensure(store)
    rows = _rows(
        store,
        f"""SELECT l.tool_name AS tool_name,
                   COUNT(*) AS links,
                   COUNT(DISTINCT l.execution_id) AS calls,
                   SUM(COALESCE(e.success, 0)) AS succeeded
            FROM {TABLE} l LEFT JOIN tool_executions e ON e.execution_id = l.execution_id
            GROUP BY l.tool_name ORDER BY calls DESC LIMIT ?""",
        (int(limit),),
    )
    detail: dict[str, dict[str, Any]] = {}
    for row in rows:
        calls = int(row["calls"] or 0)
        succeeded = int(row["succeeded"] or 0)
        detail[str(row["tool_name"] or "")] = {
            "links": int(row["links"] or 0),
            "calls": calls,
            "succeeded": succeeded,
            "success_share": round(succeeded / calls, 4) if calls else 0.0,
        }
    return detail


def report(store: Any) -> dict[str, Any]:
    _ensure(store)
    totals = _rows(
        store,
        f"""SELECT COUNT(*) AS links, COUNT(DISTINCT memory_id) AS memories,
                   COUNT(DISTINCT tool_name) AS tools FROM {TABLE}""",
    )
    row = totals[0] if totals else {}
    return {
        "links": int(row.get("links") or 0),
        "memories": int(row.get("memories") or 0),
        "tools": int(row.get("tools") or 0),
        "by_tool": by_tool(store, limit=10),
        "note": "a link says the memory was in context, not that it caused the outcome",
    }
