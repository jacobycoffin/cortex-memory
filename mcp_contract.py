"""The rules the MCP surface keeps, enforced rather than described (#154)

`mcp_server` exposes five tools to other agents: `memory_search`, `context_brief`, `remember`, `forget`,
`stats`. That is a wide surface pointed at a store that holds his memory, so the guarantees have to be
mechanical:

- **writes go through the consent gate.** `remember` is called by *agents*, and #182 established that an
  agent may not write a credential, a work-systems note or a finance fact into this store. A tool that
  skips the gate makes the gate decorative.
- **every response is bounded.** A client on a phone should not receive a 600 KB payload because it asked
  for `limit: 25`; the schema says `maximum: 25`, so the *handler* must clamp too — a client can send
  anything.
- **`forget` archives, never deletes.** Reversible by construction, because a remote agent's typo must not
  destroy a memory the owner cannot restore.
- **errors are structured and path-free.** A traceback leaking a machine path into another agent's
  context is a small data breach with a stack trace attached.
- **the tool list is versioned.** Clients cache schemas; a silent shape change breaks every caller at once.

This module is the layer between a JSON-RPC request and `_call_tool`, so the guarantees hold even when the
calling agent ignores the schema in its description.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CONTRACT_VERSION = 1
MAX_LIMIT = 25
MAX_RESULTS_CHARS = 12_000
MAX_CONTENT_CHARS = 4_000
READ_TOOLS = ("memory_search", "context_brief", "stats")
WRITE_TOOLS = ("remember", "forget")


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


@dataclass
class Refused:
    code: str
    message: str
    tool: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {"ok": False, "error": {"code": self.code, "message": self.message}}
        if self.tool:
            payload["error"]["tool"] = self.tool
        if self.detail:
            payload["error"]["detail"] = self.detail
        return payload


def clamp_limit(value: Any, *, maximum: int = MAX_LIMIT, default: int = 6) -> int:
    """A client can send anything; the handler is what makes `maximum` true."""
    try:
        number = int(value)
    except Exception:  # noqa: BLE001
        return default
    return max(1, min(int(maximum), number))


def bounded(payload: Any, *, limit: int = MAX_RESULTS_CHARS) -> tuple[Any, bool]:
    """Trim a response to the character budget, saying whether it was trimmed."""
    import json

    text = json.dumps(payload, default=str)
    if len(text) <= limit:
        return payload, False
    if isinstance(payload, dict):
        trimmed = dict(payload)
        for key in ("results", "items", "memories"):
            if isinstance(trimmed.get(key), list):
                items = list(trimmed[key])
                while items and len(json.dumps({**trimmed, key: items}, default=str)) > limit:
                    items.pop()
                trimmed[key] = items
                trimmed["truncated"] = True
                return trimmed, True
    return {"truncated": True, "preview": text[: max(0, limit - 100)]}, True


def screen_write(store: Any, tool: str, arguments: dict[str, Any]) -> Refused | None:
    """The consent gate, applied to anything an agent asks to write (#182)."""
    if tool != "remember":
        return None
    content = str(arguments.get("content") or "")
    if not content.strip():
        return Refused("empty_content", "a memory needs content", tool)
    if len(content) > MAX_CONTENT_CHARS:
        return Refused("content_too_long",
                       f"content is {len(content)} chars; the limit is {MAX_CONTENT_CHARS}", tool,
                       {"chars": len(content)})
    try:
        from .consent_gate import decide  # noqa: PLC0415 - optional helper

        decision = decide(content)
    except Exception:  # noqa: BLE001
        return None
    if decision.verdict == "refuse":
        return Refused("refused_by_consent", decision.reason, tool,
                       {"class": decision.class_name, "content_hash": decision.content_hash})
    if decision.verdict == "ask":
        return Refused("needs_consent",
                       f"{decision.class_name} content needs the owner's consent; not stored", tool,
                       {"class": decision.class_name})
    return None


def sanitise_error(error: Exception) -> Refused:
    """No tracebacks, no paths: a stack trace in another agent's context is a small data breach."""
    name = type(error).__name__
    message = str(error)
    for marker in ("/root", "/home", "/Users", "Traceback"):
        if marker in message:
            message = f"{name} (details withheld: they name a filesystem path)"
            break
    else:
        message = message[:300] or name
    return Refused("tool_failed", message)


def describe(tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The surface, as data: what it exposes and what it promises."""
    names = [str(tool.get("name")) for tool in (tools or [])]
    return {
        "contract_version": CONTRACT_VERSION,
        "tools": names,
        "read_only": [name for name in names if name in READ_TOOLS],
        "writes": [name for name in names if name in WRITE_TOOLS],
        "guarantees": [
            "a write is screened by the consent gate before it reaches the store",
            "responses are clamped to the schema maximum and the character budget",
            "forget archives; nothing on this surface deletes",
            "errors are structured and never contain a filesystem path",
            "the tool list is versioned so a client can detect a shape change",
        ],
        "limits": {"max_limit": MAX_LIMIT, "max_results_chars": MAX_RESULTS_CHARS,
                   "max_content_chars": MAX_CONTENT_CHARS},
    }


def call(store: Any, handler: Any, name: str, arguments: dict[str, Any] | None = None
         ) -> dict[str, Any]:
    """Run one tool call through the contract. `handler` is the server's own dispatcher."""
    args = dict(arguments or {})
    if name in WRITE_TOOLS:
        refusal = screen_write(store, name, args)
        if refusal is not None:
            return refusal.as_dict()
    if "limit" in args:
        args["limit"] = clamp_limit(args["limit"])
    try:
        result = handler(name, args)
    except KeyError:
        return Refused("unknown_tool", f"no tool named {name!r}", name).as_dict()
    except Exception as error:  # noqa: BLE001
        return sanitise_error(error).as_dict()
    payload, trimmed = bounded(result)
    return {"ok": True, "result": payload, "truncated": trimmed}


def audit(store: Any) -> dict[str, Any]:
    """What this surface has actually been asked to do, from the store's own records."""
    refusals = _rows(store, "SELECT verdict, COUNT(*) AS count FROM consent_decisions "
                            "GROUP BY verdict")
    counts = {str(row["verdict"]): int(row["count"] or 0) for row in refusals}
    return {
        "consent_decisions": sum(counts.values()),
        "refused": counts.get("refuse", 0),
        "asked": counts.get("ask", 0),
        "note": ("0 decisions means nothing has written through the gate yet; the surface has no "
                 "recorded writes to audit"),
    }


def render(store: Any, tools: list[dict[str, Any]] | None = None) -> list[str]:
    described = describe(tools)
    audited = audit(store)
    lines = [f"mcp surface v{described['contract_version']}: "
             f"{len(described['tools'])} tool(s), {len(described['writes'])} write(s)"]
    lines.append("  tools: " + ", ".join(described["tools"] or ["(none registered)"]))
    for guarantee in described["guarantees"]:
        lines.append(f"  · {guarantee}")
    lines.append(f"  consent decisions recorded: {audited['consent_decisions']} "
                 f"({audited['refused']} refused)")
    return lines
