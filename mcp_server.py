"""Standalone stdio MCP server for Cortex (no Hermes required).

Run it with ``python -m cortex.mcp_server`` (or the ``cortex-mcp`` console
script). It speaks MCP over stdin/stdout using newline-delimited JSON-RPC 2.0
and needs no third-party MCP SDK.

Storage lives under ``CORTEX_HOME`` (default ``~/.cortex``):

* ``$CORTEX_HOME/cortex.db``                    -- the memory database
* ``$CORTEX_HOME/models/bge-small-en-v1.5``     -- optional local ONNX embedder
* ``$CORTEX_HOME/backups``                      -- ``cortex-mcp backup`` output

Fast recall on this path is on by default:

* the embedder is loaded in a background thread at startup, so the first
  query does not pay model-load latency;
* memories that lack an embedding are backfilled in small background batches;
* recall results are cached per (query, context, store revision) for
  ``CORTEX_CACHE_TTL_SECONDS`` (default 45 s);
* ``context_brief`` with a task runs the same recall a default
  ``memory_search`` would, so the follow-up search is a cache hit;
* the ``preload`` helper (``CortexService.preload``) uses the bounded
  BackgroundPreloader; ``CORTEX_PRELOAD=0`` disables it.

Semantic fusion uses weight ``CORTEX_SEMANTIC_WEIGHT`` (default 10, the
measured LoCoMo setting) when the model is present; with no model, recall is
lexical + graph only and still works.

Optional remote components (the auto-judge and its Jev engine) are never
started by this server.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

SERVER_NAME = "cortex"
# Same kinds the Hermes cortex_memory tool accepts.
MEMORY_KINDS = ("semantic", "preference", "decision", "procedure", "identity", "prospective", "operational")
DEFAULT_PROTOCOL = "2025-06-18"
MODEL_SUBDIR = Path("models") / "bge-small-en-v1.5"


def cortex_home() -> Path:
    raw = os.environ.get("CORTEX_HOME", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".cortex"


def _configure_model_dir(home: Path) -> Path:
    model_dir = Path(os.environ.get("CORTEX_MODEL_DIR", "").strip() or home / MODEL_SUBDIR).expanduser()
    os.environ["CORTEX_MODEL_DIR"] = str(model_dir)
    return model_dir


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _runtime_available() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(name) is not None for name in ("onnxruntime", "tokenizers", "numpy"))


def _log(message: str) -> None:
    if os.environ.get("CORTEX_MCP_QUIET", "") not in {"1", "true"}:
        print(f"[cortex-mcp] {message}", file=sys.stderr, flush=True)


class CortexService:
    """Thin, thread-safe wrapper over CortexMemory for MCP and hooks."""

    def __init__(self, home: Path | None = None, *, warm: bool = True):
        self.home = (home or cortex_home()).expanduser()
        self.home.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.home, 0o700)
        except OSError:
            pass
        self.model_dir = _configure_model_dir(self.home)
        from .client import CortexMemory
        from .embeddings import get_embedder

        self.db_path = Path(os.environ.get("CORTEX_DB", "").strip() or self.home / "cortex.db").expanduser()
        self.memory = CortexMemory(self.db_path, cache_ttl_seconds=_env_float("CORTEX_CACHE_TTL_SECONDS", 45))
        self.store = self.memory.store
        self.embedder = get_embedder()
        weight = _env_float("CORTEX_SEMANTIC_WEIGHT", 10.0)
        self.semantic_enabled = bool(weight > 0 and self.embedder.available and _runtime_available())
        if self.semantic_enabled:
            self.memory.retriever.semantic_weight = weight
        self.preload_enabled = os.environ.get("CORTEX_PRELOAD", "1").strip().lower() not in {"0", "false", "no", "off"}
        self.backfilled = 0
        self.warm_ms: float | None = None
        self._warm_thread: threading.Thread | None = None
        if warm and self.semantic_enabled:
            self._warm_thread = threading.Thread(target=self._warm, name="cortex-warm", daemon=True)
            self._warm_thread.start()

    # ------------------------------------------------------------ embeddings
    def _embed_ids(self, ids: list[str]) -> int:
        if not ids or not self.semantic_enabled:
            return 0
        rows = self.store.get_memories(ids)
        texts = [str(row.get("content") or "") for row in rows]
        vectors = self.embedder.embed(texts)
        count = 0
        for row, vector in zip(rows, vectors):
            if vector:
                self.store.set_memory_embedding(str(row["id"]), self.embedder.model_id, vector)
                count += 1
        return count

    def _warm(self) -> None:
        started = time.perf_counter()
        try:
            self.embedder.embed_one("warm up")
            self.warm_ms = round((time.perf_counter() - started) * 1000, 1)
            while True:
                ids = self.store.memory_ids_missing_embeddings(self.embedder.model_id, limit=64)
                if not ids:
                    break
                done = self._embed_ids(ids)
                self.backfilled += done
                if done == 0:
                    break
        except Exception as exc:  # noqa: BLE001 - background warmup never kills the server
            _log(f"background warmup stopped: {type(exc).__name__}: {exc}")

    def wait_warm(self, timeout: float = 30.0) -> None:
        if self._warm_thread is not None:
            self._warm_thread.join(timeout)

    # ------------------------------------------------------------ operations
    def _context(self, query: str, project: str | None):
        from .retrieval import RetrievalContext

        return RetrievalContext(active_project=project or None, goal=query,
                                scope={"task_type": "general"})

    def search(self, query: str, *, limit: int = 6, project: str | None = None,
               include_archived: bool = False, token_budget: int = 900) -> list[dict[str, Any]]:
        query = " ".join(str(query or "").split())[:2000]
        if not query:
            return []
        results, _diag = self.memory.retriever.search_cached(
            query,
            limit=max(1, min(int(limit), 25)),
            token_budget=max(100, min(int(token_budget), 4000)),
            include_archived=bool(include_archived),
            temporal_mode="current",
            as_of=None,
            graph_depth=1,
            threshold=None,
            context=self._context(query, project),
        )
        out = []
        for result in results:
            record = result.as_dict()
            out.append({
                "id": record["id"],
                "content": record["content"],
                "kind": record["kind"],
                "score": round(float(record.get("score", result.score)), 4),
                "source": record.get("source_category"),
                "state": record.get("state"),
                "updated_at": result.memory.get("updated_at"),
            })
        return out

    def preload(self, query: str, project: str | None = None) -> bool:
        if not self.preload_enabled or not query.strip():
            return False
        try:
            return bool(self.memory.preload(query, active_project=project, token_budget=900))
        except Exception:  # noqa: BLE001
            return False

    def core_memories(self, limit: int = 8) -> list[dict[str, Any]]:
        with self.store._lock:  # noqa: SLF001 - read-only summary query
            rows = self.store._conn.execute(  # noqa: SLF001
                """SELECT id, kind, content, pinned, importance, updated_at FROM memories
                   WHERE state='active'
                   ORDER BY pinned DESC, importance DESC, updated_at DESC LIMIT ?""",
                (max(1, min(int(limit), 30)),),
            ).fetchall()
        return [{"id": r["id"], "kind": r["kind"], "content": r["content"], "pinned": bool(r["pinned"])} for r in rows]

    def brief(self, task: str = "", *, project: str | None = None, limit: int = 8) -> str:
        lines: list[str] = []
        core = self.core_memories(limit=min(limit, 8))
        # Same options as a default memory_search, so this also warms the
        # recall cache for the agent's follow-up search on the same task.
        relevant = self.search(task, limit=6, project=project) if task.strip() else []
        seen: set[str] = set()
        if relevant:
            lines.append("Relevant memories (fallible evidence; verify before acting):")
            for item in relevant:
                seen.add(item["id"])
                lines.append(f"- [{item['id'][:8]}] ({item['kind']}) {item['content']}")
        rest = [item for item in core if item["id"] not in seen]
        if rest:
            lines.append("Core memories:")
            for item in rest:
                flag = " [pinned]" if item["pinned"] else ""
                lines.append(f"- [{item['id'][:8]}] ({item['kind']}){flag} {item['content']}")
        if not lines:
            return "Cortex has no stored memories yet."
        return "\n".join(lines)

    def remember(self, content: str, *, kind: str = "semantic", source: str = "agent",
                 importance: float = 0.6, pinned: bool = False, project: str | None = None,
                 entities: list[str] | None = None, session_id: str | None = None) -> dict[str, Any]:
        text = str(content or "").strip()
        if not text:
            raise ValueError("content is required")
        if kind not in MEMORY_KINDS:
            kind = "semantic"
        category = "USER_EXPLICIT" if source == "user" else "AGENT_INFERENCE"
        memory_id, created = self.memory.remember(
            text,
            kind=kind,
            source_type="mcp",
            source_category=category,
            session_id=session_id,
            confidence=0.85 if source == "user" else 0.65,
            importance=max(0.0, min(1.0, float(importance))),
            pinned=bool(pinned),
            scope={"project": project} if project else None,
            entities=[str(e) for e in (entities or [])][:20],
            source_context="cortex MCP remember",
        )
        embedded = False
        if self.semantic_enabled and memory_id:
            try:
                embedded = self._embed_ids([memory_id]) == 1
            except Exception:  # noqa: BLE001
                embedded = False
        stored = self.store.get_memory(memory_id) if memory_id else None
        return {
            "id": memory_id,
            "created": bool(created),
            "state": (stored or {}).get("state"),
            "quarantined": bool((stored or {}).get("state") == "quarantined"),
            "embedded": embedded,
        }

    def forget(self, memory_id: str, reason: str = "forget") -> dict[str, Any]:
        resolved = self.store.resolve_id(str(memory_id or ""))
        if not resolved:
            return {"archived": False, "error": "no unambiguous memory matches that id"}
        changed = bool(self.store.set_state(resolved, "archived", reason=str(reason or "forget")[:200]))
        return {"id": resolved, "archived": changed, "reversible": True}

    def stats(self) -> dict[str, Any]:
        raw = self.store.stats()
        keep = {k: raw[k] for k in ("memories", "edges", "episodes", "recall_runs",
                                    "pending_memory_creation_proposals") if k in raw}
        keep.update({
            "states": raw.get("states") or raw.get("by_state"),
            "cortex_home": str(self.home),
            "db_path": str(self.db_path),
            "semantic_recall": self.semantic_enabled,
            "model_dir": str(self.model_dir),
            "embeddings": self.store.count_memory_embeddings(self.embedder.model_id),
            "embedder_warm_ms": self.warm_ms,
            "backfilled_this_session": self.backfilled,
            "preload": self.preload_enabled,
            "preload_stats": self.memory.preload_stats(),
        })
        return keep

    def backup(self, target: Path | None = None) -> Path:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        dest = (target or self.home / "backups" / f"cortex-{stamp}.db").expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self.store._lock:  # noqa: SLF001
            dst = sqlite3.connect(str(dest))
            try:
                self.store._conn.backup(dst)  # noqa: SLF001
            finally:
                dst.close()
        os.chmod(dest, 0o600)
        return dest

    def close(self) -> None:
        self.memory.close()


# ---------------------------------------------------------------- MCP tools
TOOLS: list[dict[str, Any]] = [
    {
        "name": "memory_search",
        "description": (
            "Search Cortex long-term memory. Call before answering anything that may depend on "
            "earlier sessions, user preferences, project decisions, or environment facts. Results are "
            "fallible evidence with ids; cite or verify them."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you need to know, in natural language."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 6},
                "project": {"type": "string", "description": "Optional project name to bias recall."},
                "include_archived": {"type": "boolean", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "context_brief",
        "description": (
            "Compact briefing from Cortex: pinned/important memories plus memories relevant to the "
            "current task. Call once at the start of a session or task."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Optional description of the current task."},
                "project": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
            },
        },
    },
    {
        "name": "remember",
        "description": (
            "Store one durable, self-contained memory (a decision, preference, procedure, or fact that "
            "will matter in a future session). Never store secrets, passwords, tokens, or keys; store "
            "where they live instead. Set source='user' only when the user explicitly asked you to remember it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "kind": {"type": "string", "enum": list(MEMORY_KINDS), "default": "semantic"},
                "source": {"type": "string", "enum": ["agent", "user"], "default": "agent"},
                "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.6},
                "pinned": {"type": "boolean", "default": False},
                "project": {"type": "string"},
                "entities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["content"],
        },
    },
    {
        "name": "forget",
        "description": "Archive a memory by id or unambiguous id prefix (reversible; history is kept).",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["id"],
        },
    },
    {
        "name": "stats",
        "description": "Cortex health: memory counts, database path, whether semantic recall and preload are active.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _call_tool(service: CortexService, name: str, args: dict[str, Any]) -> Any:
    if name == "memory_search":
        return {"results": service.search(str(args.get("query") or ""), limit=int(args.get("limit") or 6),
                                          project=args.get("project"),
                                          include_archived=bool(args.get("include_archived", False)))}
    if name == "context_brief":
        task = str(args.get("task") or "")
        text = service.brief(task, project=args.get("project"), limit=int(args.get("limit") or 8))
        return text
    if name == "remember":
        return service.remember(
            str(args.get("content") or ""), kind=str(args.get("kind") or "semantic"),
            source=str(args.get("source") or "agent"), importance=float(args.get("importance", 0.6)),
            pinned=bool(args.get("pinned", False)), project=args.get("project"),
            entities=list(args.get("entities") or []),
        )
    if name == "forget":
        return service.forget(str(args.get("id") or ""), str(args.get("reason") or "forget"))
    if name == "stats":
        return service.stats()
    raise KeyError(name)


def handle_message(service_factory: Callable[[], CortexService], message: dict[str, Any],
                   state: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    msg_id = message.get("id")
    is_request = "id" in message

    def ok(result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def err(code: int, text: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": text}}

    if method == "initialize":
        params = message.get("params") or {}
        from . import __version__ as version  # type: ignore[attr-defined]

        return ok({
            "protocolVersion": str(params.get("protocolVersion") or DEFAULT_PROTOCOL),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": str(version)},
            "instructions": (
                "Cortex is the user's long-term memory. Call context_brief at the start of a task, "
                "memory_search before answering questions that may depend on past sessions, and remember "
                "for durable decisions/preferences/procedures. Never store secrets."
            ),
        })
    if method in {"notifications/initialized", "initialized"} or (method or "").startswith("notifications/"):
        return None
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": TOOLS})
    if method == "tools/call":
        params = message.get("params") or {}
        name = str(params.get("name") or "")
        args = params.get("arguments") or {}
        if name not in {tool["name"] for tool in TOOLS}:
            return err(-32602, f"unknown tool: {name}")
        try:
            service = state.get("service") or service_factory()
            state["service"] = service
            value = _call_tool(service, name, args)
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
            return ok({"content": [{"type": "text", "text": text}], "isError": False})
        except Exception as exc:  # noqa: BLE001 - tool errors are reported, not fatal
            return ok({"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True})
    if not is_request:
        return None
    return err(-32601, f"method not found: {method}")


def serve(stdin=None, stdout=None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    state: dict[str, Any] = {}
    # Open the store eagerly so model warmup and backfill start right away.
    try:
        state["service"] = CortexService()
        svc = state["service"]
        _log(f"ready: db={svc.db_path} semantic={svc.semantic_enabled} preload={svc.preload_enabled}")
    except Exception as exc:  # noqa: BLE001 - surface on first tool call instead
        _log(f"startup failed, will retry on first call: {type(exc).__name__}: {exc}")
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32700, "message": "parse error"}}) + "\n")
            stdout.flush()
            continue
        batch = message if isinstance(message, list) else [message]
        for item in batch:
            if not isinstance(item, dict):
                continue
            response = handle_message(CortexService, item, state)
            if response is not None:
                stdout.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
                stdout.flush()
    service = state.get("service")
    if service is not None:
        service.close()
    return 0


# ---------------------------------------------------------------- Claude Code hooks
def _hook(event: str) -> int:
    """Claude Code hook entry point. Prints context on stdout; never blocks the session."""

    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:  # noqa: BLE001
        payload = {}
    os.environ.setdefault("CORTEX_MCP_QUIET", "1")
    try:
        service = CortexService(warm=False)
    except Exception:  # noqa: BLE001
        return 0
    try:
        project = Path(str(payload.get("cwd") or os.getcwd())).name or None
        if event == "session-start":
            text = service.brief("", project=project, limit=int(os.environ.get("CORTEX_BRIEF_LIMIT", "8")))
            if text and not text.startswith("Cortex has no stored memories"):
                print("Cortex memory brief (from the cortex MCP server; fallible evidence):\n" + text)
        elif event == "prompt":
            prompt = str(payload.get("prompt") or "").strip()
            _stash_prompt(service.home, payload.get("session_id"), prompt)
            if len(prompt) < 12:
                return 0
            results = service.search(prompt[:2000], limit=int(os.environ.get("CORTEX_HOOK_LIMIT", "5")),
                                     project=project)
            if results:
                lines = ["Cortex recalled these possibly relevant memories (fallible evidence):"]
                lines += [f"- [{r['id'][:8]}] ({r['kind']}) {r['content']}" for r in results]
                print("\n".join(lines))
        elif event == "stop":
            if os.environ.get("CORTEX_RECORD_EPISODES", "1").strip().lower() in {"0", "false", "no", "off"}:
                return 0
            prompt = _pop_prompt(service.home, payload.get("session_id"))
            answer = str(payload.get("last_assistant_message") or "").strip()
            if prompt and answer:
                # Episodes are raw local history; `cortex-memory sleep` consolidates them later.
                service.memory.record_episode(prompt[:8000], answer[:8000],
                                              session_id=str(payload.get("session_id") or "claude-code"))
    except Exception:  # noqa: BLE001 - a memory hook must never break the session
        return 0
    finally:
        service.close()
    return 0


def _prompt_path(home: Path, session_id: Any) -> Path:
    safe = "".join(ch for ch in str(session_id or "default") if ch.isalnum() or ch in "-_")[:80] or "default"
    return home / "hook-state" / f"{safe}.prompt"


def _stash_prompt(home: Path, session_id: Any, prompt: str) -> None:
    path = _prompt_path(home, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt[:8000], encoding="utf-8")
    os.chmod(path, 0o600)


def _pop_prompt(home: Path, session_id: Any) -> str:
    path = _prompt_path(home, session_id)
    try:
        text = path.read_text(encoding="utf-8")
        path.unlink()
        return text
    except OSError:
        return ""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "serve":
        return serve()
    command = argv[0]
    if command == "hook" and len(argv) > 1:
        return _hook(argv[1])
    if command in {"stats", "doctor"}:
        service = CortexService(warm=command == "doctor")
        if command == "doctor":
            service.wait_warm(120)
        data = service.stats()
        if command == "doctor":
            data["audit_ok"] = bool(service.memory.audit().get("ok"))
            data["model_files_present"] = service.embedder.available
            data["embedder_load_error"] = service.embedder.load_error
        print(json.dumps(data, indent=2, default=str))
        service.close()
        return 0
    if command == "backup":
        service = CortexService(warm=False)
        dest = service.backup(Path(argv[1]) if len(argv) > 1 else None)
        service.close()
        print(dest)
        return 0
    if command == "home":
        print(cortex_home())
        return 0
    print("usage: cortex-mcp [serve|stats|doctor|backup [PATH]|home|hook session-start|hook prompt|hook stop]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
