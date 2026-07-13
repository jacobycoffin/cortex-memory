"""Tool-call outcome extraction for Cortex procedural memory."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .security import sanitize_memory
from .semantics import semantic_features


_PLAIN_FAILURE = re.compile(
    r"(?:^|\n)\s*(?:error|fatal|traceback|exception|failed|failure)\s*[:\-]|"
    r"\b(?:permission denied|timed? out|unauthorized|forbidden|no such file|connection refused)\b",
    re.I,
)
_MEMORY_META_TOOLS = {"cortex_memory", "memory", "session_search", "memory_search", "memory_store"}

_TASK_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("web_research", re.compile(r"\b(?:search|research|web|website|online|latest|news|source)\b", re.I)),
    ("filesystem", re.compile(r"\b(?:file|folder|directory|read|write|edit|patch|workspace|repo)\b", re.I)),
    ("shell", re.compile(r"\b(?:terminal|shell|command|process|server|ssh|build|test|install)\b", re.I)),
    ("github", re.compile(r"\b(?:github|pull request|\bpr\b|issue|commit|branch|actions|ci)\b", re.I)),
    ("calendar", re.compile(r"\b(?:calendar|meeting|schedule|availability|event|appointment)\b", re.I)),
    ("email", re.compile(r"\b(?:email|gmail|inbox|reply|forward|message)\b", re.I)),
    ("memory", re.compile(r"\b(?:remember|memory|recall|forget|preference)\b", re.I)),
    ("data_analysis", re.compile(r"\b(?:data|metric|spreadsheet|csv|chart|dashboard|analy[sz]e)\b", re.I)),
)


@dataclass(frozen=True)
class ToolExecution:
    execution_id: str
    session_id: str
    task_type: str
    task_context: str
    tool_name: str
    argument_keys: tuple[str, ...]
    success: bool
    error_type: str | None
    result_summary: str
    step_index: int


@dataclass(frozen=True)
class ToolWorkflow:
    workflow_id: str
    session_id: str
    task_type: str
    task_fingerprint: str
    task_context: str
    workflow_key: str
    steps: tuple[dict[str, Any], ...]
    success: bool
    error_type: str | None


def classify_task(query: str, tool_name: str = "") -> str:
    combined = f"{query} {tool_name.replace('_', ' ')}"
    for name, pattern in _TASK_RULES:
        if pattern.search(combined):
            return name
    return "general"


def extract_tool_executions(messages: Sequence[dict[str, Any]], *, session_id: str) -> list[ToolExecution]:
    calls: dict[str, dict[str, Any]] = {}
    last_user = ""
    for message in messages:
        role = str(message.get("role") or "")
        if role == "user" and isinstance(message.get("content"), str):
            last_user = message["content"]
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                call_id = str(call.get("id") or hashlib.sha256(repr(call).encode()).hexdigest()[:24])
                calls[call_id] = {
                    "name": str(function.get("name") or call.get("name") or "unknown_tool"),
                    "arguments": function.get("arguments") or call.get("arguments") or "{}",
                    "task_context": last_user,
                }
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id in calls:
                calls[call_id]["result"] = message.get("content")

    executions: list[ToolExecution] = []
    for step_index, (call_id, call) in enumerate(calls.items()):
        if "result" not in call:
            continue
        tool_name = call["name"]
        if tool_name in _MEMORY_META_TOOLS:
            continue
        argument_keys = _argument_keys(call["arguments"])
        result_text = _result_text(call["result"])
        sanitized = sanitize_memory(result_text)
        success = not _looks_failed(result_text)
        error_type = None if success else _error_type(result_text)
        task_context = sanitize_memory(str(call["task_context"])).text[:400]
        stable = json.dumps(
            {
                "session": session_id,
                "call_id": call_id,
                "tool": tool_name,
                "keys": argument_keys,
                "result": sanitized.text[:500],
            },
            sort_keys=True,
        )
        executions.append(
            ToolExecution(
                execution_id=hashlib.sha256(stable.encode()).hexdigest(),
                session_id=session_id,
                task_type=classify_task(task_context, tool_name),
                task_context=task_context,
                tool_name=tool_name,
                argument_keys=argument_keys,
                success=success,
                error_type=error_type,
                result_summary=sanitized.text[:500],
                step_index=step_index,
            )
        )
    return executions


def build_tool_workflow(executions: Sequence[ToolExecution]) -> ToolWorkflow | None:
    if not executions:
        return None
    ordered = sorted(executions, key=lambda execution: execution.step_index)
    task_context = ordered[0].task_context
    task_type = classify_task(task_context, " ".join(execution.tool_name for execution in ordered))
    steps = tuple(
        {
            "tool": execution.tool_name,
            "argument_keys": list(execution.argument_keys),
            "success": execution.success,
            "error_type": execution.error_type,
        }
        for execution in ordered
    )
    workflow_key = hashlib.sha256(json.dumps(steps, sort_keys=True).encode()).hexdigest()
    stable = json.dumps(
        {
            "session": ordered[0].session_id,
            "executions": [execution.execution_id for execution in ordered],
            "workflow": workflow_key,
        },
        sort_keys=True,
    )
    return ToolWorkflow(
        workflow_id=hashlib.sha256(stable.encode()).hexdigest(),
        session_id=ordered[0].session_id,
        task_type=task_type,
        task_fingerprint=task_fingerprint(task_context),
        task_context=task_context,
        workflow_key=workflow_key,
        steps=steps,
        success=all(execution.success for execution in ordered),
        error_type=next((execution.error_type for execution in ordered if not execution.success), None),
    )


def task_fingerprint(query: str) -> str:
    features = semantic_features(query, max_features=48)
    # Tool workflows need a task *family*, not a hash of every noun in one
    # request. Concepts stay stable across distinct tasks; stem fallbacks keep
    # unrelated, uncategorized work separated.
    concepts = [
        key
        for key in features
        if key.startswith("concept:") and key not in {"concept:current", "concept:historical"}
    ]
    preferred = concepts or [key for key in features if key.startswith("stem:")][:4]
    if not preferred:
        preferred = [key for key in features if key.startswith("tok:")][:4]
    return "|".join(sorted(dict.fromkeys(preferred))) or "general"


def _argument_keys(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, dict):
        return tuple(sorted(str(key) for key in raw))
    try:
        parsed = json.loads(str(raw))
        if isinstance(parsed, dict):
            return tuple(sorted(str(key) for key in parsed))
    except (ValueError, TypeError):
        pass
    return ()


def _result_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False, default=str)
    except TypeError:
        return str(raw)


def _looks_failed(result: str) -> bool:
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            if parsed.get("success") is False or parsed.get("ok") is False or parsed.get("isError") is True:
                return True
            status = str(parsed.get("status") or "").casefold()
            if status in {"error", "failed", "failure", "fatal"}:
                return True
            if parsed.get("error") not in (None, "", False, [], {}):
                return True
            if parsed.get("success") is True or parsed.get("ok") is True:
                return False
    except (ValueError, TypeError):
        pass
    return bool(_PLAIN_FAILURE.search(result[:1200]))


def _error_type(result: str) -> str:
    lowered = result.casefold()
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "unauthorized" in lowered or "forbidden" in lowered or "permission" in lowered:
        return "authorization"
    if "not found" in lowered or "no such file" in lowered:
        return "not_found"
    if "invalid" in lowered or "validation" in lowered:
        return "validation"
    if "connection" in lowered or "network" in lowered or "dns" in lowered:
        return "connectivity"
    return "tool_error"
