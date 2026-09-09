"""Deterministic JSON serialization helpers for Cortex trace payloads.

Stage 0 of the store split plan (`docs/STORE_SPLIT_PLAN.md`): pure,
dependency-free helpers moved verbatim out of `store.py`. `store.py`
re-exports every name defined here, so `from cortex.store import
_trace_json, ...` keeps working.
"""

from __future__ import annotations

import json
from typing import Any


def _trace_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _trace_json_list(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in _trace_json_array(value) if isinstance(item, dict)]


def _trace_json_array(value: Any) -> list[Any]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return parsed


def _trace_json_object(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


__all__ = [
    "_trace_json",
    "_trace_json_list",
    "_trace_json_array",
    "_trace_json_object",
]
