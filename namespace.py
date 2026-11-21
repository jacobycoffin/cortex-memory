"""Namespace isolation: a finance or work memory must never reach a personal turn (#20).

Scoped memory is admitted by applicability, not by permission. If a memory is tagged as
belonging to a namespace (`finance`, `work`, …) and the turn declares a different channel,
the memory must not appear — not in retrieval, not in the always-on floor, not through
prospective items, not through graph expansion.

The rule is deliberately simple and explicit:

- a memory with **no** namespace is shared, and reaches every channel;
- a memory with namespaces reaches a channel only if the channel declares one of them;
- a channel that declares **nothing** sees everything (no namespace declared means no
  isolation was asked for).

Explicitness matters: guessing a namespace from prose would both leak and over-block. This
module never inspects content.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

NAMESPACE_KEYS = ("namespace", "namespaces", "domain")
MAX_NAMESPACES = 8


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.strip().lower()} if value.strip() else set()
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip().lower() for v in value if str(v).strip()}
    return set()


def _scope(memory: dict[str, Any]) -> dict[str, Any]:
    raw = memory.get("scope_json") or memory.get("scope") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


def memory_namespaces(memory: dict[str, Any]) -> set[str]:
    scope = _scope(memory)
    names: set[str] = set()
    for key in NAMESPACE_KEYS:
        names |= _as_set(scope.get(key))
    # Metadata is where derived stores keep it; scope is where capture writes it.
    raw_meta = memory.get("metadata") or memory.get("metadata_json") or {}
    if isinstance(raw_meta, str):
        try:
            raw_meta = json.loads(raw_meta)
        except ValueError:
            raw_meta = {}
    if isinstance(raw_meta, dict):
        for key in NAMESPACE_KEYS:
            names |= _as_set(raw_meta.get(key))
    return set(sorted(names)[:MAX_NAMESPACES])


def channel_namespaces(context: Any) -> set[str]:
    """Namespaces the current turn declares. Empty means no isolation was requested."""
    if context is None:
        return set()
    scope: dict[str, Any] = {}
    for attribute in ("scope", "system_state"):
        value = getattr(context, attribute, None)
        if isinstance(value, dict):
            scope.update(value)
    if not scope and isinstance(context, dict):
        scope = context
    names: set[str] = set()
    for key in NAMESPACE_KEYS:
        names |= _as_set(scope.get(key))
    return set(sorted(names)[:MAX_NAMESPACES])


def admissible(memory: dict[str, Any], channel: set[str] | Iterable[str]) -> bool:
    """True when this memory may appear in a turn carrying `channel` namespaces."""
    required = memory_namespaces(memory or {})
    if not required:
        return True
    allowed = {str(c).strip().lower() for c in (channel or []) if str(c).strip()}
    if not allowed:
        return True
    return bool(required & allowed)


def filter_results(results: Iterable[Any], channel: Iterable[str]) -> tuple[list[Any], int]:
    """Drop inadmissible results. Returns (kept, dropped)."""
    allowed = set(channel or [])
    kept: list[Any] = []
    dropped = 0
    for result in results:
        memory = getattr(result, "memory", {}) or {}
        if admissible(memory, allowed):
            kept.append(result)
        else:
            dropped += 1
    return kept, dropped
