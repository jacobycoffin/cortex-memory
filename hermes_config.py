"""Hermes adapter defaults, config parsing, and tool response encoding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "db_path": "$HERMES_HOME/cortex/cortex.db",
    # --- multi-tenant fleet layer -------------------------------------------
    # Unset shared_db_path = fleet disabled: one store, today's behaviour.
    "shared_db_path": "",
    "shared_weight": 0.85,
    "shared_pool": 4,
    "shared_max_share": 0.5,
    "shared_min_score": 0.0,
    "shared_promotion_enabled": False,
    "fleet_tenant": "",
    "auto_capture": True,
    "primary_memory": True,
    "top_k": 6,
    "token_budget": 700,
    "retrieval_threshold": 0.16,
    # Local embedding fusion. 0.0 = OFF — the code default stays conservative on
    # purpose; enabling it is an explicit deployment choice. Measured on LoCoMo
    # across 10 conversations: weight 10 gives +9.6 multi-hop hit@10 with no
    # single-hop regression; weight 30 regresses single-hop.
    "semantic_fusion_weight": 0.0,
    "semantic_fusion_pool": 20,
    "semantic_fusion_min_similarity": 0.0,
    "adaptive_recall": True,
    "adaptive_budget_learning": True,
    "attentional_learning": False,
    "metacognition_mode": "shadow",
    "query_cache_ttl_seconds": 45,
    "background_preload": False,
    "background_preload_continuity": False,
    "compact_context": True,
    "memory_receipts": True,
    "memory_receipt_url": "",
    "attribution_threshold": 0.18,
    "regret_mode": "shadow",
    "consolidation_mode": "shadow",
    "pruning_mode": "shadow",
    "cold_after_days": 90,
    "archive_after_days": 180,
}


def _coerce_scalar(value: str) -> Any:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    lowered = text.casefold()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_plugins_fallback(text: str) -> dict[str, Any]:
    """Minimal nested-mapping reader used when PyYAML is absent.

    Only the shape a Cortex config block needs is understood: nested mappings of
    scalars with space indentation. Lists, anchors, tab indentation and anything
    else make it give up and return ``{}`` — a half-understood config block is
    worse than none, because fleet settings would silently be ignored.

    The repository ships no runtime dependencies, but Hermes profiles always have
    PyYAML, so this path exists for CI and bare checkouts. Without it
    ``_read_config`` returned ``{}`` and a configured shared layer looked
    "not configured" to the operator CLI.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if "\t" in raw[: indent + 1]:
            return {}
        if stripped.startswith("-") or ":" not in stripped:
            return {}
        key, _, value = stripped.partition(":")
        key = key.strip()
        if not key or any(ch in key for ch in "[]{}"):
            return {}
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            return {}
        parent = stack[-1][1]
        if value.strip():
            scalar = value.strip()
            if " #" in scalar:
                scalar = scalar.split(" #", 1)[0].strip()
            # Flow collections, anchors, tags and block scalars are outside what
            # this parser understands; refuse rather than import a wrong value.
            if not scalar or scalar[0] in "[{&*|>!@`":
                return {}
            parent[key] = _coerce_scalar(scalar)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _cortex_block(root: dict[str, Any]) -> dict[str, Any]:
    """Extract the cortex plugin block from a parsed config tree.

    Live layout is ``plugins.cortex``; ``plugins.entries.cortex`` is the nested
    style some Hermes builds write. Accept both, nested first.
    """

    plugins = root.get("plugins")
    if not isinstance(plugins, dict):
        return {}
    block: dict[str, Any] = {}
    entries = plugins.get("entries")
    if isinstance(entries, dict):
        cortex = entries.get("cortex")
        if isinstance(cortex, dict):
            block.update(cortex)
    cortex = plugins.get("cortex")
    if isinstance(cortex, dict):
        block.update(cortex)
    return block


def _read_config(hermes_home: Path) -> dict[str, Any]:
    path = hermes_home / "config.yaml"
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    try:
        import yaml
    except ModuleNotFoundError:
        return _cortex_block(_parse_plugins_fallback(text))
    try:
        root = yaml.safe_load(text) or {}
    except Exception:
        return {}
    if not isinstance(root, dict):
        return {}
    return _cortex_block(root)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in value or ():
        cleaned = " ".join(str(item).split())[:200]
        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            result.append(cleaned)
    return tuple(result)


def _json_ok(**payload: Any) -> str:
    return json.dumps({"success": True, **payload}, ensure_ascii=False, default=str)


def _json_error(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)
