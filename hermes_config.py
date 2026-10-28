"""Hermes adapter defaults, config parsing, and tool response encoding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "db_path": "$HERMES_HOME/cortex/cortex.db",
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


def _read_config(hermes_home: Path) -> dict[str, Any]:
    path = hermes_home / "config.yaml"
    if not path.exists():
        return {}
    try:
        root = _load_yaml(path.read_text(encoding="utf-8-sig")) or {}
        return dict((root.get("plugins") or {}).get("cortex") or {})
    except Exception:
        return {}


def _load_yaml(text: str) -> Any:
    """Parse YAML with PyYAML, or ruamel.yaml when PyYAML is absent.

    Current Hermes environments ship ruamel.yaml but not PyYAML; without this
    fallback every ``plugins.cortex`` setting was silently ignored there.
    """

    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        from ruamel.yaml import YAML

        return YAML(typ="safe", pure=True).load(text)


def write_cortex_config(path: Path, values: dict[str, Any], *, provider: str | None = None,
                        only_missing: bool = False) -> dict[str, Any]:
    """Write ``plugins.cortex`` (and optionally ``memory.provider``) to config.yaml.

    Uses ruamel.yaml round-trip mode when available so Hermes's comments and key
    order survive; falls back to PyYAML. With ``only_missing`` existing operator
    choices are never overwritten. Returns the resulting ``plugins.cortex`` map.
    """

    text = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    try:
        from ruamel.yaml import YAML

        yaml_rt = YAML()
        yaml_rt.preserve_quotes = True
        root = yaml_rt.load(text) if text.strip() else None
        dump = yaml_rt.dump
    except ImportError:
        import yaml

        root = yaml.safe_load(text) if text.strip() else None

        def dump(data: Any, stream: Any) -> None:
            stream.write(yaml.safe_dump(data, sort_keys=False))
    if root is None:
        root = {}
    if provider is not None:
        if root.get("memory") is None:
            root["memory"] = {}
        root["memory"]["provider"] = provider
    if root.get("plugins") is None:
        root["plugins"] = {}
    if root["plugins"].get("cortex") is None:
        root["plugins"]["cortex"] = {}
    section = root["plugins"]["cortex"]
    if not only_missing:
        for key in list(section.keys()):
            if key not in values:
                del section[key]
    for key, value in values.items():
        if only_missing and key in section:
            continue
        section[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    import io

    buffer = io.StringIO()
    dump(root, buffer)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return dict(section)


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
