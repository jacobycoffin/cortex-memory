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
        import yaml

        root = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        return dict((root.get("plugins") or {}).get("cortex") or {})
    except Exception:
        return {}


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
