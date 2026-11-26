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
    text = path.read_text(encoding="utf-8-sig")
    try:
        root = _load_yaml(text) or {}
        return dict((root.get("plugins") or {}).get("cortex") or {})
    except Exception:
        # No YAML library is guaranteed to be importable: the Hermes runtime ships
        # neither PyYAML nor ruamel.yaml, so leaning on them lost every
        # ``plugins.cortex`` setting there. The block is a flat mapping of scalars, so
        # read exactly that and nothing else.
        return _read_cortex_block(text)


def _read_cortex_block(text: str) -> dict[str, Any]:
    """Read ``plugins.cortex`` out of YAML text without a YAML library.

    Deliberately narrow: it understands the shape this setting has (a two-level
    mapping of scalars), because a general YAML parser written here would be a bug
    farm. Nested containers are skipped rather than guessed at.
    """

    block: dict[str, Any] = {}
    in_plugins = False
    in_cortex = False
    cortex_indent = 0
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if indent == 0:
            in_plugins = line == "plugins:"
            in_cortex = False
            continue
        if not in_plugins:
            continue
        if not in_cortex:
            if line == "cortex:":
                in_cortex = True
                cortex_indent = indent
            continue
        if indent <= cortex_indent:
            in_cortex = False
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if not value:
            continue  # nested container: not a scalar, so not something this reader claims to support
        if value[:1] in {'"', "'"}:
            # Unquote first: a '#' inside quotes is content, not a comment.
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        block[key] = _coerce_scalar(value)
    return block


def _coerce_scalar(value: str) -> Any:
    """Turn a YAML scalar into the type the plugin expects, without guessing past the obvious."""

    lowered = value.casefold()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_yaml(text: str) -> Any:
    """Parse YAML with PyYAML, or ruamel.yaml when PyYAML is absent.

    Neither is guaranteed on this host: ``_read_config`` falls back to
    ``_read_cortex_block`` when both imports fail, so a missing library degrades the
    reader instead of silently losing every setting.
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
