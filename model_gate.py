"""Only a strong model may write a memory without a human (#26).

mem0's audit found 97.8% of model-extracted memories were junk. The lesson is not "extraction is
bad" — it is that a *cheap* model's fluent output looks exactly like a *strong* model's, so a write
path that accepts whatever model is configured will flood the store the first time someone points
it at a small model to save money.

The rule that follows is about the model's identity, not its confidence:

- a **cheap family** may still reject or defer — saying "no" is always safe — but its "yes" is not
  enough to write a memory. The proposal stays staged for a human and the store records why;
- a **strong family** may write, because that is what it is for;
- an **unknown** name is allowed by default, because self-hosted and synthetic models are real and
  a gate that blocks anything unfamiliar would be turned off within a week. `CORTEX_STRONG_MODELS_STRICT=1`
  flips that for an environment that wants an allow-list instead of a deny-list.

Matching is on prefixes, so a version bump (`gpt-6.1` → `gpt-6.2`) does not silently change the
path, and adding a bad model to the deny-list is one line.
"""

from __future__ import annotations

import os
from typing import Any

WRITE_ACTIONS = frozenset({"remember", "remember_edited", "evidence_only"})

# Families measured to be too cheap for this job. Prefixes, matched case-insensitively.
WEAK_PREFIXES: tuple[str, ...] = (
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-3.5",
    "o1-mini",
    "claude-haiku",
    "claude-3-5-haiku",
    "gemini-flash",
    "gemini-2.0-flash",
    "flash-lite",
    "llama-3",
    "llama-4-scout",
    "mistral-small",
    "qwen-7b",
    "qwen-14b",
    "deepseek-chat",
    "hy4-flash",
    "grok-fast",
)

# Families known to be strong at this. Used only when strict mode asks for an allow-list.
STRONG_PREFIXES: tuple[str, ...] = (
    "gpt-6",
    "gpt-5",
    "o3",
    "o4",
    "claude-opus",
    "claude-sonnet",
    "deepseek-v4",
    "gemini-3-pro",
    "gemini-2.5-pro",
)


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def weak_prefixes() -> tuple[str, ...]:
    return _env_tuple("CORTEX_WEAK_MODELS", WEAK_PREFIXES)


def strong_prefixes() -> tuple[str, ...]:
    return _env_tuple("CORTEX_STRONG_MODELS", STRONG_PREFIXES)


def strict() -> bool:
    return os.environ.get("CORTEX_STRONG_MODELS_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def classify(model: str | None) -> str:
    """`weak`, `strong`, or `unknown`."""
    name = str(model or "").strip().lower()
    if not name:
        return "unknown"
    if any(name.startswith(prefix) for prefix in weak_prefixes()):
        return "weak"
    if any(name.startswith(prefix) for prefix in strong_prefixes()):
        return "strong"
    return "unknown"


def may_auto_write(model: str | None) -> bool:
    verdict = classify(model)
    if verdict == "weak":
        return False
    if verdict == "strong":
        return True
    return not strict()


def gate_action(model: str | None, action: str) -> tuple[str, str]:
    """Return `(action, reason)`. A cheap model's write becomes a deferral, never a write."""
    value = str(action or "").strip().lower()
    if value not in WRITE_ACTIONS:
        return value, ""
    if may_auto_write(model):
        return value, ""
    return (
        "defer",
        f"automatic write withheld: extractor model '{model or 'unset'}' is "
        f"{classify(model)}, and only a strong model may write a memory without a human (#26)",
    )


def enabled() -> bool:
    return os.environ.get("CORTEX_STRONG_MODEL_GATE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def describe() -> dict[str, Any]:
    return {
        "enabled": enabled(),
        "strict": strict(),
        "weak_prefixes": list(weak_prefixes()),
        "strong_prefixes": list(strong_prefixes()),
        "write_actions": sorted(WRITE_ACTIONS),
    }
