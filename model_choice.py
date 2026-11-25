"""Which model the night is allowed to think with (#118).

Sleep's reflection is the one place in the store that spends real money on real reasoning: it reads a night's
episodes and proposes what they mean. Today it is wired to three env vars that somebody must set by hand
(`CORTEX_SLEEP_ENDPOINT`, `CORTEX_SLEEP_MODEL`, `CORTEX_SLEEP_API_KEY_ENV`) and, measured on the live store, it has
run **0 times in 82 sleep runs** — every one recorded `disabled` with a token budget of 0. So the item's premise
("sleep uses the strongest available model, not the chat model") is not yet true in either direction: sleep uses no
model at all.

This module is the resolver that makes the choice explicit and auditable, and it deliberately does **not** turn
anything on. Six rules:

- **the ladder is declared data, not intuition.** `LADDER` orders candidate providers strongest first, each naming
  the env vars that configure it. The ordering is an assertion by the operator about *which job deserves the
  strongest model* — this module does not measure model quality and does not pretend to.
- **availability is checked, not assumed.** A tier is only usable when its model, endpoint and key are all present.
  A model you cannot call is not a candidate, and the skip reason names the missing variable (never its value).
- **the chat model is never a candidate.** The ladder lists deep-work providers only; inheriting whatever the
  interactive turn uses is precisely the failure this item exists to prevent.
- **an explicit pin is authoritative; ambient config is not.** A model passed as `pin=` (or the CLI's
  `--reflection-model`) is a per-run instruction: if its key is missing the answer is `refused`, never a silent
  downgrade to something cheaper. `CORTEX_SLEEP_MODEL` is ambient configuration, so a stale one falls back *loudly*
  instead of blocking the night — the two are deliberately different, because one is a command and one is a default.
- **falling back is loud.** The result carries `fell_back` plus every tier that was skipped and why, so a cheap
  night is visible in the run record instead of looking like a strong one.
- **nothing available is a clear refusal**, with the list of what was looked for. It does not invent a model name or
  a default endpoint.

The resolver spends nothing: it reads environment names and returns a decision.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

# Ordered strongest -> weakest. Each entry names the variables that configure that provider; the model is never
# constructed from parts, and an entry whose variables are unset simply drops out of the ladder.
LADDER: tuple[dict[str, str], ...] = (
    {
        "tier": "sleep_explicit",
        "why": "a model configured specifically for sleep reflection",
        "model_env": "CORTEX_SLEEP_MODEL",
        "endpoint_env": "CORTEX_SLEEP_ENDPOINT",
        "key_env": "CORTEX_SLEEP_API_KEY_ENV",
        "key_default": "OPENROUTER_API_KEY",
    },
    {
        "tier": "judge",
        "why": "the admission judge's provider, already trusted with reasoning about memories",
        "model_env": "CORTEX_AUTO_JUDGE_MODEL",
        "endpoint_env": "CORTEX_AUTO_JUDGE_ENDPOINT",
        "key_env": "CORTEX_AUTO_JUDGE_API_KEY_ENV",
        "key_default": "OPENROUTER_API_KEY",
    },
    {
        "tier": "mechanics",
        "why": "the brain-mechanics provider",
        "model_env": "CORTEX_BRAIN_MECHANICS_MODEL",
        "endpoint_env": "CORTEX_BRAIN_MECHANICS_ENDPOINT",
        "key_env": "CORTEX_BRAIN_MECHANICS_API_KEY_ENV",
        "key_default": "OPENROUTER_API_KEY",
    },
)


@dataclass
class Choice:
    usable: bool = False
    tier: str = ""
    model: str = ""
    endpoint: str = ""
    api_key_env: str = ""
    fell_back: bool = False
    pinned: bool = False
    reason: str = ""
    skipped: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Goes into the sleep run's details: a bad consolidation must be attributable to a model."""
        return {
            "usable": self.usable,
            "tier": self.tier,
            "model": self.model,
            "endpoint": self.endpoint,
            "api_key_env": self.api_key_env,
            "fell_back": self.fell_back,
            "pinned": self.pinned,
            "reason": self.reason,
            "skipped": self.skipped,
        }

    def render(self) -> str:
        if not self.usable:
            return f"no model available: {self.reason}"
        fallback = " (fell back)" if self.fell_back else ""
        return f"{self.tier}: {self.model}{fallback} via {self.endpoint}, key from {self.api_key_env}"


def _get(env: Mapping[str, str], name: str) -> str:
    return str(env.get(name, "") or "").strip()


def choose(*, env: Mapping[str, str] | None = None, pin: str | None = None) -> Choice:
    """Resolve the model for nighttime reasoning. Never raises, never spends, never invents a name."""
    source = env if env is not None else os.environ
    explicit_pin = (pin or "").strip()
    choice = Choice()
    tried: list[dict[str, str]] = []

    for entry in LADDER:
        tier = entry["tier"]
        model = explicit_pin if (tier == "sleep_explicit" and explicit_pin) else _get(source, entry["model_env"])
        if not model:
            tried.append({"tier": tier, "why": f"{entry['model_env']} is unset"})
            continue
        endpoint = _get(source, entry["endpoint_env"])
        if not endpoint:
            tried.append({"tier": tier, "why": f"{entry['endpoint_env']} is unset"})
            continue
        key_env = _get(source, entry["key_env"]) or entry["key_default"]
        if not _get(source, key_env):
            # The variable NAME is reported; the value never is.
            tried.append({"tier": tier, "why": f"{key_env} is unset"})
            if tier == "sleep_explicit" and explicit_pin:
                choice.skipped = tried
                choice.pinned = True
                choice.reason = (f"a model was pinned for sleep ({model}) but {key_env} is unset: refusing rather "
                                 f"than silently using a weaker one")
                return choice
            continue

        choice.usable = True
        choice.tier = tier
        choice.model = model
        choice.endpoint = endpoint
        choice.api_key_env = key_env
        choice.fell_back = bool(tried)
        choice.pinned = tier == "sleep_explicit" and bool(explicit_pin)
        choice.skipped = tried
        choice.reason = entry["why"] if not choice.fell_back else (
            f"{entry['why']}; {len(tried)} stronger or earlier tier(s) were unavailable")
        return choice

    choice.skipped = tried
    looked = ", ".join(f"{item['tier']} ({item['why']})" for item in tried) or "no tier configured"
    choice.reason = f"no tier is usable: {looked}"
    return choice


def available(env: Mapping[str, str] | None = None) -> bool:
    return choose(env=env).usable
