"""Refuse secrets; do not merely scrub them (#43).

Redaction is not enough. A store that turns `password: hunter2` into `password: [REDACTED]` and keeps
the row has produced a memory that says *the store knows this password* — which is worse than saying
nothing, because a later reader (or a later model) will believe it and act on it. The value was
already read, logged and copied into a database that gets backed up nightly.

So the decision is triaged rather than uniform:

| verdict | when | what happens |
|---|---|---|
| `refuse` | a private-key block, a recognised provider token, a password/secret **with a value** | nothing is stored |
| `redact` | secret-shaped text with no extractable value ("my password is stored in the vault") | stored with the phrase neutralised |
| `allow` | no secret-shaped content | stored unchanged |

The distinction is *value present or not*. Mentioning a secret is ordinary conversation and has to
keep working; carrying one is a different act, and refusal is the only verdict that does not leave a
copy behind.

Refusals are counted, not silent: a user who pastes a key five times should be able to see that the
store declined five times, and why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

REFUSE = "refuse"
REDACT = "redact"
ALLOW = "allow"

# Anything matching one of these carries a usable credential.
_HARD_SECRETS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.I), "private_key"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws_access_key"),
    (re.compile(r"\bASIA[0-9A-Z]{16}\b"), "aws_session_key"),
    (re.compile(r"\b(?:sk|rk)-[A-Za-z0-9_-]{20,}\b"), "provider_api_key"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"), "anthropic_api_key"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b"), "github_token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b"), "github_token"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "slack_token"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "google_api_key"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"), "jwt"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b", re.I), "bearer_token"),
)

# A labelled credential with something after the separator.
_LABELLED_VALUE = re.compile(
    r"\b(password|passwd|passphrase|passcode|pin|secret|token|api[_ -]?key|private[_ -]?key|"
    r"access[_ -]?key|client[_ -]?secret)\b"
    r"\s*(?:[:=]|\bis\b|\bwas\b)\s*"
    r"(?P<value>[^\s,;]{3,})",
    re.IGNORECASE,
)

# A placeholder stands where a value would be: the row is safe to keep, but the wording should be
# neutralised so a later reader does not think the store holds the value.
_PLACEHOLDERS = re.compile(
    r"\[redacted\]|<redacted>|\*\*\*|\.\.\.|…|^x+$", re.IGNORECASE
)

# State verbs describe what happened to a credential rather than quoting one. "the api key was
# rotated" is a mention, not a secret, and refusing it would make ordinary conversation fail.
_STATE_VERBS = re.compile(
    r"\b(?:rotated|changed|updated|reset|expired|leaked|revoked|created|issued|stored|saved|"
    r"managed|kept|located|required|needed)\b",
    re.IGNORECASE,
)

# A mention of a credential with no value attached is not a secret.
_MENTION_ONLY = re.compile(
    r"\b(password|passwd|passphrase|passcode|pin|secret|token|api[_ -]?key)\b", re.IGNORECASE
)

_COUNTER = "secret_refusals"


@dataclass
class Verdict:
    action: str
    reasons: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.action == REFUSE

    def describe(self) -> str:
        if self.action == ALLOW:
            return "no credential detected"
        detail = ", ".join(self.reasons) or "unspecified"
        verb = "refused" if self.refused else "redact before storing"
        return f"{verb} ({detail})"


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {_COUNTER} (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    count INTEGER NOT NULL DEFAULT 0,
                    reasons_json TEXT,
                    updated_at TEXT
                )"""
        )


def assess(text: str) -> Verdict:
    """Decide whether this text may be stored at all. Pure and deterministic."""
    value = str(text or "")
    if not value.strip():
        return Verdict(ALLOW)

    reasons: list[str] = []
    matched: list[str] = []
    for pattern, label in _HARD_SECRETS:
        match = pattern.search(value)
        if match:
            reasons.append(label)
            matched.append(match.group(0)[:12] + "…" if len(match.group(0)) > 12 else match.group(0))
    if reasons:
        return Verdict(REFUSE, sorted(set(reasons)), matched)

    labelled = _LABELLED_VALUE.search(value)
    if labelled:
        candidate = labelled.group("value")
        if _PLACEHOLDERS.search(candidate):
            return Verdict(REDACT, [f"{labelled.group(1).strip().lower()} (placeholder value)"])
        if not _STATE_VERBS.search(candidate):
            return Verdict(
                REFUSE,
                [f"{labelled.group(1).strip().lower()} with a value"],
                ["<credential>"],
            )

    if _MENTION_ONLY.search(value):
        return Verdict(ALLOW, ["credential mentioned without a value"])

    return Verdict(ALLOW)


def refusal_count(store: Any) -> int:
    try:
        row = store._conn.execute(f"SELECT count FROM {_COUNTER} WHERE id = 1").fetchone()
    except Exception:  # noqa: BLE001 - a missing table means no refusals yet
        return 0
    return int(row["count"]) if row else 0


def note_refusal(store: Any, verdict: Verdict, *, at: Any = None) -> int:
    """Count a refusal so a repeat offender is visible rather than silent."""
    from datetime import datetime, timezone
    import json

    _ensure(store)
    stamp = (at or datetime.now(timezone.utc)).isoformat()
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {_COUNTER} (id, count, reasons_json, updated_at)
                VALUES (1, 1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    count = count + 1,
                    reasons_json = excluded.reasons_json,
                    updated_at = excluded.updated_at""",
            (json.dumps(sorted(set(verdict.reasons))), stamp),
        )
    return refusal_count(store)


def report(store: Any) -> dict[str, Any]:
    import json

    try:
        row = store._conn.execute(
            f"SELECT count, reasons_json, updated_at FROM {_COUNTER} WHERE id = 1"
        ).fetchone()
    except Exception:  # noqa: BLE001
        return {"refusals": 0, "reasons": [], "last": None}
    if not row:
        return {"refusals": 0, "reasons": [], "last": None}
    try:
        reasons = json.loads(str(row["reasons_json"] or "[]"))
    except ValueError:
        reasons = []
    return {
        "refusals": int(row["count"]),
        "reasons": reasons if isinstance(reasons, list) else [],
        "last": str(row["updated_at"] or ""),
    }
