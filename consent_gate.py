"""What may be remembered, decided before the write (#182)

Every other trust surface in this system answers a question *after* the fact: what was recalled, what was
injected, why a belief is held. Consent is the one that has to run **before** the row exists, because a
memory that should never have been written cannot be un-written by a better explanation.

The design starts from the failure it prevents. The store already holds 3,843 memories and the largest
source category is `DOCUMENT_EXTRACTED` (3,028) — content lifted from documents, not said in
conversation. That is fine for a homelab note and wrong for a bank statement, so a blanket "remember
things" is not a consent model; **consent is per class**.

Six rules, each of which is a decision I do not get to make on his behalf:

- **default is `ask`, not `store`, for anything in a protected class.** A gate that defaults to writing
  is a gate that only works when someone remembers to configure it.
- **a credential is refused** (#43), not redacted into the store, because a redacted credential still
  records where the credential lives.
- **work systems are refused outright** — not asked about, not stored, not summarised. That boundary is
  his, stated once, and a policy engine that re-opens it every write is not respecting it.
- **consent is per class.** Authorising technical facts never authorises finance; that is the entire
  point of naming classes.
- **a refusal names the class and the rule, never the content.** A gate that echoes the sentence it
  blocked has just written the memory into a log.
- **every decision is recorded, and the record is the hash.** "Why is this in here" is answerable from
  `consent_decisions` without a second copy of what was said.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

STORE = "store"
STORE_REDACTED = "store_redacted"
ASK = "ask"
REFUSE = "refuse"

# Ordered: the first class that matches wins, so the most protective rule gets the first word.
CLASSES: tuple[tuple[str, str, str], ...] = (
    ("credential", REFUSE, "a credential is refused, not scrubbed (#43)"),
    ("work_systems", REFUSE, "work systems are off limits; he stated this once"),
    ("finance", ASK, "finance-derived content needs explicit consent per class"),
    ("health", ASK, "health content needs explicit consent per class"),
    ("relationship", ASK, "personal-relationship content needs explicit consent per class"),
    ("technical", STORE, "ordinary technical facts are stored"),
)

_PATTERNS: dict[str, re.Pattern[str]] = {
    "credential": re.compile(
        r"\b(?:api[_\s-]?key|secret|token|password|passphrase|private key|BEGIN [A-Z ]*PRIVATE KEY)\b",
        re.I,
    ),
    "work_systems": re.compile(
        r"\b(?:work (?:laptop|system|email|account)|employer|payroll|ticket system|corporate|"
        r"southern tier|continuum of care)\b",
        re.I,
    ),
    "finance": re.compile(
        r"\b(?:bank|routing number|account number|balance|statement|mortgage|loan|401k|ira|"
        r"paycheck|salary|invoice|credit card|debit|net worth|budget)\b",
        re.I,
    ),
    "health": re.compile(
        r"\b(?:diagnosis|prescription|medication|therapy|doctor|symptom|blood pressure|"
        r"insurance claim)\b",
        re.I,
    ),
    "relationship": re.compile(
        r"\b(?:my wife|my husband|my partner|my mother|my father|my brother|my sister|"
        r"my son|my daughter|divorce|argument with)\b",
        re.I,
    ),
}

DEFAULT_CONSENT = {"technical": True}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def classify(text: str) -> str:
    """The most protective class the content matches. `technical` is the fallback, not the default.

    This is a **keyword screen**, not a classifier: it finds mentions. A note *about* rotating a token
    matches `credential` exactly as a leaked token does, so a count of matches is an upper bound on
    sensitive content, not a count of secrets.
    """
    body = str(text or "")
    for name, _verdict, _reason in CLASSES:
        pattern = _PATTERNS.get(name)
        if pattern is not None and pattern.search(body):
            return name
    return "technical"


def rule_for(name: str) -> tuple[str, str]:
    for class_name, verdict, reason in CLASSES:
        if class_name == name:
            return verdict, reason
    return ASK, "an unknown class is asked about, never stored by default"


@dataclass
class Decision:
    verdict: str = REFUSE
    class_name: str = "unknown"
    reason: str = ""
    content_hash: str = ""
    classes_seen: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "class": self.class_name, "reason": self.reason,
                "content_hash": self.content_hash, "classes_seen": self.classes_seen}


def decide(text: str, *, consent: dict[str, bool] | None = None) -> Decision:
    """The gate. `consent` maps class → allowed; absent means not allowed for protected classes."""
    body = str(text or "")
    granted = dict(DEFAULT_CONSENT)
    granted.update({str(key): bool(value) for key, value in (consent or {}).items()})
    seen = [name for name, _verdict, _reason in CLASSES
            if _PATTERNS.get(name) is not None and _PATTERNS[name].search(body)]
    primary = classify(body)
    verdict, reason = rule_for(primary)
    if verdict == ASK and granted.get(primary):
        verdict, reason = STORE, f"{primary} content is stored because consent was granted for it"
    decision = Decision(verdict=verdict, class_name=primary, reason=reason,
                        content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest()[:16],
                        classes_seen=seen)
    return decision


def ensure_schema(store: Any) -> bool:
    """The decision log holds the hash and the verdict — never the text."""
    try:
        with store.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS consent_decisions ("
                "content_hash TEXT PRIMARY KEY, verdict TEXT NOT NULL, class TEXT NOT NULL, "
                "reason TEXT NOT NULL, decided_at TEXT NOT NULL, memory_id TEXT)"
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def record_decision(store: Any, decision: Decision, *, memory_id: str | None = None,
                    now: datetime | None = None) -> bool:
    ensure_schema(store)
    try:
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO consent_decisions(content_hash, verdict, class, reason, decided_at, "
                "memory_id) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(content_hash) DO UPDATE SET verdict = excluded.verdict, "
                "reason = excluded.reason, decided_at = excluded.decided_at, "
                "memory_id = COALESCE(excluded.memory_id, consent_decisions.memory_id)",
                (decision.content_hash, decision.verdict, decision.class_name, decision.reason,
                 (now or _now()).isoformat(), memory_id),
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def gate(store: Any, text: str, *, consent: dict[str, bool] | None = None,
         now: datetime | None = None) -> Decision:
    """Decide, then log the decision. The caller still owns the write."""
    decision = decide(text, consent=consent)
    record_decision(store, decision, now=now)
    return decision


def history(store: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    rows = _rows(
        store,
        "SELECT content_hash, verdict, class, reason, decided_at, memory_id FROM consent_decisions "
        "ORDER BY decided_at DESC LIMIT ?",
        (max(1, int(limit)),),
    )
    return rows


def summary(store: Any) -> dict[str, Any]:
    rows = _rows(store, "SELECT verdict, COUNT(*) AS count FROM consent_decisions GROUP BY verdict")
    counts = {str(row["verdict"]): int(row["count"] or 0) for row in rows}
    total = sum(counts.values())
    note = "the log holds content hashes, not content"
    if total == 0:
        note = ("no decisions recorded: the gate exists but nothing writes through it yet, so the "
                "store's existing contents were never gated")
    return {
        "decisions": total,
        "by_verdict": dict(sorted(counts.items())),
        "note": note,
    }


def render(store: Any, *, limit: int = 10) -> list[str]:
    described = summary(store)
    lines = [f"consent decisions: {described['decisions']} recorded "
             + ("(" + ", ".join(f"{name} {count}"
                                for name, count in described["by_verdict"].items()) + ")"
                if described["by_verdict"] else "")]
    for row in history(store, limit=limit):
        lines.append(f"  · {row['verdict']} {row['class']} {row['content_hash']} — {row['reason']}")
    lines.append(f"  {described['note']}")
    return lines
