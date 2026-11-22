"""Say what was redacted, without saying what it was (#44).

Redaction used to be silent. A secret came in, came out as `[REDACTED]`, and nobody — not the user,
not the reviewer, not the audit — could tell that it had happened. Two failures follow from that
silence: the user thinks the store has their password (it does not), and nobody can see that a
particular source keeps leaking credentials until something worse finds out.

So every redaction leaves a receipt:

    redacted 1 credential (api key) — the value was replaced, nothing was lost but the secret

The receipt is deliberately **about the pattern, never the value**. A receipt that quotes the secret
is the leak it was supposed to prevent, and a test asserts no stored field contains a digit from the
original text.

Receipts are attached to whatever the redaction produced — a proposal or a memory — and are read back
by `for_subject()` and by the turn-level write receipt (#37), so the user learns "I noticed a key and
dropped it" in the same breath as "remembered 2 things".
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

TABLE = "redaction_receipts"

# Anything digit-or-token-shaped is stripped before a field is stored, as a second line of defence:
# labels are built from pattern names, and a bug there must not become a leak here.
_DIGITS = re.compile(r"[A-Za-z0-9._~+/=-]{12,}")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    receipt_id TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    subject_kind TEXT NOT NULL,
                    labels TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )"""
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_redaction_receipts_subject ON {TABLE}(subject_id)"
        )


def _receipt_id(subject_id: str, labels: str) -> str:
    return hashlib.sha256(f"{subject_id}:{labels}".encode("utf-8")).hexdigest()[:20]


def _safe(label: str) -> str:
    """Keep a label short and pattern-like; never a token."""
    text = str(label or "").strip().lower()
    text = _DIGITS.sub("…", text)
    return text[:40] or "credential"


@dataclass
class Receipt:
    receipt_id: str
    subject_id: str
    kind: str
    labels: tuple[str, ...]
    count: int

    def describe(self) -> str:
        what = ", ".join(self.labels) or "credential"
        things = "credential" if self.count == 1 else "credentials"
        return (
            f"redacted {self.count} {things} ({what}) — the value was replaced, "
            "nothing was lost but the secret"
        )


def record(
    store: Any,
    subject_id: str,
    labels: list[str] | tuple[str, ...],
    *,
    kind: str = "memory",
    at: datetime | None = None,
) -> Receipt | None:
    """Record that a redaction happened. Idempotent per subject and label set."""
    cleaned = tuple(sorted({_safe(label) for label in labels if str(label).strip()}))
    if not subject_id or not cleaned:
        return None
    _ensure(store)
    joined = ",".join(cleaned)
    receipt_id = _receipt_id(str(subject_id), joined)
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE}
                (receipt_id, subject_id, subject_kind, labels, count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(receipt_id) DO UPDATE SET count = excluded.count""",
            (receipt_id, str(subject_id), str(kind), joined, len(cleaned),
             (at or _now()).isoformat()),
        )
    return Receipt(receipt_id, str(subject_id), str(kind), cleaned, len(cleaned))


def record_sanitized(
    store: Any,
    subject_id: str,
    sanitized: Any,
    *,
    kind: str = "memory",
    at: datetime | None = None,
) -> Receipt | None:
    """Record from a `SanitizedMemory`, when it actually redacted something."""
    labels = getattr(sanitized, "redactions", ()) or ()
    if not getattr(sanitized, "redacted", False) or not labels:
        if not getattr(sanitized, "redacted", False):
            return None
        labels = ("credential",)
    return record(store, subject_id, tuple(labels), kind=kind, at=at)


def for_subject(store: Any, subject_id: str) -> list[Receipt]:
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT receipt_id, subject_id, subject_kind, labels, count
            FROM {TABLE} WHERE subject_id = ? ORDER BY created_at ASC""",
        (str(subject_id),),
    ).fetchall()
    return [
        Receipt(str(row["receipt_id"]), str(row["subject_id"]), str(row["subject_kind"]),
                tuple(str(row["labels"]).split(",")) if row["labels"] else (),
                int(row["count"]))
        for row in rows
    ]


def summary_line(store: Any, subject_id: str) -> str:
    """One clause for a turn receipt, or empty when nothing was redacted."""
    receipts = for_subject(store, subject_id)
    if not receipts:
        return ""
    total = sum(item.count for item in receipts)
    labels = sorted({label for item in receipts for label in item.labels})
    things = "credential" if total == 1 else "credentials"
    return f"redacted {total} {things} ({', '.join(labels)})"


def report(store: Any, *, limit: int = 50) -> dict[str, Any]:
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT subject_kind, labels, count, created_at FROM {TABLE}
            ORDER BY created_at DESC LIMIT ?""",
        (int(limit),),
    ).fetchall()
    by_label: dict[str, int] = {}
    for row in rows:
        for label in str(row["labels"] or "").split(","):
            if label:
                by_label[label] = by_label.get(label, 0) + int(row["count"])
    total = store._conn.execute(f"SELECT COALESCE(SUM(count), 0) FROM {TABLE}").fetchone()[0]
    return {
        "redactions": int(total or 0),
        "receipts": len(rows),
        "by_label": dict(sorted(by_label.items(), key=lambda item: -item[1])),
    }


def scrub(store: Any, subject_id: str) -> int:
    """Remove the receipts for a subject (the memory itself is going away)."""
    _ensure(store)
    with store.transaction() as conn:
        cursor = conn.execute(f"DELETE FROM {TABLE} WHERE subject_id = ?", (str(subject_id),))
        return int(cursor.rowcount or 0)
