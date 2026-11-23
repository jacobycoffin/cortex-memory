"""Cheap pass proposes, strong pass accepts, disagreements queue for review (#27).

Extraction has two jobs with opposite economics. **Finding** candidate facts is a recall problem: a
cheap model reading every message is affordable, and missing a fact costs more than proposing a bad
one. **Accepting** a fact is a precision problem: it writes something a future turn will treat as
true, so it deserves the strongest reader available.

Doing both in one pass forces a single model to be good at the job that is cheap and the job that is
expensive at the same time. So they are separated, and the interesting output is not the accepted
facts — it is the **disagreements**:

| cheap | strong | what it means |
|---|---|---|
| propose | accept | agreed fact — written |
| propose | reject | the cheap pass over-reached (its usual failure) |
| — | propose | the cheap pass missed something (rare, and worth knowing) |

Both kinds of disagreement become a **review candidate**, with the text that caused them and both
verdicts. A system that only keeps the agreements learns nothing about where its cheap reader is
wrong, which is the one thing you need to know to tune it.

The passes are injected callables, so the shape is testable without a model, and the default is off:
this changes what gets written, so it should be turned on with the queue watched, not by default.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

ENABLED_ENV = "CORTEX_TWO_PASS"
TABLE = "extraction_disagreements"
MIN_CHARS = 12

Proposer = Callable[[str], Sequence[str]]
Accepter = Callable[[str], bool]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    disagreement_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    cheap_verdict TEXT NOT NULL,
                    strong_verdict TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    resolution TEXT
                )"""
        )


def heuristic_propose(text: str) -> list[str]:
    """A deliberately dumb cheap pass: sentences that look like statements.

    It exists so the two-pass shape can be exercised (and the queue populated) without a model, and
    so a test can assert the disagreement path rather than the happy path.
    """
    body = " ".join(str(text or "").split())
    if len(body) < MIN_CHARS:
        return []
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", body) if part.strip()]
    return [part for part in parts if len(part) >= MIN_CHARS and not part.endswith("?")]


def accept_all(text: str) -> bool:  # pragma: no cover - trivial default
    return True


@dataclass
class Outcome:
    accepted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    disagreements: int = 0

    @property
    def wrote_something(self) -> bool:
        return bool(self.accepted)

    def describe(self) -> str:
        return (
            f"{len(self.accepted)} accepted, {len(self.rejected)} rejected, "
            f"{len(self.missed)} missed, {self.disagreements} queued for review"
        )


def run(
    store: Any,
    text: str,
    *,
    propose: Proposer = heuristic_propose,
    accept: Accepter | None = None,
    also_propose: Proposer | None = None,
    at: datetime | None = None,
) -> Outcome:
    """Cheap pass proposes, strong pass decides. Disagreements are recorded, never discarded.

    `also_propose` is the strong pass's own reading of the text, used only to notice facts the cheap
    pass missed; when it is omitted the only disagreements are over-reaches.
    """
    checker = accept or accept_all
    outcome = Outcome()
    cheap_candidates = [str(item) for item in propose(str(text or "")) if str(item).strip()]
    for candidate in cheap_candidates:
        try:
            verdict = bool(checker(candidate))
        except Exception:  # noqa: BLE001 - a failed gate is a refusal, not an acceptance
            verdict = False
        if verdict:
            outcome.accepted.append(candidate)
            continue
        outcome.rejected.append(candidate)
        _record(store, candidate, "propose", "reject", "over_reach", at=at)
        outcome.disagreements += 1

    if also_propose is not None:
        strong_only = [
            str(item) for item in also_propose(str(text or ""))
            if str(item).strip() and str(item) not in cheap_candidates
        ]
        for candidate in strong_only:
            outcome.missed.append(candidate)
            _record(store, candidate, "none", "propose", "miss", at=at)
            outcome.disagreements += 1
    return outcome


def _record(store: Any, text: str, cheap: str, strong: str, kind: str,
            *, at: datetime | None = None) -> str:
    import uuid

    _ensure(store)
    disagreement_id = uuid.uuid4().hex[:12]
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE}
                (disagreement_id, text, cheap_verdict, strong_verdict, kind, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
            (disagreement_id, str(text)[:1000], cheap, strong, kind,
             (at or _now()).isoformat()),
        )
    return disagreement_id


def queue(store: Any, *, limit: int = 20, unresolved_only: bool = True
          ) -> list[dict[str, Any]]:
    try:
        _ensure(store)
        rows = store._conn.execute(
            f"SELECT * FROM {TABLE} ORDER BY created_at DESC LIMIT ?", (int(limit) * 5,)
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    out = []
    for row in rows:
        item = dict(row)
        if unresolved_only and item.get("resolved_at"):
            continue
        out.append(item)
        if len(out) >= limit:
            break
    return out


def resolve(store: Any, disagreement_id: str, resolution: str, *, at: datetime | None = None) -> bool:
    """A person decides: `accept`, `reject`, or `ignore`. Kept, so the queue has a history."""
    try:
        _ensure(store)
        with store.transaction() as conn:
            cursor = conn.execute(
                f"""UPDATE {TABLE} SET resolved_at = ?, resolution = ?
                    WHERE disagreement_id = ? AND resolved_at IS NULL""",
                ((at or _now()).isoformat(), str(resolution)[:40], str(disagreement_id)),
            )
            return bool(cursor.rowcount)
    except Exception:  # noqa: BLE001
        return False


def report(store: Any) -> dict[str, Any]:
    rows = queue(store, limit=10000, unresolved_only=False)
    over_reach = sum(1 for row in rows if str(row["kind"]) == "over_reach")
    miss = sum(1 for row in rows if str(row["kind"]) == "miss")
    return {
        "enabled": enabled(),
        "queued": sum(1 for row in rows if not row.get("resolved_at")),
        "total": len(rows),
        "over_reach": over_reach,
        "missed": miss,
        "note": (
            "over-reach is the cheap pass's usual failure; misses are rare and worth reading"
            if rows else "no disagreements recorded yet"
        ),
    }
