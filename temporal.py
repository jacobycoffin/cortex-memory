"""Temporal validity for stored memories — deterministic, no model, no network.

WHY THIS EXISTS. A memory can be perfectly *relevant* and still be dangerous: it
mixes durable facts with values that were only true at capture. Real examples
from the live corpus, all three flagged by the operator unprompted:

  * ``Proxmox > Host Specs``      — "Uptime 36.5 days", "Load avg 1.64", "PVE 8.4.19"
  * ``Hermes Agent > Default model`` — a model pin plus "(verified 2026-09-02)"
  * ``Debt Overview``             — balances and APRs under "as of 2026-08-07"

Each was a CORRECT retrieval. The defect is that nothing records *which* parts of
the record were only true at a moment, so a stale number can be served as current.

Why the existing machinery is not enough: `record_role` plus the role-gated
temporal window in ``retrieval.py`` only fire for ``role == "event"``, and the
live corpus contains zero event-role records. Those three memories are
*durable-role records that merely contain* volatile values — a distinction no
role can express.

This module classifies, and nothing more. It does not rewrite, summarise, or
score. Callers decide what to do with the verdict (see ``ACTION_*`` below).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

__all__ = [
    "TemporalVerdict",
    "classify_temporal",
    "DURABLE",
    "MIXED",
    "VOLATILE",
    "MOSTLY_DURABLE",
    "CLARITY_SAFE_FLAGS",
]

DURABLE = "durable"
MOSTLY_DURABLE = "mostly_durable"
MIXED = "mixed"
VOLATILE = "volatile"

# Flags this module may add to a record's readability flags. They are
# deliberately NOT members of refinery.CLARITY_FLAGS: clarity flags feed role
# classification, and a temporal observation must never change a record's role.
CLARITY_SAFE_FLAGS = frozenset(
    {
        "temporal_volatile_values",
        "temporal_mixed",
        "temporal_undated",
        "temporal_as_of_known",
    }
)

VOLATILE_MARKERS: list[tuple[str, int]] = [
    (r"\buptime\b", 3),
    (r"\bload (avg|average)\b", 3),
    (r"\b(as of|as-of|snapshot|at the time of|point-in-time)\b", 3),
    (r"\b(estimated|live) balance\b", 3),
    (r"\bstatement balance\b", 3),
    (r"\b(apr|interest charged)\b", 2),
    (r"\b\d+(\.\d+)?\s?%\s*(used|free|utilisation|utilization)\b", 3),
    (r"\bused \(\d+(\.\d+)?%\)", 3),
    (r"\b(days|hours|minutes)\s+(uptime|ago)\b", 2),
    (r"\blast (updated|checked|recheck|verified)\b", 2),
    # "currently / right now / at present" is deliberately ABSENT: it fires on
    # preferences ("build a website like mine currently") and produces nonsense.
    (r"\bverified \d{4}-\d{2}-\d{2}", 3),
    (r"\bupdated from\b", 2),
    (r"\b(live|current) (snapshot|plaid|data|state|balances?)\b", 3),
    (r"\bcurrent balances?\b", 3),
    (r"\bmonthly statements?\b", 2),
    (r"\best\.?\s*\d{1,2}/\d{1,2}", 3),
    (r"\brecheck\b|re-check", 2),
    (r"\b(free|used)\s+(space|disk|memory|ram)\b", 2),
    (r"\b(pending|due|scheduled)\b", 1),
    (r"\b(yesterday|today|tonight|this (week|month))\b", 2),
    # A product + version pin ("Muse Spark 1.3", "PVE 8.4.19", "HomeBox v0.25.x").
    # (?-i: ...) keeps it case-sensitive: a case-insensitive scan would match any
    # lowercase word before a decimal, e.g. "the 5.49".
    (r"(?-i:\b[A-Z][\w.-]*\s+v?\d+\.\d+(\.\d+|\.x)?\b)", 2),
    # A version-LABELLED field, which is how tables actually store it
    # ("| PVE version | 8.4.19 |"): the number is not adjacent to a product name.
    (r"\b(version|revision|build)\b[^\n]{0,24}\b\d+\.\d+(\.\d+)?", 2),
    (r"\bdowntime\b|\bcrash(ed)?\b", 1),
]

DURABLE_MARKERS: list[tuple[str, int]] = [
    (r"\b(hostname|ip address|mac address|host|ip)\b", 2),
    (r"\b\d{1,3}(\.\d{1,3}){3}\b", 2),
    (r"\b(model|chipset|cpu)\b", 1),
    (r"\b(prefer|prefers|preference|always|never|do not|don't)\b", 2),
    (r"\b(path|directory|repo|repository|url|endpoint|port)\b", 1),
    (r"\b(policy|rule|convention|house style)\b", 2),
    (r"(?<![\w/])(/[\w./-]{3,})", 1),
    (r"\b(username|account|credential|token|key)\b", 1),
    (r"\b(installed|configured|set up|built)\b", 1),
]

# A value that only means something at one moment. Deliberately excludes bare
# digits: a date is itself digits, so `\b\d{2,}\b` would flag every dated record.
DATED_VALUE = [
    r"\$\s?\d[\d,]*(\.\d{2})?",
    r"\d+(\.\d+)?\s?%",
    r"\bv?\d+\.\d+(\.\d+|\.x)?\b",
]

DATE_PATTERNS = [
    r"\b(\d{4}-\d{2}-\d{2})\b",
    r"\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})",
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
    r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",
]

ACTION_DURABLE = "leave as-is; no volatile values detected"
ACTION_VOLATILE_ONLY = (
    "dated observation only: never present as a current fact; decay currentness from as_of"
)
ACTION_MIXED = (
    "split the durable record from a dated observation, or render with an explicit as-of "
    "stamp and mark the volatile values; do not let the volatile half decay the durable half"
)
ACTION_MOSTLY_DURABLE = "stamp volatile values at render time; keep in normal recall"


@dataclass
class TemporalVerdict:
    classification: str = DURABLE
    volatile_score: int = 0
    durable_score: int = 0
    volatile_markers: list[str] = field(default_factory=list)
    durable_markers: list[str] = field(default_factory=list)
    as_of: str | None = None
    undated: bool = False
    action: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def needs_stamp(self) -> bool:
        """True when a caller should surface an as-of / staleness note."""
        return self.classification in {MIXED, VOLATILE} or self.classification == MOSTLY_DURABLE

    def flags(self) -> list[str]:
        out: list[str] = []
        if self.classification == MIXED:
            out.append("temporal_mixed")
        if self.classification == VOLATILE:
            out.append("temporal_volatile_values")
        if self.volatile_score > 0:
            out.append("temporal_volatile_values")
        if self.volatile_score > 0 and not self.as_of:
            out.append("temporal_undated")
        if self.as_of:
            out.append("temporal_as_of_known")
        return sorted(set(out))


def _scan(text: str, patterns: list[tuple[str, int]]) -> tuple[list[str], int]:
    hits: list[str] = []
    score = 0
    for pattern, weight in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            hits.append(match.group(0)[:40])
            score += weight
    return hits, score


def _valid_date(value: str) -> bool:
    """True only for a genuinely valid calendar date.

    The classifier must not adopt a stamp it cannot trust: a malformed value
    like "2026-99-99" is evidence the text carries a date-shaped token, but it
    must never become the record's as_of.
    """

    value = value.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M"):
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue
    for fmt in ("%b %d, %Y", "%b %d %Y"):
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue
    slash = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", value)
    if slash:
        month, day = int(slash.group(1)), int(slash.group(2))
        return 1 <= month <= 12 and 1 <= day <= 31
    return False


def find_as_of(text: str) -> str | None:
    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            candidate = match.group(1)
            if _valid_date(candidate):
                return candidate
    return None


def classify_temporal(text: str) -> TemporalVerdict:
    """Classify one record's text. Deterministic and side-effect free."""
    verdict = TemporalVerdict()
    if not text or not text.strip():
        verdict.action = "empty"
        return verdict

    verdict.volatile_markers, verdict.volatile_score = _scan(text, VOLATILE_MARKERS)
    verdict.durable_markers, verdict.durable_score = _scan(text, DURABLE_MARKERS)
    verdict.as_of = find_as_of(text)

    # A date beside a value is a dated observation even with no "as of" wording.
    if verdict.as_of and any(re.search(p, text, re.I) for p in DATED_VALUE):
        if not any("date+value" in m for m in verdict.volatile_markers):
            verdict.volatile_markers.append("date+value with no explicit marker")
            verdict.volatile_score += 3

    verdict.undated = verdict.volatile_score > 0 and not verdict.as_of

    if verdict.volatile_score == 0:
        verdict.classification = DURABLE
        verdict.action = ACTION_DURABLE
    elif verdict.durable_score == 0:
        verdict.classification = VOLATILE
        verdict.action = ACTION_VOLATILE_ONLY
    elif verdict.volatile_score >= 2:
        # A real volatile signal PLUS at least one durable anchor. One anchor is
        # enough: the point is that this record holds both, so the volatile half
        # must be stamped rather than left to rot the durable half.
        verdict.classification = MIXED
        verdict.action = ACTION_MIXED
    else:
        verdict.classification = MOSTLY_DURABLE
        verdict.action = ACTION_MOSTLY_DURABLE

    # Any verdict that carries point-in-time values deserves the recovered date,
    # not only the MIXED/VOLATILE cases.
    if verdict.as_of and verdict.needs_stamp:
        verdict.action = f"{verdict.action} [as_of={verdict.as_of}]"
    return verdict
