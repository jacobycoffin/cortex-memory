"""A frozen extraction set, so a model change has to prove itself (#47).

Extraction is the one stage a model swap silently rewrites: the schema stays, the tests stay green,
and the facts that come out are different. The only way to see that is to keep a set of conversations
with the facts they are *supposed* to yield and re-run it whenever the reader changes.

Three properties of this suite:

- **synthetic by construction.** Every conversation here is invented — invented hosts, invented names,
  no live memory ids — because a fixture that carries real facts cannot live in a repository.
- **the expectations are about facts, not wording.** A case says which subject must appear and what
  must not, matched on tokens, so an extractor that rephrases a clause still passes and one that
  drops the qualifier does not.
- **a baseline is what makes a diff readable.** A run records what was extracted per case; the next
  run compares against it and reports facts that appeared, facts that vanished, and cases whose
  outcome flipped. "It still passes" is not the same as "it still says the same thing".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

FIXTURE = Path(__file__).resolve().parent / "tests" / "fixtures" / "extraction_cases.jsonl"


def default_extractor() -> Callable[[str, str], list[dict[str, Any]]]:
    """The deterministic extractor, wrapped into the shape the suite expects."""
    from . import extraction

    def run(text: str, role: str) -> list[dict[str, Any]]:
        return [{"content": candidate.content, "kind": candidate.kind}
                for candidate in extraction.extract_candidates(text, role=role)]

    return run


@dataclass
class Case:
    case_id: str
    turns: list[dict[str, str]]
    expect: list[dict[str, Any]] = field(default_factory=list)
    forbid: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.case_id, "turns": self.turns, "expect": self.expect,
                "forbid": self.forbid, "note": self.note}


@dataclass
class Outcome:
    case_id: str
    ok: bool
    extracted: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.case_id, "ok": self.ok, "extracted": self.extracted,
                "missing": self.missing, "forbidden": self.forbidden, "reason": self.reason}


def load(path: Path | None = None) -> list[Case]:
    """Read the frozen cases. Each line is one conversation."""
    target = Path(path) if path else FIXTURE
    cases: list[Case] = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        payload = json.loads(stripped)
        cases.append(
            Case(
                case_id=str(payload.get("id") or f"case-{number}"),
                turns=[{"role": str(turn.get("role") or "user"),
                        "text": str(turn.get("text") or "")}
                       for turn in payload.get("turns") or []],
                expect=list(payload.get("expect") or []),
                forbid=[str(item) for item in payload.get("forbid") or []],
                note=str(payload.get("note") or ""),
            )
        )
    return cases


_SPLIT = __import__("re").compile(r"[^a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Split on everything that is not alphanumeric.

    Wording differs in punctuation far more than in words — "included." and "included", "sk-live"
    inside "sk-live-abc" — and a matcher that treats those as different tokens fails cases the
    extractor got right.
    """
    return {word for word in _SPLIT.split(str(text or "").lower()) if len(word) > 1}


def _contains(content: str, wanted: str) -> bool:
    """Every token of the expectation appears in the candidate, in any order."""
    vocabulary = _tokens(content)
    needed = _tokens(wanted)
    return bool(needed) and needed.issubset(vocabulary)


def run_case(case: Case, extractor: Callable[[str, str], list[dict[str, Any]]] | None = None) -> Outcome:
    reader = extractor or default_extractor()
    extracted: list[dict[str, Any]] = []
    for turn in case.turns:
        extracted.extend(reader(turn["text"], turn["role"]))
    outcome = Outcome(case.case_id, True, extracted=extracted)
    for expectation in case.expect:
        wanted = str(expectation.get("contains") or expectation.get("fact") or "")
        kind = str(expectation.get("kind") or "")
        matches = [
            item for item in extracted
            if _contains(str(item.get("content") or ""), wanted)
            and (not kind or str(item.get("kind") or "") == kind)
        ]
        if not matches:
            outcome.ok = False
            outcome.missing.append(wanted)
    for forbidden in case.forbid:
        if any(_contains(str(item.get("content") or ""), forbidden) for item in extracted):
            outcome.ok = False
            outcome.forbidden.append(forbidden)
    if not outcome.ok:
        parts = []
        if outcome.missing:
            parts.append(f"missing {len(outcome.missing)} expected fact(s)")
        if outcome.forbidden:
            parts.append(f"extracted {len(outcome.forbidden)} forbidden fact(s)")
        outcome.reason = "; ".join(parts)
    return outcome


def suite(*, path: Path | None = None, extractor: Callable[..., Any] | None = None,
          only: Iterable[str] = ()) -> dict[str, Any]:
    """Run the whole frozen set and say what happened, case by case."""
    wanted = {name for name in only}
    cases = [case for case in load(path) if not wanted or case.case_id in wanted]
    outcomes = [run_case(case, extractor) for case in cases]
    failed = [outcome for outcome in outcomes if not outcome.ok]
    return {
        "cases": len(cases),
        "passed": len(outcomes) - len(failed),
        "failed": len(failed),
        "failures": [outcome.as_dict() for outcome in failed],
        "results": [outcome.as_dict() for outcome in outcomes],
    }


def baseline(result: dict[str, Any], *, model: str = "") -> dict[str, Any]:
    """What this run produced, per case — the thing a later run is compared against."""
    return {
        "model": model,
        "cases": {
            entry["id"]: sorted(
                f"{item.get('kind') or 'fact'}: {str(item.get('content') or '')}"
                for item in entry["extracted"]
            )
            for entry in result["results"]
        },
    }


def compare(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Facts that appeared, facts that vanished, and cases that flipped.

    A model change is allowed to improve extraction; it is not allowed to change it silently.
    """
    before = previous.get("cases") or {}
    after = current.get("cases") or {}
    appeared: dict[str, list[str]] = {}
    vanished: dict[str, list[str]] = {}
    for case_id, facts in after.items():
        new = [fact for fact in facts if fact not in (before.get(case_id) or [])]
        if new:
            appeared[case_id] = new
    for case_id, facts in before.items():
        gone = [fact for fact in facts if fact not in (after.get(case_id) or [])]
        if gone:
            vanished[case_id] = gone
    return {
        "model_before": previous.get("model", ""),
        "model_after": current.get("model", ""),
        "appeared": appeared,
        "vanished": vanished,
        "cases_added": sorted(set(after) - set(before)),
        "cases_removed": sorted(set(before) - set(after)),
        "changed": bool(appeared or vanished),
    }
