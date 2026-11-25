"""The as-of slider: what the store believed on a date, and what changed since (#164).

`as_of.py` reconstructs one instant from the **version log**, and it is careful to say that it answers
system time. This module is the surface on top of it: a date typed into a slider, a count that can be
checked, and a difference between two dates. It reads `memories` (`created_at` / `valid_to`) rather than
re-deriving the version log, because the slider asks a coarser question than the log does - *which
beliefs existed then* - and answering it from the current row plus its retirement date keeps the number
reproducible against a plain `SELECT`.

Five rules, each one a bug that a naive filter ships:

- **the window is half-open and stated.** A belief covers `[created_at, valid_to)`, so a memory retired
  exactly at the boundary belongs to the "before" view and not to the "after" one. Closed-at-both-ends
  would show it in both (double-counted by a slider dragged forward), open-at-both-ends in neither (a
  belief that vanishes exactly when you look at it). The rule is printed with every view, so a number
  can be re-derived by hand instead of trusted.
- **a memory with no `valid_to` is still believed now**, so it appears in every as-of date after its
  creation. An empty retirement date is the store saying "this has not been retired", not "unknown".
- **undecidable is reported, never dropped.** A row whose `created_at` is missing, or whose `valid_to`
  is unparseable, does not go into the count as absent - an absence is a claim, and this store cannot
  make it. It goes into `unknown` with the reason, and a view that could not read the table at all
  says so (`readable = False`) instead of returning a confident zero.
- **the difference is a list, not a text diff.** "What changed between A and B" is *added* and
  *retired*, because a line diff of two views makes the reader do the set arithmetic and hides it.
- **the count travels with the window.** `4,102 beliefs` means nothing without "on 2026-10-01, over
  this rule"; both are in the sentence.

Nothing is written. The live store has 8,371 memories and **0 of them carry a `valid_to`**, so today
the retirement half of the predicate is inert and the slider is a creation-date filter - which is
itself the finding: the store retires by `state`, not by a belief window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from . import as_of

# A tombstone is not a demotion: `forget_ladder` uses it for right-to-be-forgotten and conflict
# resolution, where the content is meant to be unreachable. A historical view that re-printed it would
# quietly undo the erasure - so these never appear in an as-of view, and the number excluded is stated.
# `archived` and `cold` are *not* here: they were believed once, and archival is a ranking decision,
# not an erasure.
FORGOTTEN = ("tombstoned", "deleted")

# Printed with every answer so the arithmetic can be checked without reading this file.
RULE = "created_at <= t < valid_to (an empty valid_to stays open)"


def _stamp(value: Any) -> datetime | None:
    """An aware UTC instant, or None if the text is not a timestamp this store can place."""
    text = str(value or "").strip()
    if not text:
        return None
    if text[-1] in "Zz":
        # Python 3.10's fromisoformat rejects a trailing Z, and this repo keeps 3.10 compatibility.
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        # A timestamp with no offset is the store's own local convention; reading it as UTC is the
        # only choice that cannot shift a boundary by an arbitrary number of hours.
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _instant(when: Any) -> datetime:
    """Resolve the requested date through `as_of.instant` so both modules agree on what a bare date means."""
    moment = _stamp(as_of.instant(when))
    if moment is None:
        # `as_of.instant` falls back to "now" for empty input. Answering a different date than the one
        # asked for is worse than refusing: a slider that silently snaps to today looks like a working
        # slider and reports the wrong store.
        raise ValueError(f"not an instant: {when!r}")
    return moment


def _load(store: Any) -> tuple[list[dict[str, Any]] | None, str]:
    """Every row the view could need, or a reason the table could not be read."""
    conn = getattr(store, "_conn", None)
    if conn is None:
        return None, "no store connection"
    try:
        rows = conn.execute(
            "SELECT id, content, kind, state, confidence, created_at, valid_to FROM memories"
        ).fetchall()
    except Exception:  # noqa: BLE001 - the contract is that this view reports, not raises
        # The exception text is deliberately not echoed: a sqlite failure names the database file, and
        # this string is destined for a terminal or an agent's context.
        return None, "the memories table could not be read"
    return [dict(row) for row in rows], ""


@dataclass
class AsOfView:
    when: str = ""
    rule: str = RULE
    beliefs: list[as_of.Belief] = field(default_factory=list)
    unknown: list[dict[str, Any]] = field(default_factory=list)
    count: int = 0
    total: int = 0
    forgotten: int = 0
    readable: bool = True
    note: str = ""
    sentence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "when": self.when,
            "rule": self.rule,
            "count": self.count,
            "total": self.total,
            "forgotten": self.forgotten,
            "unknown": self.unknown,
            "readable": self.readable,
            "note": self.note,
            "sentence": self.sentence,
            "beliefs": [belief.as_dict() for belief in self.beliefs],
        }

    @property
    def ids(self) -> set[str]:
        return {belief.memory_id for belief in self.beliefs}


def at(store: Any, when: Any, *, limit: int = 200) -> AsOfView:
    """The beliefs held at one instant, the window they were tested against, and what could not be decided."""
    moment = _instant(when)
    stamp = moment.isoformat()
    rows, error = _load(store)
    if rows is None:
        # Not a zero. A broken store and an empty store are different answers, and only one of them
        # means "the store held nothing then".
        return AsOfView(
            when=stamp,
            readable=False,
            unknown=[{"memory_id": "", "reason": error}],
            note=f"cannot say what was held at {stamp}: {error}",
            sentence=f"unknown: {error}, so no count is reported for {stamp}",
        )

    beliefs: list[as_of.Belief] = []
    unknown: list[dict[str, Any]] = []
    forgotten = 0
    for row in rows:
        raw_created = str(row.get("created_at") or "").strip()
        created = _stamp(raw_created)
        if created is None:
            unknown.append({
                "memory_id": str(row.get("id") or ""),
                "reason": "created_at missing" if not raw_created else "created_at unparseable",
            })
            continue
        raw_until = str(row.get("valid_to") or "").strip()
        until = _stamp(raw_until) if raw_until else None
        if raw_until and until is None:
            unknown.append({
                "memory_id": str(row.get("id") or ""),
                "reason": "valid_to unparseable",
            })
            continue
        if created > moment:
            continue  # not yet written: the store did not hold it, so it cannot have believed it
        state = str(row.get("state") or "")
        if state in FORGOTTEN:
            forgotten += 1
            continue
        if until is not None and until <= moment:
            # Half-open: a memory retired exactly at `moment` is gone. `<=` here is the whole rule.
            continue
        beliefs.append(
            as_of.Belief(
                memory_id=str(row.get("id") or ""),
                content=str(row.get("content") or ""),
                kind=str(row.get("kind") or ""),
                state=state,
                confidence=float(row.get("confidence") or 0.0),
                since=raw_created,
                until=raw_until,
            )
        )

    beliefs.sort(key=lambda belief: (belief.since, belief.memory_id))
    total = len(beliefs)
    kept = beliefs[: max(0, int(limit))]
    view = AsOfView(
        when=stamp,
        beliefs=kept,
        unknown=unknown,
        count=len(kept),
        total=total,
        forgotten=forgotten,
    )
    view.sentence = (
        f"{total} belief(s) held at {stamp} [{view.rule}], "
        f"{len(unknown)} undecidable, {forgotten} erased"
    )
    if len(kept) < total:
        view.note = f"showing {len(kept)} of {total}"
    if not total and not unknown:
        view.note = "the store held nothing that satisfies the window"
    return view


@dataclass
class AsOfDiff:
    start: str = ""
    end: str = ""
    added: list[as_of.Belief] = field(default_factory=list)
    retired: list[as_of.Belief] = field(default_factory=list)
    still: int = 0
    added_total: int = 0
    retired_total: int = 0
    unknown: list[dict[str, Any]] = field(default_factory=list)
    readable: bool = True
    note: str = ""
    sentence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "added": [belief.as_dict() for belief in self.added],
            "retired": [belief.as_dict() for belief in self.retired],
            "still": self.still,
            "added_total": self.added_total,
            "retired_total": self.retired_total,
            "unknown": self.unknown,
            "readable": self.readable,
            "note": self.note,
            "sentence": self.sentence,
        }


def between(store: Any, start: Any, end: Any, *, limit: int = 200) -> AsOfDiff:
    """Added and retired beliefs between two instants - the set difference, not a text diff.

    Both endpoints go through the same half-open rule as `at()`, which is what makes the pair
    consistent: a memory retired exactly at `end` is in the first view and not the second, so it is
    reported as retired once, not as both retired and added.
    """
    first, second = _instant(start), _instant(end)
    before = at(store, first.isoformat(), limit=100000)
    after = at(store, second.isoformat(), limit=100000)
    diff = AsOfDiff(start=first.isoformat(), end=second.isoformat())
    if not before.readable or not after.readable:
        diff.readable = False
        diff.unknown = before.unknown + after.unknown
        diff.note = "one of the two endpoints could not be read, so no difference is claimed"
        diff.sentence = f"unknown: {diff.note}"
        return diff

    before_ids, after_ids = before.ids, after.ids
    # Sets, not text: an id present in one view and not the other is exactly one thing - a belief that
    # arrived, or one that stopped being held. A content diff would report a reworded belief as a
    # retirement plus an addition, which is a different claim.
    diff.added = [belief for belief in after.beliefs if belief.memory_id not in before_ids]
    diff.retired = [belief for belief in before.beliefs if belief.memory_id not in after_ids]
    diff.still = len(before_ids & after_ids)
    diff.added_total, diff.retired_total = len(diff.added), len(diff.retired)
    diff.added = diff.added[: max(0, int(limit))]
    diff.retired = diff.retired[: max(0, int(limit))]
    diff.unknown = _merge_unknown(before.unknown, after.unknown)
    diff.sentence = (
        f"{diff.added_total} belief(s) added and {diff.retired_total} retired between "
        f"{first.isoformat()} and {second.isoformat()}, {diff.still} unchanged, "
        f"{len(diff.unknown)} undecidable"
    )
    if first > second:
        diff.note = "start is after end: 'added' and 'retired' are read backwards"
    return diff


def _merge_unknown(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []
    for group in groups:
        for item in group:
            key = (str(item.get("memory_id") or ""), str(item.get("reason") or ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def render(view: AsOfView) -> list[str]:
    """A terminal rendering: the sentence first, then the beliefs, then the caveats."""
    lines = [view.sentence]
    for belief in view.beliefs:
        lines.append("  " + belief.describe())
    if view.unknown:
        reasons: dict[str, int] = {}
        for item in view.unknown:
            reason = str(item.get("reason") or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        lines.append("  undecidable: " + ", ".join(f"{reason} {count}"
                                                   for reason, count in sorted(reasons.items())))
    if view.note:
        lines.append("  " + view.note)
    return lines


def render_diff(diff: AsOfDiff) -> list[str]:
    lines = [diff.sentence]
    for belief in diff.added:
        lines.append("  + " + belief.describe())
    for belief in diff.retired:
        lines.append("  - " + belief.describe())
    if diff.note:
        lines.append("  " + diff.note)
    return lines
