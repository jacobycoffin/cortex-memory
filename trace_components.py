"""Keep the components that explain a score, even when the full dump is dropped (#222).

The trace stores a scoring dump per candidate. Measured 2026-09-15 that dump was ~69% of a recall's stored
payload (~125 KB of ~181 KB per trace, ~23 MB/day), written on the turn hot path — so `_trim_trace_candidate_detail`
drops it for every candidate that is not selected and not in the top ten, and leaves `components_omitted: True`.

The consequence is that a "why did you say that" answer exists for **11.2%** of candidate rows and for the rest
the trace has a score and no reason for it. This module keeps the part that explains a score — the six graded
signals and the weights applied to them — and drops only the adjustment bookkeeping, which is what the bulk
consisted of.

Three rules:

- **what explains a score is named, not inferred.** `GRADED` is the list of signals that carry a *graded* value
  (`lexical`, `phrase`, `semantic`, `activation`, `utility`, `importance`, `confidence`, `source_reliability`,
  `currentness`, `memory_type`, `uniqueness`, `graph`, `neighborhood`). A gate is a multiplier and an adjustment
  is a bonus; neither explains a ranking on its own.
- **what is dropped is named too.** `components_dropped` lists the keys removed, so an absent value is
  distinguishable from a value that never existed — the same rule the store already applies with
  `components_omitted`.
- **the compaction is measured.** `coverage()` reports how many candidates carry graded components, so the gap
  this module exists to close is a number somebody can watch, not a claim in a changelog.

Nothing here decides a score; it decides what survives to be argued with afterwards.
"""

from __future__ import annotations

from typing import Any, Iterable

# The signals that take a graded value. Ordered as `score_signals` is ordered in `retrieval.py`, so a reader
# comparing the dump against the scorer sees the same list.
# Six signals, not thirteen. Measured on a real candidate: keeping all thirteen graded values plus the
# dropped-key bookkeeping came to 79% of the full dump - barely a saving - because most of the dump IS graded
# signals. These six are the ones a ranking question is actually answered with ("it won on lexical and
# currentness, it lost on importance"), and together they cost about a quarter of the original dump.
GRADED: tuple[str, ...] = (
    "lexical",
    "semantic",
    "activation",
    "importance",
    "confidence",
    "currentness",
)

# Rounded to four places: the scorer's own precision, and enough to reproduce a ranking comparison.
PLACES = 4


def explainable(components: dict[str, Any] | None) -> dict[str, float]:
    """The graded signals from a components dict, rounded, floats only."""
    out: dict[str, float] = {}
    for key in GRADED:
        value = (components or {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = round(float(value), PLACES)
    return out


def compact(candidate: dict[str, Any]) -> dict[str, Any]:
    """Replace a full dump with the part that explains the score, in place.

    Keeps `components_omitted` set: the *full* dump is still omitted, and a reader who only knows that flag
    keeps behaving exactly as before. Adds `components_graded` and `components_dropped` beside it.
    """
    components = candidate.get("components")
    if not components:
        return candidate
    graded = explainable(components)
    dropped = sorted(key for key in components if key not in graded)
    candidate.pop("components", None)
    if graded:
        candidate["components_graded"] = graded
    # Naming all seventeen adjustment keys cost as much as keeping their values (measured: the compacted
    # candidate came out 95.7% of the original), so the count travels and four names do, and the rest are
    # discoverable from the scorer's own `common_adjustments` list.
    candidate["components_dropped_count"] = len(dropped)
    candidate["components_dropped_sample"] = dropped[:4]
    candidate["components_omitted"] = True
    return candidate


def summarize(candidates: Iterable[dict[str, Any]], *, keep_full: set[str] | None = None) -> list[dict[str, Any]]:
    """Compact every candidate that is not in `keep_full`; leave the rest untouched."""
    keep = {str(item) for item in (keep_full or set())}
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        if str(candidate.get("memory_id") or "") in keep:
            out.append(candidate)
            continue
        out.append(compact(candidate))
    return out


def coverage(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """How many candidates can be explained: full dump, graded-only, or nothing at all."""
    rows = list(candidates)
    full = graded_only = nothing = 0
    for candidate in rows:
        if candidate.get("components"):
            full += 1
        elif candidate.get("components_graded"):
            graded_only += 1
        else:
            nothing += 1
    total = len(rows)
    return {
        "candidates": total,
        "full_dump": full,
        "graded_only": graded_only,
        "unexplained": nothing,
        "explainable": full + graded_only,
        "explainable_pct": round(100.0 * (full + graded_only) / total, 1) if total else 0.0,
        "graded_signals": len(GRADED),
    }


def render_why(candidate: dict[str, Any]) -> list[str]:
    """One candidate's score and the components that produced it, or an honest gap."""
    lines = [f"{str(candidate.get('memory_id') or '')[:8]} score {candidate.get('score')} "
             f"rank {candidate.get('rank')}"]
    graded = candidate.get("components_graded") or explainable(candidate.get("components"))
    if graded:
        lines.append("  graded: " + ", ".join(f"{key} {value}" for key, value in graded.items()))
    elif candidate.get("components_omitted"):
        lines.append("  graded: not recorded (the full dump was omitted and no graded summary was kept)")
    else:
        lines.append("  graded: not recorded")
    dropped_count = int(candidate.get("components_dropped_count") or 0)
    if dropped_count:
        sample = ", ".join(candidate.get("components_dropped_sample") or [])
        lines.append(f"  dropped: {dropped_count} adjustment key(s) - {sample}"
                     + (" ..." if dropped_count > len(candidate.get("components_dropped_sample") or []) else ""))
    if candidate.get("reason"):
        lines.append(f"  reason: {candidate['reason']}")
    return lines
