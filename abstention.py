"""Abstention: say "I have nothing for this" instead of the least-bad match (#14).

The mechanism is here; the policy that would use it is **off by default**, because the data
says no available signal can tell a good recall from a bad one. Measured on the frozen set
(240 cases, 152 hits):

    top-1 score      AUC 0.562   hits 0.589  misses 0.580   (ranges overlap almost fully)
    query coverage   AUC 0.480   hits 0.506  misses 0.520   (worse than chance)
    score spread     AUC 0.559   hits 0.142  misses 0.126
    result count     AUC 0.617   hits 5.099  misses 4.602
    metacognitive    AUC 0.447   hits 0.694  misses 0.700   (below chance)

The only cell with tolerable precision — `score < 0.55 and results <= 3` — silences 13
cases of which 3 are hits. Abstaining there would trade three right answers for ten wrong
ones and a "don't know", which is a worse memory than one that tries.

So this module does the defensible part: it defines the decision, applies protective
guards that must never be overridden, and stays inert until a signal that actually
separates hits from misses exists (see `scripts/abstention_features.py`,
`scripts/abstention_metacog.py`, `scripts/calibrate_abstention.py`).
"""

from __future__ import annotations

from typing import Any, Sequence

# A threshold below this cannot be meaningfully enforced: every score is above it.
MIN_USEFUL_THRESHOLD = 0.05


def should_abstain(
    results: Sequence[Any],
    *,
    threshold: float | None = None,
    min_results: int = 1,
) -> tuple[bool, str]:
    """Decide whether to return nothing. Returns (abstain, reason).

    `threshold=None` means the policy is disabled and never abstains — the shipped default.
    """
    if threshold is None:
        return False, "abstention disabled"
    if threshold <= MIN_USEFUL_THRESHOLD:
        return False, f"threshold {threshold:.3f} too low to mean anything"
    if not results:
        return False, "nothing was retrieved; nothing to abstain from"
    if len(results) < max(1, int(min_results)):
        return False, f"only {len(results)} candidate(s); not enough to judge"

    top = results[0]
    memory = getattr(top, "memory", {}) or {}
    components = getattr(top, "components", {}) or {}

    # Guards: evidence that this memory has worked before outranks a score cut-off.
    if memory.get("pinned"):
        return False, "top result is pinned"
    if float(components.get("use_boost", 0.0) or 0.0) > 0.0:
        return False, "top result has been used before"
    if float(components.get("stability", 0.0) or 0.0) > 0.0:
        return False, "top result has repeated corroboration"
    if int(components.get("negative_cue", 0) or 0) > 0:
        return False, "top result was already demoted by a negative cue"

    score = float(getattr(top, "score", 0.0) or 0.0)
    if score < threshold:
        return True, f"top score {score:.3f} below threshold {threshold:.3f}"
    return False, f"top score {score:.3f} at or above threshold {threshold:.3f}"
