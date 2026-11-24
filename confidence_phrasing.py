"""Say no more than the memory can support (#176)

A memory has a confidence and a provenance. The sentence that carries it into a conversation has neither,
unless the phrasing is chosen from the number. Two failures follow from that gap, and they point in
opposite directions:

- **overclaiming.** "The backup runs at 3am" is stated as fact whether the memory was verified by a tool
  (0.95) or inferred by an agent (0.55). The reader cannot tell, so the weakest belief travels with the
  strongest voice.
- **underclaiming.** Hedging everything — #147 measured a fixed 0.8 hedge threshold hedging **53.3% of
  the whole store** — makes a verified fact sound like a guess, which is its own kind of lying.

So the bands are **derived from the store's own distribution**, not from a constant somebody liked. In
this store confidence piles up at exactly **0.78 (1,859 memories) and 0.80 (973)**, which is what a
decaying or defaulted score looks like — and it means a fixed threshold would cut through the middle of
a pile-up and put near-identical beliefs on opposite sides of a hedge.

Four rules:

- **a phrase must not claim more certainty than the memory has**, and the cut points come from quantiles;
- **an absolute word in a weak memory is an overclaim** — counted and reported, never silently rewritten,
  because the stored content is the owner's text and a checker does not get to edit it;
- **age is part of certainty.** A fact past its kind's refresh window is phrased with its date, not with
  a hedge that hides the date;
- **unknown confidence is phrased as unknown**, not as the average, because the average is a guess about
  a distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

WINDOW = 2000
# Below this gap the 25th and 75th percentiles describe the same pile-up, not two populations.
MIN_SEPARATION = 0.05
ABSOLUTE_WORDS = ("always", "never", "definitely", "certainly", "exactly", "guaranteed", "must",
                  "impossible", "every", "none")
# Days after which a fact of each kind should be stated with its date. Mirrors the forget ladder's
# refresh windows rather than inventing a second set of numbers.
STALE_AFTER_DAYS = {
    "semantic": 90,
    "preference": 365,
    "identity": 3650,
    "procedure": 180,
    "operational": 30,
    "episode": 30,
}


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


@dataclass
class Band:
    name: str
    floor: float
    phrase: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "floor": round(self.floor, 3), "phrase": self.phrase}


def bands(store: Any, *, window: int = WINDOW) -> list[Band]:
    """Cut points from the store's own quantiles: a constant here would be a guess about a distribution."""
    rows = _rows(store, "SELECT confidence FROM memories WHERE state <> 'archived' "
                        "ORDER BY created_at DESC LIMIT ?", (int(window),))
    values = [float(row["confidence"] or 0.0) for row in rows if row["confidence"] is not None]
    low = _quantile(values, 0.25)
    high = _quantile(values, 0.75)
    if low is None or high is None or high - low < MIN_SEPARATION:
        # A quantile of a degenerate distribution is as much of a guess as a constant: this store piles
        # up at 0.78-0.80, so p25 and p75 come out 0.02 apart and cannot separate anything. Say so and
        # fall back to the documented ceilings rather than pretending the cut points were measured.
        low, high = 0.60, 0.85
    return [
        Band("unsure", 0.0, "I'm not sure, but I have a note that"),
        Band("likely", low, "I think"),
        Band("confident", high, "I'm fairly sure"),
        Band("verified", max(high, 0.95), "I know"),
    ]


def phrase(confidence: float | None, *, band_list: list[Band] | None = None,
           text: str = "", days_old: int | None = None, kind: str = "semantic") -> str:
    """Choose the wording from the number, then add the date if the memory is past its refresh window."""
    ladder = band_list or [Band("unsure", 0.0, "I'm not sure, but I have a note that"),
                           Band("likely", 0.60, "I think"),
                           Band("confident", 0.85, "I'm fairly sure"),
                           Band("verified", 0.95, "I know")]
    body = str(text or "").strip()
    if confidence is None:
        opening = "I have a note (confidence unknown) that"
    else:
        chosen = ladder[0]
        for band in ladder:
            if float(confidence) >= band.floor:
                chosen = band
        opening = chosen.phrase
    sentence = f"{opening} {body}".strip() if body else opening
    limit = STALE_AFTER_DAYS.get(str(kind or ""), None)
    if days_old is not None and limit is not None and int(days_old) > limit:
        sentence += f" (as of {int(days_old)} days ago)"
    return sentence


def overclaims(text: str) -> list[str]:
    """Absolute words in a memory body. Reported, never rewritten."""
    body = str(text or "").lower()
    found = []
    for word in ABSOLUTE_WORDS:
        if body.count(word):
            found.append(word)
    return found


@dataclass
class Audit:
    examined: int = 0
    overclaiming: int = 0
    by_band: dict[str, int] = field(default_factory=dict)
    worst_words: dict[str, int] = field(default_factory=dict)
    floors: list[float] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        share = round(self.overclaiming / self.examined, 4) if self.examined else None
        return {
            "examined": self.examined, "overclaiming": self.overclaiming, "share": share,
            "by_band": self.by_band, "worst_words": self.worst_words,
            "floors": [round(value, 3) for value in self.floors], "note": self.note,
        }


def audit(store: Any, *, window: int = WINDOW) -> Audit:
    """How much stored content states more certainty than its own score supports."""
    ladder = bands(store, window=window)
    rows = _rows(store, "SELECT content, confidence FROM memories WHERE state <> 'archived' "
                        "ORDER BY created_at DESC LIMIT ?", (int(window),))
    found = Audit(examined=len(rows), floors=[band.floor for band in ladder])
    if not rows:
        found.note = "no memories in the window"
        return found
    for row in rows:
        words = overclaims(str(row.get("content") or ""))
        if not words:
            continue
        found.overclaiming += 1
        confidence = row.get("confidence")
        name = "unknown"
        if confidence is not None:
            for band in ladder:
                if float(confidence) >= band.floor:
                    name = band.name
        found.by_band[name] = found.by_band.get(name, 0) + 1
        for word in words:
            found.worst_words[word] = found.worst_words.get(word, 0) + 1
    found.worst_words = dict(sorted(found.worst_words.items(), key=lambda item: -item[1])[:5])
    if found.overclaiming == 0:
        found.note = "no absolute words found in the window"
    else:
        found.note = ("a lexical screen: it counts words, it does not read sentences — 'every 30 "
                      "minutes' is not an overclaim, and a rewrite would be an edit to the owner's text")
    return found


def render(store: Any, *, window: int = WINDOW) -> list[str]:
    described = audit(store, window=window)
    share = described.as_dict()["share"]
    lines = [f"confidence phrasing over {described.examined} memor(ies): "
             f"{described.overclaiming} overclaim ({share:.1%})" if share is not None
             else "confidence phrasing: nothing examined"]
    lines.append("  bands from the store's own quantiles: " +
                 ", ".join(f"{band.name} ≥ {band.floor:.2f}" for band in bands(store, window=window)))
    if described.by_band:
        lines.append("  overclaims by band: " +
                     ", ".join(f"{name} {count}" for name, count in sorted(described.by_band.items())))
    if described.worst_words:
        lines.append("  words: " + ", ".join(f"{word} {count}"
                                             for word, count in described.worst_words.items()))
    if described.note:
        lines.append(f"  {described.note}")
    return lines
