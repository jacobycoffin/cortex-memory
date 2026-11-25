"""Repeated episodes becoming one durable fact, with citations (#104).

This is the first genuinely *learning* item in the ladder: the store has 4,717 episodes — what was asked and what
was said — and a fact is what survives repetition. The failure this prevents is the opposite of forgetting: an
assistant that treats a single conversation turn as knowledge. A one-off mention is an episode; the third time a
person says the same thing in a different session, it is something they know.

Five rules, each a way a promotion loop corrupts a store:

- **repetition must cross sessions.** Two identical sentences in one conversation are one thought restated, not
  corroboration: the bar is `MIN_SESSIONS` distinct sessions and `MIN_EPISODES` occurrences, and both numbers
  travel with the candidate.
- **citations are mandatory and travel with the fact.** The promoted memory's `source_ref` names the episodes it
  came from. A durable fact nobody can trace back to a moment is a rumour that will be defended later.
- **a contradiction blocks the promotion instead of picking a side.** If two clusters say the same thing with a
  different number or a negation, neither is promoted, and the refusal names both.
- **the promoted text is the most common spelling, not the longest.** The longest variant is where the rambling
  lives; the mode is what the person actually keeps saying.
- **promotion is idempotent and reversible.** Promoting the same cluster twice reports `already` rather than
  writing a second fact, and the fact is written by the ordinary `add_memory` path so it can be corrected or
  retired like anything else.

Matching here is exact after normalisation. That is deliberately the conservative end, and the measurement on the
live store says what it costs: see `report()`.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

EPISODE_TABLE = "episodes"
MEMORY_TABLE = "memories"

# Three mentions across two sessions. Both numbers are stated in every decision so a reader can argue with them.
MIN_EPISODES = 3
MIN_SESSIONS = 2

# Episodes shorter than this carry no assertable fact ("ok", "yes", "thanks"), and a promoted memory made of one
# is noise the store will keep forever.
MIN_WORDS = 4

_NOISE = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")

# A question is not a fact. Promoting one would make the store assert the thing it was asked.
_QUESTION_STARTS = ("what", "how", "why", "when", "where", "who", "which", "can", "could", "should", "is",
                    "are", "do", "does", "did")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def normalise(text: Any) -> str:
    """Case, punctuation and spacing folded - nothing else. Two statements match only if they say the same thing."""
    body = _SPACE.sub(" ", _NOISE.sub(" ", str(text or "").lower())).strip()
    return body


# Scaffolding that arrives as user content: the cron wrapper, the task preamble, the system block. Measured on
# the live store, the most-repeated multi-session string is the cron preamble (19 sessions) - exact repetition
# finds boilerplate before it finds knowledge, and promoting it would write the harness into memory as a fact.
BOILERPLATE = (
    re.compile(r"\[\s*IMPORTANT", re.I),
    re.compile(r"\[\s*system note", re.I),
    re.compile(r"you are running as a", re.I),
    re.compile(r"you are hermes", re.I),
    re.compile(r"^deliver\b", re.I),
    re.compile(r"scheduled cron job", re.I),
    # Instruction-shaped: a command aimed at the assistant is not something the person knows. Measured on the
    # live store, every one of the ten "eligible" clusters was in this class or the one above - the most
    # repeated multi-session strings in a real store are smoke tests ("Reply with exactly PONG.") and system
    # notes, so verbatim repetition finds the harness before it finds knowledge.
    re.compile(r"^(reply|respond|say|send|print|output|answer)\b", re.I),
    re.compile(r"^\s*\["),
    re.compile(r"can you\b|could you\b|would you\b|will you\b", re.I),
)

# A string said in more than this share of *all* sessions is a template, not a belief: a person does not repeat
# one sentence in a fifth of everything they ever say.
BROADCAST_SHARE = 0.10


def is_boilerplate(text: Any) -> bool:
    body = str(text or "").strip()
    return any(pattern.search(body) for pattern in BOILERPLATE)


def is_assertable(text: Any) -> tuple[bool, str]:
    body = str(text or "").strip()
    if is_boilerplate(body):
        return False, ("scaffolding or an instruction, not a statement: this is the harness or a request "
                       "aimed at the assistant, not something the person knows")
    if len(body.split()) < MIN_WORDS:
        return False, f"fewer than {MIN_WORDS} words: not enough to assert"
    first = body.split()[0].lower().strip(".,:;!?")
    if first in _QUESTION_STARTS and body.endswith("?"):
        return False, "question-shaped: a promoted fact must not assert the question"
    if body.endswith("?"):
        return False, "question-shaped: a promoted fact must not assert the question"
    return True, ""


@dataclass
class Cluster:
    """One repeated statement and the episodes that carry it."""

    key: str = ""
    text: str = ""
    episode_ids: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)
    count: int = 0
    promoted_id: str = ""
    refusal: str = ""

    @property
    def eligible(self) -> bool:
        return (not self.refusal and self.count >= MIN_EPISODES
                and len(self.sessions) >= MIN_SESSIONS and not self.promoted_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "text": self.text,
            "count": self.count,
            "sessions": len(self.sessions),
            "episode_ids": self.episode_ids,
            "eligible": self.eligible,
            "promoted_id": self.promoted_id,
            "refusal": self.refusal,
        }

    def render(self) -> list[str]:
        state = "eligible" if self.eligible else (self.refusal or "below the bar")
        return [f"{self.key}: {self.text[:60]!r} x{self.count} in {len(self.sessions)} session(s) - {state}"]


def _episodes(store: Any, *, limit: int) -> list[dict[str, Any]] | None:
    return _rows(store, f"SELECT id, session_id, user_content, created_at FROM {EPISODE_TABLE} "
                        "ORDER BY created_at DESC LIMIT ?", (int(limit),))


def _promoted_keys(store: Any) -> set[str]:
    rows = _rows(store, f"SELECT content FROM {MEMORY_TABLE} WHERE source_type = 'episode_promotion' "
                        "AND coalesce(state, '') <> 'tombstoned'")
    return {normalise(row.get("content")) for row in rows or []}


def _session_total(store: Any) -> int:
    rows = _rows(store, f"SELECT COUNT(DISTINCT session_id) AS n FROM {EPISODE_TABLE}")
    return int(rows[0]["n"]) if rows else 0


def clusters(store: Any, *, limit: int = 5000) -> list[Cluster]:
    """Group episodes that say the same thing, best-supported first. Reads only."""
    rows = _episodes(store, limit=limit)
    if rows is None:
        return []
    session_total = _session_total(store)
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        text = str(row.get("user_content") or "")
        key = normalise(text)
        if not key:
            continue
        ok, why = is_assertable(text)
        group = groups.setdefault(key, {"spellings": Counter(), "ids": [], "sessions": set(), "why": ""})
        group["spellings"][text.strip()] += 1
        group["ids"].append(str(row.get("id")))
        session = str(row.get("session_id") or "")
        if session:
            group["sessions"].add(session)
        if not ok and not group["why"]:
            group["why"] = why

    promoted = _promoted_keys(store)
    out: list[Cluster] = []
    for key, group in groups.items():
        cluster = Cluster(
            key=hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
            text=group["spellings"].most_common(1)[0][0],
            episode_ids=group["ids"],
            sessions=sorted(group["sessions"]),
            count=len(group["ids"]),
            refusal=group["why"],
        )
        if key in promoted:
            cluster.refusal = "already promoted"
        elif not cluster.refusal and session_total and len(cluster.sessions) > max(
                3, int(BROADCAST_SHARE * session_total)):
            # Said in more than a tenth of every session this store has: a template, not a belief.
            cluster.refusal = (f"appears in {len(cluster.sessions)} of {session_total} sessions: that is a "
                               "broadcast or a template, not something a person keeps saying")
        out.append(cluster)
    out.sort(key=lambda item: (item.count, len(item.sessions)), reverse=True)
    return out


def pending(store: Any, *, limit: int = 5000) -> list[Cluster]:
    """Clusters that clear the bar and have not been promoted yet."""
    return [cluster for cluster in clusters(store, limit=limit) if cluster.eligible]


def _contradiction(store: Any, cluster: Cluster, others: Iterable[Cluster]) -> Cluster | None:
    """Another cluster that shares this one's skeleton but differs in a number or a negation.

    Exact matching means two phrasings of the same claim land in different clusters; when the difference is a
    number or a "not", that is a contradiction and neither may be promoted.
    """
    base = set(normalise(cluster.text).split())
    for other in others:
        if other.key == cluster.key:
            continue
        words = set(normalise(other.text).split())
        if not base or not words:
            continue
        shared = base & words
        if len(shared) / max(1, len(base | words)) < 0.6:
            continue
        digits_differ = bool({w for w in base if w.isdigit()} ^ {w for w in words if w.isdigit()})
        negation_differs = ("not" in base) != ("not" in words)
        if digits_differ or negation_differs:
            return other
    return None


@dataclass
class Result:
    ok: bool = False
    status: str = ""  # promoted | already | refused | unknown | stale
    reason: str = ""
    cluster: str = ""
    fact_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "status": self.status, "reason": self.reason,
                "cluster": self.cluster, "fact_id": self.fact_id}

    def render(self) -> list[str]:
        return [f"{self.status}: {self.reason}"]


def promote(store: Any, cluster_key: str, *, consent: bool = False, at: datetime | None = None) -> Result:
    """Write one durable fact from a repeated cluster. Never raises."""
    out = Result(cluster=str(cluster_key))
    if not consent:
        out.status, out.reason = "refused", "promotion asks first; pass consent to write the fact"
        return out
    everything = clusters(store)
    if not everything and _episodes(store, limit=1) is None:
        out.status, out.reason = "refused", "the episode table could not be read"
        return out
    cluster = next((item for item in everything if item.key == str(cluster_key)), None)
    if cluster is None:
        out.status, out.reason = "unknown", f"no episode cluster {cluster_key}"
        return out
    if cluster.refusal == "already promoted":
        out.status, out.reason = "already", "this statement is already a fact"
        return out
    if cluster.refusal:
        out.status, out.reason = "refused", cluster.refusal
        return out
    if cluster.count < MIN_EPISODES:
        out.status = "refused"
        out.reason = f"{cluster.count} episode(s): a fact needs {MIN_EPISODES}"
        return out
    if len(cluster.sessions) < MIN_SESSIONS:
        out.status = "refused"
        out.reason = (f"{len(cluster.sessions)} session(s): repetition inside one conversation is one thought "
                      f"restated, not corroboration (needs {MIN_SESSIONS})")
        return out
    clash = _contradiction(store, cluster, everything)
    if clash is not None:
        out.status = "refused"
        out.reason = (f"contradicts another repeated statement ({clash.text[:60]!r}); neither is promoted "
                      "until one of them stops being said")
        return out

    moment = at or _now()
    confidence = min(0.9, 0.6 + 0.05 * cluster.count)
    fact_id = store.add_memory(
        content=cluster.text,
        kind="semantic",
        source_type="episode_promotion",
        source_category="USER_STATED",
        source_ref="episodes:" + ",".join(cluster.episode_ids[:20]),
        session_id=cluster.sessions[-1] if cluster.sessions else None,
        source_context=f"said {cluster.count} times across {len(cluster.sessions)} sessions",
        confidence=confidence,
        observed_at=moment.isoformat(),
    )[0]
    out.ok = True
    out.status = "promoted"
    out.fact_id = fact_id
    out.reason = (f"{cluster.count} episodes across {len(cluster.sessions)} sessions -> one fact "
                  f"({fact_id[:8]}), cited to {len(cluster.episode_ids)} episode(s)")
    return out


def report(store: Any, *, limit: int = 5000) -> dict[str, Any]:
    """What the loop can see: how many clusters exist, how many clear the bar, and why the rest do not."""
    everything = clusters(store, limit=limit)
    reasons = Counter()
    for cluster in everything:
        if cluster.refusal:
            reasons[cluster.refusal] += 1
        elif cluster.count < MIN_EPISODES:
            reasons["seen once or twice"] += 1
        elif len(cluster.sessions) < MIN_SESSIONS:
            reasons["one session only"] += 1
        else:
            reasons["eligible"] += 1
    return {
        "episodes": sum(cluster.count for cluster in everything),
        "clusters": len(everything),
        "eligible": len(pending(store, limit=limit)),
        "reasons": dict(reasons),
        "bar": {"min_episodes": MIN_EPISODES, "min_sessions": MIN_SESSIONS, "min_words": MIN_WORDS},
    }
