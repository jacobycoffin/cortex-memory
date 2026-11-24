"""Ask about the memories that matter before stating them as current (#177)

A memory past its refresh window is not wrong — it is *unconfirmed*. The failure mode is not the stale
fact, it is the assistant asserting it in the present tense, and the fix is a question rather than a
deletion:

> Still true? "the backup runs at 3am" — noted 128 days ago, used 14 times.

Five rules, each of which exists because the opposite behaviour is annoying:

- **ask about what is used, not what is old.** Priority is *how far past its window* × *how often it has
  been recalled*. A 400-day memory nobody ever recalled is not worth his attention; a 100-day memory
  that goes into every conversation is.
- **pinned and correction memories are never prompted.** A standing instruction does not expire, and
  re-litigating a correction is how an assistant argues with its owner.
- **one question per memory, bounded.** A prompt that needs scrolling is not a prompt.
- **never ask twice inside the cooldown.** The prompt log is the module's own table, so a question asked
  yesterday is not asked again today; an assistant that repeats itself gets muted.
- **a stale memory is not demoted, rewritten or hidden** — only asked about, and the answer is his.

The refresh windows are **not redefined here**: they come from `confidence_phrasing.STALE_AFTER_DAYS`, so
"when does a fact go stale" has one answer in the codebase rather than two that drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .confidence_phrasing import STALE_AFTER_DAYS

COOLDOWN_DAYS = 45
# A memory needs an actual assertion before it is worth confirming. Below this many words of real
# content it is a stub ("Vault note: … Path: … Section: …"), and "still true?" about a stub is not a
# question, it is noise — the same template-text problem #223 measures, arriving here as a queue.
MIN_CLAIM_WORDS = 4
MAX_PROMPT_CHARS = 160
MAX_SNIPPET = 70
DEFAULT_LIMIT = 10
# States that never need confirming: a standing instruction and a correction outrank recency.
NEVER_PROMPT = ("pinned", "protected")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _parse(stamp: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def age_days(row: dict[str, Any], *, now: datetime | None = None) -> int | None:
    """Age from when the fact was observed, falling back to when it was written."""
    moment = (now or _now())
    for key in ("observed_at", "created_at"):
        stamp = _parse(row.get(key))
        if stamp is not None:
            return max(0, (moment - stamp).days)
    return None


def window_days(kind: str) -> int | None:
    return STALE_AFTER_DAYS.get(str(kind or ""), None)


def is_stale(row: dict[str, Any], *, now: datetime | None = None) -> tuple[bool, int | None, int | None]:
    limit = window_days(str(row.get("kind") or ""))
    age = age_days(row, now=now)
    if limit is None or age is None:
        return False, age, limit
    return age > limit, age, limit


def priority(row: dict[str, Any], *, now: datetime | None = None) -> float:
    """How much a confirmation is worth: past-window-ness times how often the memory is actually used."""
    stale, age, limit = is_stale(row, now=now)
    if not stale or not limit or age is None:
        return 0.0
    overdue = age / float(limit)
    uses = int(row.get("used_count") or 0) + int(row.get("retrieved_count") or 0)
    return round(overdue * (1.0 + uses), 4)


def _strip_scaffolding(text: str) -> str:
    """Reuse the vault-scaffolding stripper if it is present; a second regex would drift from the first."""
    try:
        from .topic_nodes import _strip_scaffolding as stripper  # noqa: PLC0415 - optional helper

        return stripper(text)
    except Exception:  # noqa: BLE001
        return str(text or "")


def has_claim(text: str, *, minimum: int = MIN_CLAIM_WORDS) -> bool:
    """Does this memory assert anything, once its scaffolding is removed?"""
    body = _strip_scaffolding(str(text or ""))
    return len(body.split()) >= max(1, int(minimum))


def snippet(text: str, *, limit: int = MAX_SNIPPET) -> str:
    body = " ".join(str(text or "").split())
    return body if len(body) <= limit else body[: limit - 1].rstrip() + "…"


def prompt(row: dict[str, Any], *, now: datetime | None = None) -> str:
    """One bounded question. It names the fact, its age and its use — nothing else."""
    stale, age, limit = is_stale(row, now=now)
    body = snippet(str(row.get("content") or ""))
    uses = int(row.get("used_count") or 0)
    if not stale:
        return ""
    spoken = f'Still true? "{body}" — noted {age} days ago'
    if uses:
        spoken += f", used {uses} time" + ("" if uses == 1 else "s")
    if limit:
        spoken += f" (refresh window {limit}d)"
    spoken += "."
    if len(spoken) > MAX_PROMPT_CHARS:
        spoken = spoken[: MAX_PROMPT_CHARS - 1].rstrip() + "…"
    return spoken


def ensure_schema(store: Any) -> bool:
    """The module owns one small table: when each memory was last asked about."""
    try:
        with store.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS staleness_prompts ("
                "memory_id TEXT PRIMARY KEY, prompted_at TEXT NOT NULL)"
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def record(store: Any, memory_id: str, *, now: datetime | None = None) -> bool:
    """Log that the question was asked, so the cooldown has something to measure against."""
    ensure_schema(store)
    try:
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO staleness_prompts(memory_id, prompted_at) VALUES(?, ?) "
                "ON CONFLICT(memory_id) DO UPDATE SET prompted_at = excluded.prompted_at",
                (str(memory_id), (now or _now()).isoformat()),
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def _recently_prompted(store: Any, *, now: datetime | None = None) -> set[str]:
    cutoff = (now or _now()) - timedelta(days=COOLDOWN_DAYS)
    rows = _rows(store, "SELECT memory_id, prompted_at FROM staleness_prompts")
    fresh = set()
    for row in rows:
        stamp = _parse(row.get("prompted_at"))
        if stamp is not None and stamp >= cutoff:
            fresh.add(str(row.get("memory_id")))
    return fresh


@dataclass
class Due:
    examined: int = 0
    stale: int = 0
    items: list[dict[str, Any]] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"examined": self.examined, "stale": self.stale, "items": self.items,
                "skipped": self.skipped, "note": self.note}


def due(store: Any, *, limit: int = DEFAULT_LIMIT, now: datetime | None = None,
        kinds: list[str] | None = None) -> Due:
    """The questions worth asking, worst first — never the whole store."""
    moment = now or _now()
    where = "state <> 'archived'"
    params: list[Any] = []
    if kinds:
        where += f" AND kind IN ({','.join('?' for _ in kinds)})"
        params.extend(kinds)
    rows = _rows(store, f"SELECT * FROM memories WHERE {where}", tuple(params))
    found = Due(examined=len(rows))
    prompted = _recently_prompted(store, now=moment)
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        stale, age, window = is_stale(row, now=moment)
        if not stale:
            continue
        found.stale += 1
        memory_id = str(row.get("id") or "")
        if str(row.get("state") or "") in NEVER_PROMPT or row.get("pinned"):
            found.skipped["pinned"] = found.skipped.get("pinned", 0) + 1
            continue
        if str(row.get("source_category") or "") == "USER_CORRECTION":
            found.skipped["correction"] = found.skipped.get("correction", 0) + 1
            continue
        if memory_id in prompted:
            found.skipped["cooldown"] = found.skipped.get("cooldown", 0) + 1
            continue
        if not has_claim(str(row.get("content") or "")):
            found.skipped["no assertion"] = found.skipped.get("no assertion", 0) + 1
            continue
        score = priority(row, now=moment)
        if score <= 0:
            continue
        scored.append((score, {
            "memory_id": memory_id,
            "kind": str(row.get("kind") or ""),
            "age_days": age,
            "window_days": window,
            "uses": int(row.get("used_count") or 0),
            "priority": score,
            "prompt": prompt(row, now=moment),
        }))
    scored.sort(key=lambda item: (-item[0], item[1]["memory_id"]))
    found.items = [item for _, item in scored[: max(1, int(limit))]]
    if not found.items:
        found.note = "nothing is both stale and worth asking about"
    return found


def render(store: Any, *, limit: int = DEFAULT_LIMIT, now: datetime | None = None) -> list[str]:
    described = due(store, limit=limit, now=now)
    lines = [f"staleness prompts: {described.stale} stale of {described.examined} examined, "
             f"{len(described.items)} question(s) due"]
    if described.skipped:
        lines.append("  skipped: " + ", ".join(f"{name} {count}"
                                               for name, count in sorted(described.skipped.items())))
    for item in described.items:
        lines.append(f"  · {item['prompt']}")
    if described.note:
        lines.append(f"  {described.note}")
    return lines
