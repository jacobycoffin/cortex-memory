"""Live view of Jev decisions for the Brain dashboard.

The auto-judge service appends one JSON line per Jev judgment to the decision
log (``CORTEX_JEV_DECISION_LOG``). The log deliberately carries no memory
content. This module follows that file by byte offset and, for display only,
joins bounded content previews from the local store. ``sort_queue`` joins the
newest verdict per proposal onto the pending queue for the Sort screen.
Nothing here writes to the log or to the database.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .store import CortexStore

PREVIEW_CHARS = 280
_MAX_READ_BYTES = 2_000_000
_TAIL_WINDOW_BYTES = 262_144


def log_identity(path: Path | None) -> str | None:
    """Stable identity for detecting rename/replacement log rotation."""
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return f"{stat.st_dev}:{stat.st_ino}"


def decision_log_path(environ: dict[str, str] | None = None) -> Path | None:
    """Resolve the log path exactly like ``jev.JevSettings.from_env``."""
    env = os.environ if environ is None else environ
    raw = env.get("CORTEX_JEV_DECISION_LOG")
    if raw is None:
        return Path.home() / ".hermes" / "cortex" / "jev-decisions.jsonl"
    raw = str(raw).strip()
    return Path(raw).expanduser() if raw else None


def _parse_lines(chunk: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in chunk.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def read_events_since(path: Path | None, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Return complete events after ``offset`` and the offset to resume from.

    A trailing line without its newline stays unread until it is finished. A
    file shorter than ``offset`` was truncated or rotated, so reading restarts
    at the beginning.
    """
    if path is None:
        return [], 0
    try:
        size = path.stat().st_size
    except OSError:
        return [], 0
    start = int(offset) if 0 <= int(offset) <= size else 0
    if start == size:
        return [], size
    try:
        with path.open("rb") as handle:
            handle.seek(start)
            chunk = handle.read(min(size - start, _MAX_READ_BYTES))
    except OSError:
        return [], 0
    cut = chunk.rfind(b"\n")
    if cut < 0:
        return [], start
    complete = chunk[: cut + 1]
    return _parse_lines(complete), start + len(complete)


def tail_events(path: Path | None, *, limit: int = 100) -> tuple[list[dict[str, Any]], int]:
    """The most recent ``limit`` events, plus the end offset for resuming."""
    if path is None:
        return [], 0
    try:
        size = path.stat().st_size
    except OSError:
        return [], 0
    start = max(0, size - _TAIL_WINDOW_BYTES)
    try:
        with path.open("rb") as handle:
            handle.seek(start)
            chunk = handle.read(size - start)
    except OSError:
        return [], 0
    skipped = 0
    if start > 0:
        # The window may begin mid-line; drop that fragment.
        first_newline = chunk.find(b"\n")
        skipped = first_newline + 1 if first_newline >= 0 else len(chunk)
    body = chunk[skipped:]
    cut = body.rfind(b"\n")
    complete = body[: cut + 1] if cut >= 0 else b""
    bounded = max(1, min(int(limit), 1000))
    # Resume right after the last complete line so an unfinished one is re-read.
    return _parse_lines(complete)[-bounded:], start + skipped + len(complete)


def read_event_batch(
    path: Path | None,
    *,
    offset: int | None,
    expected_identity: str | None,
    limit: int = 100,
) -> tuple[list[dict[str, Any]], int, str | None, bool]:
    """Read live events with identity, size, and bytes from one descriptor.

    Opening first and using ``fstat`` closes the path-stat/open race during log
    rotation. An identity change forces a tail reset regardless of file size.
    """
    if path is None:
        return [], 0, None, expected_identity is not None
    try:
        with path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            identity = f"{stat.st_dev}:{stat.st_ino}"
            size = int(stat.st_size)
            reset = expected_identity is not None and identity != expected_identity
            bounded = max(1, min(int(limit), 1000))
            if offset is None or reset:
                start = max(0, size - _TAIL_WINDOW_BYTES)
                handle.seek(start)
                chunk = handle.read(size - start)
                skipped = 0
                if start > 0:
                    first_newline = chunk.find(b"\n")
                    skipped = first_newline + 1 if first_newline >= 0 else len(chunk)
                body = chunk[skipped:]
                cut = body.rfind(b"\n")
                complete = body[: cut + 1] if cut >= 0 else b""
                return (
                    _parse_lines(complete)[-bounded:],
                    start + skipped + len(complete),
                    identity,
                    reset,
                )
            start = int(offset) if 0 <= int(offset) <= size else 0
            if start == size:
                return [], size, identity, False
            handle.seek(start)
            chunk = handle.read(min(size - start, _MAX_READ_BYTES))
    except OSError:
        return [], 0, None, expected_identity is not None
    cut = chunk.rfind(b"\n")
    if cut < 0:
        return [], start, identity, False
    complete = chunk[: cut + 1]
    return _parse_lines(complete)[-bounded:], start + len(complete), identity, False


def _event_ids(event: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("proposal_id", "candidate_id", "memory_id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            ids.append(value)
    for key in ("related_ids",):
        values = event.get(key)
        if isinstance(values, list):
            ids.extend(str(v) for v in values if isinstance(v, str) and v)
    links = event.get("links")
    if isinstance(links, list):
        ids.extend(
            str(link.get("memory_id"))
            for link in links
            if isinstance(link, dict) and link.get("memory_id")
        )
    return list(dict.fromkeys(ids))


def _preview(store: CortexStore, item_id: str) -> dict[str, Any] | None:
    memory = store.get_memory(item_id)
    if memory:
        text = str(memory.get("content") or "")
        return {
            "type": "memory",
            "kind": str(memory.get("kind") or ""),
            "state": str(memory.get("state") or ""),
            "text": text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else ""),
        }
    proposal = store.get_memory_creation_proposal(item_id)
    if proposal:
        text = str(proposal.get("content") or "")
        return {
            "type": "proposal",
            "kind": str(proposal.get("kind") or ""),
            "state": str(proposal.get("status") or ""),
            "text": text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else ""),
        }
    return None


def enrich_events(store: CortexStore, events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach bounded content previews for every id an event mentions."""
    cache: dict[str, dict[str, Any] | None] = {}
    enriched: list[dict[str, Any]] = []
    for event in events:
        previews: dict[str, Any] = {}
        for item_id in _event_ids(event)[:24]:
            if item_id not in cache:
                try:
                    cache[item_id] = _preview(store, item_id)
                except Exception:  # a bad id must never break the feed
                    cache[item_id] = None
            if cache[item_id] is not None:
                previews[item_id] = cache[item_id]
        enriched.append({**event, "previews": previews})
    return enriched


def summarize(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Counters the live view shows above the stream."""
    actions: dict[str, int] = {}
    paths: dict[str, int] = {}
    latencies: list[int] = []
    guards = links_created = link_judgments = prune_judgments = 0
    for event in events:
        kind = event.get("event")
        latency = event.get("latency_ms")
        if isinstance(latency, int) and not isinstance(latency, bool):
            latencies.append(latency)
        if kind == "admission":
            action = str(event.get("action") or "unknown")
            actions[action] = actions.get(action, 0) + 1
            path = str(event.get("path") or "unknown")
            paths[path] = paths.get(path, 0) + 1
        elif kind == "guard":
            guards += 1
        elif kind == "links":
            link_judgments += 1
            links = event.get("links")
            links_created += len(links) if isinstance(links, list) else 0
        elif kind == "prune":
            prune_judgments += 1
    admissions = sum(actions.values())
    ordered = sorted(latencies)
    return {
        "admissions": admissions,
        "actions": dict(sorted(actions.items())),
        "paths": dict(sorted(paths.items())),
        "defer_rate": (actions.get("defer", 0) / admissions) if admissions else 0.0,
        "guard_overrides": guards,
        "link_judgments": link_judgments,
        "links_suggested": links_created,
        "prune_judgments": prune_judgments,
        "latency_p50_ms": ordered[(len(ordered) - 1) // 2] if ordered else None,
    }


# ---------------------------------------------------------------------------
# Sort screen: Jev's verdicts joined onto the pending proposal queue.
# ---------------------------------------------------------------------------

# The whole log is read for the join (it is small and content-free); this cap
# keeps a runaway file from being read into memory. Only the newest bytes are
# kept, so the newest verdicts always win.
_QUEUE_MAX_READ_BYTES = 16_000_000
SORT_CONTENT_CHARS = 4000
VERDICT_SCORE_KEYS = (
    "worth_saving",
    "worth_adjusted",
    "durable",
    "standalone",
    "useful_again",
    "scope_clear",
    "curated",
    "duplicate_of_related",
)
_LANE_FOR_ACTION = {"remember": "keep", "reject": "drop"}
_AGREEMENT_NOTE = re.compile(r"^jev:([a-z_]+) run:([0-9a-z]+) op:(accepted|overrode|decided)$")


def read_all_events(path: Path | None) -> tuple[list[dict[str, Any]], bool]:
    """Every complete event in the log (newest ``_QUEUE_MAX_READ_BYTES``).

    Returns ``(events, log_available)``. Uses the same line parser as the
    live feed; a torn trailing line is ignored until it is finished.
    """
    if path is None:
        return [], False
    try:
        with path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            start = max(0, size - _QUEUE_MAX_READ_BYTES)
            handle.seek(start)
            chunk = handle.read(size - start)
    except OSError:
        return [], False
    if start > 0:
        first_newline = chunk.find(b"\n")
        chunk = chunk[first_newline + 1:] if first_newline >= 0 else b""
    cut = chunk.rfind(b"\n")
    return (_parse_lines(chunk[: cut + 1]) if cut >= 0 else []), True


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def latest_verdicts(events: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The newest admission verdict per proposal, with any guard that held it.

    Log order is append order, so a later line always supersedes an earlier
    one. A guard applies only to the admission from its own run.
    """
    verdicts: dict[str, dict[str, Any]] = {}
    guards: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        kind = event.get("event")
        proposal_id = event.get("proposal_id")
        if not isinstance(proposal_id, str) or not proposal_id:
            continue
        if kind == "admission":
            action = str(event.get("action") or "defer")
            verdicts[proposal_id] = {
                "action": action,
                "jev_action": action,
                "scores": {
                    key: _number(event.get(key)) for key in VERDICT_SCORE_KEYS if _number(event.get(key)) is not None
                },
                "gate": event.get("gate"),
                "path": event.get("path"),
                "model": event.get("model"),
                "run_ref": str(event.get("run_ref") or "") or None,
                "ts": event.get("ts"),
                "audit": bool(event.get("audit")),
                "guard": None,
            }
        elif kind == "guard":
            guards[(proposal_id, str(event.get("run_ref") or ""))] = {
                "reason": str(event.get("reason") or "")[:300],
                "from_action": event.get("from_action"),
                "to_action": event.get("to_action") or "needs_context",
            }
    for proposal_id, verdict in verdicts.items():
        guard = guards.get((proposal_id, str(verdict["run_ref"] or "")))
        if guard:
            verdict["guard"] = guard
            verdict["action"] = str(guard["to_action"])
    return verdicts


def _path_thresholds() -> dict[str, dict[str, float | None]]:
    from . import jev

    paths = [jev.DEFAULT_ADMIT_PATH, *jev.ADMISSION_PATHS.values()]
    return {path.label: {"admit": path.admit, "reject": path.reject} for path in paths}


# Plain-English descriptors for Jev's closed questions: (strong, weak).
_REASON_PHRASES = {
    "durable": ("lasting", "short-lived"),
    "useful_again": ("likely useful again", "unlikely to be needed again"),
    "standalone": ("makes sense on its own", "doesn't make sense on its own"),
    "scope_clear": ("clear about what it applies to", "vague about what it applies to"),
    "curated": ("well formed", "rough or noisy"),
}


def plain_reason(verdict: dict[str, Any] | None) -> str:
    """One sentence built from Jev's two strongest and one weakest answer."""
    if not verdict:
        return "Not judged by Jev yet."
    guard = verdict.get("guard")
    if guard:
        reason = str(guard.get("reason") or "a safety check needs your context")
        reason = re.sub(r"^Safety override:\s*", "", reason).rstrip(".")
        return f"Held for you by a safety check: {reason[:1].lower()}{reason[1:]}."
    if verdict.get("path") == "builtin_memory_review" and verdict.get("action") == "defer":
        return "Built-in memory notes always come to you; Jev never auto-decides them."
    scores = dict(verdict.get("scores") or {})
    strong = sorted(
        ((value, key) for key, value in scores.items() if key in _REASON_PHRASES and value >= 0.65),
        reverse=True,
    )[:2]
    weak_parts: list[str] = []
    if float(scores.get("duplicate_of_related") or 0.0) >= 0.5:
        weak_parts.append("may duplicate an existing memory")
    else:
        weak = sorted((value, key) for key, value in scores.items() if key in _REASON_PHRASES and value <= 0.35)
        if weak:
            weak_parts.append(_REASON_PHRASES[weak[0][1]][1])
    good = [_REASON_PHRASES[key][0] for _value, key in strong]
    if not good and not weak_parts:
        return "Nothing stood out either way, so Jev left it to you."
    if not good:
        sentence = weak_parts[0]
    else:
        sentence = " and ".join(good) + (f", but {weak_parts[0]}" if weak_parts else "")
    return sentence[:1].upper() + sentence[1:] + "."


def _review_agreement(conn: Any) -> dict[str, int]:
    tally = {"judged": 0, "agreed": 0, "overrode": 0, "decided_without_call": 0}
    rows = conn.execute(
        """SELECT reason_text FROM operator_review_decisions
           WHERE item_type='creation' AND reversed_at IS NULL AND reason_text LIKE 'jev:%'"""
    ).fetchall()
    for (note,) in rows:
        match = _AGREEMENT_NOTE.match(str(note or ""))
        if not match:
            continue
        outcome = match.group(3)
        if outcome == "accepted":
            tally["agreed"] += 1
        elif outcome == "overrode":
            tally["overrode"] += 1
        else:
            tally["decided_without_call"] += 1
    tally["judged"] = tally["agreed"] + tally["overrode"]
    return tally


def _read_only_connection(store: CortexStore) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{store.path}?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only=ON")
    return conn


def fresh_verdict_ids(store: CortexStore, path: Path | None) -> set[str]:
    """Pending proposals whose newest verdict is not older than their last sighting.

    Used by an "unjudged only" run so a new batch never re-asks about items
    Jev has already sorted (the offset paging it replaces skips or repeats
    items once the pending set shifts).
    """
    events, _available = read_all_events(path)
    verdicts = latest_verdicts(events)
    if not verdicts:
        return set()
    conn = _read_only_connection(store)
    try:
        rows = conn.execute(
            "SELECT proposal_id, last_seen_at FROM memory_creation_proposals WHERE status='pending'"
        ).fetchall()
    finally:
        conn.close()
    return {
        str(proposal_id)
        for proposal_id, last_seen in rows
        if str(proposal_id) in verdicts and not _is_stale(verdicts[str(proposal_id)].get("ts"), last_seen)
    }


def _parse_ts(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _is_stale(verdict_ts: Any, last_seen_at: Any) -> bool:
    judged, seen = _parse_ts(verdict_ts), _parse_ts(last_seen_at)
    return bool(judged and seen and judged < seen)


def sort_queue(store: CortexStore, path: Path | None) -> dict[str, Any]:
    """Pending proposals with Jev's newest verdict, lanes, runs and agreement.

    Read-only: a separate ``mode=ro`` + ``query_only`` connection, so neither
    the queue nor a failure here can write the database.
    """
    events, available = read_all_events(path)
    verdicts = latest_verdicts(events)
    thresholds = _path_thresholds()
    conn = _read_only_connection(store)
    try:
        rows = conn.execute(
            """SELECT proposal_id, content, kind, source_type, source_category, first_seen_at,
                      last_seen_at, recurrence_count, quarantine_reason, redacted
               FROM memory_creation_proposals WHERE status='pending'
               ORDER BY last_seen_at DESC, rowid DESC"""
        ).fetchall()
        agreement = _review_agreement(conn)
    finally:
        conn.close()
    now = datetime.now(timezone.utc)
    items: list[dict[str, Any]] = []
    lanes = {"keep": 0, "drop": 0, "decide": 0, "unjudged": 0}
    runs: dict[str, dict[str, Any]] = {}
    for (proposal_id, content, kind, source_type, source_category, first_seen, last_seen,
         recurrence, quarantine, redacted) in rows:
        text = str(content or "")
        verdict = verdicts.get(str(proposal_id))
        if verdict is not None:
            verdict = {
                **verdict,
                "thresholds": thresholds.get(str(verdict.get("path") or ""), {"admit": None, "reject": None}),
                "stale": _is_stale(verdict.get("ts"), last_seen),
            }
        lane = _LANE_FOR_ACTION.get(str(verdict["action"]), "decide") if verdict else "decide"
        lanes[lane] += 1
        if verdict is None:
            lanes["unjudged"] += 1
        else:
            run_ref = verdict.get("run_ref") or "unknown"
            run = runs.setdefault(
                run_ref, {"run_ref": run_ref, "pending": 0, "keep": 0, "drop": 0, "decide": 0, "latest_ts": None}
            )
            run["pending"] += 1
            run[lane] += 1
            if str(verdict.get("ts") or "") > str(run["latest_ts"] or ""):
                run["latest_ts"] = verdict.get("ts")
        seen = _parse_ts(first_seen)
        items.append({
            "proposal_id": str(proposal_id),
            "content": text[:SORT_CONTENT_CHARS],
            "content_truncated": len(text) > SORT_CONTENT_CHARS,
            "kind": str(kind or ""),
            "source_type": str(source_type or ""),
            "source_category": str(source_category or ""),
            "first_seen_at": first_seen,
            "last_seen_at": last_seen,
            "age_seconds": int((now - seen).total_seconds()) if seen else None,
            "recurrence_count": int(recurrence or 0),
            "quarantined": bool(quarantine),
            "redacted": bool(redacted),
            "lane": lane,
            "reason": plain_reason(verdict),
            "verdict": verdict,
        })
    return {
        "generated_at": now.isoformat(),
        "log_available": available,
        "pending_total": len(items),
        "lanes": lanes,
        "runs": sorted(runs.values(), key=lambda run: str(run["latest_ts"] or ""), reverse=True),
        "agreement": agreement,
        "thresholds": thresholds,
        "items": items,
    }
