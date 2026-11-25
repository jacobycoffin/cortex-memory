"""One week of this memory system, told as a story with its receipts (#181)

Every other report answers a narrow question — `audit_digest` asks what happened to my memory,
`health` asks whether anything is broken, `timeline_view` asks what the shape was. Nobody answered
the compound question a person asks on a Sunday: **what did it do, what did it learn, and what about
it should worry me?**

This is that page. It is deliberately not a dashboard: a dashboard shows every number and hides the
ranking, and the ranking is the whole point. Five rules, each of which exists because the obvious
version of this report flatters:

- **a week with no data is `quiet`, never an error and never an empty page.** Silence is a finding
  ("nothing was written") and must be distinguishable from a broken query. So a quiet week prints
  the window, the tables it could read and the count of zero, and says `quiet` — while a store it
  could not read at all says `unavailable`, because *unknown is not zero*.
- **every figure names its table and its window.** `42 written` is not checkable; `42 written
  (memories.source_category, 2026-11-17 → 2026-11-24)` is. The window is carried on the section, not
  inferred by the reader from the header, so a figure cannot drift away from its dates.
- **the next action is chosen by the worst number, not by a list.** A static "top three tips" advises
  the same thing in a good week and a bad one. Each issue carries its own remedy, and the digest
  prints the remedy belonging to the largest deviation from its limit. That is why `action` is a
  property of `issues[0]` rather than a constant.
- **it must not flatter.** A bad week states its worst metric *first*, above the totals, because a
  reader who stops after the first line must stop on the true thing. The sections follow; the
  headline is not a pleasant summary of them.
- **the output is deterministic for a given window.** Nothing reads the clock except the window
  bounds themselves: no "generated at", no relative ages ("3 days ago"), no iteration over dict
  order that a different insertion order could change. Two runs over the same store and window are
  byte-identical, so a diff between two weeks is a difference in the store and not in the printer.

The score behind "worst" is a share past the limit: `(observed - limit) / limit` for a metric that
must stay below its limit, `(limit - observed) / limit` for one that must stay above (bounded by 1.0
because an observed count cannot go below zero). That makes a 100%-dry-run week and a never-corrected
week comparable without inventing a currency, and ties break on the issue key so the same store
always prints the same winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

DEFAULT_DAYS = 7

# Limits are stated as numbers rather than prose so the severity is computable and disputable.
DRY_RUN_LIMIT = 0.5            # more than half of maintenance being rehearsal is not maintenance
UNUSED_LIMIT = 0.5             # half of what retrieval selected should be used at least once
ABSTENTION_LIMIT = 0.10        # one turn in ten with nothing to offer is a recall gap
LATENCY_LIMIT_MS = 150.0       # the pre-turn deadline the hook contract already promises
IMPORT_LIMIT = 0.90            # one source making nine in ten writes is an importer, not a memory
CORRECTION_FLOOR = 1.0         # at least one correction per 1,000 writes


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    """`None` means the table could not be read — which is not the same as zero rows."""
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def _table_exists(store: Any, name: str) -> bool:
    rows = _rows(store, "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,))
    return bool(rows)


def _parse(stamp: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


@dataclass
class Section:
    """One figure, its table, and the window it was counted over."""

    name: str
    table: str
    window: str
    count: int | None = None
    detail: dict[str, int] = field(default_factory=dict)
    unavailable: bool = False
    # Only windowed sections decide whether a week was quiet: "3 corrections stand" is stock, not
    # activity, and letting stock decide would call an untouched store busy.
    windowed: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "table": self.table, "window": self.window,
                "count": self.count, "detail": dict(sorted(self.detail.items())),
                "unavailable": self.unavailable, "windowed": self.windowed}

    def sentence(self) -> str:
        where = f"{self.table}, {self.window}"
        if self.unavailable:
            return f"{self.name}: unavailable ({where} could not be read)"
        if self.count is None:
            return f"{self.name}: unknown ({where})"
        if not self.count:
            return f"{self.name}: none ({where})"
        extra = ""
        if self.detail:
            extra = " (" + ", ".join(f"{key} {value}" for key, value in sorted(self.detail.items())
                                     if value) + ")"
        return f"{self.name}: {self.count}{extra} ({where})"


@dataclass(frozen=True)
class Issue:
    """A number that looks wrong, the limit it crossed, and the one thing to do about it."""

    key: str
    table: str
    window: str
    reading: str
    observed: float
    limit: float
    direction: str  # "at_most" or "at_least"
    action: str

    @property
    def severity(self) -> float:
        """How far past the limit, as a share of the limit.

        Deliberately not clipped at the top: a metric three times its limit is three times as wrong
        as one at twice, and clipping would turn every catastrophe into a tie broken by name.
        """
        if self.limit <= 0:
            return 0.0
        if self.direction == "at_least":
            # Cannot overshoot below zero, so this is naturally bounded by 1.0 (observed == 0).
            return round(min(1.0, max(0.0, (self.limit - self.observed) / self.limit)), 3)
        return round(max(0.0, (self.observed - self.limit) / self.limit), 3)

    @property
    def wrong(self) -> bool:
        return self.severity > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "table": self.table, "window": self.window,
                "reading": self.reading, "observed": round(float(self.observed), 3),
                "limit": round(float(self.limit), 3), "direction": self.direction,
                "severity": self.severity, "action": self.action}

    def line(self) -> str:
        return f"{self.reading} ({self.table}, {self.window})"


@dataclass
class WeeklyDigest:
    days: int = DEFAULT_DAYS
    start: str = ""
    end: str = ""
    sections: list[Section] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def window(self) -> str:
        return f"{self.start[:10]} → {self.end[:10]}"

    @property
    def status(self) -> str:
        readable = [section for section in self.sections if section.windowed and not section.unavailable]
        if not readable:
            # Nothing could be read: that is a broken instrument, and saying "quiet" would be a lie.
            return "unavailable"
        if all(not section.count for section in readable):
            return "quiet"
        return "reported"

    @property
    def worst(self) -> Issue | None:
        ranked = [issue for issue in self.issues if issue.wrong]
        if not ranked:
            return None
        return sorted(ranked, key=lambda issue: (-issue.severity, issue.key))[0]

    @property
    def action(self) -> str:
        """The remedy for the worst number — never a canned list, so it changes with the week."""
        worst = self.worst
        if worst is not None:
            return worst.action
        if self.status == "quiet":
            return ("confirm the writer is running: 0 rows landed in memories between "
                    f"{self.start[:10]} and {self.end[:10]}")
        if self.status == "unavailable":
            return "restore the store: no table this digest reads could be read"
        return ("no figure crossed its limit this week; re-run the digest for a longer window "
                "before concluding the system is fine")

    def as_dict(self) -> dict[str, Any]:
        return {
            "days": self.days,
            "window": {"start": self.start, "end": self.end},
            "status": self.status,
            "sections": [section.as_dict() for section in self.sections],
            "issues": [issue.as_dict() for issue in self.issues],
            "worst": self.worst.key if self.worst else "",
            "action": self.action,
            "notes": list(self.notes),
        }

    def headline(self) -> str:
        """The first line is the worst number, because the first line is what gets read."""
        if self.status == "unavailable":
            return f"week {self.window}: unavailable — no table this digest reads could be read"
        if self.status == "quiet":
            return (f"week {self.window}: quiet — 0 writes, 0 retrievals, 0 maintenance actions "
                    f"(memories, recall_runs, maintenance_log)")
        worst = self.worst
        if worst is None:
            return f"week {self.window}: nothing crossed its limit"
        return f"week {self.window}: worst — {worst.line()}"

    def render(self) -> list[str]:
        lines = [self.headline()]
        for issue in self.issues[:3]:
            mark = "!" if issue.wrong else "·"
            lines.append(f"  {mark} {issue.line()}")
        for section in self.sections:
            if section.unavailable or section.count:
                lines.append("  - " + section.sentence())
        for note in self.notes:
            lines.append(f"  ? {note}")
        lines.append(f"next: {self.action}")
        return lines


def _window(days: int, now: datetime) -> tuple[datetime, datetime]:
    """Both bounds inclusive: a row written exactly at `now` belongs to this window, not the next."""
    end = now
    start = end - timedelta(days=max(1, int(days)))
    return start, end


def _window_text(start: datetime, end: datetime) -> str:
    return f"{start.isoformat()[:10]} → {end.isoformat()[:10]}"


def _writes(store: Any, start: datetime, end: datetime, window: str) -> tuple[Section, int]:
    section = Section("memories written", "memories", window)
    rows = _rows(store, "SELECT source_category, COUNT(*) AS count FROM memories "
                        "WHERE created_at >= ? AND created_at <= ? GROUP BY source_category",
                 (start.isoformat(), end.isoformat()))
    if rows is None:
        section.unavailable = True
        return section, 0
    section.count = sum(int(row["count"] or 0) for row in rows)
    section.detail = {str(row["source_category"] or "unknown"): int(row["count"] or 0)
                      for row in rows}
    return section, section.count


def _entities(store: Any, start: datetime, end: datetime, window: str) -> Section:
    """New entity names: seen in this window, absent from every memory written before it.

    `entity_nodes` is created lazily by its own module, so a store that has never run it has no
    table to count; the memory rows still carry `entities_json`, which is where the names were
    born. Counting from there keeps the figure available on a store without the entity module.
    """
    section = Section("new entities", "memories.entities_json", window)
    fresh = _rows(store, "SELECT entities_json FROM memories "
                         "WHERE created_at >= ? AND created_at <= ? AND entities_json NOT IN ('', '[]', 'null')",
                  (start.isoformat(), end.isoformat()))
    if fresh is None:
        section.unavailable = True
        return section
    known = _rows(store, "SELECT entities_json FROM memories "
                         "WHERE created_at < ? AND entities_json NOT IN ('', '[]', 'null')",
                  (start.isoformat(),))
    if known is None:
        section.unavailable = True
        return section
    import json  # noqa: PLC0415 - only this section parses JSON

    def names(rows: list[dict[str, Any]]) -> set[str]:
        found: set[str] = set()
        for row in rows:
            try:
                values = json.loads(str(row["entities_json"] or "[]"))
            except Exception:  # noqa: BLE001
                continue
            if isinstance(values, list):
                found.update(str(value) for value in values if str(value).strip())
            elif isinstance(values, dict):
                found.update(str(key) for key in values)
        return found

    new = names(fresh) - names(known)
    section.count = len(new)
    return section


def _topics(store: Any, start: datetime, end: datetime, window: str) -> Section:
    section = Section("new topics", "topic_nodes", window)
    if not _table_exists(store, "topic_nodes"):
        section.unavailable = True
        return section
    rows = _rows(store, "SELECT COUNT(*) AS count FROM topic_nodes "
                        "WHERE created_at >= ? AND created_at <= ?",
                 (start.isoformat(), end.isoformat()))
    if rows is None:
        section.unavailable = True
        return section
    section.count = int(rows[0]["count"] or 0) if rows else 0
    return section


def _corrections(store: Any, start: datetime, end: datetime, window: str) -> Section:
    section = Section("corrections recorded", "memories(source_category)", window)
    rows = _rows(store, "SELECT COUNT(*) AS count FROM memories "
                        "WHERE source_category = 'USER_CORRECTION' "
                        "AND created_at >= ? AND created_at <= ?",
                 (start.isoformat(), end.isoformat()))
    if rows is None:
        section.unavailable = True
        return section
    section.count = int(rows[0]["count"] or 0) if rows else 0
    return section


def _retirements(store: Any, start: datetime, end: datetime, window: str) -> Section:
    section = Section("beliefs retired", "memories(valid_to)", window)
    rows = _rows(store, "SELECT COUNT(*) AS count FROM memories "
                        "WHERE valid_to IS NOT NULL AND valid_to >= ? AND valid_to < ?",
                 (start.isoformat(), end.isoformat()))
    if rows is None:
        section.unavailable = True
        return section
    section.count = int(rows[0]["count"] or 0) if rows else 0
    return section


def _recall(store: Any, start: datetime, end: datetime, window: str
            ) -> tuple[Section, Section, float, int]:
    runs = Section("retrieval runs", "recall_runs", window)
    rows = _rows(store, "SELECT abstained, prepare_ms, selected_count, candidate_count "
                        "FROM recall_runs WHERE created_at >= ? AND created_at <= ?",
                 (start.isoformat(), end.isoformat()))
    abstained = 0
    p50 = 0.0
    if rows is None:
        runs.unavailable = True
    else:
        timings = [float(row["prepare_ms"]) for row in rows if row["prepare_ms"] is not None]
        abstained = sum(1 for row in rows if int(row["abstained"] or 0))
        p50 = _median(timings)
        runs.count = len(rows)
        if abstained:
            runs.detail["abstained"] = abstained
        if timings:
            runs.detail["p50 ms"] = int(round(p50))

    use = Section("retrieval use", "usage_records", window)
    # `usage_records` rows are offers; `selected` says the turn showed it. Both come from the same
    # rows, because a `used` on a memory the turn never selected is not evidence about retrieval.
    rows = _rows(store, "SELECT COUNT(*) AS offers, COALESCE(SUM(selected), 0) AS selected, "
                        "COALESCE(SUM(used), 0) AS used "
                        "FROM usage_records WHERE created_at >= ? AND created_at <= ?",
                 (start.isoformat(), end.isoformat()))
    if rows is None:
        use.unavailable = True
    else:
        offers = int(rows[0]["offers"] or 0)
        selected = int(rows[0]["selected"] or 0)
        used = int(rows[0]["used"] or 0)
        use.count = offers
        if selected:
            use.detail["selected"] = selected
        if used:
            use.detail["used"] = used
    return runs, use, p50, abstained


def _maintenance(store: Any, start: datetime, end: datetime, window: str
                 ) -> tuple[Section, Section, float]:
    section = Section("maintenance actions", "maintenance_log", window)
    rows = _rows(store, "SELECT action, dry_run, COUNT(*) AS count FROM maintenance_log "
                        "WHERE created_at >= ? AND created_at <= ? GROUP BY action, dry_run",
                 (start.isoformat(), end.isoformat()))
    dry_share = 0.0
    if rows is None:
        section.unavailable = True
    else:
        section.count = sum(int(row["count"] or 0) for row in rows)
        actions: dict[str, int] = {}
        dry = 0
        for row in rows:
            actions[str(row["action"] or "unknown")] = actions.get(str(row["action"] or "unknown"), 0) + int(row["count"] or 0)
            if int(row["dry_run"] or 0):
                dry += int(row["count"] or 0)
        section.detail = dict(sorted(actions.items(), key=lambda item: (-item[1], item[0]))[:4])
        if dry:
            section.detail["dry runs"] = dry
            dry_share = dry / section.count if section.count else 0.0

    lifecycle = Section("lifecycle changes", "lifecycle_events", window)
    rows = _rows(store, "SELECT to_state, COUNT(*) AS count FROM lifecycle_events "
                        "WHERE created_at >= ? AND created_at <= ? GROUP BY to_state",
                 (start.isoformat(), end.isoformat()))
    if rows is None:
        lifecycle.unavailable = True
    else:
        lifecycle.count = sum(int(row["count"] or 0) for row in rows)
        lifecycle.detail = {str(row["to_state"] or "unknown"): int(row["count"] or 0)
                            for row in rows}
    return section, lifecycle, dry_share


def _issues(writes: Section, corrections: Section, use: Section, runs: Section,
            maintenance: Section, p50: float, abstained: int, dry_share: float,
            window: str) -> list[Issue]:
    """Every measurable deviation, ordered later by how far past its limit it is."""
    found: list[Issue] = []
    if not maintenance.unavailable and maintenance.count:
        found.append(Issue(
            key="maintenance_rehearsal", table="maintenance_log", window=window,
            reading=f"{round(dry_share * 100)}% of maintenance actions were dry runs "
                    f"({int(round(dry_share * maintenance.count))} of {maintenance.count})",
            observed=dry_share, limit=DRY_RUN_LIMIT, direction="at_most",
            action=f"run one maintenance action for real and record it: "
                   f"{int(round(dry_share * maintenance.count))} of {maintenance.count} "
                   f"maintenance_log rows in {window} were dry runs",
        ))
    if not writes.unavailable and writes.count and not corrections.unavailable:
        per_thousand = 1000.0 * (corrections.count or 0) / writes.count
        found.append(Issue(
            key="never_corrected", table="memories(source_category)", window=window,
            reading=f"{corrections.count} corrections against {writes.count} writes "
                    f"({per_thousand:.1f} per 1,000)",
            observed=per_thousand, limit=CORRECTION_FLOOR, direction="at_least",
            action=f"record one correction by hand: {writes.count} writes and "
                   f"{corrections.count} user corrections landed in {window}, so nothing in the "
                   f"store has been marked wrong",
        ))
    if not use.unavailable and (use.detail.get("selected") or 0):
        selected = int(use.detail["selected"] or 0)
        used = int(use.detail.get("used") or 0)
        unused = 1.0 - (used / selected if selected else 0.0)
        found.append(Issue(
            key="unused_retrievals", table="usage_records", window=window,
            reading=f"{round(unused * 100)}% of selected memories were never used "
                    f"({selected - used} of {selected})",
            observed=unused, limit=UNUSED_LIMIT, direction="at_most",
            action=f"drop the {selected - used} memories offered and never used in {window} from "
                   f"the retrieval pool, or find out why they were selected",
        ))
    if not runs.unavailable and runs.count:
        share = abstained / runs.count
        found.append(Issue(
            key="abstention", table="recall_runs", window=window,
            reading=f"{round(share * 100)}% of retrievals abstained ({abstained} of {runs.count})",
            observed=share, limit=ABSTENTION_LIMIT, direction="at_most",
            action=f"label {abstained} abstained retrieval(s) from {window} as recall gaps and "
                   f"check whether the memories exist at all",
        ))
    if not runs.unavailable and p50:
        found.append(Issue(
            key="latency", table="recall_runs(prepare_ms)", window=window,
            reading=f"retrieval p50 {p50:.0f} ms against a {LATENCY_LIMIT_MS:.0f} ms deadline",
            observed=p50, limit=LATENCY_LIMIT_MS, direction="at_most",
            action=f"profile the slowest retrieval stage: p50 prepare_ms is {p50:.0f} ms over "
                   f"{runs.count} runs in {window} against a {LATENCY_LIMIT_MS:.0f} ms deadline",
        ))
    if not writes.unavailable and writes.count and writes.detail:
        top_source, top_count = sorted(writes.detail.items(), key=lambda item: (-item[1], item[0]))[0]
        share = top_count / writes.count
        found.append(Issue(
            key="single_source", table="memories(source_category)", window=window,
            reading=f"{round(share * 100)}% of writes came from one source ({top_source}, "
                    f"{top_count} of {writes.count})",
            observed=share, limit=IMPORT_LIMIT, direction="at_most",
            action=f"check whether {top_source} is importing rather than learning: it wrote "
                   f"{top_count} of {writes.count} memories in {window}",
        ))
    return found


def digest(store: Any, *, days: int = DEFAULT_DAYS, now: datetime | None = None) -> WeeklyDigest:
    """One week: what was written, what was learned, what was fetched and what looks wrong."""
    moment = now or _now()
    start, end = _window(days, moment)
    window = _window_text(start, end)
    described = WeeklyDigest(days=max(1, int(days)), start=start.isoformat(), end=end.isoformat())

    writes, write_count = _writes(store, start, end, window)
    corrections = _corrections(store, start, end, window)
    retirements = _retirements(store, start, end, window)
    entities = _entities(store, start, end, window)
    topics = _topics(store, start, end, window)
    runs, use, p50, abstained = _recall(store, start, end, window)
    maintenance, lifecycle, dry_share = _maintenance(store, start, end, window)

    described.sections = [writes, corrections, retirements, entities, topics,
                          runs, use, maintenance, lifecycle]
    # Sorted once, here: the caller's issue list and the rendered top three must be in the same
    # order, and the tie-break on `key` makes the order a property of the data, not of build order.
    described.issues = sorted(_issues(writes, corrections, use, runs, maintenance, p50, abstained,
                                      dry_share, window),
                              key=lambda issue: (-issue.severity, issue.key))

    if described.status == "quiet":
        described.notes.append(
            f"quiet week: nothing was written between {window} (memories.created_at) — "
            f"silence here is a finding, not a failure of the query"
        )
    if described.status == "unavailable":
        described.notes.append("no section could be read; an unreadable store is not an empty one")
    if not corrections.unavailable and write_count and not corrections.count:
        described.notes.append(
            f"no correction was recorded in {window} while {write_count} memories were written"
        )
    if entities.unavailable:
        described.notes.append("entity names could not be counted (memories.entities_json)")
    described.notes.append("counts come from the table named beside each figure, over the same "
                           "window; an unreadable table says unavailable, never zero")
    return described


def render(store: Any, *, days: int = DEFAULT_DAYS, now: datetime | None = None) -> list[str]:
    return digest(store, days=days, now=now).render()
