"""The audit trail, written for a person rather than for a query (#185)

The store records a great deal and explains almost none of it. There are 1,051 `lifecycle_events`, 1,791
`maintenance_log` rows and 19 `reconsolidation_events` — and no way to answer the only question a person
actually asks: **what did this thing do to my memory this week, and was anything refused?**

So this is a digest, not a viewer. `audit_view` shows a table; this writes sentences:

```
last 7 days: 42 memories written (31 document extracts, 9 from conversation),
3 refused by consent (2 credential, 1 work systems), 1 correction, 0 forgotten,
12 quarantined, 4 maintenance actions (0 dry runs)
```

Five rules:

- **every number names its window and its table.** `42 written` is not checkable; `42 written
  (memories.created_at, 2026-11-17 → 2026-11-24)` is.
- **a quiet period is reported as silence**, not as an empty list — "nothing happened" is information, and
  a blank screen is indistinguishable from a broken query.
- **refusals are counted by class, never by content.** The digest must not become the leak the gate
  prevented (#182).
- **what was *not* done is part of the audit.** Zero corrections and zero retirements is a finding about
  the system, not a blank.
- **the digest reads only tables it can name.** An unknown or missing table is reported as `unavailable`
  rather than silently contributing zero — the same `unknown ≠ 0` rule that runs through this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

DEFAULT_DAYS = 7


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


@dataclass
class Section:
    name: str
    table: str
    count: int | None = None
    detail: dict[str, int] = field(default_factory=dict)
    unavailable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "table": self.table, "count": self.count,
                "detail": self.detail, "unavailable": self.unavailable}

    def sentence(self) -> str:
        if self.unavailable:
            return f"{self.name}: unavailable ({self.table} could not be read)"
        if self.count is None:
            return f"{self.name}: unknown"
        if not self.count:
            return f"{self.name}: none"
        extra = ""
        if self.detail:
            extra = " (" + ", ".join(f"{name} {count}"
                                     for name, count in sorted(self.detail.items()) if count) + ")"
        return f"{self.name}: {self.count}{extra}"


@dataclass
class Digest:
    window: str = ""
    days: int = DEFAULT_DAYS
    sections: list[Section] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"window": self.window, "days": self.days,
                "sections": [section.as_dict() for section in self.sections],
                "notes": self.notes}

    @property
    def quiet(self) -> bool:
        return all(not section.count for section in self.sections if not section.unavailable)


def _since(days: int, now: datetime | None = None) -> str:
    return ((now or _now()) - timedelta(days=max(1, int(days)))).isoformat()


def digest(store: Any, *, days: int = DEFAULT_DAYS, now: datetime | None = None) -> Digest:
    """One readable week: what was written, refused, corrected, forgotten and flagged."""
    moment = now or _now()
    since = _since(days, moment)
    described = Digest(window=f"{since[:10]} → {moment.isoformat()[:10]}", days=days)

    written = Section("memories written", "memories")
    rows = _rows(store, "SELECT source_category, COUNT(*) AS count FROM memories "
                        "WHERE created_at >= ? GROUP BY source_category", (since,))
    if rows is None:
        written.unavailable = True
    else:
        written.count = sum(int(row["count"] or 0) for row in rows)
        written.detail = {str(row["source_category"] or "unknown"): int(row["count"] or 0)
                          for row in rows}
    described.sections.append(written)

    lifecycle = Section("lifecycle changes", "lifecycle_events")
    rows = _rows(store, "SELECT to_state, COUNT(*) AS count FROM lifecycle_events "
                        "WHERE created_at >= ? GROUP BY to_state", (since,))
    if rows is None:
        lifecycle.unavailable = True
    else:
        lifecycle.count = sum(int(row["count"] or 0) for row in rows)
        lifecycle.detail = {str(row["to_state"] or "unknown"): int(row["count"] or 0)
                            for row in rows}
    described.sections.append(lifecycle)

    maintenance = Section("maintenance actions", "maintenance_log")
    rows = _rows(store, "SELECT action, dry_run, COUNT(*) AS count FROM maintenance_log "
                        "WHERE created_at >= ? GROUP BY action, dry_run", (since,))
    if rows is None:
        maintenance.unavailable = True
    else:
        maintenance.count = sum(int(row["count"] or 0) for row in rows)
        dry = sum(int(row["count"] or 0) for row in rows if int(row["dry_run"] or 0))
        if dry:
            maintenance.detail["dry runs"] = dry
    described.sections.append(maintenance)

    reconsolidation = Section("reconsolidation", "reconsolidation_events")
    rows = _rows(store, "SELECT status, COUNT(*) AS count FROM reconsolidation_events "
                        "WHERE corrected_at >= ? OR trigger_access_at >= ? GROUP BY status",
                 (since, since))
    if rows is None:
        reconsolidation.unavailable = True
    else:
        reconsolidation.count = sum(int(row["count"] or 0) for row in rows)
        reconsolidation.detail = {str(row["status"] or "unknown"): int(row["count"] or 0)
                                  for row in rows}
    described.sections.append(reconsolidation)

    consent = Section("consent decisions", "consent_decisions")
    if not _table_exists(store, "consent_decisions"):
        consent.unavailable = True
    else:
        rows = _rows(store, "SELECT verdict, class, COUNT(*) AS count FROM consent_decisions "
                            "WHERE decided_at >= ? GROUP BY verdict, class", (since,))
        if rows is None:
            consent.unavailable = True
        else:
            consent.count = sum(int(row["count"] or 0) for row in rows)
            refused = sum(int(row["count"] or 0) for row in rows
                          if str(row["verdict"]) == "refuse")
            asked = sum(int(row["count"] or 0) for row in rows
                        if str(row["verdict"]) == "ask")
            if refused:
                consent.detail["refused"] = refused
            if asked:
                consent.detail["asked"] = asked
    described.sections.append(consent)

    corrections = Section("corrections in force", "memories")
    rows = _rows(store, "SELECT COUNT(*) AS count FROM memories "
                        "WHERE source_category = 'USER_CORRECTION' AND valid_to IS NULL")
    if rows is None:
        corrections.unavailable = True
    else:
        corrections.count = int(rows[0]["count"] or 0) if rows else 0
    described.sections.append(corrections)

    retired = Section("beliefs retired", "memories")
    rows = _rows(store, "SELECT COUNT(*) AS count FROM memories WHERE valid_to >= ?", (since,))
    if rows is None:
        retired.unavailable = True
    else:
        retired.count = int(rows[0]["count"] or 0) if rows else 0
    described.sections.append(retired)

    if described.quiet:
        described.notes.append("nothing happened in this window")
    dry = maintenance.detail.get("dry runs") or 0
    if maintenance.count and dry == maintenance.count:
        described.notes.append(f"every maintenance action was a dry run ({dry} of {dry}): "
                               "the maintenance path has been rehearsing, not acting")
    if not corrections.count:
        described.notes.append("no correction has ever been recorded (#178): "
                              "every correction so far is indistinguishable from a fact")
    if consent.unavailable or not consent.count:
        described.notes.append("no consent decisions logged: nothing writes through the gate yet (#182)")
    described.notes.append("counts read from the tables named beside them; "
                           "an unreadable table says unavailable, never zero")
    return described


def render(store: Any, *, days: int = DEFAULT_DAYS, now: datetime | None = None) -> list[str]:
    described = digest(store, days=days, now=now)
    lines = [f"what happened to my memory, {described.window}:"]
    for section in described.sections:
        lines.append("  · " + section.sentence())
    for note in described.notes:
        lines.append(f"  ? {note}")
    return lines
