#!/usr/bin/env python3
"""Validity probe — the cheap tier of the memory validity judge.

Zero-LLM, zero-dependency, READ-ONLY. Reports the validity signals that were
measured to actually carry information, and audits the signals that do not.

Design context: docs/VALIDITY_JUDGE.md

Signals that survived measurement (carry information today):
  1. Evidence inventory      - how much real truth evidence exists at all
  2. Opportunity-adjusted use - used/offered, not raw counts
  3. Invisible rot           - never-recalled memories, by importance
  4. Protection audit        - high-stakes memories exposed to usage decay
  5. Source staleness        - vault chunks whose source file moved on

Signals that FAILED measurement (kept here as explicit non-features):
  * Contradiction scan over subject/predicate/object_value: 0 conflicts found.
    98.5% of stored triples use a single placeholder predicate ("documents"),
    so they are bookkeeping, not semantic claims. Needs better extraction
    before a scanner is worth building.

Usage:
    python3 scripts/validity_probe.py [--db PATH] [--vault PATH] [--json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys

HIGH_STAKES_TERMS = (
    "password", "passwd", "token", "api key", "apikey", "secret", "credential",
    "login", "allerg", "medication", "prescription", "insurance", "passport",
    "emergency", "ssn", "social security", "bank account", "routing number",
)
# source categories that represent something a person actually told Cortex,
# as opposed to reference material scraped from documents.
PERSONAL_SOURCES = ("USER_EXPLICIT", "OPERATOR_APPROVED", "REFLECTION", "TOOL_VERIFIED")

# A memory must have had at least this many chances before its use-rate is
# treated as meaningful. Below this we cannot distinguish "rare" from "bad".
MIN_OPPORTUNITIES = 5


def connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        sys.exit(f"database not found: {db_path}")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def evidence_inventory(con: sqlite3.Connection) -> dict:
    """How much real truth evidence exists, per signal."""
    cols = [
        ("retrieved", "retrieved_count"), ("selected", "selected_count"),
        ("injected", "injected_count"), ("used", "used_count"),
        ("helpful", "helpful_count"), ("harmful", "harmful_count"),
        ("confirmed", "confirmed_count"), ("validated", "validated_count"),
        ("corrected", "correction_count"), ("false_positive", "false_positive_count"),
    ]
    out = {}
    for label, col in cols:
        row = con.execute(
            f"SELECT COALESCE(SUM({col}),0) total, "
            f"COALESCE(SUM(CASE WHEN {col}>0 THEN 1 ELSE 0 END),0) memories FROM memories"
        ).fetchone()
        out[label] = {"total_events": row["total"], "memories": row["memories"]}
    out["_truth_events_total"] = sum(
        out[k]["total_events"]
        for k in ("helpful", "harmful", "confirmed", "validated", "corrected", "false_positive")
    )
    return out


def counter_separation(con: sqlite3.Connection) -> dict:
    """Do the observation counters actually measure different things?

    If retrieved == selected == injected, the store cannot distinguish
    'Cortex offered this' from 'the agent used this' - which makes any
    precision metric built on them circular.
    """
    row = con.execute(
        "SELECT COALESCE(SUM(retrieved_count),0) r, COALESCE(SUM(selected_count),0) s, "
        "COALESCE(SUM(injected_count),0) i, COALESCE(SUM(used_count),0) u FROM memories"
    ).fetchone()
    r, s, i, u = row["r"], row["s"], row["i"], row["u"]
    return {
        "retrieved": r, "selected": s, "injected": i, "used": u,
        "retrieval_counters_are_identical": (r == s == i) and r > 0,
        "used_per_offered": round(u / r, 4) if r else None,
        "note": (
            "retrieved/selected/injected are one counter, not three - do not build "
            "precision or validity metrics on them"
            if (r == s == i) and r > 0 else "counters are separated"
        ),
    }


def opportunity_adjusted_use(con: sqlite3.Connection, minimum: int = MIN_OPPORTUNITIES) -> dict:
    """Use-rate = used / offered, for memories with enough chances to judge."""
    row = con.execute(
        "SELECT COUNT(*) c FROM memories WHERE injected_count >= ?", (minimum,)
    ).fetchone()
    eligible = row["c"]
    buckets = {"never_used": 0, "1-24%": 0, "25-49%": 0, "50%+": 0}
    for r in con.execute(
        "SELECT used_count, injected_count FROM memories WHERE injected_count >= ?", (minimum,)
    ):
        rate = r["used_count"] / r["injected_count"]
        if rate == 0:
            buckets["never_used"] += 1
        elif rate < 0.25:
            buckets["1-24%"] += 1
        elif rate < 0.5:
            buckets["25-49%"] += 1
        else:
            buckets["50%+"] += 1
    return {
        "eligible_memories": eligible,
        "minimum_opportunities": minimum,
        "buckets": buckets,
        "signals": {
            "prune_candidates_offered_never_used": buckets["never_used"],
            "retain_candidates_consistently_useful": buckets["50%+"],
        },
    }


def invisible_rot(con: sqlite3.Connection, high_importance: float = 0.7) -> dict:
    """Memories that no recall has ever touched, so no recall-triggered
    judge will ever see them. These need a scheduled sweep."""
    total = con.execute("SELECT COUNT(*) c FROM memories WHERE retrieved_count = 0").fetchone()["c"]
    active = con.execute(
        "SELECT COUNT(*) c FROM memories WHERE retrieved_count = 0 AND state = 'active'"
    ).fetchone()["c"]
    important = con.execute(
        "SELECT COUNT(*) c FROM memories WHERE retrieved_count = 0 AND importance >= ?",
        (high_importance,),
    ).fetchone()["c"]
    archived = con.execute(
        "SELECT COUNT(*) c FROM memories WHERE retrieved_count = 0 AND state = 'archived'"
    ).fetchone()["c"]
    return {
        "never_retrieved_total": total,
        "never_retrieved_active": active,
        "never_retrieved_archived": archived,
        "never_retrieved_important": important,
        "high_importance_threshold": high_importance,
        "note": "recall-triggered judging cannot see these; a scheduled sweep is required",
    }


def _high_stakes_clause() -> tuple[str, list]:
    clause = " OR ".join("LOWER(content) LIKE ?" for _ in HIGH_STAKES_TERMS)
    params = [f"%{t}%" for t in HIGH_STAKES_TERMS]
    return f"({clause})", params


def protection_audit(con: sqlite3.Connection) -> dict:
    """Which high-stakes memories are actually exposed to usage decay?

    Keyword matching alone overcounts badly: most hits are daily/reference
    notes that merely mention a protected word. So we separate reference
    material from things a person actually told Cortex.
    """
    clause, params = _high_stakes_clause()
    placeholders = ",".join("?" for _ in PERSONAL_SOURCES)
    exposed = []
    query = f"""
        SELECT id, content, source_category, importance, volatility, protected, pinned
        FROM memories
        WHERE state = 'active' AND protected = 0 AND pinned = 0
          AND source_category IN ({placeholders}) AND {clause}
        ORDER BY importance DESC
    """
    for r in con.execute(query, [*PERSONAL_SOURCES, *params]):
        exposed.append({
            "id": r["id"],
            "source_category": r["source_category"],
            "importance": r["importance"],
            "volatility": r["volatility"],
            "excerpt": (r["content"] or "")[:100],
        })
    reference_hits = con.execute(
        f"SELECT COUNT(*) c FROM memories WHERE state='active' AND protected=0 "
        f"AND source_category='DOCUMENT_EXTRACTED' AND {clause}",
        params,
    ).fetchone()["c"]
    return {
        "unprotected_personal_memories": exposed,
        "count": len(exposed),
        "unprotected_reference_chunks": reference_hits,
        "note": "reference chunks are informational only; personal memories are the real exposure",
    }


def source_staleness(con: sqlite3.Connection, vault: str, slop_seconds: int = 3600) -> dict:
    """Vault-sourced memories whose source file moved on after ingest.

    This is a PROXY for content change: it compares file mtime against when the
    memory was observed. A same-second comparison false-positives on the ingest
    write itself, hence the slop window. The exact check already exists in the
    ingest path (vault.py compares the stored chunk digest against a freshly
    computed one); this probe approximates it without re-running the chunker.
    """
    rows = con.execute(
        "SELECT id, source_ref, observed_at FROM memories "
        "WHERE source_category = 'DOCUMENT_EXTRACTED' AND source_ref LIKE 'vault:%' "
        "AND state = 'active'"
    ).fetchall()
    missing, stale, fresh, unparseable = [], [], 0, 0
    for r in rows:
        rel = r["source_ref"][len("vault:"):].split("#", 1)[0].strip()
        path = os.path.join(vault, rel)
        if not os.path.exists(path):
            missing.append(rel)
            continue
        try:
            mtime = dt.datetime.fromtimestamp(os.path.getmtime(path), dt.timezone.utc)
            observed = dt.datetime.fromisoformat(r["observed_at"])
        except (ValueError, OSError):
            unparseable += 1
            continue
        if (mtime - observed).total_seconds() > slop_seconds:
            stale.append({"source_ref": r["source_ref"], "observed": observed.date().isoformat(),
                          "modified": mtime.date().isoformat()})
        else:
            fresh += 1
    return {
        "chunks_checked": len(rows),
        "source_missing": len(missing),
        "source_changed_since_ingest": len(stale),
        "confirmed_current": fresh,
        "unparseable": unparseable,
        "examples": stale[:5],
        "method": f"file mtime > observed_at + {slop_seconds}s (proxy, not content diff)",
    }


def lifecycle(con: sqlite3.Connection) -> dict:
    states = {r["state"]: r["n"] for r in con.execute(
        "SELECT state, COUNT(*) n FROM memories GROUP BY 1 ORDER BY 2 DESC")}
    one = lambda sql: con.execute(sql).fetchone()["c"]  # noqa: E731
    return {
        "states": states,
        "protected": one("SELECT COUNT(*) c FROM memories WHERE protected=1"),
        "pinned": one("SELECT COUNT(*) c FROM memories WHERE pinned=1"),
        "stranded": one("SELECT COUNT(*) c FROM memories WHERE stranded=1"),
        "with_valid_to": one("SELECT COUNT(*) c FROM memories WHERE valid_to IS NOT NULL"),
        "with_supersedes_link": one("SELECT COUNT(*) c FROM memories WHERE supersedes_id IS NOT NULL"),
        "pruning_regret_rows": one("SELECT COUNT(*) c FROM pruning_regret"),
    }


def report(db: str, vault: str) -> dict:
    con = connect(db)
    try:
        return {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "database": db,
            "evidence_inventory": evidence_inventory(con),
            "counter_separation": counter_separation(con),
            "opportunity_adjusted_use": opportunity_adjusted_use(con),
            "invisible_rot": invisible_rot(con),
            "protection_audit": protection_audit(con),
            "source_staleness": source_staleness(con, vault),
            "lifecycle": lifecycle(con),
        }
    finally:
        con.close()


def render(rep: dict) -> str:
    L = []
    add = L.append
    inv = rep["evidence_inventory"]
    add("CORTEX VALIDITY PROBE (read-only)")
    add("=" * 62)
    add(f"generated {rep['generated_at']}   db={rep['database']}")
    add("")
    add("1. EVIDENCE INVENTORY  (the judge's food supply)")
    for name in ("retrieved", "used", "helpful", "harmful", "confirmed",
                 "validated", "corrected", "false_positive"):
        e = inv[name]
        add(f"   {name:16} {e['total_events']:>7} events   {e['memories']:>5} memories")
    add(f"   {'TRUTH EVENTS':16} {inv['_truth_events_total']:>7}  <- helpful+harmful+confirmed+validated+corrected+false_positive")
    add("")
    cs = rep["counter_separation"]
    add("2. COUNTER SEPARATION")
    add(f"   retrieved={cs['retrieved']}  selected={cs['selected']}  injected={cs['injected']}  used={cs['used']}")
    add(f"   identical? {cs['retrieval_counters_are_identical']}   used/offered={cs['used_per_offered']}")
    add(f"   {cs['note']}")
    add("")
    ou = rep["opportunity_adjusted_use"]
    add(f"3. OPPORTUNITY-ADJUSTED USE  (memories offered >= {ou['minimum_opportunities']} times)")
    add(f"   eligible: {ou['eligible_memories']}")
    for k, v in ou["buckets"].items():
        bar = "#" * min(40, v // 5)
        add(f"   {k:12} {v:>4}  {bar}")
    add(f"   -> prune candidates (offered, never used): {ou['signals']['prune_candidates_offered_never_used']}")
    add(f"   -> retain candidates (consistent use):      {ou['signals']['retain_candidates_consistently_useful']}")
    add("")
    ir = rep["invisible_rot"]
    add("4. INVISIBLE ROT  (never recalled -> invisible to any recall-triggered judge)")
    add(f"   total never retrieved : {ir['never_retrieved_total']}")
    add(f"   ...still active       : {ir['never_retrieved_active']}")
    add(f"   ...already archived   : {ir['never_retrieved_archived']}")
    add(f"   ...importance >= {ir['high_importance_threshold']:<4} : {ir['never_retrieved_important']}")
    add("")
    pa = rep["protection_audit"]
    add("5. PROTECTION AUDIT  (high-stakes memories exposed to usage decay)")
    add(f"   unprotected PERSONAL memories : {pa['count']}")
    for m in pa["unprotected_personal_memories"][:5]:
        add(f"      [{m['source_category']}] imp={m['importance']:.2f} {m['excerpt'][:60]!r}")
    add(f"   unprotected reference chunks  : {pa['unprotected_reference_chunks']} (informational)")
    add("")
    ss = rep["source_staleness"]
    add("6. SOURCE STALENESS  (vault chunks whose source moved on)")
    add(f"   chunks checked        : {ss['chunks_checked']}")
    add(f"   source missing        : {ss['source_missing']}")
    add(f"   changed since ingest  : {ss['source_changed_since_ingest']}")
    add(f"   confirmed current     : {ss['confirmed_current']}")
    for e in ss["examples"]:
        add(f"      {e['source_ref'][:58]:60} {e['observed']} -> {e['modified']}")
    add(f"   method: {ss['method']}")
    add("")
    lc = rep["lifecycle"]
    add("7. LIFECYCLE")
    add(f"   states: {lc['states']}")
    add(f"   protected={lc['protected']} pinned={lc['pinned']} stranded={lc['stranded']} "
        f"valid_to_set={lc['with_valid_to']} supersedes={lc['with_supersedes_link']} "
        f"pruning_regret={lc['pruning_regret_rows']}")
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", default=os.path.expanduser("~/.hermes/cortex/cortex.db"))
    p.add_argument("--vault", default=os.path.expanduser("~/.hermes/obsidian-vault"))
    p.add_argument("--json", action="store_true", help="emit raw JSON instead of a report")
    args = p.parse_args()
    rep = report(args.db, args.vault)
    print(json.dumps(rep, indent=2) if args.json else render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
