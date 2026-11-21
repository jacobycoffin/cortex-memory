#!/usr/bin/env python3
"""Freeze a retrieval eval set from real usage evidence.

The cases are not hand-authored questions: each one is a query Cortex was
actually asked, whose expected memories are the memories the agent actually
marked as used on that query (`access_log.event = 'used'`). That makes the set
a replay of real traffic with real outcomes instead of an operator's guess.

The frozen file and its manifest are written outside the repository on
purpose: queries describe personal projects (docs/EVALUATION.md). The manifest
pins the set so `scripts/frozen_evalset.py run` can prove it is unchanged.

Usage:
    python3 scripts/build_frozen_evalset.py \
        --db ~/.hermes/cortex/cortex.db \
        --out ~/cortex-evaluations/frozen/frozen-v1.jsonl \
        --set-id frozen-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUILDER_VERSION = 1
LABEL_SCHEMA_VERSION = 1
DEFAULT_MIN_CASES = 200
DEFAULT_LIMIT = 240
MIN_QUERY_CHARS = 12
MAX_QUERY_CHARS = 400
NON_LANGUAGE_MARKERS = ("task-outcome:", "tool_result", "\"role\"", "<tool")

WHITESPACE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a retrieval eval set from real 'used' feedback in the Cortex database. "
            "Writes private labels plus a pinned manifest; never writes inside the repository."
        )
    )
    parser.add_argument("--db", type=Path, required=True, help="Cortex database to read (opened read-only).")
    parser.add_argument("--out", type=Path, required=True, help="Frozen labels JSONL path (outside the repo).")
    parser.add_argument("--set-id", default="frozen-v1", help="Set identifier recorded in the manifest.")
    parser.add_argument(
        "--min-cases",
        type=int,
        default=DEFAULT_MIN_CASES,
        help="Refuse to freeze below this many cases (default %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum cases to freeze, strongest evidence first (default %(default)s).",
    )
    parser.add_argument(
        "--max-relevant",
        type=int,
        default=4,
        help="Cap expected memories per case, most recently used first (default %(default)s).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing frozen set.")
    parser.add_argument(
        "--allow-repo-path",
        action="store_true",
        help="Permit writing inside the repository. Off by default: queries are private.",
    )
    return parser.parse_args()


def normalize_query(query: str) -> str:
    return WHITESPACE.sub(" ", query.strip().lower())


def is_evaluable_query(query: str) -> bool:
    if not (MIN_QUERY_CHARS <= len(query) <= MAX_QUERY_CHARS):
        return False
    if not any(char.isalpha() for char in query):
        return False
    lowered = query.lower()
    return not any(marker in lowered for marker in NON_LANGUAGE_MARKERS)


def build_cases(
    connection: sqlite3.Connection,
    *,
    limit: int,
    max_relevant: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return frozen cases plus the counts that explain the selection."""

    rows = connection.execute(
        """
        SELECT a.query AS query, a.memory_id AS memory_id, a.created_at AS created_at
        FROM access_log a
        JOIN memories m ON m.id = a.memory_id
        WHERE a.event = 'used'
          AND a.query IS NOT NULL
          AND length(trim(a.query)) >= ?
          AND m.state = 'active'
          AND length(trim(coalesce(m.content, ''))) > 0
        ORDER BY a.created_at ASC, a.id ASC
        """,
        (MIN_QUERY_CHARS,),
    ).fetchall()

    grouped: dict[str, dict[str, Any]] = {}
    skipped_short = 0
    for query, memory_id, created_at in rows:
        normalized = normalize_query(query)
        if not is_evaluable_query(normalized):
            skipped_short += 1
            continue
        entry = grouped.setdefault(
            normalized,
            {"query": query.strip(), "strength": 0, "used_at": {}, "case_id": ""},
        )
        entry["strength"] += 1
        entry["query"] = query.strip()  # last observed original wording wins
        entry["used_at"][memory_id] = created_at

    eligible = list(grouped.values())
    # Deterministic on a given database: strongest evidence first, ties broken by
    # the normalized query itself so two runs cannot disagree.
    eligible.sort(key=lambda item: (-item["strength"], normalize_query(item["query"])))
    selected = eligible[:limit]

    cases: list[dict[str, Any]] = []
    for entry in selected:
        normalized = normalize_query(entry["query"])
        case_id = "fz1-" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
        ranked = sorted(entry["used_at"].items(), key=lambda pair: (pair[1], pair[0]), reverse=True)
        relevant = [memory_id for memory_id, _ in ranked[:max_relevant]]
        cases.append(
            {
                "schema_version": LABEL_SCHEMA_VERSION,
                "case_id": case_id,
                "query": entry["query"],
                "relevant_memory_ids": relevant,
                "group": "real-usage",
            }
        )
    cases.sort(key=lambda case: case["case_id"])

    counts = {
        "used_rows_considered": len(rows),
        "used_rows_not_evaluable": skipped_short,
        "eligible_queries": len(eligible),
        "frozen_cases": len(cases),
    }
    return cases, counts


def _inside_repository(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
    except ValueError:
        return False
    return True


def manifest_for(
    *,
    set_id: str,
    labels_path: Path,
    cases: list[dict[str, Any]],
    counts: dict[str, int],
    connection: sqlite3.Connection,
    selection: dict[str, Any],
) -> dict[str, Any]:
    digest = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    fingerprint = {
        "memories": connection.execute("SELECT count(*) FROM memories").fetchone()[0],
        "memories_active": connection.execute("SELECT count(*) FROM memories WHERE state='active'").fetchone()[0],
        "used_events": connection.execute("SELECT count(*) FROM access_log WHERE event='used'").fetchone()[0],
    }
    return {
        "manifest_version": 1,
        "set_id": set_id,
        "builder_version": BUILDER_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_count": len(cases),
        "labels_file": labels_path.name,
        "labels_sha256": digest,
        "evidence": "access_log event='used' joined to active memories",
        "label_provenance": "model-attributed 'used' events, not human judgments",
        "claim_boundary": (
            "A deterministic replay of real traffic with real usage signals. Read it as a "
            "relative regression gate, not as an absolute retrieval-quality score: some "
            "'used' attributions are noisy, and a noisy positive that was never retrievable "
            "counts as a miss on every run, including the baseline."
        ),
        "source_fingerprint": fingerprint,
        "selection": selection,
        "counts": counts,
        "privacy": {
            "contains_private_queries": True,
            "do_not_commit": True,
            "keep_outside_repository": True,
        },
    }


def main() -> int:
    args = parse_args()
    if args.min_cases < 1 or args.limit < 1 or args.max_relevant < 1:
        raise SystemExit("--min-cases, --limit and --max-relevant must be positive")
    out = args.out.expanduser()
    if _inside_repository(out) and not args.allow_repo_path:
        raise SystemExit(
            "refusing to write private queries inside the repository "
            f"({out}); choose a path outside {ROOT} or pass --allow-repo-path"
        )
    if out.exists() and not args.overwrite:
        raise SystemExit(f"frozen set already exists: {out}; pass --overwrite to rebuild it")
    database = args.db.expanduser().resolve()
    if not database.is_file():
        raise SystemExit(f"Cortex database does not exist: {args.db}")

    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=5.0)
    try:
        cases, counts = build_cases(connection, limit=args.limit, max_relevant=args.max_relevant)
        if len(cases) < args.min_cases:
            raise SystemExit(
                f"only {len(cases)} usable case(s) with 'used' evidence; "
                f"refusing to freeze below {args.min_cases}"
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases), encoding="utf-8")
        out.chmod(0o600)
        selection = {
            "limit": args.limit,
            "min_cases": args.min_cases,
            "max_relevant": args.max_relevant,
            "min_query_chars": MIN_QUERY_CHARS,
            "max_query_chars": MAX_QUERY_CHARS,
            "order": "strength desc, query asc",
        }
        manifest = manifest_for(
            set_id=args.set_id,
            labels_path=out,
            cases=cases,
            counts=counts,
            connection=connection,
            selection=selection,
        )
    finally:
        connection.close()

    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.chmod(0o600)
    print(
        f"Froze {len(cases)} cases from {counts['eligible_queries']} eligible queries "
        f"({counts['used_rows_considered']} used rows) -> {out}"
    )
    print(f"manifest {manifest_path.name} sha256={manifest['labels_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
