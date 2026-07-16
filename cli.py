"""Small local inspection CLI for a Cortex database."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .retrieval import MemoryRetriever
from .security import sanitize_memory
from .store import CortexStore


def main() -> int:
    parser = argparse.ArgumentParser(prog="cortex-memory", description="Inspect and test Cortex Memory")
    parser.add_argument(
        "--db",
        default=os.environ.get("CORTEX_DB", str(Path.home() / ".hermes" / "cortex" / "cortex.db")),
        help="Path to cortex.db",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stats")
    sub.add_parser("audit")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=6)
    search.add_argument("--include-archived", action="store_true")
    search.add_argument("--historical", action="store_true")
    search.add_argument("--as-of", help="ISO timestamp used by historical retrieval")
    remember = sub.add_parser("remember")
    remember.add_argument("content")
    remember.add_argument("--kind", default="semantic")
    remember.add_argument("--pin", action="store_true")
    explain = sub.add_parser("explain")
    explain.add_argument("memory_id")
    maintenance = sub.add_parser("maintenance")
    maintenance.add_argument("--apply", action="store_true")
    consolidate = sub.add_parser("consolidate", help="Preview or apply reversible near-duplicate folding")
    consolidate.add_argument("--apply", action="store_true")
    consolidate.add_argument("--threshold", type=float, default=0.78)
    undo = sub.add_parser("undo-consolidation", help="Restore members from an applied consolidation run")
    undo.add_argument("run_id")
    sleep = sub.add_parser("sleep", help="Run bounded offline replay and consolidation")
    sleep.add_argument("--mode", choices=("shadow", "apply"), default="shadow")
    sleep.add_argument(
        "--apply",
        action="store_true",
        help="Required confirmation when --mode apply is selected",
    )
    sleep.add_argument("--min-episode-age-hours", type=int, default=12)
    sleep.add_argument("--max-episodes", type=int, default=250)
    sleep.add_argument("--min-association-witnesses", type=int, default=2)
    sleep.add_argument("--replay-threshold", type=float, default=0.24)
    sleep.add_argument("--decay-after-days", type=int, default=120)
    sleep.add_argument("--cold-after-days", type=int, default=90)
    sleep.add_argument("--archive-after-days", type=int, default=180)
    sleep.add_argument(
        "--reflection-token-budget",
        type=int,
        default=int(os.environ.get("CORTEX_SLEEP_TOKEN_BUDGET", "0")),
        help="Separate per-cycle provider-token ceiling; 0 disables model reflection",
    )
    sleep.add_argument("--reflection-endpoint", default=os.environ.get("CORTEX_SLEEP_ENDPOINT"))
    sleep.add_argument("--reflection-model", default=os.environ.get("CORTEX_SLEEP_MODEL"))
    sleep.add_argument(
        "--reflection-api-key-env",
        default=os.environ.get("CORTEX_SLEEP_API_KEY_ENV", "OPENROUTER_API_KEY"),
        help="Name of the environment variable containing the provider key; never the key itself",
    )
    sleep_undo = sub.add_parser("sleep-undo", help="Undo reversible changes from an applied sleep run")
    sleep_undo.add_argument("run_id")
    sub.add_parser("recall-stats", help="Show attention-gate latency and context-budget evidence")
    traces = sub.add_parser("traces", help="Inspect task-level memory decisions or export append-only JSONL")
    traces.add_argument("--limit", type=int, default=100)
    traces.add_argument("--task-id")
    traces.add_argument("--jsonl", action="store_true")
    traces.add_argument("--summary", action="store_true")
    writes = sub.add_parser("write-decisions", help="Inspect durable create, update, and ignore decisions")
    writes.add_argument("--limit", type=int, default=100)
    writes.add_argument("--summary", action="store_true")
    context_feedback = sub.add_parser(
        "context-feedback", help="Inspect project/task-specific memory usefulness evidence"
    )
    context_feedback.add_argument("--limit", type=int, default=100)
    sub.add_parser(
        "quality-report",
        help="Show retrieval precision, false positives, context failures, health, and Sleep evidence",
    )
    sub.add_parser(
        "refinery-report",
        help="Dry-run role classification report: aggregate counts and redacted examples, no mutation",
    )
    sub.add_parser(
        "refinery-summary",
        help="Show record-role, readability-flag, presentation, and refinery-proposal aggregates",
    )
    refinery_shadow = sub.add_parser(
        "refinery-shadow",
        help="Compare live retrieval with the shadow role-tier policy for one query (no live effect)",
    )
    refinery_shadow.add_argument("query")
    harness_contract = sub.add_parser(
        "harness-contract",
        help="Print the portable Cortex-first lifecycle and bootstrap pointer for any agent harness",
    )
    harness_contract.add_argument("--tool-name", default="cortex_memory")
    dashboard = sub.add_parser("dashboard")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--no-open", action="store_true")
    dashboard_password = sub.add_parser(
        "dashboard-password", help="Generate a temporary dashboard password and require a change at login"
    )
    dashboard_password.add_argument(
        "--username", default=os.environ.get("CORTEX_DASHBOARD_USER", "cortex")
    )
    dashboard_password.add_argument(
        "--auth-file", help="Override the dashboard auth file path (defaults beside cortex.db)"
    )
    vault_index = sub.add_parser("vault-index", help="Plan or apply an incremental Obsidian vault import")
    vault_index.add_argument("vault_path")
    vault_index.add_argument("--apply", action="store_true", help="Write the planned import to Cortex")
    vault_index.add_argument("--max-chars", type=int, default=1600, help="Maximum source characters per chunk")
    vault_index.add_argument("--max-file-bytes", type=int, default=1_000_000)
    sub.add_parser("vault-status", help="Show indexed vault sources and chunks")
    args = parser.parse_args()

    if args.command == "dashboard":
        from .dashboard import serve_dashboard

        serve_dashboard(args.db, port=args.port, open_browser=not args.no_open)
        return 0
    if args.command == "dashboard-password":
        from .dashboard_auth import DashboardAuth

        auth_path = Path(args.auth_file).expanduser() if args.auth_file else Path(args.db).expanduser().parent / "dashboard-auth.json"
        temporary_password = DashboardAuth(auth_path).reset(username=args.username, must_change=True)
        print(
            json.dumps(
                {
                    "auth_file": str(auth_path),
                    "must_change_password": True,
                    "temporary_password": temporary_password,
                    "username": args.username,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "harness-contract":
        from .harness import harness_contract_manifest

        print(json.dumps(harness_contract_manifest(tool_name=args.tool_name), indent=2, ensure_ascii=False))
        return 0

    store = CortexStore(args.db)
    try:
        if args.command == "stats":
            result = store.stats()
        elif args.command == "audit":
            result = store.audit()
        elif args.command == "search":
            result = [
                r.as_dict()
                for r in MemoryRetriever(store).search(
                    args.query,
                    limit=args.limit,
                    include_archived=args.include_archived,
                    temporal_mode="historical" if args.historical or args.as_of else "current",
                    as_of=args.as_of,
                )
            ]
        elif args.command == "remember":
            sanitized = sanitize_memory(args.content)
            memory_id, created = store.add_memory(
                sanitized.text,
                kind=args.kind,
                pinned=args.pin,
                source_type="cli",
                quarantine_reason=sanitized.quarantine_reason,
            )
            result = {"memory_id": memory_id, "created": created, "quarantined": bool(sanitized.quarantine_reason)}
        elif args.command == "explain":
            memory_id = store.resolve_id(args.memory_id)
            result = store.explain(memory_id) if memory_id else None
        elif args.command == "vault-index":
            from .vault import VaultIndexer

            indexer = VaultIndexer(
                store,
                args.vault_path,
                max_chars=args.max_chars,
                max_file_bytes=args.max_file_bytes,
            )
            if args.apply:
                result = indexer.apply()
            else:
                _scan, result = indexer.plan()
                result = {**result, "applied": False}
        elif args.command == "vault-status":
            sources = list(store.document_manifest().values())
            result = {
                "sources": sources,
                "active_sources": sum(source["status"] == "active" for source in sources),
                "missing_sources": sum(source["status"] == "missing" for source in sources),
                "active_chunks": sum(
                    len(store.document_chunks(source["source_path"], active_only=True)) for source in sources
                ),
            }
        elif args.command == "consolidate":
            result = store.consolidate(dry_run=not args.apply, similarity_threshold=args.threshold)
        elif args.command == "undo-consolidation":
            result = store.undo_consolidation(args.run_id)
        elif args.command == "sleep":
            from .sleep import SleepConfig, run_sleep

            if args.mode == "apply" and not args.apply:
                parser.error("--mode apply requires the explicit --apply confirmation")
            result = run_sleep(
                store,
                SleepConfig(
                    mode=args.mode,
                    min_episode_age_hours=args.min_episode_age_hours,
                    max_episodes=args.max_episodes,
                    min_association_witnesses=args.min_association_witnesses,
                    replay_threshold=args.replay_threshold,
                    decay_after_days=args.decay_after_days,
                    cold_after_days=args.cold_after_days,
                    archive_after_days=args.archive_after_days,
                    reflection_token_budget=args.reflection_token_budget,
                    reflection_endpoint=args.reflection_endpoint,
                    reflection_model=args.reflection_model,
                    reflection_api_key_env=args.reflection_api_key_env,
                ),
            )
        elif args.command == "sleep-undo":
            from .sleep import undo_sleep

            result = undo_sleep(store, args.run_id)
        elif args.command == "recall-stats":
            snapshot = store.dashboard_snapshot(memory_limit=1)
            result = {
                "summary": snapshot["recall_summary"],
                "modes": snapshot["recall_modes"],
                "by_day": snapshot["recall_by_day"],
            }
        elif args.command == "traces":
            if args.jsonl:
                print(store.memory_trace_jsonl(limit=args.limit, task_id=args.task_id))
                return 0
            if args.summary:
                result = store.memory_trace_summary(limit=args.limit)
            else:
                result = store.memory_traces(limit=args.limit, task_id=args.task_id)
        elif args.command == "write-decisions":
            result = (
                store.memory_write_summary(limit=args.limit)
                if args.summary
                else store.memory_write_decisions(limit=args.limit)
            )
        elif args.command == "context-feedback":
            result = store.context_feedback_summary(limit=args.limit)
        elif args.command == "quality-report":
            result = store.memory_quality_report()
        elif args.command == "refinery-report":
            result = store.refinery_classification_report()
        elif args.command == "refinery-summary":
            result = store.refinery_summary()
        elif args.command == "refinery-shadow":
            result = MemoryRetriever(store).shadow_tiered_comparison(args.query)
        else:
            result = store.maintenance(dry_run=not args.apply)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
