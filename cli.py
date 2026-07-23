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
    search.add_argument(
        "--evidence-lookup",
        action="store_true",
        help="Include lookup-only reference evidence; graph expansion remains primary-only",
    )
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
    auto_judge = sub.add_parser(
        "auto-judge",
        help="Silently review a bounded batch of staged memory candidates with an LLM",
    )
    auto_judge.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress normal output for timer/service use",
    )
    auto_judge.add_argument(
        "--link-orphans",
        action="store_true",
        help="Link existing memories with no edges (uses LLM + contradiction detection)",
    )
    auto_judge.add_argument(
        "--consolidate",
        action="store_true",
        help="Judge up to five oldest related-memory pairs; records shadow proposals by default",
    )
    auto_judge.add_argument(
        "--apply-consolidation",
        action="store_true",
        help="Explicitly apply high-confidence consolidation judgments (requires --consolidate)",
    )
    auto_judge.add_argument(
        "--prune",
        action="store_true",
        help="Judge up to 50 low-relevance memories; records shadow proposals by default",
    )
    auto_judge.add_argument(
        "--apply-pruning",
        action="store_true",
        help="Explicitly apply high-confidence pruning judgments (requires --prune)",
    )
    auto_judge.add_argument(
        "--tune-weights",
        action="store_true",
        help="Audit recent outcomes and stage task-specific scoring weight proposals",
    )
    auto_judge.add_argument(
        "--reconsolidate",
        action="store_true",
        help="Judge same-task evidence for recently used memories; always stages proposals",
    )
    auto_judge.add_argument(
        "--reconsolidation-task-id",
        help="Limit --reconsolidate to one completed task trace",
    )
    auto_judge.add_argument(
        "--schemas",
        action="store_true",
        help="Review evidence-qualified repeated-memory clusters; always stages abstractions",
    )
    mechanics = sub.add_parser(
        "brain-mechanics",
        help="Run due opt-in proposal-only consolidation, pruning, reconsolidation, schema, and weight passes",
    )
    mechanics.add_argument("--quiet", action="store_true")
    sub.add_parser(
        "semantic-consolidation-report",
        help="Inspect semantic consolidation runs and proposed/applied decisions",
    )
    apply_semantic = sub.add_parser(
        "apply-semantic-consolidation",
        help="Apply one reviewed semantic consolidation proposal",
    )
    apply_semantic.add_argument("decision_id")
    undo_semantic = sub.add_parser(
        "undo-semantic-consolidation",
        help="Restore both source memories from one applied semantic consolidation",
    )
    undo_semantic.add_argument("decision_id")
    semantic_feedback = sub.add_parser(
        "semantic-consolidation-feedback",
        help="Label a consolidation judgment correct or wrong; wrong applied merges are undone",
    )
    semantic_feedback.add_argument("decision_id")
    semantic_feedback.add_argument("label", choices=("correct", "wrong"))
    semantic_feedback.add_argument("--reason", default="")
    sub.add_parser(
        "pruning-report",
        help="Inspect outcome-aware pruning proposals, strands, reversals, and regret rate",
    )
    apply_pruning = sub.add_parser(
        "apply-pruning",
        help="Apply one reviewed adaptive-pruning proposal",
    )
    apply_pruning.add_argument("decision_id")
    undo_pruning = sub.add_parser(
        "undo-pruning",
        help="Undo one applied adaptive-pruning decision",
    )
    undo_pruning.add_argument("decision_id")
    sub.add_parser(
        "weight-proposals",
        help="Inspect staged, active, rolled-back, and factory scoring profiles",
    )
    apply_weights = sub.add_parser(
        "apply-weight-proposal",
        help="Explicitly approve one staged task-specific scoring profile",
    )
    apply_weights.add_argument("proposal_id")
    apply_weights.add_argument("--confirm-large-change", action="store_true")
    apply_weights.add_argument("--note", default="")
    reject_weights = sub.add_parser(
        "reject-weight-proposal",
        help="Reject one staged scoring profile without affecting retrieval",
    )
    reject_weights.add_argument("proposal_id")
    reject_weights.add_argument("--note", default="")
    rollback_weights = sub.add_parser(
        "rollback-weights",
        help="Restore the baseline captured by one approved scoring proposal",
    )
    rollback_weights.add_argument("proposal_id")
    rollback_weights.add_argument("--reason", required=True)
    reset_weights = sub.add_parser(
        "reset-weights",
        help="Restore immutable code-default weights for one task type",
    )
    reset_weights.add_argument("task_type")
    reset_weights.add_argument("--note", default="factory reset from CLI")
    sub.add_parser(
        "reconsolidation-proposals",
        help="Inspect lability-gated supersede, extend, and conflict proposals",
    )
    apply_recon = sub.add_parser(
        "apply-reconsolidation",
        help="Apply one reviewed reconsolidation proposal",
    )
    apply_recon.add_argument("proposal_id")
    apply_recon.add_argument("--confirm-protected", action="store_true")
    undo_recon = sub.add_parser(
        "undo-reconsolidation",
        help="Reverse one applied adaptive reconsolidation",
    )
    undo_recon.add_argument("proposal_id")
    recon_feedback = sub.add_parser(
        "reconsolidation-feedback",
        help="Label a reconsolidation proposal correct or wrong; wrong applies undo",
    )
    recon_feedback.add_argument("proposal_id")
    recon_feedback.add_argument("label", choices=("correct", "wrong"))
    recon_feedback.add_argument("--reason", default="")
    sub.add_parser(
        "schema-proposals",
        help="Inspect proposed, applied, dirty, reviewed, and reversed schemas",
    )
    apply_schema = sub.add_parser(
        "apply-schema",
        help="Apply one reviewed schema abstraction while preserving all sources",
    )
    apply_schema.add_argument("proposal_id")
    undo_schema = sub.add_parser(
        "undo-schema",
        help="Archive one applied schema and restore ordinary source weighting",
    )
    undo_schema.add_argument("proposal_id")
    schema_feedback = sub.add_parser(
        "schema-feedback",
        help="Label a schema judgment correct or wrong; wrong applied schemas are undone",
    )
    schema_feedback.add_argument("proposal_id")
    schema_feedback.add_argument("label", choices=("correct", "wrong"))
    schema_feedback.add_argument("--reason", default="")
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
    neighborhood_report = sub.add_parser(
        "neighborhood-report",
        help="Explain why named project/service neighborhoods were admitted or held back",
    )
    neighborhood_report.add_argument("--limit", type=int, default=100)
    decision_log = sub.add_parser(
        "decision-log", help="Show the unified admission, review, policy, schema, and lifecycle timeline"
    )
    decision_log.add_argument("--limit", type=int, default=200)
    learning_dataset = sub.add_parser(
        "learning-dataset",
        help="Export privacy-safe Cortex learning experiences for replay or policy training",
    )
    learning_dataset.add_argument("--limit", type=int, default=10000)
    learning_dataset.add_argument("--jsonl", action="store_true")
    learning_dataset.add_argument(
        "--include-text",
        action="store_true",
        help="Explicit local opt-in to include sanitized memory/proposal text",
    )
    sub.add_parser(
        "quality-report",
        help="Show retrieval precision, false positives, context failures, health, and Sleep evidence",
    )
    scoring_trend = sub.add_parser(
        "scoring-trend",
        help="Show weekly resolved retrieval precision, helpfulness, false positives, and token waste",
    )
    scoring_trend.add_argument("--weeks", type=int, default=12)
    scoring_trend.add_argument("--limit", type=int, default=10000)
    attention_learning = sub.add_parser(
        "attention-learning",
        help="Show per-topic shadow attention evidence, decay, and promotion readiness",
    )
    attention_learning.add_argument("--limit", type=int, default=100)
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
                    evidence_lookup=args.evidence_lookup,
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
        elif args.command == "auto-judge":
            from .autojudge import AutoJudge, AutoJudgeConfig, link_orphan_memories

            config = AutoJudgeConfig.from_env()
            if args.apply_consolidation and not args.consolidate:
                parser.error("--apply-consolidation requires --consolidate")
            if args.apply_pruning and not args.prune:
                parser.error("--apply-pruning requires --prune")
            selected_passes = sum(
                bool(value)
                for value in (
                    args.link_orphans,
                    args.consolidate,
                    args.prune,
                    args.tune_weights,
                    args.reconsolidate,
                    args.schemas,
                )
            )
            if selected_passes > 1:
                parser.error(
                    "--link-orphans, --consolidate, --prune, --tune-weights, "
                    "--reconsolidate, and --schemas are separate bounded passes"
                )
            if args.consolidate:
                from .semantic_consolidation import run_semantic_consolidation

                result = run_semantic_consolidation(
                    store,
                    config,
                    apply=bool(args.apply_consolidation),
                )
            elif args.prune:
                from .relevance_pruning import run_adaptive_pruning

                result = run_adaptive_pruning(
                    store,
                    config,
                    relevance_threshold=float(
                        os.environ.get("CORTEX_AUTO_JUDGE_PRUNE_THRESHOLD", "0.25")
                    ),
                    max_candidates=int(
                        os.environ.get("CORTEX_AUTO_JUDGE_PRUNE_MAX_PER_RUN", "50")
                    ),
                    apply=bool(args.apply_pruning),
                )
            elif args.tune_weights:
                from .adaptive_weights import run_adaptive_weight_learning

                result = run_adaptive_weight_learning(
                    store,
                    config,
                    lookback_days=int(
                        os.environ.get("CORTEX_AUTO_JUDGE_WEIGHT_AUDIT_DAYS", "7")
                    ),
                )
            elif args.reconsolidate:
                from .adaptive_reconsolidation import run_adaptive_reconsolidation

                result = run_adaptive_reconsolidation(
                    store,
                    config,
                    task_id=args.reconsolidation_task_id,
                    lability_minutes=int(
                        os.environ.get("CORTEX_LABILITY_WINDOW_MINUTES", "30")
                    ),
                )
            elif args.schemas:
                from .schema_formation import run_schema_formation

                result = run_schema_formation(
                    store,
                    config,
                    minimum_cluster=int(
                        os.environ.get("CORTEX_AUTO_JUDGE_SCHEMA_MIN_CLUSTER", "3")
                    ),
                )
            elif args.link_orphans:
                result = link_orphan_memories(store, config)
            else:
                result = AutoJudge(config).run(store)
                from .brain_mechanics import run_due_brain_mechanics

                result["brain_mechanics"] = run_due_brain_mechanics(store, config)
        elif args.command == "brain-mechanics":
            from .autojudge import AutoJudgeConfig
            from .brain_mechanics import run_due_brain_mechanics

            result = run_due_brain_mechanics(store, AutoJudgeConfig.from_env())
        elif args.command == "semantic-consolidation-report":
            result = store.semantic_consolidation_snapshot()
        elif args.command == "apply-semantic-consolidation":
            result = store.apply_semantic_consolidation(
                args.decision_id,
                actor="cortex-operator:cli",
            )
        elif args.command == "undo-semantic-consolidation":
            result = store.undo_semantic_consolidation(args.decision_id)
        elif args.command == "semantic-consolidation-feedback":
            result = store.record_semantic_consolidation_feedback(
                args.decision_id,
                args.label,
                reason=args.reason,
                actor="cortex-operator:cli",
            )
        elif args.command == "pruning-report":
            result = store.adaptive_pruning_snapshot()
        elif args.command == "apply-pruning":
            result = store.apply_adaptive_pruning(
                args.decision_id,
                actor="cortex-operator:cli",
            )
        elif args.command == "undo-pruning":
            result = store.undo_adaptive_pruning(args.decision_id)
        elif args.command == "weight-proposals":
            result = {
                **store.scoring_weight_snapshot(),
                "auto_reverted": store.auto_revert_scoring_weights(),
            }
        elif args.command == "apply-weight-proposal":
            result = store.apply_scoring_weight_proposal(
                args.proposal_id,
                actor="cortex-operator:cli",
                note=args.note,
                confirm_large_change=bool(args.confirm_large_change),
            )
        elif args.command == "reject-weight-proposal":
            result = store.reject_scoring_weight_proposal(
                args.proposal_id,
                actor="cortex-operator:cli",
                note=args.note,
            )
        elif args.command == "rollback-weights":
            result = store.rollback_scoring_weights(
                args.proposal_id,
                reason=args.reason,
                actor="cortex-operator:cli",
            )
        elif args.command == "reset-weights":
            result = store.factory_reset_scoring_weights(
                args.task_type,
                actor="cortex-operator:cli",
                note=args.note,
            )
        elif args.command == "reconsolidation-proposals":
            result = store.adaptive_reconsolidation_snapshot()
        elif args.command == "apply-reconsolidation":
            result = store.apply_adaptive_reconsolidation(
                args.proposal_id,
                actor="cortex-operator:cli",
                confirm_protected=bool(args.confirm_protected),
            )
        elif args.command == "undo-reconsolidation":
            result = store.undo_adaptive_reconsolidation(args.proposal_id)
        elif args.command == "reconsolidation-feedback":
            result = store.record_adaptive_reconsolidation_feedback(
                args.proposal_id,
                args.label,
                reason=args.reason,
            )
        elif args.command == "schema-proposals":
            result = store.schema_formation_snapshot()
        elif args.command == "apply-schema":
            result = store.apply_schema_formation(
                args.proposal_id,
                actor="cortex-operator:cli",
            )
        elif args.command == "undo-schema":
            result = store.undo_schema_formation(args.proposal_id)
        elif args.command == "schema-feedback":
            result = store.record_schema_formation_feedback(
                args.proposal_id,
                args.label,
                reason=args.reason,
            )
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
        elif args.command == "neighborhood-report":
            result = store.neighborhood_training_snapshot(limit=args.limit)
        elif args.command == "decision-log":
            result = store.decision_log(limit=args.limit)
        elif args.command == "learning-dataset":
            result = store.learning_experience_dataset(
                limit=args.limit, include_text=args.include_text
            )
            if args.jsonl:
                for item in result["experiences"]:
                    print(json.dumps(item, ensure_ascii=False, default=str))
                return 0
        elif args.command == "quality-report":
            result = store.memory_quality_report()
        elif args.command == "scoring-trend":
            result = store.scoring_health(weeks=args.weeks, limit=args.limit)
        elif args.command == "attention-learning":
            result = store.attention_learning_summary(limit=args.limit)
        elif args.command == "refinery-report":
            result = store.refinery_classification_report()
        elif args.command == "refinery-summary":
            result = store.refinery_summary()
        elif args.command == "refinery-shadow":
            result = MemoryRetriever(store).shadow_tiered_comparison(args.query)
        else:
            result = store.maintenance(dry_run=not args.apply)
        if not (
            (args.command == "auto-judge" and args.quiet)
            or (args.command == "brain-mechanics" and args.quiet)
        ):
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
