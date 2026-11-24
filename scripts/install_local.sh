#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
TARGET="$HERMES_HOME/plugins/cortex"
BACKUP_ROOT="$HERMES_HOME/backups"

if [[ -d "$TARGET" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$BACKUP_ROOT"
  cp -a "$TARGET" "$BACKUP_ROOT/cortex-plugin-$timestamp"
  printf 'Backed up the previous Cortex plugin to %s\n' "$BACKUP_ROOT/cortex-plugin-$timestamp"
fi

mkdir -p "$TARGET"
for file in __init__.py __main__.py abstention.py abstention_ledger.py adaptive_reconsolidation.py adaptive_weights.py as_of.py assoc_feedback.py attribution.py autojudge.py backfill.py benchmarking.py boost_caps.py brain_mechanics.py calibration.py client.py cli.py cognition.py cortex_schema.py cortex_records.py continuity.py cortex_trace.py dashboard.py dashboard_auth.py dashboard.html density.py derived_facts.py diversity.py embeddings.py entity_anchor.py evaluation.py favicon.svg favicon.ico apple-touch-icon.png extraction.py floor_context.py fusion.py harness.py hermes_config.py hermes_hooks.py hermes_provider.py mcp_server.py runtime_path.py ab_policy.py confirmation.py contradiction.py contradiction_cluster.py core_backup.py cross_project.py core_block.py forget_ladder.py forget_preview.py regret.py suppression.py triage.py decay.py degree_caps.py derived_metadata.py dormancy.py edge_decay.py edge_explain.py edge_provenance.py edge_quarantine.py forget_entity.py forget_range.py legal_hold.py prune_receipts.py quota.py restore_ladder.py aliases.py anchoring.py archaeology.py co_retrieval.py compaction.py constraints.py context_tags.py importance_rubric.py dispute.py dry_run.py entity_merge.py entity_nodes.py entity_resolve.py entity_timeline.py extraction_regression.py fact_families.py expiry_classes.py model_gate.py near_miss.py negative_facts.py path_query.py person_profiles.py preference_strength.py quotes.py recall_events.py redaction_receipt.py reject_learn.py secret_refusal.py serial_position.py session_dedupe.py source_weight.py rtbf.py splitting.py volatility.py write_dedup.py write_receipts.py write_throttle.py storage_cost.py core_diff.py core_voice.py precompute.py project_scope.py promotion.py promotion_gate.py provenance.py prediction_error.py preload.py task_class.py hermes_receipts.py project_detect.py jev.py latency_budget.py lesson_demo.py merge_review.py metacognition.py migrations.py namespace.py negative_cues.py outcome_lessons.py outcomes.py recall_explain.py recall_replay.py redundancy.py refinery.py relationship_edges.py rerank.py relevance_pruning.py research.py retrieval.py review_copilot.py rewrite.py schema_formation.py scope_capture.py self_correction.py security.py semantic_consolidation.py semantics.py serializers.py session_scope.py sleep.py sleep_proposals.py source_authority.py stability.py store.py store_read.py task_budget.py temporal.py usage_weight.py validity.py citations.py shadow_rank.py counterfactuals.py summaries.py consolidation_queue.py bootstrap_gate.py plugin_drift.py label_independence.py audit_view.py phone_surface.py recall_receipt.py voice_summary.py turn_panel.py vault_header.py bootstrap_block.py layout_cache.py saved_views.py path_finder.py health.py latency_reconcile.py timeline_view.py share_export.py render_budget.py accessibility.py confidence_phrasing.py staleness_prompts.py correction_memory.py explain_memory.py consent_gate.py audit_digest.py trust_score.py preturn_contract.py tooling.py idle_window.py tool_links.py two_pass.py topic_nodes.py two_pass.py undo_learned.py vault.py plugin.yaml README.md AGENTS.md LICENSE CHANGELOG.md; do
  cp "$SOURCE_DIR/$file" "$TARGET/$file"
done
mkdir -p "$TARGET/scripts"
cp "$SOURCE_DIR/scripts/cortex-vault-index.service.in" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/cortex-vault-index.timer" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/install_vault_timer.sh" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/cortex-dashboard.service.in" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/install_dashboard_service.sh" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/cortex-sleep.service.in" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/cortex-sleep.timer" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/install_sleep_timer.sh" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/cortex-auto-judge.service.in" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/cortex-auto-judge.timer" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/install_auto_judge_timer.sh" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/uninstall_local.sh" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/hermes_python.sh" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/benchmark.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/benchmark_compare.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/benchmark_e2e.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/benchmark_aggregate.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/benchmark_social_card.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/benchmark_adaptive.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/benchmark_recall.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/evaluate_real_history.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/evaluate_preload.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/compare_retrieval_reports.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/download_embedding_model.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/backfill_embeddings.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/verify_hermes_install.py" "$TARGET/scripts/"
mkdir -p "$TARGET/benchmarks"
cp "$SOURCE_DIR/benchmarks/__init__.py" "$TARGET/benchmarks/"
cp "$SOURCE_DIR/benchmarks/core.py" "$TARGET/benchmarks/"
mkdir -p "$TARGET/docs"
cp "$SOURCE_DIR/docs/"*.md "$TARGET/docs/"
chmod +x "$TARGET/scripts/install_vault_timer.sh"
chmod +x "$TARGET/scripts/install_dashboard_service.sh"
chmod +x "$TARGET/scripts/install_sleep_timer.sh"
chmod +x "$TARGET/scripts/install_auto_judge_timer.sh"
chmod +x "$TARGET/scripts/uninstall_local.sh"
chmod +x "$TARGET/scripts/benchmark.py"
chmod +x "$TARGET/scripts/benchmark_compare.py"
chmod +x "$TARGET/scripts/benchmark_e2e.py"
chmod +x "$TARGET/scripts/benchmark_aggregate.py"
chmod +x "$TARGET/scripts/benchmark_social_card.py"
chmod +x "$TARGET/scripts/benchmark_adaptive.py"

printf 'Installed Cortex to %s\n' "$TARGET"
printf 'Agent install/configure checklist: %s/AGENTS.md\n' "$TARGET"
if [[ "${CORTEX_INSTALL_AUTO_JUDGE_TIMER:-0}" == "1" ]]; then
  if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    HERMES_HOME="$HERMES_HOME" CORTEX_PLUGIN_DIR="$TARGET" \
      "$TARGET/scripts/install_auto_judge_timer.sh"
  else
    printf 'Cortex auto-judge timer files were installed but not enabled: no user systemd session.\n' >&2
  fi
fi
printf 'Cortex is installed but not activated. Run: hermes memory setup\n'
