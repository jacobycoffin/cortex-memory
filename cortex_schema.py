"""Schema definition and migrations for the Cortex SQLite store.

Stage 1 of the store split plan (`docs/STORE_SPLIT_PLAN.md`): the schema /
migration cluster moved out of `store.py`. `store.py` re-imports every name
defined here and `CortexStore` keeps thin facades for the instance-level entry
points, so `from cortex.store import ...` and every `self._...` call site keep
working unchanged.

Transaction rule (acceptance criterion 2): nothing here opens a connection or
a transaction of its own. Every function takes the caller's connection, so the
store keeps its existing transaction boundaries -- `CortexStore._create_schema`
remains the single commit point for the open-time schema sequence.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Sequence

from .security import normalize_text
from .semantics import semantic_features


SCHEMA_VERSION = 32

# Moved verbatim out of ``CortexStore._create_schema`` (indentation included,
# so the stored DDL is byte-identical). It stays a single ``executescript``
# call, so SQLite's implicit pre-script COMMIT stays exactly where it was.
SCHEMA_SQL = """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            DROP TRIGGER IF EXISTS cortex_revision_memory_insert;
            DROP TRIGGER IF EXISTS cortex_revision_memory_delete;
            DROP TRIGGER IF EXISTS cortex_revision_memory_material_update;
            DROP TRIGGER IF EXISTS cortex_revision_edge_insert;
            DROP TRIGGER IF EXISTS cortex_revision_edge_update;
            DROP TRIGGER IF EXISTS cortex_revision_edge_delete;
            DROP TRIGGER IF EXISTS cortex_revision_tool_stats_insert;
            DROP TRIGGER IF EXISTS cortex_revision_tool_stats_update;
            DROP TRIGGER IF EXISTS cortex_revision_tool_stats_delete;
            DROP TRIGGER IF EXISTS cortex_revision_workflow_stats_insert;
            DROP TRIGGER IF EXISTS cortex_revision_workflow_stats_update;
            DROP TRIGGER IF EXISTS cortex_revision_workflow_stats_delete;

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL DEFAULT 'semantic',
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'conversation',
                source_category TEXT NOT NULL DEFAULT 'AGENT_INFERENCE',
                origin_source_category TEXT NOT NULL DEFAULT 'AGENT_INFERENCE',
                approval_state TEXT NOT NULL DEFAULT 'unreviewed',
                source_ref TEXT,
                session_id TEXT,
                context_mode TEXT NOT NULL DEFAULT 'standalone',
                scope_json TEXT NOT NULL DEFAULT '{}',
                entities_json TEXT NOT NULL DEFAULT '[]',
                preconditions_json TEXT NOT NULL DEFAULT '{}',
                source_context TEXT,
                applicable_systems_json TEXT NOT NULL DEFAULT '[]',
                applicable_versions_json TEXT NOT NULL DEFAULT '[]',
                metadata_completeness REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                subject TEXT,
                predicate TEXT,
                object_value TEXT,
                extraction_method TEXT NOT NULL DEFAULT 'unknown',
                confidence REAL NOT NULL DEFAULT 0.6,
                currentness_confidence REAL NOT NULL DEFAULT 0.7,
                importance REAL NOT NULL DEFAULT 0.5,
                uniqueness REAL NOT NULL DEFAULT 1.0,
                volatility REAL NOT NULL DEFAULT 0.4,
                trust REAL NOT NULL DEFAULT 0.7,
                record_role TEXT NOT NULL DEFAULT 'canonical',
                role_method TEXT NOT NULL DEFAULT 'legacy_default',
                role_version TEXT,
                role_reviewed_at TEXT,
                state TEXT NOT NULL DEFAULT 'active',
                stranded INTEGER NOT NULL DEFAULT 0,
                stranded_reason TEXT,
                stranded_at TEXT,
                pinned INTEGER NOT NULL DEFAULT 0,
                protected INTEGER NOT NULL DEFAULT 0,
                dirty INTEGER NOT NULL DEFAULT 0,
                dirty_reason TEXT,
                supersedes_id TEXT REFERENCES memories(id),
                quarantine_reason TEXT,
                retrieved_count INTEGER NOT NULL DEFAULT 0,
                selected_count INTEGER NOT NULL DEFAULT 0,
                injected_count INTEGER NOT NULL DEFAULT 0,
                used_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                confirmed_count INTEGER NOT NULL DEFAULT 0,
                validated_count INTEGER NOT NULL DEFAULT 0,
                helpful_count INTEGER NOT NULL DEFAULT 0,
                harmful_count INTEGER NOT NULL DEFAULT 0,
                correction_count INTEGER NOT NULL DEFAULT 0,
                false_positive_count INTEGER NOT NULL DEFAULT 0,
                duplicate_count INTEGER NOT NULL DEFAULT 0,
                last_retrieved_at TEXT,
                last_injected_at TEXT,
                last_used_at TEXT,
                last_helpful_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
            CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state, pinned, updated_at);
            CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);

            CREATE TABLE IF NOT EXISTS memory_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                state TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                system_from TEXT NOT NULL,
                system_to TEXT,
                reason TEXT,
                source_ref TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_versions_memory ON memory_versions(memory_id, system_from);

            CREATE TABLE IF NOT EXISTS edges (
                src_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                dst_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.25,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_reinforced_at TEXT NOT NULL,
                PRIMARY KEY (src_id, dst_id, relation),
                CHECK (src_id <> dst_id)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id, relation);

            CREATE TABLE IF NOT EXISTS edge_evidence (
                evidence_id TEXT PRIMARY KEY,
                src_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                dst_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                evidence_key TEXT NOT NULL,
                summary TEXT NOT NULL,
                source_ref TEXT,
                task_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                UNIQUE(src_id,dst_id,relation,evidence_type,evidence_key),
                CHECK(src_id <> dst_id)
            );
            CREATE INDEX IF NOT EXISTS idx_edge_evidence_edge
              ON edge_evidence(src_id,dst_id,relation,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_edge_evidence_task
              ON edge_evidence(task_id,created_at DESC);

            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                event TEXT NOT NULL,
                query TEXT,
                session_id TEXT,
                score REAL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_access_memory ON access_log(memory_id, created_at);

            CREATE TABLE IF NOT EXISTS access_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                accessed_at TEXT NOT NULL,
                task_type TEXT,
                context_hash TEXT,
                context_summary TEXT,
                outcome TEXT NOT NULL DEFAULT 'pending',
                event TEXT NOT NULL DEFAULT 'retrieved',
                reconsolidated INTEGER NOT NULL DEFAULT 0,
                shadow INTEGER NOT NULL DEFAULT 1,
                strength_before REAL,
                strength_after REAL
            );
            CREATE INDEX IF NOT EXISTS idx_access_history_memory ON access_history(memory_id, accessed_at);
            CREATE INDEX IF NOT EXISTS idx_access_history_shadow ON access_history(shadow, accessed_at);

            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_content TEXT NOT NULL,
                assistant_content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS usage_records (
                usage_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                session_id TEXT,
                query TEXT,
                score REAL,
                selected INTEGER NOT NULL DEFAULT 1,
                used INTEGER NOT NULL DEFAULT 0,
                attribution REAL NOT NULL DEFAULT 0.0,
                outcome TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_usage_task ON usage_records(task_id, outcome);
            CREATE INDEX IF NOT EXISTS idx_usage_memory ON usage_records(memory_id, created_at);

            CREATE TABLE IF NOT EXISTS memory_dependencies (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                evidence_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                relation TEXT NOT NULL DEFAULT 'derived_from',
                weight REAL NOT NULL DEFAULT 1.0,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                PRIMARY KEY(memory_id, evidence_id, relation),
                CHECK(memory_id <> evidence_id)
            );
            CREATE INDEX IF NOT EXISTS idx_dependencies_evidence ON memory_dependencies(evidence_id, active);

            CREATE TABLE IF NOT EXISTS tool_executions (
                execution_id TEXT PRIMARY KEY,
                session_id TEXT,
                task_type TEXT NOT NULL,
                task_context TEXT,
                tool_name TEXT NOT NULL,
                argument_keys TEXT NOT NULL,
                success INTEGER NOT NULL,
                error_type TEXT,
                result_summary TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tool_exec_lookup ON tool_executions(task_type, tool_name, success, created_at);

            CREATE TABLE IF NOT EXISTS tool_stats (
                task_type TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                helpful_count INTEGER NOT NULL DEFAULT 0,
                harmful_count INTEGER NOT NULL DEFAULT 0,
                argument_keys TEXT NOT NULL DEFAULT '[]',
                last_success_at TEXT,
                last_failure_at TEXT,
                last_error_type TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(task_type, tool_name)
            );

            CREATE TABLE IF NOT EXISTS tool_workflows (
                workflow_id TEXT PRIMARY KEY,
                session_id TEXT,
                task_type TEXT NOT NULL,
                task_fingerprint TEXT NOT NULL,
                task_context TEXT,
                steps_json TEXT NOT NULL,
                workflow_key TEXT NOT NULL,
                success INTEGER NOT NULL,
                error_type TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tool_workflows_lookup
              ON tool_workflows(task_type,task_fingerprint,success,created_at);

            CREATE TABLE IF NOT EXISTS tool_workflow_stats (
                task_type TEXT NOT NULL,
                task_fingerprint TEXT NOT NULL,
                workflow_key TEXT NOT NULL,
                steps_json TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                distinct_tasks INTEGER NOT NULL DEFAULT 0,
                last_error_type TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(task_type,task_fingerprint,workflow_key)
            );

            CREATE TABLE IF NOT EXISTS document_sources (
                source_path TEXT PRIMARY KEY,
                source_type TEXT NOT NULL DEFAULT 'obsidian_markdown',
                title TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                imported_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_document_sources_status
              ON document_sources(status,last_seen_at);

            CREATE TABLE IF NOT EXISTS document_chunks (
                source_path TEXT NOT NULL REFERENCES document_sources(source_path) ON DELETE CASCADE,
                chunk_key TEXT NOT NULL,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                chunk_hash TEXT NOT NULL,
                heading TEXT,
                ordinal INTEGER NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                imported_at TEXT NOT NULL,
                PRIMARY KEY(source_path,chunk_key)
            );
            CREATE INDEX IF NOT EXISTS idx_document_chunks_memory
              ON document_chunks(memory_id,active);

            CREATE TABLE IF NOT EXISTS maintenance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                dry_run INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS recall_runs (
                recall_id TEXT PRIMARY KEY,
                task_id TEXT,
                session_id TEXT,
                query TEXT,
                mode TEXT NOT NULL,
                reason TEXT,
                requested_limit INTEGER NOT NULL,
                token_budget INTEGER NOT NULL,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                selected_count INTEGER NOT NULL DEFAULT 0,
                estimated_tokens INTEGER NOT NULL DEFAULT 0,
                prepare_ms REAL NOT NULL DEFAULT 0,
                abstained INTEGER NOT NULL DEFAULT 0,
                stage_ms_json TEXT NOT NULL DEFAULT '{}',
                rendered_count INTEGER NOT NULL DEFAULT 0,
                withheld_count INTEGER NOT NULL DEFAULT 0,
                rendered_tokens INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_recall_runs_created ON recall_runs(created_at);

            CREATE TABLE IF NOT EXISTS memory_traces (
                trace_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL UNIQUE,
                session_id TEXT,
                goal TEXT NOT NULL,
                context_summary TEXT NOT NULL,
                retrieval_context_json TEXT NOT NULL DEFAULT '{}',
                task_type TEXT NOT NULL,
                recall_mode TEXT NOT NULL,
                retrieval_used INTEGER NOT NULL DEFAULT 0,
                retrieval_reason TEXT NOT NULL,
                queries_json TEXT NOT NULL DEFAULT '[]',
                candidate_memories_json TEXT NOT NULL DEFAULT '[]',
                selected_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                rejected_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                influence_json TEXT NOT NULL DEFAULT '[]',
                evaluations_json TEXT NOT NULL DEFAULT '[]',
                memory_actions_json TEXT NOT NULL DEFAULT '[]',
                outcome TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memory_traces_created
              ON memory_traces(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_traces_outcome
              ON memory_traces(outcome,created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_trace_events (
                sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                task_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_trace_events_task
              ON memory_trace_events(task_id,sequence_id);

            CREATE TABLE IF NOT EXISTS memory_write_decisions (
                decision_id TEXT PRIMARY KEY,
                session_id TEXT,
                candidate_hash TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'conversation',
                context_mode TEXT NOT NULL,
                reusable_score REAL NOT NULL,
                durability TEXT NOT NULL,
                quality_flags_json TEXT NOT NULL DEFAULT '[]',
                duplicate_memory_id TEXT,
                contradiction_ids_json TEXT NOT NULL DEFAULT '[]',
                independently_understandable INTEGER NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                memory_id TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_write_decisions_created
              ON memory_write_decisions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_write_decisions_memory
              ON memory_write_decisions(memory_id,created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_creation_proposals (
                proposal_id TEXT PRIMARY KEY,
                proposal_key TEXT NOT NULL,
                candidate_hash TEXT NOT NULL,
                content TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'semantic',
                source_type TEXT NOT NULL DEFAULT 'conversation',
                source_category TEXT NOT NULL DEFAULT 'AGENT_INFERENCE',
                source_ref TEXT,
                session_id TEXT,
                context_mode TEXT NOT NULL DEFAULT 'standalone',
                candidate_json TEXT NOT NULL DEFAULT '{}',
                assessment_json TEXT NOT NULL DEFAULT '{}',
                storage_policy TEXT NOT NULL DEFAULT 'automatic',
                quarantine_reason TEXT,
                redacted INTEGER NOT NULL DEFAULT 0,
                recurrence_count INTEGER NOT NULL DEFAULT 1,
                positive_feedback_count INTEGER NOT NULL DEFAULT 0,
                strong_feedback_count INTEGER NOT NULL DEFAULT 0,
                last_feedback_at TEXT,
                status TEXT NOT NULL DEFAULT 'pending'
                  CHECK(status IN ('pending','remembered','evidence_only','rejected','needs_context')),
                decision_action TEXT,
                decision_note TEXT,
                result_memory_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                review_id TEXT,
                actor TEXT,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                decided_at TEXT,
                review_started_at TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_creation_pending_key
              ON memory_creation_proposals(proposal_key)
              WHERE status IN ('pending','needs_context');
            CREATE INDEX IF NOT EXISTS idx_memory_creation_status
              ON memory_creation_proposals(status,last_seen_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_creation_candidate
              ON memory_creation_proposals(candidate_hash,last_seen_at DESC);

            CREATE TABLE IF NOT EXISTS memory_creation_feedback (
                feedback_id TEXT PRIMARY KEY,
                proposal_id TEXT NOT NULL REFERENCES memory_creation_proposals(proposal_id) ON DELETE CASCADE,
                session_id TEXT,
                strength TEXT NOT NULL CHECK(strength IN ('positive','strong')),
                source_type TEXT NOT NULL DEFAULT 'implicit_user_feedback',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_creation_feedback_proposal
              ON memory_creation_feedback(proposal_id,created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_experience_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL CHECK(event_type IN (
                  'candidate_observed','memory_selected','memory_used','outcome_labeled'
                )),
                proposal_id TEXT REFERENCES memory_creation_proposals(proposal_id) ON DELETE SET NULL,
                memory_id TEXT REFERENCES memories(id) ON DELETE CASCADE,
                task_id TEXT,
                session_id TEXT,
                source_type TEXT NOT NULL DEFAULT '',
                source_category TEXT NOT NULL DEFAULT '',
                outcome TEXT NOT NULL DEFAULT '',
                evidence_key TEXT NOT NULL UNIQUE,
                event_day TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_experience_memory
              ON memory_experience_events(memory_id,event_type,event_day,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_experience_proposal
              ON memory_experience_events(proposal_id,event_type,event_day,created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_recall_sets (
                recall_set_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'trained'
                  CHECK(kind IN ('legacy','trained')),
                status TEXT NOT NULL DEFAULT 'inactive'
                  CHECK(status IN ('active','inactive')),
                parent_recall_set_id TEXT REFERENCES memory_recall_sets(recall_set_id),
                actor TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                activated_at TEXT,
                deactivated_at TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_recall_sets_one_active
              ON memory_recall_sets(status) WHERE status='active';
            CREATE INDEX IF NOT EXISTS idx_memory_recall_sets_created
              ON memory_recall_sets(created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_recall_memberships (
                recall_set_id TEXT NOT NULL REFERENCES memory_recall_sets(recall_set_id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                eligibility TEXT NOT NULL DEFAULT 'primary'
                  CHECK(eligibility IN ('primary','evidence_only')),
                origin TEXT NOT NULL DEFAULT 'trusted_write',
                review_id TEXT,
                actor TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                revoked_at TEXT,
                PRIMARY KEY(recall_set_id,memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_recall_memberships_active
              ON memory_recall_memberships(recall_set_id,eligibility,memory_id)
              WHERE revoked_at IS NULL;
            CREATE INDEX IF NOT EXISTS idx_memory_recall_memberships_memory
              ON memory_recall_memberships(memory_id,recall_set_id);

            CREATE TABLE IF NOT EXISTS memory_recall_set_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                recall_set_id TEXT NOT NULL REFERENCES memory_recall_sets(recall_set_id) ON DELETE CASCADE,
                prior_recall_set_id TEXT,
                memory_id TEXT,
                eligibility TEXT,
                actor TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_recall_set_events_created
              ON memory_recall_set_events(created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_neighborhoods (
                neighborhood_id TEXT PRIMARY KEY,
                slug TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'topic',
                parent_neighborhood_id TEXT REFERENCES memory_neighborhoods(neighborhood_id),
                description TEXT NOT NULL DEFAULT '',
                safety_class TEXT NOT NULL DEFAULT 'normal'
                  CHECK(safety_class IN ('normal','secret_reference_only')),
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_neighborhood_memberships (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                neighborhood_id TEXT NOT NULL REFERENCES memory_neighborhoods(neighborhood_id) ON DELETE CASCADE,
                confidence REAL NOT NULL DEFAULT 1.0,
                origin TEXT NOT NULL DEFAULT 'deterministic',
                explanation TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                PRIMARY KEY(memory_id,neighborhood_id)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_neighborhood_memberships_neighborhood
              ON memory_neighborhood_memberships(neighborhood_id,memory_id);

            CREATE TABLE IF NOT EXISTS memory_neighborhood_evaluations (
                category TEXT NOT NULL CHECK(category IN ('project','service')),
                name_key TEXT NOT NULL,
                display_name TEXT NOT NULL,
                memory_count INTEGER NOT NULL DEFAULT 0,
                context_count INTEGER NOT NULL DEFAULT 0,
                decision TEXT NOT NULL CHECK(decision IN ('admitted','held','inactive')),
                reason TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                PRIMARY KEY(category,name_key)
            );
            CREATE TABLE IF NOT EXISTS memory_neighborhood_decision_events (
                event_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                name_key TEXT NOT NULL,
                display_name TEXT NOT NULL,
                prior_decision TEXT,
                decision TEXT NOT NULL,
                memory_count INTEGER NOT NULL DEFAULT 0,
                context_count INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_neighborhood_events_created
              ON memory_neighborhood_decision_events(created_at DESC);

            CREATE TRIGGER IF NOT EXISTS cortex_recall_membership_on_memory_insert
            AFTER INSERT ON memories BEGIN
              INSERT OR IGNORE INTO memory_recall_memberships(
                recall_set_id,memory_id,eligibility,origin,actor,reason,created_at
              )
              SELECT recall_set_id,NEW.id,
                     CASE WHEN kind='trained' AND (
                       NEW.source_type='vault_markdown' OR NEW.record_role='reference'
                     ) THEN 'evidence_only' ELSE 'primary' END,
                     'trusted_write','system','memory created while this recall set was active',NEW.created_at
              FROM memory_recall_sets WHERE status='active' LIMIT 1;
            END;

            CREATE TABLE IF NOT EXISTS memory_context_outcomes (
                task_id TEXT NOT NULL,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                context_key TEXT NOT NULL,
                context_json TEXT NOT NULL,
                selected INTEGER NOT NULL DEFAULT 1,
                used INTEGER NOT NULL DEFAULT 0,
                outcome TEXT NOT NULL DEFAULT 'pending',
                updated_at TEXT NOT NULL,
                PRIMARY KEY(task_id,memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_context_outcomes_lookup
              ON memory_context_outcomes(memory_id,context_key,outcome);

            CREATE TABLE IF NOT EXISTS recall_budget_observations (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                mode TEXT NOT NULL,
                requested_budget INTEGER NOT NULL,
                estimated_tokens INTEGER NOT NULL,
                selected_count INTEGER NOT NULL,
                used_count INTEGER NOT NULL DEFAULT 0,
                outcome TEXT NOT NULL DEFAULT 'pending',
                rendered_count INTEGER NOT NULL DEFAULT 0,
                rendered_tokens INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_budget_observations_lookup
              ON recall_budget_observations(task_type,mode,outcome,created_at);

            CREATE TABLE IF NOT EXISTS attention_observations (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                topics_json TEXT NOT NULL DEFAULT '[]',
                live_mode TEXT NOT NULL,
                live_budget INTEGER NOT NULL,
                shadow_mode TEXT NOT NULL,
                shadow_budget INTEGER NOT NULL,
                policy_version TEXT NOT NULL,
                selected_count INTEGER NOT NULL DEFAULT 0,
                usage_outcome TEXT NOT NULL DEFAULT 'pending'
                  CHECK(usage_outcome IN ('pending','used','ignored','no_context','control')),
                final_outcome TEXT NOT NULL DEFAULT 'pending'
                  CHECK(final_outcome IN ('pending','helpful','harmful','validated','corrected')),
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_attention_observations_lookup
              ON attention_observations(task_type,live_mode,usage_outcome,created_at);

            CREATE TABLE IF NOT EXISTS attention_weights (
                task_type TEXT NOT NULL,
                topic_key TEXT NOT NULL,
                mode TEXT NOT NULL,
                used_count INTEGER NOT NULL DEFAULT 0,
                helpful_count INTEGER NOT NULL DEFAULT 0,
                ignored_count INTEGER NOT NULL DEFAULT 0,
                harmful_count INTEGER NOT NULL DEFAULT 0,
                weight_delta REAL NOT NULL DEFAULT 0.0,
                last_observed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(task_type,topic_key,mode)
            );
            CREATE INDEX IF NOT EXISTS idx_attention_weights_topic
              ON attention_weights(topic_key,task_type,updated_at DESC);

            CREATE TABLE IF NOT EXISTS metacognitive_predictions (
                prediction_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                task_type TEXT NOT NULL,
                recall_mode TEXT NOT NULL,
                monitor_mode TEXT NOT NULL DEFAULT 'shadow',
                source_category TEXT NOT NULL,
                raw_probability REAL NOT NULL,
                calibrated_probability REAL NOT NULL,
                decision TEXT NOT NULL,
                applied INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL,
                calibration_scope TEXT NOT NULL DEFAULT 'prior',
                calibration_samples INTEGER NOT NULL DEFAULT 0,
                features_json TEXT NOT NULL DEFAULT '{}',
                outcome TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                UNIQUE(task_id,memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_metacognitive_outcomes
              ON metacognitive_predictions(task_type,source_category,outcome,created_at);
            CREATE INDEX IF NOT EXISTS idx_metacognitive_created
              ON metacognitive_predictions(created_at DESC);

            CREATE TABLE IF NOT EXISTS task_outcome_labels (
                label_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                actor TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'dashboard',
                prior_outcome TEXT NOT NULL DEFAULT 'used',
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                reversed_at TEXT,
                reversed_by TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_task_outcome_active
              ON task_outcome_labels(task_id) WHERE active=1;
            CREATE INDEX IF NOT EXISTS idx_task_outcome_created
              ON task_outcome_labels(created_at DESC);

            CREATE TABLE IF NOT EXISTS evaluation_cases (
                case_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL UNIQUE,
                query TEXT NOT NULL,
                relevant_memory_ids TEXT NOT NULL,
                task_type TEXT NOT NULL DEFAULT 'general',
                source_label_id TEXT NOT NULL REFERENCES task_outcome_labels(label_id),
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_evaluation_cases_active
              ON evaluation_cases(active,updated_at DESC);

            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id TEXT PRIMARY KEY,
                suite TEXT NOT NULL,
                suite_version TEXT NOT NULL,
                status TEXT NOT NULL,
                phase TEXT NOT NULL DEFAULT 'queued',
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                case_count INTEGER NOT NULL DEFAULT 0,
                adaptive_hit_at_k REAL,
                adaptive_recall_at_k REAL,
                adaptive_mrr REAL,
                fixed_hit_at_k REAL,
                fixed_recall_at_k REAL,
                fixed_mrr REAL,
                context_delta_p50 REAL,
                latency_delta_p95 REAL,
                result_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_evaluation_runs_started
              ON evaluation_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS tool_guidance_exposures (
                exposure_id TEXT PRIMARY KEY,
                session_id TEXT,
                task_id TEXT,
                task_type TEXT NOT NULL,
                guidance_type TEXT NOT NULL,
                guidance_key TEXT NOT NULL,
                recommended_tool TEXT,
                predicted_reliability REAL NOT NULL,
                followed INTEGER,
                success INTEGER,
                task_outcome TEXT,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_guidance_exposure_session
              ON tool_guidance_exposures(session_id,resolved_at,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_guidance_exposure_type
              ON tool_guidance_exposures(guidance_type,created_at DESC);

            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id TEXT PRIMARY KEY,
                suite TEXT NOT NULL,
                suite_version TEXT NOT NULL,
                status TEXT NOT NULL,
                phase TEXT NOT NULL DEFAULT 'queued',
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                score REAL,
                quality_score REAL,
                speed_score REAL,
                recall_at_k REAL,
                mrr REAL,
                precision_at_k REAL,
                p50_ms REAL,
                p95_ms REAL,
                context_tokens_p50 REAL,
                default_coverage REAL,
                corpus_memories INTEGER NOT NULL,
                queries INTEGER NOT NULL,
                result_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_benchmark_runs_started
              ON benchmark_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS lifecycle_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                from_state TEXT NOT NULL,
                to_state TEXT NOT NULL,
                reason TEXT,
                retention_score REAL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_lifecycle_memory ON lifecycle_events(memory_id,created_at);

            CREATE TABLE IF NOT EXISTS pruning_regret (
                regret_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                query TEXT,
                score REAL NOT NULL,
                restored INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pruning_regret_created ON pruning_regret(created_at);

            CREATE TABLE IF NOT EXISTS adaptive_pruning_runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL CHECK(mode IN ('shadow','apply')),
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                relevance_threshold REAL NOT NULL,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                judgment_count INTEGER NOT NULL DEFAULT 0,
                applied_count INTEGER NOT NULL DEFAULT 0,
                usage_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_adaptive_pruning_runs_started
              ON adaptive_pruning_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS adaptive_pruning_decisions (
                decision_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES adaptive_pruning_runs(run_id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                memory_hash TEXT NOT NULL,
                relevance_score REAL NOT NULL,
                score_evidence_json TEXT NOT NULL DEFAULT '{}',
                action TEXT NOT NULL
                  CHECK(action IN ('cool','archive','quarantine','keep','orphan_strand')),
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL
                  CHECK(status IN ('proposed','applied','kept','skipped','reversed')),
                prior_state TEXT NOT NULL,
                result_state TEXT,
                edge_snapshot_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                applied_at TEXT,
                reversed_at TEXT,
                UNIQUE(run_id,memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_adaptive_pruning_decisions_status
              ON adaptive_pruning_decisions(status,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_adaptive_pruning_decisions_memory
              ON adaptive_pruning_decisions(memory_id,created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_strands (
                strand_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL UNIQUE
                  REFERENCES adaptive_pruning_decisions(decision_id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                prior_state TEXT NOT NULL,
                reason TEXT NOT NULL,
                edge_snapshot_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active','restored','reversed')),
                created_at TEXT NOT NULL,
                restored_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memory_strands_active
              ON memory_strands(memory_id,status,created_at DESC);

            CREATE TABLE IF NOT EXISTS scoring_weight_runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL DEFAULT 'shadow' CHECK(mode='shadow'),
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                lookback_days INTEGER NOT NULL DEFAULT 7,
                task_type_count INTEGER NOT NULL DEFAULT 0,
                proposal_count INTEGER NOT NULL DEFAULT 0,
                usage_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_scoring_weight_runs_started
              ON scoring_weight_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS scoring_weight_proposals (
                proposal_id TEXT PRIMARY KEY,
                run_id TEXT REFERENCES scoring_weight_runs(run_id) ON DELETE SET NULL,
                task_type TEXT NOT NULL,
                baseline_version TEXT NOT NULL,
                baseline_weights_json TEXT NOT NULL,
                proposed_weights_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '{}',
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                max_deviation REAL NOT NULL,
                explicit_confirmation_required INTEGER NOT NULL DEFAULT 0,
                baseline_precision REAL,
                status TEXT NOT NULL CHECK(status IN (
                  'proposed','approved','rejected','rolled_back','auto_reverted'
                )),
                created_at TEXT NOT NULL,
                decided_at TEXT,
                decided_by TEXT,
                decision_note TEXT,
                rolled_back_at TEXT,
                rollback_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_scoring_weight_proposals_status
              ON scoring_weight_proposals(status,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_scoring_weight_proposals_task
              ON scoring_weight_proposals(task_type,created_at DESC);

            CREATE TABLE IF NOT EXISTS scoring_weight_history (
                history_id TEXT PRIMARY KEY,
                proposal_id TEXT REFERENCES scoring_weight_proposals(proposal_id) ON DELETE SET NULL,
                task_type TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                weights_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active','superseded','rolled_back')),
                precision_at_activation REAL,
                activation_sample_count INTEGER NOT NULL DEFAULT 0,
                activated_at TEXT NOT NULL,
                deactivated_at TEXT,
                actor TEXT NOT NULL,
                note TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_scoring_weight_active_task
              ON scoring_weight_history(task_type) WHERE status='active';

            CREATE TABLE IF NOT EXISTS adaptive_reconsolidation_runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                task_id TEXT,
                lability_minutes INTEGER NOT NULL DEFAULT 30,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                proposal_count INTEGER NOT NULL DEFAULT 0,
                usage_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_adaptive_reconsolidation_runs_started
              ON adaptive_reconsolidation_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS adaptive_reconsolidation_proposals (
                proposal_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL
                  REFERENCES adaptive_reconsolidation_runs(run_id) ON DELETE CASCADE,
                task_id TEXT NOT NULL,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                evidence_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                memory_hash TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('supersede','extend','conflict')),
                replacement_content TEXT,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                protected_confirmation_required INTEGER NOT NULL DEFAULT 0,
                source_snapshot_json TEXT NOT NULL DEFAULT '{}',
                edge_snapshot_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL CHECK(status IN (
                  'proposed','applied','rejected','reversed','skipped'
                )),
                created_at TEXT NOT NULL,
                applied_at TEXT,
                applied_by TEXT,
                reversed_at TEXT,
                feedback_label TEXT CHECK(feedback_label IN ('correct','wrong')),
                feedback_reason TEXT,
                feedback_at TEXT,
                UNIQUE(run_id,task_id,memory_id,evidence_memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_adaptive_reconsolidation_status
              ON adaptive_reconsolidation_proposals(status,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_adaptive_reconsolidation_memory
              ON adaptive_reconsolidation_proposals(memory_id,created_at DESC);

            CREATE TABLE IF NOT EXISTS schema_formation_runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                minimum_cluster INTEGER NOT NULL DEFAULT 3,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                proposal_count INTEGER NOT NULL DEFAULT 0,
                usage_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_schema_formation_runs_started
              ON schema_formation_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS schema_formation_proposals (
                proposal_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES schema_formation_runs(run_id) ON DELETE CASCADE,
                cluster_signature TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('abstract','no_schema','partial')),
                abstract_content TEXT,
                source_ids_json TEXT NOT NULL,
                included_source_ids_json TEXT NOT NULL,
                source_hashes_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '{}',
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN (
                  'proposed','applied','dismissed','rejected','reversed','skipped'
                )),
                result_memory_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                created_at TEXT NOT NULL,
                applied_at TEXT,
                applied_by TEXT,
                reversed_at TEXT,
                feedback_label TEXT CHECK(feedback_label IN ('correct','wrong')),
                feedback_reason TEXT,
                feedback_at TEXT,
                UNIQUE(run_id,cluster_signature)
            );
            CREATE INDEX IF NOT EXISTS idx_schema_formation_status
              ON schema_formation_proposals(status,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_schema_formation_signature
              ON schema_formation_proposals(cluster_signature,created_at DESC);

            CREATE TABLE IF NOT EXISTS consolidation_runs (
                run_id TEXT PRIMARY KEY,
                dry_run INTEGER NOT NULL,
                cluster_count INTEGER NOT NULL,
                member_count INTEGER NOT NULL,
                details TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS consolidation_members (
                run_id TEXT NOT NULL REFERENCES consolidation_runs(run_id) ON DELETE CASCADE,
                canonical_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                member_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                similarity REAL NOT NULL,
                prior_state TEXT NOT NULL,
                PRIMARY KEY(run_id,member_id)
            );

            CREATE TABLE IF NOT EXISTS semantic_consolidation_runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL CHECK(mode IN ('shadow','apply')),
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                judgment_count INTEGER NOT NULL DEFAULT 0,
                merge_count INTEGER NOT NULL DEFAULT 0,
                keep_count INTEGER NOT NULL DEFAULT 0,
                link_count INTEGER NOT NULL DEFAULT 0,
                applied_count INTEGER NOT NULL DEFAULT 0,
                usage_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_semantic_consolidation_runs_started
              ON semantic_consolidation_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS semantic_consolidation_decisions (
                decision_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES semantic_consolidation_runs(run_id) ON DELETE CASCADE,
                left_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                right_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                left_hash TEXT NOT NULL,
                right_hash TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('merge','keep_separate','link_as_related')),
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                merged_content TEXT,
                candidate_evidence_json TEXT NOT NULL DEFAULT '[]',
                source_snapshot_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL CHECK(
                    status IN ('proposed','applied','kept_separate','linked','skipped','reversed')
                ),
                result_memory_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                created_at TEXT NOT NULL,
                applied_at TEXT,
                reversed_at TEXT,
                UNIQUE(run_id,left_id,right_id)
            );
            CREATE INDEX IF NOT EXISTS idx_semantic_consolidation_decisions_status
              ON semantic_consolidation_decisions(status,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_semantic_consolidation_decisions_pair
              ON semantic_consolidation_decisions(left_id,right_id,created_at DESC);

            CREATE TABLE IF NOT EXISTS semantic_consolidation_feedback (
                feedback_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL
                  REFERENCES semantic_consolidation_decisions(decision_id) ON DELETE CASCADE,
                label TEXT NOT NULL CHECK(label IN ('correct','wrong')),
                reason TEXT NOT NULL DEFAULT '',
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_semantic_consolidation_feedback_decision
              ON semantic_consolidation_feedback(decision_id,created_at DESC);

            CREATE TABLE IF NOT EXISTS sleep_runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                cutoff_at TEXT NOT NULL,
                episodes_scanned INTEGER NOT NULL DEFAULT 0,
                episodes_replayed INTEGER NOT NULL DEFAULT 0,
                association_proposals INTEGER NOT NULL DEFAULT 0,
                interference_proposals INTEGER NOT NULL DEFAULT 0,
                edge_decay_proposals INTEGER NOT NULL DEFAULT 0,
                lifecycle_candidates INTEGER NOT NULL DEFAULT 0,
                consolidation_candidates INTEGER NOT NULL DEFAULT 0,
                dependency_candidates INTEGER NOT NULL DEFAULT 0,
                applied_changes INTEGER NOT NULL DEFAULT 0,
                reflection_token_budget INTEGER NOT NULL DEFAULT 0,
                reflection_estimated_tokens INTEGER NOT NULL DEFAULT 0,
                reflection_billed_tokens INTEGER,
                reflection_status TEXT NOT NULL DEFAULT 'disabled',
                report_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_sleep_runs_started
              ON sleep_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS sleep_episode_state (
                episode_id INTEGER PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
                first_replayed_at TEXT NOT NULL,
                last_replayed_at TEXT NOT NULL,
                replay_count INTEGER NOT NULL DEFAULT 1,
                last_run_id TEXT NOT NULL REFERENCES sleep_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS sleep_usage_state (
                task_id TEXT PRIMARY KEY,
                processed_at TEXT NOT NULL,
                last_run_id TEXT NOT NULL REFERENCES sleep_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS sleep_association_evidence (
                src_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                dst_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                witness_key TEXT NOT NULL,
                evidence_kind TEXT NOT NULL,
                score REAL NOT NULL,
                episode_id INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
                run_id TEXT NOT NULL REFERENCES sleep_runs(run_id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                PRIMARY KEY(src_id,dst_id,witness_key,evidence_kind),
                CHECK(src_id < dst_id)
            );
            CREATE INDEX IF NOT EXISTS idx_sleep_evidence_pair
              ON sleep_association_evidence(src_id,dst_id,evidence_kind);

            CREATE TABLE IF NOT EXISTS sleep_proposals (
                proposal_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES sleep_runs(run_id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                src_id TEXT REFERENCES memories(id) ON DELETE CASCADE,
                dst_id TEXT REFERENCES memories(id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'proposed',
                score REAL NOT NULL DEFAULT 0.0,
                evidence_count INTEGER NOT NULL DEFAULT 0,
                rationale TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sleep_proposals_run
              ON sleep_proposals(run_id,kind,status);

            CREATE TABLE IF NOT EXISTS operator_review_decisions (
                review_id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL,
                item_key TEXT NOT NULL,
                proposal_id TEXT,
                src_id TEXT,
                dst_id TEXT,
                action TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                reason_text TEXT,
                prior_json TEXT NOT NULL DEFAULT '{}',
                effect_json TEXT NOT NULL DEFAULT '{}',
                learning_signal_json TEXT NOT NULL DEFAULT '{}',
                decision_scope TEXT NOT NULL DEFAULT 'item_only'
                  CHECK(decision_scope IN ('item_only','exact_duplicates','policy_evidence')),
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reversed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_operator_review_created
              ON operator_review_decisions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_operator_review_item
              ON operator_review_decisions(item_type,item_key,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_operator_review_proposal
              ON operator_review_decisions(proposal_id, reversed_at);

            CREATE TABLE IF NOT EXISTS review_copilot_interpretations (
                interpretation_id TEXT PRIMARY KEY,
                proposal_id TEXT REFERENCES sleep_proposals(proposal_id) ON DELETE SET NULL,
                item_key TEXT NOT NULL,
                operator_text TEXT NOT NULL,
                conversation_json TEXT NOT NULL DEFAULT '[]',
                response_mode TEXT NOT NULL CHECK(response_mode IN ('clarify','recommendation')),
                response_json TEXT NOT NULL DEFAULT '{}',
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                created_at TEXT NOT NULL,
                confirmed_review_id TEXT REFERENCES operator_review_decisions(review_id) ON DELETE SET NULL,
                confirmed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_review_copilot_item
              ON review_copilot_interpretations(item_key,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_review_copilot_confirmed
              ON review_copilot_interpretations(confirmed_review_id,confirmed_at DESC);

            CREATE TABLE IF NOT EXISTS policy_candidates (
                candidate_id TEXT PRIMARY KEY,
                signal_key TEXT NOT NULL UNIQUE,
                domain TEXT NOT NULL,
                title TEXT NOT NULL,
                explanation TEXT NOT NULL,
                selector_json TEXT NOT NULL DEFAULT '{}',
                change_json TEXT NOT NULL DEFAULT '{}',
                direction TEXT NOT NULL,
                support_count INTEGER NOT NULL DEFAULT 0,
                oppose_count INTEGER NOT NULL DEFAULT 0,
                context_count INTEGER NOT NULL DEFAULT 0,
                consistency REAL NOT NULL DEFAULT 0.0,
                stage TEXT NOT NULL DEFAULT 'collecting',
                evidence_json TEXT NOT NULL DEFAULT '{}',
                replay_json TEXT NOT NULL DEFAULT '{}',
                shadow_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                decided_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_policy_candidates_stage
              ON policy_candidates(stage,updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_policy_candidates_domain
              ON policy_candidates(domain,updated_at DESC);

            CREATE TABLE IF NOT EXISTS policy_versions (
                version_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL REFERENCES policy_candidates(candidate_id) ON DELETE RESTRICT,
                domain TEXT NOT NULL,
                activation_scope TEXT NOT NULL,
                selector_json TEXT NOT NULL DEFAULT '{}',
                change_json TEXT NOT NULL DEFAULT '{}',
                evidence_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'active',
                activated_by TEXT NOT NULL,
                activated_at TEXT NOT NULL,
                deactivated_at TEXT,
                deactivated_by TEXT,
                rollback_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_policy_versions_active
              ON policy_versions(status,domain,activated_at DESC);

            CREATE TABLE IF NOT EXISTS policy_events (
                event_id TEXT PRIMARY KEY,
                candidate_id TEXT REFERENCES policy_candidates(candidate_id) ON DELETE SET NULL,
                version_id TEXT REFERENCES policy_versions(version_id) ON DELETE SET NULL,
                event_type TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}',
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_policy_events_created
              ON policy_events(created_at DESC);

            CREATE TABLE IF NOT EXISTS policy_proposal_effects (
                version_id TEXT NOT NULL REFERENCES policy_versions(version_id) ON DELETE CASCADE,
                proposal_id TEXT NOT NULL REFERENCES sleep_proposals(proposal_id) ON DELETE CASCADE,
                prior_status TEXT NOT NULL,
                next_status TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reversed_at TEXT,
                PRIMARY KEY(version_id,proposal_id)
            );
            CREATE INDEX IF NOT EXISTS idx_policy_proposal_effects_active
              ON policy_proposal_effects(version_id,next_status,reversed_at);

            CREATE TABLE IF NOT EXISTS sleep_edge_changes (
                change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES sleep_runs(run_id) ON DELETE CASCADE,
                src_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                dst_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                prior_exists INTEGER NOT NULL,
                prior_weight REAL,
                prior_evidence_count INTEGER,
                prior_last_reinforced_at TEXT,
                next_weight REAL NOT NULL,
                next_evidence_count INTEGER NOT NULL,
                next_last_reinforced_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reversed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS sleep_state_changes (
                change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES sleep_runs(run_id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                prior_state TEXT NOT NULL,
                next_state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reversed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS sleep_edge_downscale_state (
                src_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                dst_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                source_last_reinforced_at TEXT NOT NULL,
                last_run_id TEXT NOT NULL REFERENCES sleep_runs(run_id) ON DELETE CASCADE,
                last_downscaled_at TEXT NOT NULL,
                PRIMARY KEY(src_id,dst_id,relation)
            );

            CREATE TABLE IF NOT EXISTS controlled_experiments (
                experiment_id TEXT PRIMARY KEY,
                experiment_key TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'stopped',
                conditions_json TEXT NOT NULL,
                assignment_rule TEXT NOT NULL,
                primary_metric TEXT NOT NULL,
                minimum_labeled_per_condition INTEGER NOT NULL DEFAULT 8,
                started_at TEXT,
                stopped_at TEXT,
                created_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_controlled_experiment_active
              ON controlled_experiments(experiment_key) WHERE status='active';

            CREATE TABLE IF NOT EXISTS recall_experiment_assignments (
                assignment_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL REFERENCES controlled_experiments(experiment_id) ON DELETE CASCADE,
                task_id TEXT UNIQUE,
                session_id TEXT,
                query_hash TEXT NOT NULL,
                task_type TEXT NOT NULL,
                condition TEXT NOT NULL,
                random_bucket INTEGER NOT NULL,
                outcome TEXT NOT NULL DEFAULT 'pending',
                outcome_source TEXT,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_recall_experiment_results
              ON recall_experiment_assignments(experiment_id,condition,outcome_source,created_at);

            CREATE TABLE IF NOT EXISTS agent_task_observations (
                task_id TEXT PRIMARY KEY,
                session_id TEXT,
                task_type TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                query_preview TEXT,
                recall_condition TEXT NOT NULL,
                recall_mode TEXT NOT NULL,
                memory_count INTEGER NOT NULL DEFAULT 0,
                context_tokens INTEGER NOT NULL DEFAULT 0,
                prepare_ms REAL NOT NULL DEFAULT 0,
                response_ms REAL,
                tool_calls INTEGER NOT NULL DEFAULT 0,
                tool_successes INTEGER NOT NULL DEFAULT 0,
                correction_detected INTEGER NOT NULL DEFAULT 0,
                outcome TEXT NOT NULL DEFAULT 'pending',
                outcome_source TEXT,
                experiment_assignment_id TEXT REFERENCES recall_experiment_assignments(assignment_id),
                started_at TEXT NOT NULL,
                completed_at TEXT,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_agent_tasks_day
              ON agent_task_observations(started_at,task_type,outcome);

            CREATE TABLE IF NOT EXISTS sleep_trials (
                trial_id TEXT PRIMARY KEY,
                sleep_run_id TEXT REFERENCES sleep_runs(run_id) ON DELETE SET NULL,
                status TEXT NOT NULL,
                assignment_rule TEXT NOT NULL,
                primary_metric TEXT NOT NULL,
                pair_count INTEGER NOT NULL DEFAULT 0,
                minimum_labeled_per_arm INTEGER NOT NULL DEFAULT 8,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS sleep_trial_items (
                item_id TEXT PRIMARY KEY,
                trial_id TEXT NOT NULL REFERENCES sleep_trials(trial_id) ON DELETE CASCADE,
                pair_key TEXT NOT NULL,
                proposal_id TEXT NOT NULL REFERENCES sleep_proposals(proposal_id) ON DELETE CASCADE,
                assignment TEXT NOT NULL,
                exposure TEXT NOT NULL,
                src_id TEXT REFERENCES memories(id) ON DELETE CASCADE,
                dst_id TEXT REFERENCES memories(id) ON DELETE CASCADE,
                score REAL NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(trial_id,proposal_id)
            );
            CREATE INDEX IF NOT EXISTS idx_sleep_trial_items
              ON sleep_trial_items(trial_id,assignment,pair_key);

            CREATE TABLE IF NOT EXISTS summary_candidates (
                candidate_id TEXT PRIMARY KEY,
                source_signature TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'proposed',
                approved_memory_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                reviewed_by TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_summary_candidates_status
              ON summary_candidates(status,created_at DESC);

            CREATE TABLE IF NOT EXISTS summary_candidate_claims (
                candidate_id TEXT NOT NULL REFERENCES summary_candidates(candidate_id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                claim_text TEXT NOT NULL,
                PRIMARY KEY(candidate_id,ordinal)
            );

            CREATE TABLE IF NOT EXISTS summary_candidate_sources (
                candidate_id TEXT NOT NULL,
                claim_ordinal INTEGER NOT NULL,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                excerpt TEXT NOT NULL,
                PRIMARY KEY(candidate_id,claim_ordinal,memory_id),
                FOREIGN KEY(candidate_id,claim_ordinal)
                  REFERENCES summary_candidate_claims(candidate_id,ordinal) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS prospective_items (
                memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'open',
                due_at TEXT,
                completed_at TEXT,
                abandoned_at TEXT,
                note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_prospective_due
              ON prospective_items(status,due_at);

            CREATE TABLE IF NOT EXISTS reconsolidation_events (
                event_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                prior_version_id INTEGER REFERENCES memory_versions(version_id) ON DELETE SET NULL,
                new_version_id INTEGER REFERENCES memory_versions(version_id) ON DELETE SET NULL,
                trigger_access_at TEXT,
                corrected_at TEXT NOT NULL,
                first_reused_at TEXT,
                stabilized_at TEXT,
                failed_at TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                outcome TEXT,
                source_ref TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_reconsolidation_status
              ON reconsolidation_events(status,corrected_at DESC);

            CREATE TABLE IF NOT EXISTS memory_presentations (
                memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                display_title TEXT NOT NULL,
                display_summary TEXT NOT NULL,
                applies_when TEXT NOT NULL,
                retention_reason TEXT NOT NULL,
                readability_flags_json TEXT NOT NULL DEFAULT '[]',
                presentation_method TEXT NOT NULL,
                presentation_version TEXT NOT NULL,
                source_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_refinery_proposals (
                proposal_id TEXT PRIMARY KEY,
                proposal_kind TEXT NOT NULL
                  CHECK(proposal_kind IN ('promote','rewrite','split','role_change','merge','archive','trash')),
                source_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                proposed_records_json TEXT NOT NULL DEFAULT '[]',
                dependencies_json TEXT NOT NULL DEFAULT '[]',
                rationale TEXT NOT NULL DEFAULT '',
                readability_evidence_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'proposed'
                  CHECK(status IN ('proposed','applied','rejected','undone')),
                actor TEXT NOT NULL DEFAULT '',
                review_id TEXT,
                before_state_json TEXT NOT NULL DEFAULT '{}',
                undo_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                decided_at TEXT,
                undone_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_refinery_proposals_status
              ON memory_refinery_proposals(status,created_at DESC);

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                memory_id UNINDEXED,
                content,
                tokenize='porter unicode61'
            );

            CREATE TABLE IF NOT EXISTS memory_features (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                feature TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY(memory_id,feature)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_features_feature ON memory_features(feature,memory_id);

            CREATE TABLE IF NOT EXISTS memory_context_terms (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                term_type TEXT NOT NULL,
                term_key TEXT NOT NULL DEFAULT '',
                term_value TEXT NOT NULL,
                PRIMARY KEY(memory_id,term_type,term_key,term_value)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_context_terms_lookup
              ON memory_context_terms(term_type,term_key,term_value,memory_id);

            CREATE TABLE IF NOT EXISTS feature_stats (
                feature TEXT PRIMARY KEY,
                document_frequency INTEGER NOT NULL DEFAULT 0
            );
            """


def _create_schema(conn: sqlite3.Connection) -> None:
    """Run the base CREATE TABLE / CREATE INDEX block on ``conn``."""

    conn.executescript(SCHEMA_SQL)


def _migrate_columns(conn: sqlite3.Connection) -> None:
    """Add v2 columns safely when opening an early Cortex prototype DB."""

    columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    additions = {
        "source_category": "TEXT NOT NULL DEFAULT 'AGENT_INFERENCE'",
        "origin_source_category": "TEXT NOT NULL DEFAULT 'AGENT_INFERENCE'",
        "approval_state": "TEXT NOT NULL DEFAULT 'unreviewed'",
        "observed_at": "TEXT",
        "subject": "TEXT",
        "predicate": "TEXT",
        "object_value": "TEXT",
        "extraction_method": "TEXT NOT NULL DEFAULT 'unknown'",
        "currentness_confidence": "REAL NOT NULL DEFAULT 0.7",
        "uniqueness": "REAL NOT NULL DEFAULT 1.0",
        "protected": "INTEGER NOT NULL DEFAULT 0",
        "dirty": "INTEGER NOT NULL DEFAULT 0",
        "dirty_reason": "TEXT",
        "supersedes_id": "TEXT",
        "selected_count": "INTEGER NOT NULL DEFAULT 0",
        "validated_count": "INTEGER NOT NULL DEFAULT 0",
        "helpful_count": "INTEGER NOT NULL DEFAULT 0",
        "harmful_count": "INTEGER NOT NULL DEFAULT 0",
        "last_helpful_at": "TEXT",
        "strength": "REAL NOT NULL DEFAULT 1.0",
        "lability_until": "TEXT",
        "context_mode": "TEXT NOT NULL DEFAULT 'standalone'",
        "scope_json": "TEXT NOT NULL DEFAULT '{}'",
        "entities_json": "TEXT NOT NULL DEFAULT '[]'",
        "preconditions_json": "TEXT NOT NULL DEFAULT '{}'",
        "source_context": "TEXT",
        "applicable_systems_json": "TEXT NOT NULL DEFAULT '[]'",
        "applicable_versions_json": "TEXT NOT NULL DEFAULT '[]'",
        "metadata_completeness": "REAL NOT NULL DEFAULT 1.0",
        "record_role": "TEXT NOT NULL DEFAULT 'canonical'",
        "role_method": "TEXT NOT NULL DEFAULT 'legacy_default'",
        "role_version": "TEXT",
        "role_reviewed_at": "TEXT",
        "stranded": "INTEGER NOT NULL DEFAULT 0",
        "stranded_reason": "TEXT",
        "stranded_at": "TEXT",
    }
    for name, declaration in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {declaration}")
    conn.execute("UPDATE memories SET observed_at=COALESCE(observed_at, created_at)")
    conn.execute(
        """UPDATE memories
           SET origin_source_category=CASE
             WHEN origin_source_category IS NULL OR origin_source_category='' THEN source_category
             WHEN origin_source_category='AGENT_INFERENCE'
                  AND source_category<>'AGENT_INFERENCE' THEN source_category
             ELSE origin_source_category END,
               approval_state=CASE
             WHEN approval_state IS NULL OR approval_state='' THEN
               CASE
                 WHEN source_category='OPERATOR_APPROVED' THEN 'operator_approved'
                 WHEN source_category='AUTOMATIC_APPROVED' THEN 'automatic_approved'
                 ELSE 'unreviewed' END
             WHEN approval_state='unreviewed' AND source_category='OPERATOR_APPROVED'
               THEN 'operator_approved'
             WHEN approval_state='unreviewed' AND source_category='AUTOMATIC_APPROVED'
               THEN 'automatic_approved'
             ELSE approval_state END"""
    )
    # Schema 23 did not persist claim origin separately from review
    # authority. Recover it only when the proposal decision proves that it
    # created this memory; duplicate reviews must never rewrite an existing
    # memory's provenance.
    conn.execute(
        """UPDATE memories
           SET origin_source_category=(
             SELECT p.source_category
             FROM memory_creation_proposals p
             JOIN operator_review_decisions d ON d.review_id=p.review_id
             WHERE p.result_memory_id=memories.id
               AND json_extract(d.effect_json,'$.memory_created')=1
             ORDER BY p.decided_at DESC LIMIT 1
           )
           WHERE origin_source_category IN (
             'AGENT_INFERENCE','OPERATOR_APPROVED','AUTOMATIC_APPROVED'
           )
             AND EXISTS(
               SELECT 1
               FROM memory_creation_proposals p
               JOIN operator_review_decisions d ON d.review_id=p.review_id
               WHERE p.result_memory_id=memories.id
                 AND json_extract(d.effect_json,'$.memory_created')=1
             )"""
    )
    conn.execute(
        """UPDATE memories SET context_mode='standalone',scope_json='{}',entities_json='[]',
             preconditions_json='{}',applicable_systems_json='[]',applicable_versions_json='[]',
             metadata_completeness=1.0
           WHERE context_mode IS NULL OR context_mode NOT IN ('standalone','context_dependent')"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_context_mode ON memories(context_mode,state)")
    task_label_columns = {row[1] for row in conn.execute("PRAGMA table_info(task_outcome_labels)")}
    if "prior_outcome" not in task_label_columns:
        conn.execute(
            "ALTER TABLE task_outcome_labels ADD COLUMN prior_outcome TEXT NOT NULL DEFAULT 'used'"
        )
    if "reversed_by" not in task_label_columns:
        conn.execute("ALTER TABLE task_outcome_labels ADD COLUMN reversed_by TEXT")
    recall_columns = {row[1] for row in conn.execute("PRAGMA table_info(recall_runs)")}
    if "task_id" not in recall_columns:
        conn.execute("ALTER TABLE recall_runs ADD COLUMN task_id TEXT")
    if "stage_ms_json" not in recall_columns:
        conn.execute(
            "ALTER TABLE recall_runs ADD COLUMN stage_ms_json TEXT NOT NULL DEFAULT '{}'"
        )
    if "rendered_count" not in recall_columns:
        conn.execute(
            "ALTER TABLE recall_runs ADD COLUMN rendered_count INTEGER NOT NULL DEFAULT 0"
        )
    if "withheld_count" not in recall_columns:
        conn.execute(
            "ALTER TABLE recall_runs ADD COLUMN withheld_count INTEGER NOT NULL DEFAULT 0"
        )
    if "rendered_tokens" not in recall_columns:
        conn.execute(
            "ALTER TABLE recall_runs ADD COLUMN rendered_tokens INTEGER NOT NULL DEFAULT 0"
        )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_recall_runs_task ON recall_runs(task_id)")
    budget_columns = {row[1] for row in conn.execute("PRAGMA table_info(recall_budget_observations)")}
    if "rendered_count" not in budget_columns:
        conn.execute(
            "ALTER TABLE recall_budget_observations ADD COLUMN rendered_count INTEGER NOT NULL DEFAULT 0"
        )
    if "rendered_tokens" not in budget_columns:
        conn.execute(
            "ALTER TABLE recall_budget_observations ADD COLUMN rendered_tokens INTEGER NOT NULL DEFAULT 0"
        )
    trace_columns = {row[1] for row in conn.execute("PRAGMA table_info(memory_traces)")}
    if "retrieval_context_json" not in trace_columns:
        conn.execute(
            "ALTER TABLE memory_traces ADD COLUMN retrieval_context_json TEXT NOT NULL DEFAULT '{}'"
        )
    write_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(memory_write_decisions)")
    }
    if "source_type" not in write_columns:
        conn.execute(
            "ALTER TABLE memory_write_decisions ADD COLUMN source_type TEXT NOT NULL DEFAULT 'conversation'"
        )
    if "quality_flags_json" not in write_columns:
        conn.execute(
            "ALTER TABLE memory_write_decisions ADD COLUMN quality_flags_json TEXT NOT NULL DEFAULT '[]'"
        )
    proposal_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(memory_creation_proposals)")
    }
    proposal_additions = {
        "positive_feedback_count": "INTEGER NOT NULL DEFAULT 0",
        "strong_feedback_count": "INTEGER NOT NULL DEFAULT 0",
        "last_feedback_at": "TEXT",
        "review_started_at": "TEXT",
    }
    for name, declaration in proposal_additions.items():
        if name not in proposal_columns:
            conn.execute(
                f"ALTER TABLE memory_creation_proposals ADD COLUMN {name} {declaration}"
            )
    review_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(operator_review_decisions)")
    }
    if "decision_scope" not in review_columns:
        conn.execute(
            "ALTER TABLE operator_review_decisions "
            "ADD COLUMN decision_scope TEXT NOT NULL DEFAULT 'item_only'"
        )
        # Decisions made before explicit reach controls were presented as
        # training evidence. Preserve that meaning during migration while
        # defaulting every new dashboard review to a one-off action.
        conn.execute(
            "UPDATE operator_review_decisions SET decision_scope='policy_evidence'"
        )
    # Schema 26: Remove FK constraints from operator_review_decisions so that
    # sleep-proposal reviews can be stored alongside creation-proposal reviews.
    # SQLite does not support ALTER TABLE DROP CONSTRAINT, so we recreate the table.
    if "proposal_id TEXT REFERENCES" in (
        conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='operator_review_decisions'"
        ).fetchone() or [""]
    )[0]:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.executescript(
            """CREATE TABLE IF NOT EXISTS operator_review_decisions_v26 (
                review_id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL,
                item_key TEXT NOT NULL,
                proposal_id TEXT,
                src_id TEXT,
                dst_id TEXT,
                action TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                reason_text TEXT,
                prior_json TEXT NOT NULL DEFAULT '{}',
                effect_json TEXT NOT NULL DEFAULT '{}',
                learning_signal_json TEXT NOT NULL DEFAULT '{}',
                decision_scope TEXT NOT NULL DEFAULT 'item_only'
                  CHECK(decision_scope IN ('item_only','exact_duplicates','policy_evidence')),
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reversed_at TEXT
            );
            INSERT INTO operator_review_decisions_v26
                SELECT * FROM operator_review_decisions;
            DROP TABLE operator_review_decisions;
            ALTER TABLE operator_review_decisions_v26 RENAME TO operator_review_decisions;
            CREATE INDEX IF NOT EXISTS idx_operator_review_created
              ON operator_review_decisions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_operator_review_item
              ON operator_review_decisions(item_type,item_key,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_operator_review_proposal
              ON operator_review_decisions(proposal_id, reversed_at);"""
        )
        conn.execute("PRAGMA foreign_keys=ON")


def _index_features_tx(conn: sqlite3.Connection, memory_id: str, content: str) -> None:
    old_features = [
        str(row["feature"])
        for row in conn.execute("SELECT feature FROM memory_features WHERE memory_id=?", (memory_id,)).fetchall()
    ]
    if old_features:
        conn.executemany(
            "UPDATE feature_stats SET document_frequency=MAX(0,document_frequency-1) WHERE feature=?",
            [(feature,) for feature in old_features],
        )
        conn.execute("DELETE FROM feature_stats WHERE document_frequency<=0")
    conn.execute("DELETE FROM memory_features WHERE memory_id=?", (memory_id,))
    indexed = list(semantic_features(content).items())
    conn.executemany(
        "INSERT INTO memory_features(memory_id,feature,weight) VALUES(?,?,?)",
        [(memory_id, feature, weight) for feature, weight in indexed],
    )
    conn.executemany(
        """INSERT INTO feature_stats(feature,document_frequency) VALUES(?,1)
           ON CONFLICT(feature) DO UPDATE SET document_frequency=document_frequency+1""",
        [(feature,) for feature, _weight in indexed],
    )


def _backfill_memory_features(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """SELECT m.id,m.content FROM memories m
           WHERE NOT EXISTS(SELECT 1 FROM memory_features f WHERE f.memory_id=m.id)"""
    ).fetchall()
    for row in rows:
        _index_features_tx(conn, str(row["id"]), str(row["content"]))


def _index_context_terms_tx(
    conn: sqlite3.Connection,
    memory_id: str,
    *,
    context_mode: str,
    scope: dict[str, Any],
    entities: Sequence[str],
    preconditions: dict[str, Any],
    applicable_systems: Sequence[str],
    applicable_versions: Sequence[str],
) -> None:
    conn.execute("DELETE FROM memory_context_terms WHERE memory_id=?", (memory_id,))
    terms: set[tuple[str, str, str]] = {
        ("mode", "", normalize_text(context_mode).casefold())
    }
    terms.update(
        ("scope", normalize_text(str(key)).casefold(), normalize_text(str(value)).casefold())
        for key, value in scope.items()
        if normalize_text(str(key)) and normalize_text(str(value))
    )
    terms.update(
        (
            "precondition",
            normalize_text(str(key)).casefold(),
            normalize_text(str(value)).casefold(),
        )
        for key, value in preconditions.items()
        if normalize_text(str(key)) and normalize_text(str(value))
    )
    for term_type, values in (
        ("entity", entities),
        ("system", applicable_systems),
        ("version", applicable_versions),
    ):
        terms.update(
            (term_type, "", normalize_text(str(value)).casefold())
            for value in values
            if normalize_text(str(value))
        )
    conn.executemany(
        """INSERT INTO memory_context_terms(memory_id,term_type,term_key,term_value)
           VALUES(?,?,?,?)""",
        [(memory_id, term_type, key, value) for term_type, key, value in sorted(terms)],
    )


def _rebuild_feature_stats(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM feature_stats")
    conn.execute(
        """INSERT INTO feature_stats(feature,document_frequency)
           SELECT feature,COUNT(*) FROM memory_features GROUP BY feature"""
    )


__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_SQL",
    "_backfill_memory_features",
    "_create_schema",
    "_index_context_terms_tx",
    "_index_features_tx",
    "_migrate_columns",
    "_rebuild_feature_stats",
]
