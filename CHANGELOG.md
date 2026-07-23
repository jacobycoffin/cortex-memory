# Changelog

## 0.3.0-dev.1 — Unreleased

- advanced the database to schema 32 with outcome-aware reversible pruning and graph stranding, task-specific staged scoring profiles with bounded approval and precision rollback, lability-gated adaptive reconsolidation, and evidence-qualified schema abstraction; all scheduled model passes remain opt-in and proposal-only, while authenticated dashboard and CLI actions expose confirmation, feedback, dirty-state, regret, factory-reset, and undo evidence;
- advanced the database to schema 25 with evidence-earned named project/service neighborhoods, recoverable pre-schema-24 claim origins, separate origin/operator/automatic-approval provenance preserved across trained recall-set transitions, append-only privacy-safe learning experiences and creation-feedback evidence, and one inspectable decision ledger for admission, review, schema, recall-set, policy, and lifecycle changes;
- added a silent, explicitly opted-in five-minute automatic memory judge with fully bounded serialized requests, applicability metadata, and responses, redirect refusal, strict configuration/output validation, quarantine/contradiction/truncation guards, oldest-first backlog handling, atomic and exactly reversible duplicate promotion, archived-duplicate race safety, honest `AUTOMATIC_APPROVED` labeling, and next-turn reinforcement limited to unambiguous positive feedback; remote provider data egress is disabled by default and requires a separate installation consent flag checked against the effective preserved endpoint;
- made retrieval, retention, reconsolidation, and Sleep reinforcement depend on attributed use, helpful outcomes, and independent contexts instead of repeated retrieval or prompt injection, with selected-but-unused inhibition and relation-specific graph activation;
- added conservative prospective recall, stricter transient/status-noise capture filters, preview-confirmed clean-start recall sets, unified provider context budgeting, privacy-safe dataset export, and the human-memory-inspired cognitive baseline and harness lifecycle;
- upgraded Train Kaya, the memory drawer, Outcome Lab, and the memory map so operators can see why a candidate was noticed, where it would live, why a memory was or was not selected, why a neighborhood exists, and the evidence behind every visible connection;
- added schema 20 Review Copilot interpretations: an opt-in bounded LLM can translate an operator's natural-language connection reasoning into a validated action, relationship, and item-only or pattern-training scope; it may ask one clarifying question, never applies its own recommendation, and links the original explanation and structured preview to the later confirmed review without creating recallable memory;
- added a portable Cortex-first harness contract, machine-readable `harness-contract` manifest, and `CortexHarnessAdapter` reference lifecycle so any Python agent harness can run bounded recall before inference, resolve actually used evidence afterward, and reserve harness-native memory for a bootstrap pointer and temporary session scratch;
- strengthened the Hermes system prompt and tool description so Cortex is explicitly the primary durable store for user facts, preferences, decisions, corrections, and verified procedures rather than a secondary copy of Hermes's bounded built-in memory;
- documented the Cortex-primary Hermes runtime setting that disables Hermes's redundant periodic built-in-memory writer while preserving its curated files as bootstrap context;
- added schema 19 explainable connection training: connection reviews are grouped by memory-kind and source pattern, present readable A/B decision briefs with independent-witness and shared-signal evidence, and replace ambiguous approval reasons with typed relationship choices;
- made every approved connection return an edge receipt and appear as a highlighted, explained edge on every memory-map renderer, with a direct route back to the connection queue;
- made promoted stricter connection policies remove matching weak pending proposals from the inbox while preserving an audited `policy_proposal_effects` ledger; rollback restores proposals that remain eligible;
- added the schema 18 Memory Refinery: explicit `record_role` (canonical, reference, event, claim) with method/version provenance, one derived `memory_presentations` row per memory, and audited `memory_refinery_proposals`;
- added a versioned deterministic role classifier and presentation generator (no model calls): document extractions default to reference evidence, structural signals (code, tables, configuration, diagrams, command sequences, logs, raw JSON) file records as reference, episodes become events, unsupported inferences stay claims, and readability flags are preserved for review instead of silently discarding records;
- reorganized the Index into Readable memories (default), Reference evidence, Needs clarity, and All records, with full-aggregate counts, readable cards (title, statement, applicability, source, retention reason, use evidence), collapsed **View raw evidence**, and a plain-language detail drawer (What Kaya remembers / When this applies / Why it is retained / Source and evidence / Raw record / History and connections);
- added a Clarity category to Train Kaya with outcome-phrased actions — keep as readable memory, keep only as reference, rewrite clearly, split into separate memories, archive, trash — where rewrite and split show an editable deterministic preview, store operator text as explicit evidence linked to the untouched raw record, default to item-only reach, keep reasons optional, and remain undoable;
- added bounded authenticated refinery APIs (`summary`, `items`, `shadow`, `preview`, `action`, `undo`, `rebuild-presentations`) behind the existing session, same-origin, and `CORTEX_DASHBOARD_REVIEWS` gates, plus `refinery-report`, `refinery-summary`, and `refinery-shadow` CLI commands;
- defaulted new vault imports to reference evidence, required canonical clarity checks for automatic capture (flagged automatic writes stay reviewable claims), and kept vault reindex idempotent with stable memory IDs and refreshed presentations;
- added a shadow-only role-tier retrieval comparison (`role_tier_shadow_v1`): canonical unchanged, reference requires direct lexical/scope/entity/system/version or exact technical support, events respect temporal relevance, unsupported claims stay gated — Stage 1 live selection and ordering are byte-identical and activation stays behind paired evaluation;
- defaulted the memory map to canonical records with an explicit reference-evidence toggle and role counts kept separate from kind and lifecycle counts;
- rebranded the project as `cortex-memory`, with Hermes retained as the first harness adapter;
- added a framework-neutral `CortexMemory` and `RecallBatch` API plus pip packaging and a `cortex-memory` CLI;
- added a five-event integration contract for custom harnesses and package-install CI coverage;
- added Cortex Sleep, a bounded offline replay cycle that runs outside normal inference;
- added independent-witness association evidence, structured interference review, weak-edge downscaling, and lifecycle, consolidation, and dependency previews;
- added explicit reversible apply mode plus `sleep-undo`; default scheduling remains shadow-only;
- added an optional provider reflection stage with a separate per-run token ceiling, strict output validation, proposal-only writes, and zero-token default;
- added nightly low-priority systemd scheduling, schema 6 run/evidence/proposal/change records, and dashboard observability;
- expanded dashboard Sleep observability with schedule state, selectable run history, exact proposal and before/after change ledgers, reversals, and clearly non-causal post-run observations;
- reorganized Brain Health into recall integrity, memory quality, learning-loop, and Sleep-maintenance layers;
- added a daily Accuracy × capacity view that compares outcome-backed memory helpfulness, stored capacity, and average recall context with visible sample size and descriptive-only correlation;
- fixed Timeline totals and memory-type charts to use complete daily aggregates instead of the 1,000-row interactive-detail cap;
- added schema 7 metacognitive prediction records, conservative outcome-band calibration, shadow use/verify/abstain decisions, and a dedicated Trust monitor with Brier/ECE evidence;
- added schema 8 dashboard benchmark runs with an authenticated fixed-suite runner, transparent quality/speed score, persisted history graph, host environment record, and measured improvement guidance;
- added schema 9 audited task-outcome labels, private evaluation cases and aggregate paired-run ledgers, plus tool-guidance exposure follow-through;
- added an Outcome & Causality Lab that turns reversible positive task labels into a private fixed-versus-adaptive retrieval evaluation after eight cases;
- added a read-only evidence hierarchy, proposal-level Sleep hypotheses, observational tool/workflow evaluation, and a metacognition enforcement gate that keeps configured enforcement in shadow until calibration and selective-risk thresholds pass;
- added schema 10 controlled recall assignments, agent-task evaluation, randomized matched Sleep apply trials, a failure explorer, approval-gated cited summaries, prospective-memory state, and reconsolidation follow-through;
- added schema 11 task-level memory traces with candidate score components, explicit selection/rejection reasons, answer-use attribution, six-band outcome ratings, storage actions, and append-only JSONL export;
- added schema 12 standalone/context-dependent memory metadata, project/entity/scope/precondition/system/version applicability gates, and an independently maintained context-term candidate index;
- added schema 13 pre-storage durability/duplicate/contradiction/comprehensibility decisions plus reversible project/task-specific usefulness adaptation;
- added schema 14 evidence-bearing graph links, explainable-link migration, source-aware storage quality flags, and a dashboard hygiene queue;
- added schema 15 typed operator-review decisions and a unified, reversible Review Inbox for cleanup, connection, conflict, claim, and real-answer training;
- added schema 16 policy candidates, evidence replay, new-review shadow gates, explicit scoped/core promotion, active policy versions, and rollback;
- added schema 17 per-review reach controls: item-only decisions stay out of policy training, eligible exact duplicates can be handled together without generalizing, and only explicit Teach Kaya choices feed the compiler;
- simplified Train Kaya reviews into action first, reach second, and an optional action-specific reason; no taxonomy choice blocks a decision, and free-form or no-reason paths remain valid;
- added a guided Train Kaya dashboard path with five progress stages, next-action guidance, inspectable proposed standards, and active-policy monitoring;
- wired promoted operator policies into bounded automatic-write admission, Sleep connection witnesses, retrieval ranking, and lifecycle retention without allowing code self-modification or bypassing existing safety gates;
- stopped promoting raw tool executions and aggregate telemetry into recallable memory while preserving the dedicated tool execution, statistics, and workflow ledgers;
- made vault section revisions and file restoration reuse stable memory IDs with version history instead of producing repeated archived clones;
- tightened reversible hygiene rules so lack of retrieval never prunes a memory, substantive TODO/TBD notes are preserved, and only deterministic telemetry/placeholder/status noise is staged;
- expanded shadow Sleep with context-repair proposals and automatic source-cited summary candidates that remain outside recall until operator approval;
- added compact write-decision, context-feedback, and memory-quality CLI reports with explicit non-causal claim boundaries;
- added a Learning Lab with explicit daily accuracy, sparse-chart safeguards, condition confidence intervals, reversible treatment links, task-level labeling for no-memory controls, and evidence-first causal gates;
- added psychology/neuroscience design notes with primary-source references and explicit software/biology boundaries;
- added Sleep safety, idempotency, budget, outage, and reversibility tests.

## 0.3.0-dev.0

- added outcome-driven recall budgets that learn conservatively from resolved helpful, harmful, used, and ignored evidence;
- added a short-lived, mutation-revisioned retrieval cache that preserves per-turn evidence accounting;
- tightened deterministic abstention for social-only and self-contained arithmetic turns;
- added a private real-history retrieval evaluator and paired recorded/live-fixture tool-calling evaluator;
- added dashboard evidence for learned context-budget pressure;
- advanced the database to schema 5 with automatic, non-destructive migration;
- added Python 3.10–3.14 CI, repository privacy checks, and isolated clean-install/upgrade smoke tests;
- added the measured development roadmap and public benchmark-report template.

## 0.2.0 — 2026-07-13

- added attention-gated recall with five task-sensitive plans;
- added transparent semantic-feature retrieval and bounded personalized graph activation;
- added current and historical temporal retrieval;
- added structured evidence-use attribution;
- added repeated multi-step tool workflow learning;
- added adaptive retention, lifecycle events, pruning-regret detection, reversible consolidation, and undo;
- added schema 4 migration and feature backfill;
- added live recall/token/latency/lifecycle/workflow dashboard evidence;
- expanded research, quickstart, privacy, community, and benchmark documentation.

## 0.1.0 — 2026-07-12

- initial Hermes memory provider, FTS5 retrieval, utility scoring, provenance, correction history, graph links, tool outcome learning, vault indexing, dashboard, and reversible lifecycle states.
