# Changelog

## 0.3.0-dev.0 — Unreleased

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
