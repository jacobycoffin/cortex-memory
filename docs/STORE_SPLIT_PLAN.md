# Store split plan — incremental extraction from `store.py`

`store.py` is ~17.6k lines / 231 `CortexStore` methods (measured 2026-09-09).
`CortexStore` stays the public facade forever: stages move code, never call
sites. No stage may change ranking, scoring weights, retrieval selection, or
any learning default.

## Responsibility clusters (approximate, by name family)

| # | Cluster | ~Methods | Extracts to |
|---|---------|----------|-------------|
| 1 | Migrations / schema (`_create_schema`, `_migrate_columns`, backfills) | 9 + table defs | `cortex_schema.py` |
| 2 | Memory records (CRUD, metadata, strand/quarantine/lifecycle) | 6+ | `cortex_records.py` |
| 3 | Recall sets (eligibility, active/trained transitions) | 9 | `cortex_recall_sets.py` |
| 4 | Recall runs + traces (runs, decisions, render reports, JSONL) | 30 | `cortex_traces.py` |
| 5 | Usage / feedback (resolve/apply/label, outcomes, access traces, strength) | 16 | `cortex_feedback.py` |
| 6 | Review / proposals (inbox, refinery, roles, presentations) | 26 | `cortex_review.py` |
| 7 | Policy / scoring (policies, weights, calibration, metacognition) | 17 | `cortex_policy.py` |
| 8 | Consolidation / pruning (Sleep apply paths stay transactional) | 13 | `cortex_consolidation.py` |
| 9 | Attention (topics, salience) | 4 | `cortex_attention.py` |
| 10 | Graph / edges (relations, neighborhoods, witnesses) | 12 | `cortex_graph.py` |
| 11 | Dashboard / exports (snapshots, quality, stats) | 11 | `cortex_insights.py` |
| 12 | Episodes / eval (episodes, private eval lifecycle) | 8 | `cortex_episodes.py` |

Plus module-level pure helpers (`_trace_json*`, `_normalize_*`, `_decode_*`,
retrieval-context keys) → `cortex_serializers.py`. That move is dependency-free
and is the recommended stage 0.

## Stage order (one commit each, smallest dependency-free first)

- Stage 0: `cortex_serializers.py` — pure helpers only. Zero behavior change;
  `store.py` re-imports them.
- Stage 1: `cortex_schema.py` — table defs + migrations. Prove with the
  old-DB migration tests (recall_runs render columns pattern).
- Stages 2–12 in table order above (records before sets before runs/traces
  before feedback; policy/consolidation last — they touch learning).

## Per-stage acceptance criteria (all must hold before merge)

1. `CortexStore` public method set is byte-identical (`dir()` diff empty).
2. Transaction boundaries unchanged: moved writers take the caller's
   connection/tx object, never open their own.
3. Schema compatibility: old-DB migration tests green; `SCHEMA_VERSION`
   bumps only when a stage changes DDL (with migration, never silent).
4. Audit history + undo behavior preserved (decision ledgers append-only;
   apply/undo round-trips covered by existing tests).
5. Full suite + `scripts/check_repository.py` green.
6. No ranking/selection change: retrieval + scoring diffs must be empty
   (`git diff --stat retrieval.py adaptive_weights.py` clean).

## Explicitly out of scope

- No scoring-weight, ranking, or retrieval-selection change rides along with
  a move (Astra task 5 constraint).
- No live-plugin deploy per stage; deploys stay surgical deltas per the
  divergence ledger (live-only v0 trace machinery must survive every stage).
