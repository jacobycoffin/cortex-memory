# Cortex validation

## Automated suite

Run from the repository root:

```bash
python3 -m unittest discover -v
python3 scripts/benchmark.py
python3 scripts/benchmark_compare.py --sizes 100,500,2000 --queries 200
python3 scripts/benchmark_adaptive.py --size 500 --memory-queries 30
python3 scripts/benchmark_cache.py --size 2000 --repetitions 80
PYTHON_BIN=python3 bash scripts/smoke_install_upgrade.sh
```

GitHub Actions runs the unit and install/upgrade suites on Python 3.10 through 3.14, then checks Python and shell syntax, public-file privacy, SVG validity, and whitespace. For the full benchmark methodology, paired live-model runner, metric definitions, and public-claim guardrails, see [BENCHMARKING.md](BENCHMARKING.md). For private real-history retrieval and paired tool-call measurement, see [EVALUATION.md](EVALUATION.md).

The suite currently covers:

- add, exact deduplication, search, and outcome feedback;
- version-preserving user correction;
- graph-expanded recall;
- shadow versus applied lifecycle maintenance;
- secret redaction and memory-injection quarantine;
- provider recall, explanation, auto-capture, success feedback, and safe forget;
- Cortex-first system-prompt behavior plus the harness-neutral before-turn recall and after-turn evidence-resolution contract;
- Review Copilot clarification/recommendation validation, explicit broad-scope cues, confirmation binding, non-recallable audit storage, and provider transparency;
- prompt-injection memories excluded from recall;
- time-separated claims not treated as contradictions;
- overlapping structured conflicts linked without deleting either claim;
- dependency invalidation and dry-run repair;
- protected prospective memories excluded from pruning;
- repeatedly harmful memory losing to validated memory despite high retrieval count;
- archived-memory restoration;
- repeated tool successes becoming procedural guidance only after reinforcement;
- repeated tool failures becoming task-scoped warnings;
- selected-but-unused memories receiving no positive reinforcement;
- v1 prototype databases migrating to the structured evidence schema without losing memories;
- SQLite FTS5 BM25 relevance preserving the correct best-to-worst order;
- repeated live-run aggregation accepting independent frozen query seeds;
- attention-gate abstention and task-sensitive recall plans;
- transparent semantic-feature paraphrase recall;
- current versus historical temporal retrieval;
- structured attribution without vague conceptual over-credit;
- JSON tool-result failure classification;
- repeated multi-step workflow reinforcement;
- reversible consolidation and pruning-regret restoration;
- schema 4 migration and semantic-feature backfill;
- schema 5 migration without memory loss;
- outcome-driven recall budgets that ignore pending evidence and stay within configured caps;
- cache hits preserving per-turn injection and usage evidence;
- cross-connection cache invalidation after material correction;
- private evaluation snapshots that include live WAL data but omit private fields from reports;
- paired tool-call aggregation and fixture-only live-provider behavior;
- isolated clean installation, upgrade backup, database persistence, and installed-provider import.
- schema 6 Sleep migration, independent-witness replay, shadow idempotency, reversible apply/undo, bounded reflection, and provider-outage isolation.
- schema 11 migration, candidate-level score/rejection diagnostics, task influence ratings, create/update/ignore storage decisions, and valid append-only JSONL export.
- schema 12/13 migration, cross-project isolation, required-context gates, context-independent candidate recall, storage preflight, context-specific usefulness and reversal, context-repair Sleep proposals, source-cited summary candidates, and combined quality reporting.
- schema 15 Review Inbox coverage: decision-ready proposal snapshots, explained connection approval, typed operator-learning signals, reversible denial, and tombstone/restore version history.
- schema 16 operator-policy coverage: review compilation, support and consistency gates, replay, shadow observations, scoped promotion, live retrieval adjustment, version audit, and rollback.
- schema 17 decision-reach coverage: one-off exclusion from policy training, exact-duplicate discovery and reversible multi-memory action, explicit Teach Kaya compilation, counters, audit history, and migration defaults.
- schema 18 Memory Refinery coverage: deterministic role classification (code, tables, configuration, diagrams, documents → reference; explicit user statements canonical; episodes events; unsupported inferences claims), presentation fidelity (negation, anchors, and uncertainty preserved; no invented summaries; no paths in reference titles), idempotent rebuilds with source-change invalidation, legacy-database backfill without touching state/content/IDs, vault reindex idempotency with stable IDs, item-only/exact-duplicate/Teach Kaya reach on clarity actions, editable rewrite/split previews with preserved source dependencies, optional reasons, undo that never overwrites later changes, aggregate-only privacy in summaries and reports, Stage 1 byte-identical retrieval equivalence, and mutation-free shadow role-tier comparisons.
- schema 19 connection-training coverage: stable source-and-kind pattern grouping, readable connection review metadata, typed explained edge creation and undo, default pattern-evidence reach, independent-witness policy compilation, pending weak-proposal filtering on promotion, and proposal restoration on rollback.

## Current compatibility check

The provider and built-in-memory baseline were checked against the upstream Hermes implementation at commit `bd740f203b44237dbc5c27a2de4d86ef32af4dde`. The check initializes the provider, exercises turn sync, validates the exposed tool schema, runs an audit, and shuts down cleanly.

## Manual acceptance test

Keep `pruning_mode: shadow`.

1. Install and activate Cortex through `hermes memory setup`.
2. Start a new Hermes process.
3. Tell Hermes an explicit durable decision.
4. Start another session and ask a differently phrased retrieval question.
5. Ask Hermes to explain the recalled memory and its counters.
6. Correct the memory and verify the old version remains in `explain`.
7. Archive and restore the memory.
8. Run the same multi-step tool-backed workflow on two distinct tasks and inspect Tool notes and Cognition.
9. Inspect `cortex recall-stats` and verify social/self-contained turns can record zero context.
10. Resolve at least eight comparable recall outcomes, then inspect Cognition's Context budget learning panel; pending retrievals alone must not move the budget.
11. Preview consolidation, apply it only on a backup test database, and undo the run.
12. Run `cortex audit` and verify `ok: true`.
13. Run `cortex sleep --mode shadow --reflection-token-budget 0`, verify no state/edge changes, and inspect the Sleep section in Cognition.
14. Open **Train Kaya** and verify the review initially asks only **What should happen?**. Select an action, confirm reach and optional action-specific reasons appear, and submit successfully with no suggested reason selected. Verify reach defaults to **Only this memory/review**. When an eligible duplicate exists, confirm its exact count is shown before selecting **All exact copies**. Then mark five consistent matching reviews **Teach Kaya from this** and verify only those five produce a proposed standard with an explained selector and bounded core change.
15. Filter **Train Kaya → Connections**. Choose a pattern, verify the A/B titles and summaries are readable, raw stored text is collapsed, witness counts and shared signals are visible, and no reason taxonomy is required. Approve a typed relationship and verify the exact edge opens highlighted on the memory map with the operator explanation. Return to reviews, deny five matching co-occurrence-only pairs, run replay plus three shadow reviews, approve the scoped standard, verify matching weak pending proposals leave the inbox, then roll it back and verify eligible proposals return.
15. Run its evidence replay, start shadow observation, add three new matching reviews, and confirm activation remains unavailable until the shadow gate passes.
16. Approve the scoped version, verify it appears under Active policy versions, then roll it back and confirm the version becomes inactive without deleting its evidence.

## Metrics for the shadow trial

- retrieval precision@6 and recall@6;
- percentage of selected memories actually used;
- helpful versus harmful recalled memories;
- stale and contradicted memory retrieval rate;
- unsupported inference count;
- average and p95 retrieval latency;
- injected tokens per turn;
- tool selection success by task type;
- lifecycle candidate count and later pruning regret.

## Public benchmark rule

`scripts/benchmark.py` is a local p95 performance gate and `scripts/benchmark_compare.py` is an offline capacity/retrieval evaluation. `scripts/benchmark_cache.py` measures only an identical repeated prefetch inside the cache lifetime; it does not establish a production hit rate or varied-query speed. None of these measures model inference. Only results from `scripts/benchmark_e2e.py` may be described as TTFT or total-response speed, and those results must name the exact model, provider class, corpus, paired question count, and uncertainty interval.
