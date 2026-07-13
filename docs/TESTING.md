# Cortex validation

## Automated suite

Run from the repository root:

```bash
python3 -m unittest discover -v
python3 scripts/benchmark.py
python3 scripts/benchmark_compare.py --sizes 100,500,2000 --queries 200
python3 scripts/benchmark_adaptive.py --size 500 --memory-queries 30
```

For the full benchmark methodology, paired live-model runner, metric definitions, and public-claim guardrails, see [BENCHMARKING.md](BENCHMARKING.md).

The suite currently covers:

- add, exact deduplication, search, and outcome feedback;
- version-preserving user correction;
- graph-expanded recall;
- shadow versus applied lifecycle maintenance;
- secret redaction and memory-injection quarantine;
- provider recall, explanation, auto-capture, success feedback, and safe forget;
- prompt-injection memories excluded from recall;
- time-separated claims not treated as contradictions;
- overlapping structured conflicts linked without deleting either claim;
- dependency invalidation and dry-run repair;
- protected prospective memories excluded from pruning;
- repeatedly harmful memory losing to validated memory despite high retrieval count;
- archived-memory restoration;
- repeated tool successes becoming procedural guidance only after reinforcement;
- repeated tool failures becoming task-scoped warnings;
- selected-but-unused memories receiving no positive reinforcement.
- v1 prototype databases migrating to the structured evidence schema without losing memories;
- SQLite FTS5 BM25 relevance preserving the correct best-to-worst order;
- repeated live-run aggregation accepting independent frozen query seeds.
- attention-gate abstention and task-sensitive recall plans;
- transparent semantic-feature paraphrase recall;
- current versus historical temporal retrieval;
- structured attribution without vague conceptual over-credit;
- JSON tool-result failure classification;
- repeated multi-step workflow reinforcement;
- reversible consolidation and pruning-regret restoration;
- schema 4 migration and semantic-feature backfill.

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
10. Preview consolidation, apply it only on a backup test database, and undo the run.
11. Run `cortex audit` and verify `ok: true`.

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

`scripts/benchmark.py` is a local p95 performance gate and `scripts/benchmark_compare.py` is an offline capacity/retrieval evaluation. Neither measures model inference. Only results from `scripts/benchmark_e2e.py` may be described as TTFT or total-response speed, and those results must name the exact model, provider class, corpus, paired question count, and uncertainty interval.
