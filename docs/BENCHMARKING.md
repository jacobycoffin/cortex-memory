# Benchmarking Cortex fairly

The benchmark is designed to answer four different questions without blending them into one marketing number:

1. Can the memory layer retrieve the right fact?
2. How much memory context does it place in the model prompt?
3. How much query-time overhead does the memory layer add?
4. Does the complete agent answer more accurately or faster with the same model?

The first three can be measured locally without an LLM. The fourth requires paired live-model requests. A local retrieval result by itself is not evidence of faster model inference.

## The actual Hermes baseline

Hermes built-in memory is not a vector database or a file search on every turn. It loads a frozen `MEMORY.md` and `USER.md` snapshot when the agent starts and places the enabled blocks in the system prompt. On the VPS code inspected July 13, 2026 (`bd740f203b44237dbc5c27a2de4d86ef32af4dde`), the defaults are:

- `MEMORY.md`: 2,200 characters;
- `USER.md`: 1,375 characters;
- writes that would exceed a limit are rejected until the agent consolidates or removes content;
- the frozen snapshot is reused during the session to preserve prefix caching;
- Cortex is additive to those built-in blocks in the current Hermes integration.

That makes built-in memory a strong small-context baseline with a hard capacity tradeoff. The appropriate comparison is bounded curated context versus indexed recall at scale—not SQLite versus rereading a giant Markdown file.

## Test 1: deterministic capacity and retrieval

Run:

```bash
python3 scripts/benchmark_compare.py --sizes 100,500,2000 --queries 200
```

The runner creates equally important synthetic operating facts with exact answer labels. Queries rotate between exact and mildly paraphrased wording. Both conditions receive facts in the same deterministic observed order.

Hermes built-in memory packs facts up to the configured 2,200-character `MEMORY.md` limit. Cortex indexes the full corpus and returns up to six memories within its approximate 700-token recall budget. Index construction is measured separately and excluded from query latency. One warm-up query is excluded.

Metrics:

| Metric | Meaning |
| --- | --- |
| Answer coverage | Fraction of questions whose labeled answer is present in the supplied memory context. |
| Recall@6 | Fraction whose relevant Cortex memory appears in the top six. |
| Precision@6 | Relevant results divided by returned results, averaged across questions. |
| MRR | Mean reciprocal rank; `1.0` means the relevant result was always first. |
| p50/p95 query latency | Median and tail retrieval overhead, excluding indexing and model inference. |
| Approximate context tokens | `ceil(characters / 4)` for portable comparison; not provider billing tokens. |

The checked-in 0.2 result is saved as a [human-readable report](../benchmark-results/cortex-v020-retrieval.md) with its [raw JSON](../benchmark-results/cortex-v020-retrieval.json). The fixture deliberately gives each project eight competing attributes, so matching the project alone is insufficient and paraphrase cases are harder for FTS-only retrieval. Cortex achieved 99.0%, 98.5%, and 99.5% recall@6 as the corpus grew, with MRR between 0.947 and 0.985, 12.0–34.0 ms p95 retrieval, and roughly 173 median context tokens. The built-in 2,200-character snapshot held 18 synthetic facts and covered 18.0%, 3.0%, and 1.5% of the sampled questions. These results are a capacity/retrieval finding, not yet an inference-speed finding. An earlier [comparison run](../benchmark-results/cortex-compare-20260713T123444Z.md) is included as a cross-check.

### Dashboard standard suite

The authenticated **Run benchmark** control in Insights runs a fixed, bounded version of this test directly on the dashboard host. Version 1 uses 100, 500, and 2,000 synthetic memories, 80 labeled queries per scale, seed 7, recall@6, and an approximate 700-token retrieval budget. It uses an isolated temporary database, never reads production memories, makes no provider requests, and permits only one run at a time.

The dashboard records the raw local result and four headline measures from the 2,000-memory scale:

| Measure | Dashboard role |
| --- | --- |
| Recall@6 | Primary retrieval-coverage outcome; target at least 98%. |
| MRR | Ranking-quality driver; target at least 0.95. |
| p95 retrieval latency | Local speed guardrail; target at most 50 ms on the benchmark host. |
| Approximate context tokens | Efficiency guardrail; kept outside the score so an empty result cannot look good. |

The version 1 overall score is `85% × retrieval quality + 15% × speed`. Retrieval quality is `65% × recall@6 + 35% × MRR`. Speed receives 100 points at 50 ms p95 or faster and otherwise receives `100 × 50 / measured p95`. The formula is deliberately quality-heavy: making retrieval faster cannot compensate for missing the correct memory. A score is comparable only to the same suite version under similar host load.

One run is a baseline, not a trend. The dashboard plots individual points immediately but waits for eight compatible runs before connecting them as a trend line. The recorded environment, exact components, and plain-language improvement guidance remain visible for diagnosis.

During development, this test found an inverted relevance transform: SQLite FTS5 intentionally makes better BM25 matches numerically lower (usually more negative), while the prototype's transform rewarded values closest to zero. It later caught a 0.2 regression where generic semantic features outweighed exact project identifiers. The regression suite and rarity-weighted hybrid ranker now protect both cases. See the [official SQLite FTS5 documentation](https://www.sqlite.org/fts5.html#the_bm25_function).

## Test 2: adaptive prompt-preparation ablation

Run:

```bash
python3 scripts/benchmark_adaptive.py --size 500 --memory-queries 30
```

This compares fixed verbose context, fixed compact context, and adaptive compact context on the same database and mixed workload. It reports approximate memory-context tokens, zero-context rate, labeled answer availability, and local preparation latency. It does not call a model.

The checked-in 0.2 run used 500 memories and 62 mixed queries. Adaptive compact recall used 21.7% fewer approximate memory-context tokens than fixed verbose recall while preserving the same labeled answer-context recall in that sample. This is a prompt-preparation result, not evidence of faster inference. See [the report](../benchmark-results/cortex-adaptive-v020.md) and [raw JSON](../benchmark-results/cortex-adaptive-v020.json).

## Test 3: paired end-to-end model latency

The live runner makes potentially billable requests to an OpenAI-compatible Chat Completions endpoint. Use the same endpoint, model, temperature, max output, corpus, and question for each condition. Condition order is randomized within the paired run.

```bash
export CORTEX_BENCH_BASE_URL="https://provider.example/v1"
export CORTEX_BENCH_MODEL="the-exact-model-id"
export CORTEX_BENCH_API_KEY="..."

python3 scripts/benchmark_e2e.py \
  --size 500 \
  --queries 50
```

The default `--cortex-mode additive` matches the current Hermes integration: built-in `MEMORY.md` context remains in the system prompt, Cortex adds its provider system guidance, and recalled evidence is appended after the original user question inside Hermes's authoritative `<memory-context>` fence. This is the required mode for claims about the deployed VPS. `--cortex-mode replacement` is a separately labeled optimization experiment for a configuration where built-in prompt injection has been disabled after its contents are safely migrated and verified in Cortex. Do not mix or aggregate the two modes.

The live runner now uses 0.2 adaptive compact recall by default. Pass `--fixed-recall` only for a clearly labeled ablation.

The runner defaults to `reasoning.effort=none` because labeled recall is a simple lookup task and long hidden reasoning can consume the output budget or dominate latency. If the selected model requires reasoning, pass `--reasoning-effort minimal` (or another level) and report that setting. Keep it identical in both conditions.

The script records:

- exact-answer accuracy;
- whether the answer was available in the condition's context;
- model-only time to first token and total response latency;
- memory preparation time;
- whole-agent TTFT and total latency (`memory preparation + model request`);
- provider-reported prompt/completion tokens when available;
- paired Cortex-minus-default latency differences;
- a bootstrap 95% confidence interval for the paired mean difference.

Negative paired latency differences favor Cortex. Use whole-agent timing for speed claims; model-only timing is a diagnostic. Report accuracy and latency together. A fast `UNKNOWN` response caused by a missing memory is not a successful memory-system result.

For a public claim, use at least 30 paired questions; 50–100 is better. Repeat the complete run at least three times at different times of day because network and provider load can dominate small prompt differences. Publish every run or a preregistered aggregation, not only the best run.

Aggregate repeated schema-v2 runs with:

```bash
python3 scripts/benchmark_aggregate.py \
  benchmark-results/run1.json \
  benchmark-results/run2.json \
  benchmark-results/run3.json \
  --output benchmark-results/aggregate.json
```

The aggregate reports every run, pooled medians, and a hierarchical bootstrap interval that resamples both runs and paired questions.

### July 13, 2026 VPS result

The faithful additive test used `tencent/hy3:free` through OpenRouter, a 500-memory synthetic corpus, three independently seeded runs, and 30 paired questions per run (90 pairs / 180 model requests total). It retained Hermes's built-in memory in both conditions and changed only whether Cortex guidance and retrieved evidence were available.

| Condition | Accuracy | Answer available | Median prompt tokens | p50 whole-agent TTFT | p95 whole-agent TTFT | p50 whole-agent total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hermes built-in bounded snapshot | 6.7% | 6.7% | 641 | 1,337.3 ms | 1,608.3 ms | 1,487.2 ms |
| Built-in + Cortex | 96.7% | 100.0% | 973 | 1,338.7 ms | 1,583.2 ms | 1,524.4 ms |

Cortex's own memory preparation took 11.7 ms at p50 and 15.0 ms at p95. The pooled paired median Cortex-minus-default whole-agent TTFT was -7.2 ms, with a hierarchical bootstrap median 95% interval of -53.0 to +76.7 ms. The pooled paired median total-latency difference was +31.8 ms, interval -11.4 to +102.7 ms. Both intervals cross zero, so this run supports **no clear latency difference**, not a raw-speed improvement. Cortex used 51.8% more median prompt tokens to make the needed memory available and produced a 90-percentage-point answer-accuracy gain.

One Cortex request took 19.2 seconds at the provider. It remains in the raw data; robust median intervals are reported because endpoint latency is outlier-prone. See the [aggregate report](../benchmark-results/cortex-additive-faithful-aggregate.md), [aggregate JSON](../benchmark-results/cortex-additive-faithful-aggregate.json), and the three raw run files in `benchmark-results/`.

## Test 4: real-memory retrieval quality

Synthetic facts make the benchmark reproducible but easy to label. A real evaluation should be built from private history without publishing its content:

1. Sample 100 past questions or create questions from the vault before looking at Cortex results.
2. Have a human label the memory IDs or source notes that are genuinely relevant.
3. Split exact, paraphrased, temporal/correction, and no-answer questions.
4. Freeze the dataset and scoring rules.
5. Run built-in-only and Cortex conditions on isolated copies of the same history.
6. Publish aggregate metrics only; redact content and source paths.

Track recall@6, precision@6, MRR, unsupported-answer rate, stale/contradicted recall rate, approximate injected tokens, and p95 retrieval latency. A no-answer set is necessary: otherwise a system can look good by always returning something.

## Test 5: adaptation and self-healing

Run the correctness suite:

```bash
python3 -m unittest discover -v
```

It includes scenario tests for:

- high retrieval frequency failing to rescue a repeatedly harmful memory;
- correction preserving the old version;
- changed evidence marking dependent beliefs dirty;
- dependency repair running as a dry-run before mutation;
- prospective/protected memories surviving pruning;
- archive and restore remaining reversible;
- prompt-like memory being quarantined and excluded from recall;
- selected-but-unused memory receiving no helpful or success credit.

For a longitudinal trial, snapshot the database weekly and measure pruning regret: the percentage of cold/archived memories that become relevant during the next 30, 60, or 90 days. Keep maintenance in `shadow` mode until that regret rate is acceptably low.

## Test 6: tool-calling improvement

The unit suite proves the mechanism: Cortex stores task type, tool name, argument-key shape, and sanitized outcome; it waits for at least two attempts before emitting guidance; and guidance remains task-scoped. That is not yet evidence that an LLM chooses tools better.

Use a paired agent test with isolated profiles:

- 10 web-research tasks;
- 10 filesystem tasks;
- 10 shell/operations tasks;
- 10 calendar or messaging tasks;
- the same model, tool list, prompts, and starting state;
- two seeded prior successes or failures for Cortex's training phase;
- novel but same-category tasks for evaluation.

Measure first-tool accuracy, attempts before success, tool-error rate, task completion rate, total latency, and tokens. Also include cross-category negative controls: web-search experience must not alter a filesystem recommendation. Human reviewers should score task completion while blinded to the condition.

## Claim checklist before posting

- Name the baseline precisely: `Hermes built-in bounded snapshot`, not “no memory.”
- State whether Cortex was `additive` (current deployment) or `replacement` (built-in prompt injection disabled).
- State corpus size, question count, seed, hardware, date, exact model, and endpoint class.
- Separate indexing time, retrieval latency, TTFT, and total latency.
- Report p50 and p95, not only an average.
- Report accuracy beside speed.
- Say when tokens are approximate.
- Publish the script and raw aggregate JSON.
- Call synthetic evidence synthetic.
- Avoid “brain-like,” “self-healing,” or “faster inference” as a proven claim unless the corresponding test supports it.
