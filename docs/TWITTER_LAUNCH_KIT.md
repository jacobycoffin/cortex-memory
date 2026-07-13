# Cortex Twitter / X launch kit

Use the measured version that matches the evidence you have. The July 13 live result below is the faithful additive configuration running on the Cortex VPS.

## Cortex 0.2 development update

> I’m open-sourcing Cortex 0.2, an adaptive memory layer for Hermes Agent.
>
> It now decides when recall is worth the prompt cost, combines FTS with transparent semantic features and bounded graph activation, learns repeated multi-step tool workflows, preserves corrections, consolidates duplicates reversibly, and tracks pruning regret.
>
> In a local 500-memory / 62-query mixed-workload ablation, adaptive compact recall used 21.7% fewer approximate memory-context tokens than fixed verbose recall with the same labeled answer availability. Local prep only—this is not an inference-speed claim.
>
> Psychology generated the hypotheses. Tests, raw data, and ablations decide whether the software works: https://github.com/jacobycoffin/hermes-cortex-memory

Pair this with the Cognition dashboard, which shows live recall modes, zero-context abstentions, estimated tokens, preparation latency, lifecycle repairs, pruning regret, and learned workflows.

## The headline that the evidence supports

> Cortex made long-term memory answers dramatically more accurate while keeping time-to-first-token effectively unchanged in this test. It did **not** prove faster inference.

That distinction is worth leaning into. A transparent result is more credible than forcing every metric to be a win.

The ready-to-upload result graphic is [cortex-benchmark-social.png](../benchmark-results/cortex-benchmark-social.png). Regenerate its editable SVG from the aggregate JSON with:

```bash
python3 scripts/benchmark_social_card.py \
  benchmark-results/cortex-additive-faithful-aggregate.json \
  --output benchmark-results/cortex-benchmark-social.svg
```

## Ready-to-post live result

> I built Cortex, a psychology-inspired adaptive memory layer for Hermes.
>
> In 3 paired VPS runs—90 questions against 500 synthetic memories, same `tencent/hy3:free` model—answer accuracy was 96.7% with Cortex vs 6.7% with Hermes's bounded built-in snapshot.
>
> Whole-agent p50 TTFT was effectively tied: 1,338.7 vs 1,337.3 ms. The paired TTFT difference was -7.2 ms, bootstrap median 95% CI -53.0 to +76.7 ms, so there is no clear speed difference.
>
> Cortex memory lookup itself took 11.7 ms p50 / 15.0 ms p95. The tradeoff was 51.8% more prompt tokens and a +31.8 ms paired median total latency for a 90-point accuracy gain.
>
> Psychology inspired the mechanisms; the benchmark tests the software. Code, raw runs, caveats, and method: [LINK]

## Short post

> Cortex vs Hermes built-in memory: 96.7% vs 6.7% answer accuracy across 90 paired questions / 500 memories. Whole-agent TTFT was effectively tied (1,339 vs 1,337 ms; CI crosses 0). Better long-term recall, not a proven inference speedup. Raw data + method: [LINK]

## A post you can publish from the current offline result

> I built Cortex, an adaptive local memory layer for Hermes.
>
> On a deterministic synthetic test on my Hermes VPS with 2,000 memories and 200 labeled questions, Cortex hit 99.5% recall@6 / 0.963 MRR with 19.6 ms p95 retrieval and ~173 tokens of recalled context.
>
> Hermes's built-in 2,200-character memory snapshot held 18 test facts and covered 1.5% of the sampled questions at that scale.
>
> Important caveat: this tests capacity + retrieval, not LLM inference speed. The paired live-model test is next. Host: 2 vCPU AMD EPYC 9354P, Linux x86_64, Python 3.12.3. Code, raw JSON, and methodology: [LINK]

That post is intentionally precise. It says what the current run supports and what it does not.

## Thread draft

**1/9**

> AI agents usually treat memory as either “stuff everything into the prompt” or “run semantic search.” I wanted a system that could also learn which memories are useful, preserve corrections, form associations, and age stale information safely. So I built Cortex for Hermes.

**2/9**

> The default Hermes memory is a good small-context baseline: a frozen curated snapshot capped at 2,200 chars for agent memory + 1,375 for user profile. Cortex adds a local SQLite/FTS5 index and recalls a small relevant set per question.

**3/9**

> Cortex does not treat “retrieved” as “true.” It separately tracks retrieval, injection, actual use, helpfulness, success, validation, harm, and correction. Frequency signals demand; outcomes signal utility.

**4/9**

> The psychology inspiration:
> • recency/frequency influence accessibility
> • retrieval can change later accessibility
> • semantic cues activate related concepts
> • working context is bounded
> • memories can be revised after recall
>
> These are engineering analogies, not a claim that Cortex simulates a brain.

**5/9**

> “Self-healing” means auditable repair: corrections preserve prior versions, changed evidence marks dependent beliefs dirty, contradictions remain visible, harmful memories lose rank, suspicious memories are quarantined, and pruning is reversible. Cortex never hard-deletes.

**6/9**

> First deterministic result (synthetic):
> • 2,000 memories / 200 questions
> • recall@6: 99.5%
> • MRR: 0.963
> • p95 retrieval: 19.6 ms
> • median recalled context: ~173 tokens
> • host: 2 vCPU Linux VPS
>
> Raw run: [JSON LINK]

**7/9**

> I caught a real ranking bug while building the benchmark: SQLite FTS5 makes better BM25 matches more negative, and my prototype transformed that score backward. The labeled tests exposed it immediately. This is why I’m publishing the harness, not just a screenshot of a good result.

**8/9**

> Then I ran 3 faithful additive model trials: 90 paired questions, 500 memories, same `tencent/hy3:free` model. Cortex accuracy was 96.7% vs 6.7%. The needed answer was present in 100% vs 6.7% of the memory contexts.

**9/9**

> Whole-agent TTFT was effectively tied: 1,338.7 vs 1,337.3 ms; paired median difference -7.2 ms, bootstrap median 95% CI -53.0 to +76.7 ms. Cortex used 51.8% more prompt tokens. The honest result: dramatically better recall, no demonstrated speedup yet. [METHOD + RAW DATA LINK]

## Live-model result template for future models

Only use this after `scripts/benchmark_e2e.py` produces the JSON:

> Cortex vs Hermes built-in memory, paired on [N] questions with [EXACT MODEL]:
>
> • answer accuracy: [CORTEX]% vs [DEFAULT]%
> • median prompt tokens: [CORTEX] vs [DEFAULT]
> • median whole-agent TTFT: [CORTEX] ms vs [DEFAULT] ms
> • p95 whole-agent TTFT: [CORTEX] ms vs [DEFAULT] ms
> • paired Cortex-minus-default whole-agent TTFT: [DELTA] ms, bootstrap 95% CI [LOW, HIGH]
>
> Corpus: [SIZE] synthetic memories. Date/hardware/provider: [DETAILS]. Raw data + method: [LINK]

If the confidence interval crosses zero, write “no clear TTFT difference in this run.” If Cortex is slower but more accurate, say exactly that.

## Short psychology explanation

> Cortex is psychology-inspired, not a digital brain. It borrows hypotheses from research on bounded working memory, recency/frequency, retrieval practice, associative activation, complementary episodic/semantic learning, and reconsolidation. In software, those become bounded recall, type-specific decay, outcome-weighted utility, graph links, episode/semantic stores, and versioned correction. The benchmark—not the analogy—determines whether those choices help an agent.

## Words to use carefully

| Phrase | Use it when |
| --- | --- |
| “Psychology-inspired” | The implementation mapping and research links are included. |
| “Self-healing” | You immediately define correction, dependency repair, quarantine, and reversible archival. |
| “Faster retrieval” | Cortex query latency is compared with another actual retrieval operation. Do not compare it with constant-time snapshot access. |
| “Smaller prompt” | Provider-reported prompt tokens or a clearly labeled approximation is lower. |
| “Faster inference” | Paired live-model TTFT or total latency supports it, with uncertainty reported. |
| “Better memory” | You name the metric: recall, accuracy, stale rate, pruning regret, or tool completion. |

Avoid “works like the brain,” “proven by neuroscience,” “remembers everything,” and “gets smarter forever.” None is supported by the current design.
