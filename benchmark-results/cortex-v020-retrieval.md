# Cortex benchmark result

Run: `2026-07-13T14:30:41+00:00` · Python 3.14.2 · macOS-27.0-arm64-arm-64bit-Mach-O

## What this result supports

This synthetic run measures memory capacity, retrieval quality, retrieval overhead, and approximate context size. It does **not** measure LLM inference speed. Use the live paired benchmark before claiming faster time to first token or faster total responses.

| Corpus | Default stored | Default coverage | Cortex recall@6 | Cortex MRR | Cortex p95 retrieval | Default context | Cortex context |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 18 | 18.0% | 99.0% | 0.985 | 12.000 ms | 563 tokens | 173 tokens |
| 500 | 18 | 3.0% | 98.5% | 0.947 | 19.041 ms | 568 tokens | 173 tokens |
| 2,000 | 18 | 1.5% | 99.5% | 0.958 | 33.954 ms | 568 tokens | 173 tokens |

## Method in one paragraph

The benchmark creates equally important synthetic operating facts, asks labeled exact and mildly paraphrased questions, and uses the same observed order for both conditions. Hermes built-in memory packs facts into its configured 2,200-character `MEMORY.md` limit and exposes that frozen snapshot on every turn. Cortex indexes the full corpus and retrieves up to six memories within its approximate 700-token budget. Index construction is reported separately and excluded from query latency. One warm-up query is excluded.

## Required caveats

- Synthetic results do not prove the same lift on a personal vault or real conversations.
- Approximate tokens use `ceil(characters / 4)` and are not provider billing tokens.
- The built-in system is a deliberately small curated snapshot, not a failed search engine.
- Cortex remains additive to built-in memory in the current Hermes integration unless built-in prompt injection is disabled.
- Only the paired live-model test can support an inference-speed claim.
