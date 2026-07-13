# Cortex adaptive prompt-preparation ablation

Corpus: 500 synthetic memories · 62 mixed queries.

| Condition | Approx total context tokens | Zero-context turns | Labeled answer in context | p50 prep | p95 prep |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_verbose | 11,972 | 16.1% | 93.3% | 16.82 ms | 40.98 ms |
| fixed_compact | 9,828 | 16.1% | 93.3% | 16.61 ms | 21.93 ms |
| adaptive_compact | 9,377 | 19.4% | 93.3% | 17.16 ms | 31.46 ms |

Adaptive compact context reduced approximate memory-context tokens by **21.7%** versus fixed verbose context in this mixed workload; labeled answer-context recall changed by **+0.0%**.

> Local prompt-preparation ablation only. Approximate context tokens are characters/4. This does not measure model inference, TTFT, total response latency, or task completion.
