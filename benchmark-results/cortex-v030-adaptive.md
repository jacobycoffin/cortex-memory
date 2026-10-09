# Cortex adaptive prompt-preparation ablation

Corpus: 500 synthetic memories · 62 mixed queries.

| Condition | Approx total context tokens | Zero-context turns | Labeled answer in context | p50 prep | p95 prep |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_verbose | 16,844 | 16.1% | 100.0% | 46.47 ms | 68.23 ms |
| fixed_compact | 14,764 | 16.1% | 100.0% | 46.33 ms | 69.30 ms |
| adaptive_compact | 14,087 | 19.4% | 100.0% | 43.91 ms | 63.51 ms |

Adaptive compact context reduced approximate memory-context tokens by **16.4%** versus fixed verbose context in this mixed workload; labeled answer-context recall changed by **+0.0%**.

> Local prompt-preparation ablation only. Approximate context tokens are characters/4. This does not measure model inference, TTFT, total response latency, or task completion.
