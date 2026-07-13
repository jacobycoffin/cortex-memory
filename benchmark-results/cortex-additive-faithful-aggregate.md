# Cortex repeated live benchmark

Model: `tencent/hy3:free` · Cortex mode: `additive` · runs: 3 · 30 paired questions/run · 90 total pairs · corpus: 500 memories

| Condition | Accuracy | Answer available | Median prompt tokens | p50 agent TTFT | p95 agent TTFT | p50 agent total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hermes built-in | 6.7% | 6.7% | 641.0 | 1337.3 ms | 1608.3 ms | 1487.2 ms |
| Cortex | 96.7% | 100.0% | 973.0 | 1338.7 ms | 1583.2 ms | 1524.4 ms |

Cortex used **51.8% more median prompt tokens**.

Pooled paired median agent TTFT difference: **-7.2 ms**. Hierarchical bootstrap median 95% CI: -53.0 to +76.7 ms.

Pooled paired median agent total-latency difference: **+31.8 ms**. Hierarchical bootstrap median 95% CI: -11.4 to +102.7 ms.

All requests are retained. Maximum observed total latency was 19238.5 ms for Cortex and 1975.3 ms for built-in memory; the median interval is reported because latency is outlier-prone. Mean intervals remain available in the raw JSON.

## Per-run paired medians

| Run | Cortex accuracy | Default accuracy | Agent TTFT delta | Agent total delta |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 93.3% | 3.3% | -39.2 ms | -8.3 ms |
| 2 | 96.7% | 10.0% | +78.4 ms | +102.4 ms |
| 3 | 100.0% | 6.7% | -35.3 ms | +9.7 ms |

Negative latency differences favor Cortex. The hierarchical bootstrap resamples runs and paired questions; with only three runs, treat the interval as a stability check rather than a universal provider-performance estimate.
