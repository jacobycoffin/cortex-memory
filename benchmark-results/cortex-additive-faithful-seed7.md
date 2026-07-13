# Cortex live paired benchmark

Model: `tencent/hy3:free` · Cortex mode: `additive` · corpus: 500 memories · paired questions: 30

| Condition | Accuracy | Answer available | p50 memory prep | p50 agent TTFT | p95 agent TTFT | p50 agent total | Prompt tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hermes built-in | 3.3% | 3.3% | 0.0 ms | 1355.6 ms | 1571.7 ms | 1487.8 ms | 639 |
| Cortex | 93.3% | 100.0% | 12.7 ms | 1312.5 ms | 1569.4 ms | 1492.5 ms | 966 |

Paired median Cortex-minus-default agent TTFT (memory + model): **-39.2 ms**. Mean bootstrap 95% CI: -78.9 to +40.3 ms.

Paired median Cortex-minus-default agent total latency: **-8.3 ms**. Mean bootstrap 95% CI: -23.2 to +1821.2 ms.

Model-only paired median TTFT difference: **-53.2 ms**. This excludes memory preparation and must not be described as whole-agent speed.

Negative latency differences favor Cortex. Report accuracy and latency together; a fast answer with the needed memory absent is not a successful memory-system result.
