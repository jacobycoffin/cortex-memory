# Cortex live paired benchmark

Model: `tencent/hy3:free` · Cortex mode: `additive` · corpus: 500 memories · paired questions: 30

| Condition | Accuracy | Answer available | p50 memory prep | p50 agent TTFT | p95 agent TTFT | p50 agent total | Prompt tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hermes built-in | 10.0% | 10.0% | 0.0 ms | 1321.4 ms | 1459.1 ms | 1463.8 ms | 647 |
| Cortex | 96.7% | 100.0% | 9.0 ms | 1366.2 ms | 1808.9 ms | 1541.0 ms | 979 |

Paired median Cortex-minus-default agent TTFT (memory + model): **+78.4 ms**. Mean bootstrap 95% CI: +20.3 to +184.2 ms.

Paired median Cortex-minus-default agent total latency: **+102.4 ms**. Mean bootstrap 95% CI: +52.7 to +204.4 ms.

Model-only paired median TTFT difference: **+68.1 ms**. This excludes memory preparation and must not be described as whole-agent speed.

Negative latency differences favor Cortex. Report accuracy and latency together; a fast answer with the needed memory absent is not a successful memory-system result.
