# Cortex live paired benchmark

Model: `tencent/hy3:free` · Cortex mode: `additive` · corpus: 500 memories · paired questions: 30

| Condition | Accuracy | Answer available | p50 memory prep | p50 agent TTFT | p95 agent TTFT | p50 agent total | Prompt tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hermes built-in | 6.7% | 6.7% | 0.0 ms | 1338.1 ms | 1637.6 ms | 1497.8 ms | 641 |
| Cortex | 100.0% | 100.0% | 12.3 ms | 1312.9 ms | 1487.7 ms | 1528.7 ms | 969 |

Paired median Cortex-minus-default agent TTFT (memory + model): **-35.3 ms**. Mean bootstrap 95% CI: -105.5 to +24.4 ms.

Paired median Cortex-minus-default agent total latency: **+9.7 ms**. Mean bootstrap 95% CI: -59.0 to +88.8 ms.

Model-only paired median TTFT difference: **-46.9 ms**. This excludes memory preparation and must not be described as whole-agent speed.

Negative latency differences favor Cortex. Report accuracy and latency together; a fast answer with the needed memory absent is not a successful memory-system result.
