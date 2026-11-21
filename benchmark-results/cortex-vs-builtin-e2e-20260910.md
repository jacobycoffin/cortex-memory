# Cortex live paired benchmark

Model: `deepseek-chat` · Cortex mode: `additive` · recall: `adaptive_compact` · corpus: 500 memories · paired questions: 30

| Condition | Accuracy | Answer available | p50 memory prep | p50 agent TTFT | p95 agent TTFT | p50 agent total | Prompt tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hermes built-in | 3.3% | 3.3% | 0.0 ms | 845.0 ms | 1280.8 ms | 919.0 ms | 605 |
| Cortex | 90.0% | 96.7% | 23.9 ms | 841.3 ms | 1007.4 ms | 935.1 ms | 908 |

Paired median Cortex-minus-default agent TTFT (memory + model): **-42.1 ms**. Mean bootstrap 95% CI: -132.4 to +24.9 ms.

Paired median Cortex-minus-default agent total latency: **-23.0 ms**. Mean bootstrap 95% CI: -131.9 to +33.3 ms.

Model-only paired median TTFT difference: **-68.9 ms**. This excludes memory preparation and must not be described as whole-agent speed.

Negative latency differences favor Cortex. Report accuracy and latency together; a fast answer with the needed memory absent is not a successful memory-system result.
