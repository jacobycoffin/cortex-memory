# Cortex 0.3 evaluation guide

These checkout-local runners answer two different questions without adding private evaluation data to the repository:

1. Does Cortex retrieve the operator-labeled memories from a real history?
2. Does adding Cortex context change paired tool-calling outcomes?

They deliberately produce machine-readable JSON. They do not upload a Cortex database, publish memory content, or turn an offline measurement into a model-performance claim.

## Dashboard workflow

The authenticated Outcome & Causality Lab provides the operator path for ongoing measurement:

1. Use Kaya normally on a task where Cortex memory was attributed to the response.
2. Label the task helpful, validated, harmful, or corrected. The decision is audited and reversible.
3. Helpful and validated labels become private real-history cases; negative labels deactivate the case.
4. At eight active positive cases, run the private evaluation button. Cortex evaluates fixed and adaptive retrieval over the same cases in a disposable SQLite snapshot.
5. Read hit@6, recall@6, MRR, context-token, and local retrieval-latency comparisons together. The report does not measure the model's complete answer, time to first token, or task causality.

Raw queries and relevant memory IDs remain in the local `evaluation_cases` table so the test can be rerun. Persisted `evaluation_runs` reports omit those private fields. Label coverage is a measurement-quality driver: an attractive helpfulness rate over a small or selectively labeled subset should not be treated as representative.

### Controlled recall experiment

The Learning Lab adds a prospective experiment rather than replaying only labeled history:

1. An authenticated operator starts the experiment.
2. Before each Cortex prefetch, the task is assigned inside a task-type stratum to the least-filled arm; a deterministic random bucket breaks ties. The three arms are adaptive recall, fixed recall, and no Cortex memory or tool guidance.
3. Turn sync records completion, recalled-memory count, context tokens, provider preparation time, end-to-end elapsed time, and observed tool outcomes under the same task ID.
4. The operator can label every completed task, including no-memory tasks with no usage rows.
5. Accuracy is `helpful + validated` divided by all explicit positive and negative dashboard labels. Inferred conversational feedback is diagnostic only and never unlocks the causal gate.
6. The dashboard requires at least eight explicit labels in every arm and shows Wilson intervals before describing an initial causal estimate.

Assignment is balanced randomized blocking, not retrospective matching. Starting a new experiment creates a fresh experiment ID; stopping preserves all assignments and results.

### Agent-level evaluation and failures

The daily graph keeps the primary outcome and diagnostics separate:

- primary outcome: explicit task accuracy;
- drivers: explicit-label coverage and completed turn-sync records;
- behavior diagnostics: tool result success, corrections, context tokens, provider preparation, and end-to-end elapsed time;
- guardrails: negative outcomes, missing labels, and no-memory/control balance.

The failure explorer applies bounded rules to negative labels and observed tool errors. Its candidate cause is a diagnostic routing hint—not a causal attribution—and links the operator to the task evidence and a specific improvement lever.

## Claim boundaries

| Evaluation | Measures | Does not establish |
| --- | --- | --- |
| Real-history retrieval | hit@k, recall@k, precision@k, MRR, returned context size, local retrieval latency | answer correctness, inference speed, task success, tool reliability, billed tokens |
| Recorded tool outcomes | rates and paired differences in observations collected by the operator | a controlled model result unless collection was randomized and all other variables were held fixed |
| Live provider with tool fixtures | live model tool selection, argument matching, response timing, and optional final-answer checks | production tool reliability, because the harness replays a recorded result instead of executing a tool |

Treat a confidence interval that crosses zero as inconclusive. Use at least 30 independent paired cases for exploratory comparison and more for narrow effects. Repeat live runs across seeds and provider conditions before making a public claim.

## 1. Private real-history retrieval

Create the labels outside the repository and restrict their permissions:

```bash
mkdir -p "$HOME/cortex-evaluations/private"
chmod 700 "$HOME/cortex-evaluations/private"
touch "$HOME/cortex-evaluations/private/retrieval-labels.jsonl"
chmod 600 "$HOME/cortex-evaluations/private/retrieval-labels.jsonl"
```

Each JSONL row is an operator-authored question and one or more Cortex memory IDs that would correctly answer it:

```json
{"schema_version":1,"case_id":"case-001","query":"Which release channel is approved for Project Amber?","relevant_memory_ids":["MEMORY_UUID"],"group":"durable-fact"}
{"schema_version":1,"case_id":"case-002","query":"What procedure should I use after a failed health check?","relevant_memory_ids":["MEMORY_UUID_A","MEMORY_UUID_B"],"group":"procedure"}
```

Memory IDs can be copied from the private Brain dashboard or inspected locally. Do not commit this label file: its questions and labels may describe personal projects even though the generated report omits them.

Run the adaptive policy from the Cortex repository checkout:

```bash
python3 scripts/evaluate_real_history.py \
  --db "$HOME/.hermes/cortex/cortex.db" \
  --labels "$HOME/cortex-evaluations/private/retrieval-labels.jsonl" \
  --output "$HOME/cortex-evaluations/real-history-adaptive.json" \
  --policy adaptive \
  --top-k 6 \
  --token-budget 700
```

Run a fixed-policy ablation into a different file:

```bash
python3 scripts/evaluate_real_history.py \
  --db "$HOME/.hermes/cortex/cortex.db" \
  --labels "$HOME/cortex-evaluations/private/retrieval-labels.jsonl" \
  --output "$HOME/cortex-evaluations/real-history-fixed.json" \
  --policy fixed \
  --top-k 6 \
  --token-budget 700
```

The runner first creates an ephemeral, transactionally consistent SQLite backup and evaluates that copy. It does not migrate or write retrieval activity into the live Brain database, and it deletes the copy on exit. It also fails if any labeled memory no longer exists, preventing deleted or mistyped IDs from silently becoming retrieval misses.

The output contains only aggregate metrics and numbered case outcomes. It omits queries, case IDs, memory IDs, memory content, source references, database paths, and label paths. `approximate_context_tokens` is the repository's characters-divided-by-four estimate, not a provider-billed token count.

By default, group names are also omitted. `--include-group-summary` is available for private diagnosis, but it marks the report as requiring additional review because operator-authored group names may reveal metadata.

## 2. Paired tool-calling evaluation

The tool harness has two modes. Both require exactly one `default_built_in` and one `cortex` observation for every pair.

### Recorded mode: aggregate outcomes already observed

Use recorded mode after collecting outcomes from Hermes logs, a manual test protocol, or another benchmark driver. Keep raw observations outside the repository.

Each JSONL row uses schema version 1:

```json
{"schema_version":1,"pair_id":"pair-001","condition":"default_built_in","provenance":"recorded_live","tool_selected_correctly":false,"arguments_valid":false,"tool_succeeded":false,"task_succeeded":false,"provider_latency_ms":840.2,"total_latency_ms":840.2,"prompt_tokens":710,"context_tokens":95}
{"schema_version":1,"pair_id":"pair-001","condition":"cortex","provenance":"recorded_live","tool_selected_correctly":true,"arguments_valid":true,"tool_succeeded":true,"task_succeeded":true,"provider_latency_ms":875.8,"total_latency_ms":1320.4,"prompt_tokens":790,"context_tokens":155}
```

Allowed provenance values are `recorded_live`, `recorded_replay`, and `manual_grade`. `tool_succeeded`, `task_succeeded`, latency, and token fields may be `null` when they were not observed. Do not convert missing outcomes into failures.

Run it from the checkout:

```bash
python3 scripts/benchmark_tool_calling.py recorded \
  --observations "$HOME/cortex-evaluations/private/tool-observations.jsonl" \
  --output "$HOME/cortex-evaluations/tool-recorded-report.json" \
  --seed 7
```

The report includes condition rates, paired Cortex-minus-built-in rate differences, bootstrap intervals, wins/losses/ties, and paired latency/token differences. It replaces private pair IDs with numbered indices.

### Live mode: call a provider, replay a tool result

Live mode is opt-in and may create provider charges. It calls an OpenAI-compatible Chat Completions endpoint. It does **not** execute shell commands or real tools. When the model produces the expected tool call and arguments, the harness injects the scenario's recorded result fixture and optionally grades the model's final answer.

Each private JSONL scenario contains:

- one user prompt;
- identical tool schemas for both conditions;
- the expected tool and an exact/subset argument label;
- separate built-in and Cortex memory contexts;
- a recorded tool result with an operator-authored `ok` value;
- optional case-insensitive strings required in the final answer.

Example row, expanded for readability but stored as one JSONL line:

```json
{
  "schema_version": 1,
  "case_id": "pair-001",
  "prompt": "Deploy the approved service.",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "deploy_service",
        "description": "Deploy a named service",
        "parameters": {
          "type": "object",
          "properties": {"service": {"type": "string"}},
          "required": ["service"]
        }
      }
    }
  ],
  "expected_tool": "deploy_service",
  "expected_arguments": {"service": "api"},
  "conditions": {
    "default_built_in": {"memory_context": "Bounded built-in memory for this case."},
    "cortex": {"memory_context": "Retrieved Cortex evidence for this case."}
  },
  "tool_results": {
    "deploy_service": {"ok": true, "content": {"status": "deployed"}}
  },
  "expected_final_contains": ["deployed"]
}
```

Keep the API key in an environment variable. The key is never accepted as a direct CLI value:

```bash
export CORTEX_TOOL_BENCH_API_KEY='provider-key-from-your-secret-manager'

python3 scripts/benchmark_tool_calling.py live \
  --scenarios "$HOME/cortex-evaluations/private/tool-scenarios.jsonl" \
  --output "$HOME/cortex-evaluations/tool-live-report.json" \
  --base-url "https://provider.example/v1" \
  --model "provider/model" \
  --api-key-env CORTEX_TOOL_BENCH_API_KEY \
  --seed 7

unset CORTEX_TOOL_BENCH_API_KEY
```

Condition order is deterministically shuffled by seed. The generated report omits prompts, contexts, tool names, schemas, arguments, results, model answers, endpoint URLs, credentials, case IDs, and input paths. It retains the model name, counts, timing, token usage, and outcome patterns, so an operator should still review it before publishing.

If `expected_final_contains` is absent, `task_succeeded` stays `null`; the harness does not assume that a successful call means the whole task succeeded. If arguments do not match, the recorded result is not injected.

## Fair paired collection protocol

For a result that can support a public model comparison:

1. Freeze the model identifier, provider settings, tool schemas, user prompts, grader, and maximum output tokens.
2. Give each condition only the memory context produced by that condition. Do not edit a failed context after seeing the answer.
3. Randomize condition order and avoid concurrent provider requests if latency is reported.
4. Use the same recorded tool outcome for both conditions, or record actual tool reliability as a separate metric.
5. Grade expected tools and arguments before inspecting the condition label when possible.
6. Keep retries and transient failures in the raw record; state any exclusion rule before running the benchmark.
7. Report every metric's denominator, not only successful cases.
8. Publish the sanitized JSON, protocol, model, date, and sample count. Keep private labels and raw traces local.

An offline retrieval improvement is evidence that Cortex found labeled information. A live tool-selection improvement is evidence about the tested model and scenarios. Neither alone proves that autonomous pruning, workflow learning, or the entire production agent improved.
