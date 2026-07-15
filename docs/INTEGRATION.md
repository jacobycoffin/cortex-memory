# Integrating Cortex Memory with an agent harness

`cortex-memory` separates its framework-neutral memory core from harness adapters. The core has no runtime dependency on Hermes, LangGraph, CrewAI, the OpenAI Agents SDK, or another orchestration library. An adapter translates its harness lifecycle into a small evidence contract.

## The five-event contract

1. **Durable write:** call `remember` only for information worth carrying into future sessions. Label whether it came from the user, a tool, a document, or an agent inference.
2. **Bounded recall:** before inference, call `recall` with the current query, session ID, task family, result limit, and context budget. An empty result is valid.
3. **Evidence injection:** insert `RecallBatch.context()` as fallible data, not instructions. Keep it separated from system policy and tool authorization.
4. **Outcome resolution:** after the task, call `RecallBatch.finish` with only the memory IDs actually used and, when known, `helpful`, `harmful`, `validated`, or `corrected`.
5. **Episode capture:** after a completed turn or session, call `record_episode` with the user and assistant text so offline replay can inspect older experience later.

Scheduled Cortex Sleep is outside this hot path. It can run from the CLI, a timer, or `CortexMemory.sleep()` while the harness is idle.

## Minimal adapter

```python
from cortex import CortexMemory


memory = CortexMemory("./state/cortex.db")


def before_model(session_id: str, task_type: str, user_text: str):
    batch = memory.recall(
        user_text,
        session_id=session_id,
        task_type=task_type,
        active_project="Cortex",
        scope={"project": "Cortex"},
        system_state={"environment": "production"},
        limit=6,
        token_budget=700,
    )
    return batch, batch.context()


def after_model(batch, used_memory_ids, outcome=None):
    batch.finish(used_memory_ids, outcome=outcome)


def after_turn(session_id: str, user_text: str, assistant_text: str):
    memory.record_episode(user_text, assistant_text, session_id=session_id)
```

The harness decides how it detects durable facts, which evidence the answer used, and when an outcome is known. Cortex deliberately does not infer success merely because a memory was retrieved.

Each `recall` opens a task trace automatically. `RecallBatch.finish` closes its usage phase and assigns `Neutral` to attributed evidence or `Irrelevant` to selected-but-unused evidence until a stronger outcome is supplied. Later explicit outcomes update the same trace while preserving the append-only decision and evaluation events. Inspect snapshots with `cortex-memory traces`; use `--jsonl` only for private debugging because it includes task text and candidate previews.

Pass explicit context whenever the harness knows it: `active_project`, durable `scope`, named `entities`, current `system_state`, and applicable system/version identifiers. A context-dependent memory is ineligible when its required scope or preconditions are missing. Conversation and session IDs remain in the audit trace but are excluded from the stable feedback bucket, so reuse learning can accumulate across sessions.

For durable writes, use `context_mode="standalone"` only when the text makes sense by itself. Otherwise pass `context_mode="context_dependent"` plus its project/scope/entities/preconditions/source context. The store records durability, duplicate, contradiction, and comprehensibility checks in `memory_write_decisions`; automatic capture may ignore an unresolved or under-scoped candidate instead of creating a low-quality memory.

## Source and trust labels

Use `source_category` consistently:

| Category | Use |
| --- | --- |
| `USER_EXPLICIT` | The user directly stated or confirmed it. |
| `TOOL_VERIFIED` | A tool result directly established it. |
| `DOCUMENT_EXTRACTED` | It came from an indexed document. |
| `AGENT_INFERENCE` | The agent inferred it; derived memories should identify evidence. |
| `REFLECTION` | It was proposed during reflection and must remain reviewable. |

Do not store secrets, raw tool credentials, or authorization decisions as ordinary memories. Cortex sanitizes likely secrets and quarantines instruction-like text, but the harness remains responsible for minimizing sensitive input.

## Feedback semantics

`RecallBatch.finish` first resolves selected evidence as used or ignored. The optional outcome then updates only used memories:

- `helpful`: the evidence contributed positively;
- `validated`: independent evidence confirmed it;
- `harmful`: it contributed to a bad result;
- `corrected`: the task exposed stale or incorrect evidence.

Raw call count is demand, not truth. An adapter should never mark every retrieved memory helpful automatically.

## Tool and workflow integration

The included Hermes adapter demonstrates deeper tool-event capture and workflow learning. Other harnesses can start with the five-event contract, then map normalized tool outcomes into `CortexStore.record_tool_execution` and ordered successful traces into the workflow APIs. Store tool names, task families, argument keys, and normalized outcomes—not secret argument values.

## Offline consolidation

```python
from cortex.sleep import SleepConfig

report = memory.sleep(SleepConfig(mode="shadow", reflection_token_budget=0))
```

Start with deterministic shadow mode. It records replay evidence and maintenance proposals without changing memories. Optional provider reflection is separately configured, normally billed, and may send selected memory text to that provider. Model-generated output remains proposal-only.

## Adapter acceptance checklist

- A self-contained greeting can inject zero memory tokens.
- A paraphrased question retrieves a durable fact from another session.
- Selected-but-unused evidence receives no positive credit.
- Helpful and harmful outcomes move utility in opposite directions.
- Repeated selected-but-unused evidence is downweighted only in the matching stable context.
- Required project, precondition, system, and version mismatches hard-gate a context-dependent memory.
- Corrections preserve the prior version.
- Prompt-like stored text remains quarantined from recall.
- The adapter never places memory above system policy or treats it as instructions.
- A shadow Sleep cycle makes no edge or lifecycle mutation.
- `audit()` returns `ok: true` after normal use.

Hermes-specific installation and lifecycle hooks remain in the included reference adapter. A new harness should depend on the public client/store interfaces, not copy Hermes internals.
