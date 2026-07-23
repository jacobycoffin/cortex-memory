# Integrating Cortex Memory with an agent harness

`cortex-memory` separates its framework-neutral memory core from harness adapters. The core has no runtime dependency on Hermes, LangGraph, CrewAI, the OpenAI Agents SDK, or another orchestration library. An adapter translates its harness lifecycle into a small evidence contract.

## Cortex-first durable memory

An integrated harness should treat Cortex as its durable store of record, not as an optional cache behind a second long-term memory. Harness-native memory remains useful for one small Cortex bootstrap pointer and temporary session scratch. Durable facts, preferences, decisions, corrections, and verified procedures go to Cortex so the same corpus, review evidence, connections, lifecycle, and outcome learning can follow the user across harnesses.

Print the versioned, machine-readable contract for any adapter:

```bash
cortex-memory harness-contract --tool-name cortex_memory
```

The output includes the portable system-prompt block, bootstrap pointer, before-turn recall, prompt injection, after-turn evidence resolution, write routing, never-memory categories, and enforcement requirements. A harness should inject the system block and call the lifecycle hooks in code; it should not rely on the model remembering to invoke retrieval unaided. When the harness permits it, disable or intercept its competing durable-write tool. If it cannot be disabled, make Cortex's precedence explicit in the system prompt and mirror any legacy write into Cortex as a compatibility safety net.

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

For a ready-made Python lifecycle wrapper:

```python
from cortex import CortexHarnessAdapter

with CortexHarnessAdapter("./state/cortex.db") as cortex:
    system_prompt = cortex.system_prompt_block(tool_name="cortex_memory")
    turn = cortex.before_turn(
        user_text,
        session_id=session_id,
        active_project="Cortex",
        scope={"project": "Cortex"},
    )
    # Inject turn.context as fallible evidence below system policy.
    answer, used_ids = run_agent(system_prompt, turn.context, user_text)
    turn.finish(used_ids, outcome="helpful" if user_confirmed else None)
```

The wrapper adaptively skips greetings and self-contained tasks, so “primary” does not mean blindly injecting memory into every turn. `force_recall=True` is available when a harness already knows the task must use durable history.

Hermes currently keeps its built-in memory surface available alongside an external provider. The included adapter therefore injects an explicit `cortex_memory`-first rule and intercepts a successful legacy built-in write as a Cortex creation proposal. Recall itself is still enforced by Hermes's before-turn provider hook; it does not depend on the model choosing to search. Set Hermes's `memory.nudge_interval` to `0` in a Cortex-primary deployment so the periodic legacy background reviewer does not keep filling `MEMORY.md`; Cortex's provider `sync_turn` remains responsible for bounded automatic proposals and episode recording. Neither automatic turn extraction nor a model-issued memory tool call becomes recallable until an audited operator decision—or the separately enabled, delayed automatic judge—admits the unchanged staged candidate.

The harness decides how it detects durable facts, which evidence the answer used, and when an outcome is known. Cortex deliberately does not infer success merely because a memory was retrieved.

Each `recall` opens a task trace automatically. `RecallBatch.finish` closes its usage phase and assigns `Neutral` to attributed evidence or `Irrelevant` to selected-but-unused evidence until a stronger outcome is supplied. Later explicit outcomes update the same trace while preserving the append-only decision and evaluation events. Inspect snapshots with `cortex-memory traces`; use `--jsonl` only for private debugging because it includes task text and candidate previews.

Pass explicit context whenever the harness knows it: `active_project`, durable `scope`, named `entities`, current `system_state`, and applicable system/version identifiers. A context-dependent memory is ineligible when its required scope or preconditions are missing. Conversation and session IDs remain in the audit trace but are excluded from the stable feedback bucket, so reuse learning can accumulate across sessions.

For creation proposals, use `context_mode="standalone"` only when the text makes sense by itself. Otherwise pass `context_mode="context_dependent"` plus its project/scope/entities/preconditions/source context. The store records durability, duplicate, contradiction, and comprehensibility checks before placing the candidate in the review inbox; automatic capture may ignore an unresolved or under-scoped candidate instead of creating a low-quality proposal. Use `CortexHarnessAdapter.propose` for anything selected or phrased by an agent. `CortexHarnessAdapter.remember` is the trusted commit primitive and is reserved for an audited creation-review authority or a controlled, verified import.

## Source and trust labels

Use `source_category` consistently:

| Category | Use |
| --- | --- |
| `USER_EXPLICIT` | The user explicitly asked Cortex to remember the exact statement. |
| `USER_STATED` | A deterministic extractor found it in a user turn, but the exact candidate has not been reviewed. |
| `TOOL_VERIFIED` | A tool result directly established it. |
| `DOCUMENT_EXTRACTED` | It came from an indexed document. |
| `AGENT_INFERENCE` | The agent inferred it; derived memories should identify evidence. |
| `AGENT_PROPOSED` | An agent selected, summarized, or phrased a candidate that still requires review. |
| `OPERATOR_APPROVED` | The dashboard operator approved the exact memory text through Creation review. |
| `AUTOMATIC_APPROVED` | The delayed automatic judge admitted unchanged staged text through the same audited, reversible Creation review path. |
| `REFLECTION` | It was proposed during reflection and must remain reviewable. |

`origin_source_category` continues to record where the claim came from;
`approval_state` records `operator_approved`, `automatic_approved`, or a trusted
import separately. Approval governs recall eligibility and never claims the
underlying statement was independently verified. See [Automatic memory judge](AUTO_JUDGE.md).

Do not store secrets, raw tool credentials, or authorization decisions as ordinary memories. Cortex redacts likely secret values before proposal storage. It may remember only a safe location reference such as “the Hermes deploy credential is stored in 1Password under Hermes VPS.” Those references join the protected Credential references neighborhood under Tool use and may also join the relevant service neighborhood. The harness remains responsible for minimizing sensitive input.

## Feedback semantics

`RecallBatch.finish` first resolves selected evidence as used or ignored. The optional outcome then updates only used memories:

- `helpful`: the evidence contributed positively;
- `validated`: independent evidence confirmed it;
- `harmful`: it contributed to a bad result;
- `corrected`: the task exposed stale or incorrect evidence.

Raw call count is demand, not truth. An adapter should never mark every retrieved memory helpful automatically.

### User-visible memory receipts

The Cortex-primary prompt asks the agent to append one exact, compact line only
when recalled evidence materially influenced the answer. In Hermes, Cortex also
uses the pre-delivery `transform_llm_output` hook as a conservative backstop:
when the model omits the line, only memories with strong deterministic
answer-use evidence are added mechanically.

Hermes currently loads exclusive memory providers through a collector that
does not forward general plugin hooks. The Cortex adapter therefore installs
one idempotent process-wide output dispatcher during provider initialization
and routes it to the active provider by session ID. Provider shutdown removes
its session routes, so repeated gateway sessions do not accumulate bound hook
callbacks.

```text
Cortex memory: M:1234abcd, M:5678efab
```

The receipt is transparency, not a source citation or truth claim. It is limited
to three current-turn IDs, omitted when memory did not influence the answer, and
removed before Cortex performs semantic attribution, episode replay, or automatic
capture. A valid receipt is also explicit answer-use evidence, but only when each
prefix uniquely resolves inside the current turn's bounded recall set.
Automatically injected `Cortex evidence` means prefetch already checked Cortex;
the agent must not claim it skipped Cortex merely because it did not make an
explicit search tool call.

An immediate response such as `M:1234abcd was wrong` or `M:1234abcd was not
relevant` applies individual feedback only to that listed memory. When several
IDs are referenced ambiguously, Cortex does not guess. Replacement content still
uses the version-preserving `correct` action. Set `memory_receipts: false` in the
Cortex plugin configuration to disable the user-visible line.

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
