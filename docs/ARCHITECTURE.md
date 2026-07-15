# Cortex architecture

Cortex Memory is a local evidence system around agent inference. Its architecture favors bounded work, inspectability, and reversible adaptation over an opaque "remember everything" pipeline. The core modules are harness-neutral; `__init__.py` also contains the included Hermes MemoryProvider adapter.

## Invariants

1. Events are evidence; beliefs are interpretations.
2. Retrieval is not truth and is not positive reinforcement.
3. Corrections preserve history.
4. Derived knowledge names its support.
5. Destructive maintenance is staged and reversible.
6. A tool procedure requires repeated evidence from distinct tasks.
7. The hot path must not require a network service or an extra LLM call.
8. Offline model reflection may propose, but never directly mutate, memory.

## Runtime flow

```mermaid
flowchart TD
    A["Agent query"] --> B["Deterministic recall planner"]
    B -->|"none"| C["Record abstention"]
    B -->|"lean / focused / procedural / deep"| D["FTS5 candidates"]
    B --> D2["Transparent semantic-feature candidates"]
    D --> E["Candidate union"]
    D2 --> E
    E --> F["Bounded personalized graph activation"]
    F --> G["Utility + time + source scoring"]
    G --> H["Diversity and token budget"]
    H --> I["Compact, untrusted evidence block"]
    I --> J["Harness model"]
    J --> K["Attribution and task outcome"]
    K --> L["Utility / association updates"]
    K --> M["Tool execution and workflow evidence"]
```

## Memory layers

| Layer | Representation | Behavior |
| --- | --- | --- |
| Working | recall plan, pending usage batch, short-lived exact-query cache | process-local and bounded; usage resolves after turn sync |
| Episodic | immutable `episodes`, `tool_executions`, episode memories | exact observations with deduplication |
| Semantic | semantic/decision/preference/identity memories and optional claims | versioned and time-aware |
| Procedural | procedure memories, tool stats, workflow stats | reinforced only after repeated outcomes |
| Prospective | protected prospective kind and validity fields | retained; scheduling remains external |

This is functional decomposition, not a claim that a SQLite table is a hippocampus or cortex.

## Recall planner

`cognition.py` assigns the smallest plan justified by surface cues:

| Mode | Typical cue | Default ceiling |
| --- | --- | ---: |
| `none` | greeting or self-contained transformation | 0 tokens |
| `lean` | weak durable-context signal | up to 6 / ~620 with a stricter threshold |
| `focused` | personal context, decisions, current or historical fact | 6 / ~620 |
| `procedural` | tool, build, deploy, research, or operational task | up to 6 / ~620 plus tool evidence |
| `deep` | multi-hop relationship or temporal reasoning | 6 / ~680 and depth-2 graph |

These are conservative heuristics, not a semantic classifier. Every decision is recorded in `recall_runs` for later calibration.

When adaptive budget learning is enabled, a plan may be adjusted only after at least eight resolved outcomes for the same task type and mode. Pending outcomes and raw retrieval frequency do not count. Repeated ignored or harmful context can reduce the plan by 10–15%; consistently helpful, fully used context can increase it by at most 10%, always inside the configured ceiling. A broader mode-level fallback requires twice as much evidence.

## Candidate retrieval

Two independent local indexes generate candidates:

- SQLite FTS5 with Porter stemming provides exact-term precision and BM25 ranking.
- `memory_features` stores inspectable tokens, crude stems, adjacent pairs, concept aliases, and low-weight character trigrams. It catches modest paraphrases and typos without claiming embedding-level semantics.

The union becomes the seed set for a bounded personalized PageRank-style walk over explicit associations. The neighborhood is capped, the iteration count is fixed, and graph activation cannot bypass lifecycle-state checks.

The provider may cache a retrieval result for a short bounded interval (45 seconds by default, 300 seconds maximum). Connection-local SQLite triggers invalidate the cache after material writes by the provider, while SQLite's `data_version` detects changes committed through another connection. This avoids adding a revision write to every memory transaction and remains compatible with raw SQLite clients. Cache hits still create their own recall record and pending usage batch, so speed does not erase evidence accounting.

## Scoring and selection

Ranking keeps these signals separate until the final score:

- lexical and feature relevance;
- phrase match and graph support;
- type-aware activation from recency and meaningful use;
- Bayesian-smoothed utility;
- confidence × source trust;
- time validity and supersession;
- importance and uniqueness;
- staleness and harmful-use penalties.

Selection then applies a minimum threshold, duplicate suppression, claim-family caps, type diversity, top-k, and the plan's token budget. A weak result can produce an empty evidence block.

## Metacognitive source monitoring

Retrieval relevance and memory reliability are separate judgments. After ranking, `metacognition.py` computes an inspectable pre-outcome probability from source provenance, memory confidence, source trust, currentness, direct match, outcome-backed utility, stale risk, dirty evidence, supersession, and prior harm. It records one of three proposed actions:

- `use` for sufficiently supported evidence;
- `verify` for plausible but weak, stale, inferred, or sparsely calibrated evidence;
- `abstain` below the conservative reliability boundary.

Every judgment is stored in `metacognitive_predictions` before its outcome. Explicit helpful/validated outcomes are positive calibration labels; harmful/corrected outcomes are negative labels. Used, ignored, pending, and withheld records remain visible but do not pretend to be correctness labels.

The default `metacognition_mode=shadow` records what the policy would do without changing the evidence block. `enforce` is experimental: it can withhold `abstain` candidates and labels `verify` candidates inside the model-facing evidence block. Calibration learns conservatively within probability bands, preferring task-and-source evidence and requiring progressively larger samples before task-wide or global fallback. Retrieval frequency alone never changes the probability.

## Feedback and attribution

Every injected set becomes a pending usage batch. After the answer, Cortex credits only memories with evidence of use:

1. an exact structured object value;
2. distinctive paths, URLs, identifiers, or numeric anchors;
3. meaningful token overlap;
4. capped conceptual support.

Conceptual similarity alone cannot produce high-confidence credit. Later user feedback can mark attributed memories helpful or harmful. Selected-but-unused evidence is resolved as ignored.

## Tool memory

Each completed tool call records a redacted episode: task family, tool name, argument-key names, normalized success/error, timestamp, and session. It does not put argument values into procedural guidance.

`tool_workflows` preserves ordered multi-step traces. `tool_workflow_stats` reinforces a sequence only after the same task family and workflow succeeds across at least two distinct task descriptions. Workflow guidance is similarity-ranked within the current task type.

## Consolidation and lifecycle

Near-duplicate consolidation chooses a high-value canonical memory, adds a `consolidates` relation, and moves redundant members to `cold`. `consolidation_members` stores each prior state so `undo-consolidation` can restore it.

Lifecycle maintenance calculates retention from importance, confidence, trust, currentness, uniqueness, meaningful use, positive outcomes, harm, and duplicate pressure. Protected identity, preference, and prospective memories are excluded from automatic pruning. State changes are written to `lifecycle_events`.

Archived evidence is excluded from normal retrieval. A shadow search can detect a high-scoring archived match as pruning regret; `regret_mode=restore` can return it to active state. There is no hard-delete API.

## Offline Sleep cycle

`sleep.py` runs outside normal agent inference. A cycle selects only old, unprocessed episodes and resolved usage tasks, replays them through local graph-free retrieval, and stores hashed witness evidence. An association requires at least two independent session/task witnesses; one burst or raw retrieval count cannot qualify it.

The deterministic pass also produces structured interference, stale weak-edge, lifecycle, near-duplicate, and dependency-repair proposals. Shadow mode records observations only. Explicit `--mode apply --apply` can add or strengthen a `sleep_replay` edge, reduce an eligible weak association by five percent, or commit a reversible lifecycle transition. `sleep_edge_changes` and `sleep_state_changes` preserve prior values for `sleep-undo`. Neither mode hard-deletes evidence.

Optional reflection is last and defaults to zero tokens. When an operator supplies a provider, model, key environment-variable name, and per-run ceiling, Cortex sends a bounded set of already-sanitized proposal memories. The response must match the typed JSON contract and may reference only submitted memory IDs. Valid output becomes a `reflection_*` proposal; it cannot create a memory, relation, lifecycle change, or fact. Provider failure leaves deterministic Sleep results intact.

## Trust boundaries

- SQLite is local to `$HERMES_HOME/cortex` and uses WAL mode.
- memory text is sanitized before write;
- likely secrets are redacted and prompt-like instructions quarantined;
- evidence is labeled fallible and never presented as instructions;
- the dashboard binds to localhost; memory and cognition reads stay read-only, while authentication and opt-in guided review use narrowly scoped, same-origin POST endpoints;
- guided review writes are disabled unless `CORTEX_DASHBOARD_REVIEWS=1`; enabled actions require a complete authenticated session, an explicit confirmation, bounded request bodies, lifecycle/version history, and an audit record;
- public routing requires TLS and authentication at or before the dashboard;
- vault indexing is incremental and reads source notes without modifying them.
- remote Sleep reflection is disabled by default and, when enabled, crosses the local trust boundary with selected memory text.

## Schema evolution

Schema 4 added `memory_features`, `recall_runs`, `lifecycle_events`, `pruning_regret`, `consolidation_runs`, `consolidation_members`, `tool_workflows`, and `tool_workflow_stats`.

Schema 5 adds `recall_budget_observations` plus the connection-local revision and external `data_version` invalidation needed by safe caching. Opening an older database creates and backfills required structures without deleting existing memories.

Schema 6 adds auditable Sleep runs, replay/usage processing state, independent association evidence, proposals, and reversible edge/state change journals. Migration creates the new tables without rewriting existing memories or edges.
