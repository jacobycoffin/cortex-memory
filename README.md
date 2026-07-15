# Cortex Memory

**Memory that earns its place.** `cortex-memory` is a local-first, psychology-inspired memory core for agent harnesses. It recalls bounded evidence, learns which memories and tool workflows actually help, preserves corrections, consolidates offline, and cools stale information without deleting history.

The core is harness-neutral. A small Python API handles durable writes, recall batches, evidence-use feedback, episodes, audit, and Cortex Sleep. [Hermes Agent](https://github.com/NousResearch/hermes-agent) is the first included adapter and the current reference benchmark—not a requirement for the storage, retrieval, lifecycle, dashboard, or Sleep engine.

Cortex is an engineering system, not a simulated brain. Psychology and neuroscience provide hypotheses; transparent benchmarks decide whether the software helps.

![Cortex benchmark results at a glance](docs/assets/cortex-results-at-a-glance.svg)

## Evidence at a glance

| Question | Measured result | What it means |
| --- | --- | --- |
| Can it find memory at scale? | **98.5–99.5% recall@6** from 100 to 2,000 synthetic memories. | The relevant fact was almost always in Cortex's first six results. |
| Does adaptive recall reduce Cortex context? | **21.7% fewer** approximate memory-context tokens than fixed verbose Cortex, with the same 93.3% labeled answer availability. | The attention gate and compact format avoided unnecessary memory text in this workload. |
| Did answers improve in a live paired run? | **6.7% → 96.7% exact-answer accuracy** across 90 pairs in additive mode. | Cortex made the labeled answer available; the built-in bounded snapshot usually could not hold it. |
| Is inference proven faster? | **No clear latency difference.** TTFT was 1,339 ms with Cortex and 1,337 ms built-in; the confidence interval crossed zero. | The proven benefit is memory capacity and answer availability, not raw model speed—yet. |

![Cortex capacity and retrieval scaling profile](docs/assets/cortex-scale-profile.svg)

These are reproducible synthetic benchmarks, not a promise about every agent or vault. The live accuracy comparison uses the Hermes reference adapter, intentionally tests beyond its built-in snapshot's capacity, and used 51.8% more median prompt tokens in that run to supply the missing evidence. Read the [method, raw results, and required caveats](docs/BENCHMARKING.md), then run representative tests on your own harness and history.

## Why Cortex

Most agent memory systems optimize only for storing and finding text. Cortex also asks:

- Did this request need memory at all?
- Was the recalled memory actually used?
- Did it help, fail, or later prove wrong?
- Which memories are related or superseded?
- Which tool sequence has worked across multiple distinct tasks?
- Can a stale-memory decision be reversed?

The result is a bounded evidence layer that an agent harness can call, plus a Brain dashboard that makes its behavior inspectable and can optionally enable narrowly scoped, confirmed memory reviews.

## What 0.2 adds

- **Attention-gated recall:** `none`, `lean`, `focused`, `procedural`, and `deep` modes vary the memory count, token budget, graph depth, and tool evidence by request.
- **Transparent hybrid retrieval:** SQLite FTS5 precision plus dependency-free semantic features and bounded personalized graph activation. No embedding API or pre-inference LLM call.
- **Temporal recall:** current questions demote superseded facts; historical questions can retrieve the memory valid at a requested time.
- **Evidence-use attribution:** structured values, distinctive anchors, lexical evidence, and capped conceptual similarity prevent every retrieval from being rewarded.
- **Workflow learning:** Cortex learns repeated multi-step tool paths, not only single-tool success counts, and requires evidence from distinct tasks.
- **Reversible consolidation:** safe near-duplicates can fold behind a canonical memory and later be restored.
- **Adaptive lifecycle:** retention considers importance, confidence, trust, utility, uniqueness, frequency, and harm—not age alone.
- **Pruning-regret detection:** archived evidence can be surfaced in shadow mode or restored automatically.
- **Cognition dashboard:** live recall latency, estimated context tokens, abstention, lifecycle changes, tool workflows, and scientific field notes.

## What is developing on `testing`

Cortex 0.3 begins the measurement-and-efficiency cycle. The current testing build adds:

- **Outcome-driven context budgets:** Cortex holds its default until it has enough resolved evidence, then cautiously spends less on repeatedly ignored or harmful context and slightly more when helpful context is consistently saturated.
- **Safe retrieval caching:** short-lived results can be reused after repeated queries, while every turn still records retrieval, selection, injection, and use. Material memory, graph, or tool-guidance changes invalidate the cache across database connections.
- **Private real-history evaluation:** operators can label their own memories locally and export a sanitized retrieval report without publishing queries, memory text, IDs, or database paths.
- **Paired tool-call evaluation:** recorded observations or explicit live provider fixtures compare tool choice, argument shape, task success, latency, and tokens without executing arbitrary tools.
- **Release automation:** Python 3.10–3.14 CI now covers unit tests, clean install, in-place upgrade, schema migration, syntax, public-file privacy, and SVG validation.
- **Cortex Sleep:** a nightly offline replay cycle reviews older sessions and resolved outcomes, requires independent witnesses before proposing associations, surfaces interference, previews maintenance, and records every decision for the dashboard.
- **Controlled learning lab:** randomized adaptive/fixed/no-memory tasks, matched reversible Sleep trials, explicit agent-level outcomes, a failure explorer, cited summary approval, prospective-memory states, and reconsolidation follow-through share one local task/evidence ledger.
- **Budgeted idle reflection:** optional model review can spend a separate per-run token ceiling on evidence-linked proposals. It is off by default, normally billed, privacy-sensitive, and never applies its own conclusions.
- **Harness-neutral API:** `CortexMemory` and `RecallBatch` expose storage, bounded recall, outcome feedback, episodes, audit, and Sleep without depending on a particular agent framework.
- **Inspectable memory traces:** every adapter task records why recall ran or abstained, candidate component scores, selection and rejection reasons, later answer influence, outcome ratings, and create/update/ignore storage decisions in a local JSONL-style event ledger.
- **Explicit applicability:** memories can be standalone or context-dependent, with project, entity, scope, precondition, source-context, system, and version metadata. Missing required context hard-gates retrieval instead of relying on semantic similarity.
- **Context-specific adaptation:** repeated helpful use raises a memory only inside the matching stable project/task context; repeated selection without answer use downweights it there. Outcome-label undo restores the prior evidence state.
- **Storage preflight and maintenance:** automatic capture checks durability, reuse value, exact duplicates, structured contradictions, and independent comprehensibility before writing. Shadow Sleep also flags ambiguous context and drafts source-cited summaries for approval.
- **Memory hygiene and stable document identity:** raw tool executions remain in the dedicated tool ledger instead of becoming recallable memories; sparse placeholders and transient automation statuses are rejected or staged for reversible lifecycle review. Vault section edits update one versioned memory ID rather than creating an archived clone.
- **Explainable memory links:** persistent links carry a typed evidence record and a plain-language reason. Similar wording alone does not create a link, and the default map hides legacy relations whose original evidence cannot be recovered.
- **Operator-trained policy promotion:** repeated Review Inbox decisions compile into scoped admission, connection, retrieval, or retention proposals. A proposal must pass an evidence replay and three new shadow observations before explicit activation; active versions are bounded, auditable, and reversible.

This is development evidence, not a new public performance claim. Stable installs should continue to use `main`; the [roadmap](docs/ROADMAP.md) states what is implemented, still being measured, and intentionally deferred.

## Install the agent-neutral core

Requirements: Python 3.10+ and SQLite with FTS5 (included in normal Python builds).

```bash
git clone https://github.com/jacobycoffin/cortex-memory.git
cd cortex-memory
python3 -m pip install .
```

Use the same API from any harness:

```python
from cortex import CortexMemory

with CortexMemory("./cortex.db") as memory:
    memory.remember(
        "Production deploys require a health check and verified backup.",
        kind="procedure",
        source_category="USER_EXPLICIT",
        session_id="session-a",
        context_mode="context_dependent",
        scope={"project": "Cortex", "task_type": "deployment"},
        entities=["production deployment"],
        preconditions={"environment": "production"},
        source_context="Cortex production release procedure",
        applicable_systems=["cortex"],
    )

    recall = memory.recall(
        "What checks are required before deployment?",
        session_id="session-b",
        task_type="deployment",
        active_project="Cortex",
        scope={"project": "Cortex"},
        system_state={"environment": "production"},
        applicable_systems=["cortex"],
    )
    agent_context = recall.context()

    # After the harness finishes the task, credit only evidence it actually used.
    used_ids = [item["id"] for item in recall.memories]
    recall.finish(used_ids, outcome="helpful")
```

An adapter maps its own session/turn lifecycle into `remember`, `recall`, `RecallBatch.finish`, `record_episode`, and optional offline `sleep`. See the [harness integration guide](docs/INTEGRATION.md).

Inspect recent task traces locally with `cortex-memory traces --limit 20`, summarize observed selection precision with `cortex-memory traces --summary`, or export the append-only ledger with `cortex-memory traces --jsonl`. Use `cortex-memory write-decisions --summary` for storage preflight, `cortex-memory context-feedback` for context-specific adaptation, and `cortex-memory quality-report` for the combined retrieval/health/Sleep view. Raw traces can contain private task text and memory previews; do not commit them.

## Hermes adapter

The repository includes a complete Hermes MemoryProvider adapter and installer:

```bash
HERMES_HOME="$HOME/.hermes" ./scripts/install_local.sh
hermes memory setup
```

Choose `cortex`, then restart the Hermes process so it initializes the provider. Hermes's curated `MEMORY.md` and `USER.md` files remain available; Cortex mirrors explicit built-in memory writes and adds adaptive retrieval.

If the `hermes` launcher is not on your VPS `PATH`, run the real environment directly:

```bash
~/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main memory setup
```

See [Quickstart](docs/QUICKSTART.md) for migration, vault indexing, VPS services, rollback, and the first acceptance test.

## Try the Hermes adapter

In one Hermes session:

> Remember that production deploys require the test suite and a health check.

In a fresh session:

> What checks do I require before production deploys?

Then inspect the evidence:

```bash
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex search "production deploy checks"
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex recall-stats
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex audit
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex sleep --mode shadow --reflection-token-budget 0
```

## Brain dashboard

```bash
cortex-memory --db ./cortex.db dashboard --no-open --port 8765
```

For the Hermes plugin database, use `PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex dashboard --no-open --port 8765`.

Open `http://127.0.0.1:8765`. The dashboard keeps direct memory changes behind explicit review controls. Its built-in Sleep action is shadow-only: it writes an audit report and proposals, but does not change memories, links, or lifecycle state. The dashboard includes:

- a draggable 2D physics map and orbitable 3D constellation;
- timeline, source, use-through, tool-learning insights, and daily trends for memories, connections, lifecycle pruning, and tool calls;
- a unified Review Inbox for pruning, duplicate, connection, conflict, unsupported-claim, and answer-outcome decisions, with full memory text, provenance, scope, activity, proposal evidence, action consequences, and decision reach shown before the choice;
- a guided Kaya Training path that shows review, compilation, replay, shadow, approval, and monitoring progress, plus a Policy Lab for proposed standards and active-version rollback;
- a Cognition lab for recall modes, context tokens, latency, abstention, lifecycle repair, and workflows;
- a Cortex Sleep control room with the live timer window, selectable run history, exact proposal evidence, applied/reversed edge and lifecycle journals, and explicitly non-causal post-run observations;
- an Accuracy × capacity view that plots daily outcome-backed memory helpfulness against stored memory count and reports average recall context alongside it; it is a memory-quality proxy, not a claim of general answer accuracy;
- an authenticated local benchmark lab that runs the versioned synthetic retrieval suite on the dashboard host, records a transparent quality/speed score, and graphs compatible runs without claiming model-inference speed;
- an Outcome & Causality Lab for reversible task-level labels and a paired private real-history evaluation that compares adaptive and fixed retrieval without persisting queries, memory text, or IDs in its report;
- a Learning Lab that assigns future Kaya tasks to balanced randomized adaptive, fixed, or no-memory conditions before recall, graphs explicit daily agent accuracy, and keeps task completion, tool results, context, and latency as separate diagnostics;
- a failure explorer, reversible matched Sleep apply trials, source-cited summary candidates that remain outside recall until approval, prospective commitments with due/completed/abandoned states, and reconsolidation records that preserve the replaced version;
- an evidence hierarchy that keeps raw memories addressable, shows source links for supported claims, and treats higher-level bundles as read-only candidates rather than automatic truths;
- a Trust monitor that logs a pre-outcome reliability estimate for every recalled memory, explains the proposed use/verify/abstain decision, plots calibration only after enough explicit outcomes exist, and holds enforcement in shadow until a hard evidence gate passes;
- Sleep hypotheses and tool-guidance follow-through that state what future evidence would count, while keeping after-event comparisons explicitly observational;
- plain-language health guidance split into recall integrity, memory quality, learning-loop, and Sleep-maintenance layers, plus an inspectable memory index that routes operator work into the Review Inbox;
- a tablet/PWA navigation layout with visible tab names and a readable card-style memory index;
- a clickable memory-type guide plus a dedicated Settings view for themes and account controls.

For a public hostname, terminate TLS at a reverse proxy and keep Cortex bound to localhost. The included systemd installer prints a temporary password once; the dashboard requires you to replace it at first sign-in. Cortex stores a PBKDF2 password hash rather than the readable password, uses signed 12-hour browser sessions, rate-limits failed logins, and revokes existing sessions after a password change. The [self-hosting guide](docs/DASHBOARD_HOSTING.md) shows generic Caddy, Nginx, Cloudflare Tunnel, DNS, reset, and verification examples for a hostname you control.

Guided review and Learning Lab changes are opt-in. After authentication and TLS are configured, set `CORTEX_DASHBOARD_REVIEWS=1` in the dashboard environment to enable confirmed choices. The Review Inbox can approve or deny proposed links, keep/archive/trash pruning candidates, resolve conflicts, confirm unsupported claims, and label real answer outcomes. Trash means a reversible `tombstoned` lifecycle state: the memory is excluded from recall, but its text, provenance, versions, decision reason, and restore path remain. Every review records an audit decision with one explicit reach: **This memory/review only** changes only the current item and does not train a standard; **This memory + exact duplicates** applies the same action only to eligible copies with identical content, context, and source; **Teach Kaya too** changes the current item and also contributes scoped policy evidence. Exact-duplicate actions and one-off decisions never enter the Feedback Compiler.

The Feedback Compiler groups only decisions explicitly marked **Teach Kaya** into an explainable `policy_candidates` rule. Five matching decisions at 80% agreement unlock a stored-evidence replay. A passing candidate then observes three new matching training decisions in shadow mode before the dashboard permits scoped promotion. A broader core promotion requires 15 supporting decisions across three contexts. Promoted `policy_versions` can make bounded adjustments to automatic write admission, independent link-witness requirements, retrieval rank, or lifecycle retention. They never bypass relevance, scope, provenance, review, or deletion safeguards, and the Policy Lab can roll back every active version. Summary approval and prospective commitments remain explicit source-backed writes. Controlled Sleep trials apply only randomized treatment links and expose a one-click reversal. There is no hard-delete action.

The Insights benchmark button is independent of guided review. It is authenticated, allows only one bounded run at a time, uses a temporary synthetic database, and makes no model or provider calls. Its overall score is a versioned operating index: 85% retrieval quality and 15% local p95 retrieval speed. Compare only runs from the same suite version under similar host load; use the paired live-model benchmark for whole-agent latency claims.

Outcome labels require the same opt-in review setting. A helpful or validated label creates one private real-history case; harmful or corrected labels remove that case. At eight positive cases the dashboard can run adaptive and fixed retrieval against the same disposable database snapshot. Only sanitized aggregate metrics are persisted. The result measures retrieval mechanics, not Kaya's complete answer accuracy or model inference speed.

If the password is lost, generate a new one and restart the dashboard:

```bash
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex dashboard-password --username cortex
systemctl --user restart cortex-dashboard
```

## How a recall works

```mermaid
flowchart LR
    Q["Agent request"] --> G{"Attention gate"}
    G -->|"no durable context"| Z["Inject 0 memory tokens"]
    G -->|"memory can help"| H["FTS + semantic features"]
    H --> A["Bounded graph activation"]
    A --> R["Utility and temporal ranking"]
    R --> B["Token-budgeted evidence"]
    B --> M["Source monitor · shadow by default"]
    M --> L["Harness inference"]
    L --> U["Evidence-use attribution"]
    U --> S["Utility, links, workflow, lifecycle"]
```

The hot path is deterministic and local. Cortex Memory never requires an extra LLM call to decide what to remember.

## Cortex Sleep

Sleep moves consolidation work out of normal turn latency. Its deterministic pass replays a bounded set of older episodes and successful co-use records, proposes associations only after at least two independent witnesses, identifies structured conflicts, gently downscales weak stale edges, and previews lifecycle, duplicate, and dependency work. The nightly installer keeps it in shadow mode:

```bash
HERMES_HOME="$HOME/.hermes" ./scripts/install_sleep_timer.sh
systemctl --user list-timers cortex-sleep.timer
```

Model reflection is a separate opt-in stage. A positive `--reflection-token-budget` is a ceiling for a normal provider call, not banked or free conversational tokens. Remote reflection can expose selected memory text to that provider, and validated output is stored only as a review proposal. Read the [design, research basis, and exact boundaries](docs/SLEEP.md).

The dashboard keeps **proposed**, **applied**, and **reversed** maintenance records separate. Selecting a Sleep run shows the memory text and IDs behind each connection, downscale, conflict, lifecycle move, consolidation, or dependency repair. Follow-up recall and outcome counts are labeled as observations after the run. The Learning Lab can separately randomize matched association proposals into applied treatment and withheld control arms; even there, it waits for enough explicit outcomes before enabling an initial causal estimate.

The timer reads optional settings from `$HERMES_HOME/cortex/sleep.env`, created mode `0600` with `CORTEX_SLEEP_TOKEN_BUDGET=0`. To test idle reflection later, set `CORTEX_SLEEP_ENDPOINT`, `CORTEX_SLEEP_MODEL`, the token budget, and the API-key variable named by `CORTEX_SLEEP_API_KEY_ENV`; then run one manual shadow cycle before leaving it scheduled.

## Safety model

- no hard-delete operation;
- pruning and consolidation default to shadow previews;
- scheduled Sleep defaults to deterministic shadow mode with a zero reflection-token budget;
- active → cold → archived transitions remain reversible;
- corrections preserve version history;
- metacognitive use/verify/abstain decisions default to shadow observation and do not remove prompt evidence until enforcement is explicitly configured;
- derived memories identify evidence and become dirty when it changes;
- suspicious instruction-like memories are quarantined;
- likely secrets are redacted before storage;
- tool workflow guidance stores argument **keys**, not argument values;
- the dashboard exposes no hard-delete endpoint; ordinary dashboard Sleep is fixed to deterministic shadow mode, while authenticated opt-in Learning Lab actions can approve a cited summary, create or resolve a prospective commitment, or run and reverse a matched association trial.

Read [Privacy and security](docs/PRIVACY.md) before exposing a dashboard or importing a vault.

## Hermes adapter configuration

`hermes memory setup` writes values under `plugins.cortex` in `$HERMES_HOME/config.yaml`.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `db_path` | `$HERMES_HOME/cortex/cortex.db` | Local SQLite database |
| `auto_capture` | `true` | Capture conservative durable candidates |
| `top_k` | `6` | Maximum memories on the deepest recall plan |
| `token_budget` | `700` | Maximum approximate memory context budget |
| `retrieval_threshold` | `0.16` | Minimum non-pinned retrieval score |
| `adaptive_recall` | `true` | Skip or shrink recall by task |
| `adaptive_budget_learning` | `true` | Learn bounded task-sensitive budgets from resolved outcomes |
| `metacognition_mode` | `shadow` | `off`, observe use/verify/abstain decisions, or experimental `enforce` |
| `query_cache_ttl_seconds` | `45` | Reuse unchanged retrieval results briefly; `0` disables it |
| `compact_context` | `true` | Use the lower-token evidence format |
| `attribution_threshold` | `0.18` | Minimum evidence-use score for utility credit |
| `regret_mode` | `shadow` | `off`, detect only, or `restore` archived matches |
| `consolidation_mode` | `shadow` | `manual`, `shadow`, or reversible `apply` |
| `pruning_mode` | `shadow` | Preview or apply lifecycle transitions |
| `cold_after_days` | `90` | Base low-value cooling interval |
| `archive_after_days` | `180` | Base cold-memory archive interval |

Keep mutation modes and metacognition in `shadow` until you have reviewed your own recall, calibration, and pruning-regret data. `enforce` may withhold low-reliability memories and mark borderline evidence for verification, so it should follow a representative outcome trial.

## Research and evidence

- [Scientific Foundations (PDF)](output/pdf/Cortex-Scientific-Foundations.pdf) — an 11-page visual guide to 25 primary sources, their engineering translations, and the claims Cortex should and should not make.
- [Brain and memory foundations](docs/BRAIN_FOUNDATIONS.md) — annotated primary sources, anatomy cautions, and the complete research-to-feature map.
- [Architecture](docs/ARCHITECTURE.md) — data model, retrieval, learning, repair, and trust boundaries.
- [Harness integration](docs/INTEGRATION.md) — the portable API and event contract for any agent runtime.
- [Benchmarking](docs/BENCHMARKING.md) — fair baselines, paired live-model testing, uncertainty, and claim rules.
- [0.3 evaluation guide](docs/EVALUATION.md) — private real-history labels and paired tool-calling measurement.
- [Cortex Sleep](docs/SLEEP.md) — offline replay, optional reflection budgets, safety model, and primary-source research mapping.
- [Testing](docs/TESTING.md) — automated and manual acceptance paths.
- [Development roadmap](docs/ROADMAP.md) — phased work, safety rules, and promotion gates.

The July 13, 2026 additive benchmark on 90 paired questions / 500 synthetic memories measured 96.7% answer accuracy with Cortex versus 6.7% with Hermes's bounded built-in snapshot. Whole-agent TTFT was effectively tied; Cortex added prompt tokens in that pre-0.2 fixed-recall run. Treat it as a published baseline, not proof of universal speed or accuracy. Raw aggregates and methodology live in [`benchmark-results`](benchmark-results/).

On the local 0.2 retrieval run, Cortex reached 99.5% recall@6 at 2,000 synthetic memories with 34.0 ms p95 retrieval. The 500-memory mixed-workload ablation used 21.7% fewer approximate memory-context tokens than fixed verbose recall with no change in labeled answer availability. See the [0.2 retrieval report](benchmark-results/cortex-v020-retrieval.md) and [adaptive-context report](benchmark-results/cortex-adaptive-v020.md). These are local retrieval/prompt-preparation results; re-run the paired live-model benchmark before making a new inference-speed claim.

## Development

```bash
python3 -m unittest discover -v
python3 scripts/benchmark.py
python3 scripts/benchmark_compare.py --sizes 100,500,2000 --queries 200
python3 scripts/benchmark_adaptive.py --size 500 --memory-queries 30
python3 scripts/benchmark_cache.py --size 2000 --repetitions 80
PYTHON_BIN=python3 bash scripts/smoke_install_upgrade.sh
```

Runtime dependencies are Python standard library only. Real-history and tool-call evaluation commands are in the [0.3 evaluation guide](docs/EVALUATION.md). See [Contributing](CONTRIBUTING.md) for change rules and required evidence.

## Project status

Cortex is experimental software. It is suitable for opt-in testing with backups and shadow lifecycle modes. Neural embeddings, autonomous generated summaries, and unconstrained self-modification are intentionally out of scope until simpler mechanisms show a measurable benefit.

MIT licensed. Cortex Memory is an independent agent-memory project; Hermes is its first supported harness adapter.
