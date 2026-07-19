# Cortex development roadmap

Cortex evolves through measured, reversible changes. An item is not complete because code exists; it is complete only when its intended metric improves without an unacceptable regression in accuracy, latency, privacy, or recoverability.

The stable `main` branch remains the latest released preview. Active development happens on `testing` and reaches `main` only after the relevant acceptance gates pass.

## Release gates

Every promoted change must satisfy all applicable gates:

1. unit and adversarial tests pass on every supported Python version;
2. fresh install, upgrade, database migration, and rollback are exercised in isolation;
3. retrieval changes publish recall, precision, context, and preparation-latency deltas;
4. model-facing changes use paired requests with the same model and randomized condition order;
5. lifecycle changes begin in shadow mode and publish regret evidence before applying mutations;
6. new stored data has an explicit privacy classification and export/retention behavior;
7. dashboard mutations remain disabled unless separately authenticated, CSRF-protected, audited, and opt-in.

## 0.3 — Measurement and adaptive efficiency

This phase makes Cortex easier to evaluate and teaches it to spend context according to observed task needs.

| Workstream | Status | Acceptance evidence |
| --- | --- | --- |
| Outcome & Causality Lab | Implemented; operator labeling active | Used recall tasks accept one audited, reversible outcome; label coverage and observed helpfulness stay separate from causal claims. |
| Structured memory tracing | Initial task ledger implemented; longitudinal review active | Every adapter recall or abstention records candidate components, reasons, influence, rating, storage action, and append-only JSONL events without hidden reasoning. |
| Explicit applicability and storage preflight | Implemented; longitudinal false-positive review active | Project/scope/precondition/system mismatches hard-gate dependent memories; automatic writes record reuse, durability, duplicate, contradiction, and comprehensibility decisions. |
| Context-specific usefulness | Implemented; observational evidence accumulating | Helpful/validated and selected-but-unused outcomes move an inspectable score only in a stable matching context, and dashboard undo restores prior evidence. |
| Real-history evaluation runner | Implemented in CLI and dashboard; needs eight positive operator labels | Private task queries and IDs remain local; persisted reports contain aggregate paired metrics rather than memory text or IDs. |
| Paired tool-calling benchmark | Implemented; needs representative cases | First-tool accuracy, argument validity, completion, latency, and token use are reported per condition. |
| Learned recall budgets | Implemented; longitudinal validation active | Budget changes require sufficient resolved outcomes and remain inside configured ceilings. |
| Safe query-result cache | Implemented; synthetic microbenchmark added | Repeated retrieval reuses results without skipping access/usage evidence or serving post-mutation results. |
| Attention-gate calibration | Initial expansion implemented | Social-only and arithmetic cases now abstain; a broader labeled confusion set remains active work. |
| Continuous integration matrix | Implemented | Python 3.10–3.14 run unit, syntax, install, upgrade, and privacy checks. |
| Clean install and upgrade smoke test | Implemented | An isolated Hermes home preserves its database across reinstall and imports the installed provider. |
| Replacement-mode experiment | Planned | Requires an explicit Hermes integration hook; no duplicate built-in context and no migrated fact loss. |
| Query and result cache benchmark | Initial exact-query benchmark implemented | Warm exact-query latency and context stability are reported; representative hit-rate measurement remains active. |
| Dashboard benchmark wizard | Implemented for synthetic and private-history suites | Synthetic runs measure host retrieval; private runs compare paired policies. Neither is presented as whole-agent accuracy or inference speed. |
| Offline Sleep cycle | Implemented; shadow trial and hypothesis follow-through active | Nightly bounded replay reports evidence-backed associations, interference, lifecycle, context repair, cited summary candidates, and proposal-level future observations without affecting turn latency. |

## 0.4 — Memory structure and calibration

This phase adds better representations only after the 0.3 evaluation layer can compare them fairly.

| Workstream | Safety rule | Acceptance evidence |
| --- | --- | --- |
| Memory Refinery roles and presentation | Implemented (schema 18): deterministic role classification, derived presentations, Clarity review, and future-ingestion enforcement change presentation and governance only — no automatic rewrite, merge, lifecycle mutation, or live retrieval change. | Stage 1 retrieval equivalence is byte-identical on frozen queries; migration, rollback, vault idempotency, and privacy tests pass. |
| Explainable connection training | Implemented (schema 19): readable pair briefs, typed operator relationships, pattern-grouped review, immediate map receipts, and reversible filtering of weak pending proposals after policy promotion. | Approval/undo, replay-to-shadow promotion, backlog filtering/rollback, every renderer's highlighted-edge path, and desktop/tablet/mobile review flows pass. |
| Review Copilot | Implemented (schema 20): bounded provider-backed clarification and recommendation for connection reviews, with original operator language, validated action/scope, provider/model disclosure, confirmation binding, and no recallable-memory write. | Invalid outputs and changed confirmations are rejected; item-only fallback, audit linkage, provider readiness, the desktop conversation-to-map flow, responsive styles, and manual fallback are checked. |
| Creation Trainer and recall sets | Implemented (schema 25): automatic and agent-selected writes stage outside recall; audited operator or bounded automatic-judge decisions create primary or lookup-only records; a behavior-preserving Legacy set and reversible Trained set enforce eligibility across lexical, feature, context, graph, and archived-fallback paths. | Pending candidates never enter retrieval; automatic admission has distinct provenance and cannot bypass safety guards; duplicate legacy approval promotes without copying; activation changes no memory content; reopening cannot leak Legacy into Trained recall; switching back restores the prior set. |
| Semantic neighborhoods | Implemented (schema 23): overlapping project, service, type, tool, reference, and credential-reference groupings guide inspectable search expansion and map proximity. | Neighborhood expansion requires a direct query cue and still obeys recall-set eligibility. Plaintext secrets are redacted before proposal storage; only secret-manager location references may join Credential references. |
| Role-aware tiered retrieval | Shadow-only (`role_tier_shadow_v1`); activation requires representative labeled cases and a paired evaluation showing no unacceptable loss in answer availability, technical lookup coverage, or relevant recall, then ships as a versioned, reversible policy. | Paired private evaluation over labeled cases; promotion and rollback recorded like operator policies. TurboVec or any alternate embedding backend stays out of scope until refinery labels exist. |
| Multi-resolution memory | Raw evidence, dependency-backed claims, and approval-gated cited summaries implemented. | Episodes, facts, procedures, and summaries are ablated separately before automatic candidate generation expands. |
| Evidence-backed summaries | Extractive cited candidates and authenticated approval implemented; no automatic recallable summary writes. | Every statement names active source IDs and unsupported-claim rate plus retrieval cost improve together. |
| Controlled recall experiment | Balanced randomized adaptive/fixed/no-memory assignment and explicit outcome gate implemented. | Each arm reaches its minimum label count across representative task types before policy changes are adopted. |
| Controlled Sleep apply trial | Matched association treatment/control assignment and one-click reversal implemented. | Explicit future-task outcomes reach both-arm thresholds without reversal conflicts. |
| Agent-level evaluation | Daily explicit accuracy, completion, tools, context, latency, corrections, and failure routing implemented. | Several weeks of coverage make task-type comparisons stable enough for configuration decisions. |
| Prospective state and reconsolidation | Open/due/completed/abandoned commitments and correction replacement history implemented. | Notification delivery and longer-horizon replacement-rate evaluation remain external follow-ons. |
| Contradiction and supersession detection | Detection is reviewable before state changes. | Current and historical accuracy improve on a time-labeled set. |
| Confidence calibration | Shadow instrumentation and hard promotion gate implemented; confidence cannot rise from retrieval count alone. | At least 50 representative labels, Brier ≤ 0.20, ECE ≤ 0.15, and low selective risk pass before a controlled enforcement trial. |
| Association reinforcement and decay | Co-use strengthens links; unused links decay without deleting evidence. | Graph-on beats graph-off without reducing precision. |
| Spaced-use reinforcement | Repeated events in one burst have diminishing weight. | Distributed successful use predicts future utility better than raw count. |
| Interference detection | Similarity creates a review signal, not an automatic merge. | Competing-fact error rate falls. |
| Prospective memory | Scheduling remains external to Cortex. | Due, overdue, completed, and cancelled intentions are distinguished reliably. |
| Optional local embeddings | Off by default; no remote embedding call. | Must beat transparent features after indexing cost and memory use are included. |

## 0.5 — Tool and workflow intelligence

| Workstream | Safety rule | Acceptance evidence |
| --- | --- | --- |
| Argument-shape learning | Store keys and coarse types, never secret values. | Correct-argument-shape rate improves on held-out tasks. |
| Workflow branching | Preserve successful fallback paths and failure preconditions. | Completion rate improves without increasing harmful actions. |
| Cross-task workflow transfer | Require evidence from distinct task fingerprints. | Held-out task performance improves over single-tool guidance. |
| Tool-memory confidence | Failures and recoveries update separate signals. | First-tool choice and recovery-step accuracy calibrate to observed outcomes. |
| Tool insight dashboard | Aggregates cannot reveal argument values. | Operators can explain why guidance appeared and disable it. |

## 0.6 — Lifecycle and self-repair

| Workstream | Safety rule | Acceptance evidence |
| --- | --- | --- |
| Scheduled maintenance cycle | Implemented early in 0.3; shadow evidence accumulating. | Reports consolidation, interference, dependencies, lifecycle previews, resource use, and reversible apply changes. |
| Learned retention thresholds | Never train on retrieval count alone. | Storage/interference improves within a preregistered regret ceiling. |
| Operator policy training | Compiled rules are bounded, replayed, shadow-observed, explicitly promoted, versioned, and reversible; no code self-modification. | Admission, connection, retrieval, or retention changes pass review consistency and post-activation regret gates. |
| Pruning simulation | Replay archived candidates before applying. | Estimated benefit and regret are visible per proposed transition. |
| Dependency repair | Changed evidence marks derived memory dirty before demotion. | Unsupported active inference rate decreases. |
| Longitudinal evaluation | No private memory text in published results. | Several weeks of regret, restoration, correction, and utility evidence. |

## 0.7 — Portability, privacy, and operations

| Workstream | Safety rule | Acceptance evidence |
| --- | --- | --- |
| Project and user namespaces | Every retrieval and mutation is scope-bound. | Cross-namespace leakage tests remain zero. |
| Encrypted backup/export/restore | Use audited encryption such as age or SQLCipher; do not invent cryptography. | Round-trip, wrong-key, corruption, and permission tests pass. |
| At-rest encryption guidance | Prefer host disk encryption or SQLCipher. | Threat model and recovery procedure are documented and tested. |
| Migration and rollback command | Backups are verified before schema mutation. | One command restores the previous plugin and readable database. |
| Configuration recommender | Recommendations are previews with evidence. | Suggested settings outperform defaults on the operator's frozen evaluation set. |
| Renderer/retriever plugin API | Third-party code cannot bypass state/security checks. | Contract tests cover ranking, provenance, lifecycle, and failure isolation. |
| Cortex-first harness contract | Implemented initial portable prompt, JSON manifest, and Python reference adapter; Hermes is the first concrete adapter. Harness-native stores remain bootstrap/scratch rather than a competing durable corpus. | Each harness proves before-turn recall, evidence injection, after-turn used-ID resolution, durable-write routing, and zero-memory abstention on self-contained tasks. |
| Independent dashboard security review | Public exposure remains optional. | Findings are tracked and high-severity issues fixed before stable release. |

## Dashboard feedback controls

Human feedback such as **helpful**, **wrong**, **outdated**, **important**, and **forget** is valuable, but it changes memory state. The `testing` branch now includes a unified Review Inbox and guided Kaya Training path. It includes:

- authenticated signed sessions and same-origin request verification;
- explicit confirmation before an archive, confirmation, supersession, or contextual relationship;
- preserved versions, lifecycle events, provenance, and maintenance-log audit records;
- per-review reach that separates one-off actions, eligible exact-duplicate cleanup, and explicit Teach Kaya policy evidence;
- request-size bounds and no hard-delete operation;
- typed review evidence, explainable policy compilation, stored-evidence replay, shadow observation, explicit promotion, and active-version rollback;
- `CORTEX_DASHBOARD_REVIEWS=1` as an explicit switch that leaves the dashboard read-only by default.

The first policy engine is deliberately limited to bounded admission, independent-link-witness, retrieval-rank, and retention-score adjustments. Representative longitudinal regret evidence, write-specific rate limits, and automated pause recommendations remain active follow-up work.

## Stable non-goals

- claiming to simulate brain anatomy;
- treating emotional language as factual importance;
- rewarding a memory merely because it was retrieved;
- autonomous code modification;
- opaque generated summaries without inspectable evidence;
- home-grown cryptography;
- claiming faster inference from a local retrieval benchmark.
