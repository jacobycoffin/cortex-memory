# Brain and memory foundations behind Cortex

This is the source map for Cortex's psychology-inspired design. It separates three things that are often blurred together:

1. what researchers found about human or animal memory;
2. the software hypothesis Cortex derives from it;
3. the measurement needed to show that the software hypothesis helps Hermes.

Cortex does **not** simulate a brain. Terms such as attention, activation, consolidation, reconsolidation, and forgetting are engineering analogies with explicit limits.

## Start with these foundational readings

### Memory systems and anatomy

- [Human Memory, NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK10925/) — accessible overview of memory stages and systems. Useful orientation, not a software specification.
- [Squire, Memory systems of the brain (2004)](https://pubmed.ncbi.nlm.nih.gov/15464402/) — review of declarative and nondeclarative systems and their neuroanatomical organization.
- [Squire et al., Memory consolidation (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4526749/) — review of synaptic and systems consolidation and the changing role of the medial temporal lobe over time.
- [Tonegawa et al., Memory engram storage and retrieval (2015)](https://pubmed.ncbi.nlm.nih.gov/26335640/) — review of engram-cell evidence and retrieval. Cortex does not implement biological engrams; the paper helps distinguish storage from accessibility.

### Working memory and bounded access

- [Baddeley, The episodic buffer (2000)](https://pubmed.ncbi.nlm.nih.gov/11058819/) — a limited-capacity interface integrating information from multiple sources.
- [Cowan, The magical number 4 (2001)](https://pubmed.ncbi.nlm.nih.gov/11515286/) — review of working-memory capacity estimates, often around three to five chunks under controlled conditions.

### Accessibility, association, and retrieval

- [Anderson & Schooler, Reflections of the environment in memory (1991)](https://www.psychologicalscience.org/journals/psychological-science/j.1467-9280.1991.tb00174.x/) — relates accessibility to environmental recurrence and recency.
- [Collins & Loftus, A spreading-activation theory of semantic processing (1975)](https://doi.org/10.1037/0033-295X.82.6.407) — classic semantic-network account.
- [Roediger & Karpicke, Test-enhanced learning (2006)](https://pubmed.ncbi.nlm.nih.gov/16507066/) — retrieval practice improves delayed human retention relative to repeated study in the reported experiments.
- [Cepeda et al., Distributed practice meta-analysis (2006)](https://pubmed.ncbi.nlm.nih.gov/16719566/) — spacing and retention interval interact across many verbal-recall studies.

### Consolidation, replay, and revision

- [McClelland, McNaughton & O'Reilly, Complementary learning systems (1995)](https://doi.org/10.1037/0033-295X.102.3.419) — influential account of fast hippocampal learning and slower neocortical integration.
- [Klinzing, Niethard & Born, Systems consolidation during sleep (2019)](https://pubmed.ncbi.nlm.nih.gov/31451802/) — review of sleep-related reorganization of memory.
- [Nader & Hardt, A single standard for memory (2009)](https://www.nature.com/articles/nrn2590) — review of consolidation and reconsolidation.
- [Memory reconsolidation, NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK3905/?report=reader) — historical and experimental overview of memory modification after retrieval.
- [Adaptive forgetting in humans and machines (2019)](https://pubmed.ncbi.nlm.nih.gov/30930746/) — argues that forgetting can be functional rather than merely a failure.

### Reward and learning signals

- [Schultz, Dopamine reward prediction error coding (2016)](https://pubmed.ncbi.nlm.nih.gov/27069377/) — review of prediction-error signals. Cortex uses outcome deltas only as a loose inspiration; it contains no dopamine model.

### Metamemory and monitoring

- [Ryals et al., DLPFC stimulation improves memory monitoring (2016)](https://pubmed.ncbi.nlm.nih.gov/26970142/) — memory-monitoring judgments improved without improving memory performance itself. This motivates separating Cortex retrieval from its reliability monitor; it does not imply that the software reproduces prefrontal anatomy.

## Anatomy without mythology

Human memory depends on distributed, interacting systems. A few careful lessons are useful for software design:

| Biological finding | Safe engineering lesson | What Cortex does **not** claim |
| --- | --- | --- |
| Medial temporal structures are important for forming many declarative memories, while remote knowledge depends on distributed cortical representation. | Keep raw episodes separate from gradually reinforced semantic/procedural structure. | `episodes` is not a hippocampus and `memories` is not neocortex. |
| Working access is limited and task-dependent. | Inject a small relevant evidence set, and allow zero-memory turns. | An LLM token budget is not a human chunk capacity. |
| Retrieval is cue-dependent and can involve related representations. | Combine direct text cues with bounded associative activation. | Database edges are not synapses and PageRank is not neural spreading. |
| Memories can become labile and update after retrieval. | Version corrections and invalidate derived beliefs when evidence changes. | Software revision does not reproduce molecular reconsolidation. |
| Forgetting can reduce interference and reflect changing relevance. | Cool low-value evidence reversibly and measure pruning regret. | Age alone does not prove a memory is biologically or computationally useless. |
| Skill learning differs from one-shot declarative recall. | Learn tool procedures only from repeated, outcome-backed traces. | Tool statistics are not basal-ganglia circuitry. |
| Memory performance and confidence monitoring can dissociate. | Estimate reliability after retrieval and calibrate it against later outcomes. | A probability table is not a prefrontal cortex or conscious introspection. |

The amygdala and emotional salience are intentionally **not** modeled. User emotion is not a safe proxy for factual importance, and amplifying emotionally charged content could worsen agent behavior.

## Research-to-feature ledger

| Principle | Cortex mechanism | Observable metric | Falsification / ablation |
| --- | --- | --- | --- |
| Bounded working access | attention gate and plan-specific top-k/token budgets | injected tokens, TTFT, task accuracy | fixed 700-token recall vs adaptive plans |
| Rational accessibility | type-aware recency, meaningful-use frequency, volatility | recall@k, stale recall, calibration | remove activation term |
| Cue-dependent association | FTS + transparent features + bounded graph walk | paraphrase recall, precision@k, p95 prep | FTS only; FTS+features; graph off/on |
| Retrieval practice | helpful/validated outcomes strengthen utility | future success after confirmed use | count retrieval only vs outcome weighting |
| Complementary learning | immutable episodes plus semantic/procedural aggregation | transfer to novel tasks, duplicate rate | episodes only vs aggregation |
| Reconsolidation | versioned correction and dirty dependents | correction accuracy, unsupported belief rate | overwrite-in-place baseline |
| Adaptive forgetting | retention score, cold/archive states, regret search | storage size, latency, pruning regret | no pruning; age-only; adaptive retention |
| Procedural reinforcement | single-call and ordered workflow statistics | first-tool accuracy, attempts, completion rate | no guidance; tool-only; workflow guidance |
| Temporal context | validity intervals and supersession | historical/current question accuracy | time-unaware ranker |
| Metamemory | pre-outcome use/verify/abstain monitor with outcome calibration | Brier score, expected calibration error, selective risk | rank-only baseline; monitor shadow vs enforcement |

The feature is justified only when the relevant metric improves without unacceptable harm elsewhere.

## Current implementation status

Implemented in 0.2:

- adaptive recall gate and compact context;
- FTS5 + transparent feature retrieval;
- bounded personalized graph activation;
- temporal scoring and supersession;
- outcome-aware attribution and utility;
- repeated single-tool and multi-step workflow learning;
- versioned correction and evidence invalidation;
- adaptive reversible lifecycle, consolidation, and pruning-regret tracking;
- live dashboard evidence for recall cost and behavior.
- shadow metacognitive source monitoring with inspectable probabilities, decisions, outcomes, and calibration curves.

Not implemented or not claimed:

- biological neural simulation;
- emotional salience or affective memory;
- neural embedding retrieval;
- generative "dream" summaries written without evidence;
- autonomous code/self-modification;
- proven universal token, latency, or task-quality improvements.

## Agent-memory evaluation literature

These papers are relevant to testing Cortex rather than just motivating it:

- [LoCoMo (ACL 2024)](https://aclanthology.org/2024.acl-long.747/) — long-term conversational memory benchmark.
- [LongMemEval](https://arxiv.org/abs/2410.10813) — evaluation of long-term interactive memory abilities.
- [MemoryAgentBench](https://arxiv.org/abs/2507.05257) — broad memory-agent evaluation framework.
- [HippoRAG (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html) — graph-oriented long-term retrieval inspired by hippocampal indexing.
- [RAPTOR](https://openreview.net/forum?id=GN921JHCRw) — recursive summaries for hierarchical retrieval; relevant to a future evidence-backed hierarchy ablation.
- [Mem2ActBench (ACL 2026)](https://aclanthology.org/2026.acl-long.370/) — connects memory to tool/action performance.
- [Memory as Action (ACL Findings 2026)](https://aclanthology.org/2026.findings-acl.956/) — frames memory operations as decisions.
- [LoCoMo-Plus (ACL 2026)](https://aclanthology.org/2026.acl-long.1150/) — extends evaluation toward agentic long-term memory.

## How to explain Cortex in one paragraph

Cortex borrows constraints from memory research rather than pretending to reproduce a brain. Limited working access motivates a small adaptive context; recency and meaningful use influence accessibility; cues activate related evidence; episodes remain separate from gradually reinforced knowledge and tool procedures; corrections preserve history; and forgetting is reversible and measured for regret. Each analogy maps to a database mechanism and an ablation test. Psychology generates the hypothesis; agent benchmarks determine whether it works.
