# Cortex Sleep

Cortex Sleep is a name for a bounded maintenance cycle that can run while a Hermes agent is idle. It reviews evidence already stored in Cortex, looks for useful relationships and interference, and produces an auditable set of maintenance proposals.

It is not sleep in a biological sense. Cortex has no neurons, synapses, oscillations, dreams, subjective state, or unconscious processing. It does not reproduce a hippocampus, and it does not train or modify the language model. The neuroscience below supplies design questions and constraints—not proof that the corresponding software operation works like a brain.

## The plain-language version

Normal agent activity is the “wake” phase: new events arrive, memories are retrieved, tools run, and outcomes are recorded. A Sleep pass gives Cortex a quiet, limited window to ask:

- Which memories have repeatedly helped across separate tasks?
- Which memories are stale, harmful, contradicted, or competing with newer evidence?
- Which memories are often useful together?
- Can a shorter derived memory be proposed without losing its source evidence?
- Would cooling or archiving something create retrieval regret?
- Are any derived memories unsupported because their evidence changed?

The default and recommended deployment mode is shadow-only. Sleep writes a report and proposals without changing retrieval state. A separate explicit `--mode apply --apply` path exists for bounded, reversible association, downscaling, and lifecycle changes after shadow evidence has been reviewed. Sleep never hard-deletes memories or silently applies model-generated conclusions. Every proposal must name its evidence and reason. Pinned, protected, prospective, quarantined, and corrected memories keep their existing lifecycle protections.

The Brain dashboard exposes this safe default under **Cognition → Cortex Sleep**. Its **Start shadow Sleep** control runs the deterministic pass with provider reflection disabled, rejects concurrent runs, and streams the current phase—episode replay, connection review, pruning review, lifecycle preview, consolidation, and report finalization. The same view reads the installed timer when available, keeps a selectable run history, expands proposals into their source and destination memories, and shows applied or reversed edge/state journals separately.

The authenticated **Learning Lab → Controlled Sleep apply trials** path is narrower than a general apply run. It selects only association proposals, matches them by proposal kind and evidence count, randomizes one proposal in each pair to treatment, withholds its control, journals the treatment edge through the normal reversible Sleep change table, and waits for explicit future task labels. Each arm needs eight explicit observations before the dashboard permits an initial causal estimate. An operator can reverse the treatment links; the ordinary Cognition Sleep button remains shadow-only.

## A bounded cycle

A Sleep run should have explicit ceilings rather than “thinking until finished”:

1. **Idle check.** Start only when the agent is not handling a turn or tool execution, and use a lock so two maintenance cycles cannot overlap.
2. **Consistent snapshot.** Read a stable view of memories, access outcomes, corrections, dependencies, tool outcomes, and graph edges.
3. **Candidate selection.** Inspect a bounded number of high-value, uncertain, conflicting, recently corrected, or lifecycle-eligible records. Do not sweep the full database merely because it exists.
4. **Deterministic analysis.** Compute replay, co-use, interference, retention, dependency, and pruning-regret signals using local metadata and retrieval behavior.
5. **Optional reflection.** If separately enabled, spend no more than the configured per-run token budget to draft evidence-backed proposals.
6. **Validation.** Reject proposals without active source IDs, proposals that cross protection rules, and proposals that exceed per-run change limits.
7. **Report.** Record inputs by ID, reason codes, proposed changes, rejected changes, runtime, and resource use so the operator can inspect the result.

Wall-clock time, candidate count, graph depth, proposed transitions, and optional reflection tokens should all be capped independently. A deadline ends the run cleanly; it does not justify skipping validation.

## How the research maps to the design

### Replay

Wilson and McNaughton observed that patterns of co-activity expressed during a spatial task in rats were expressed again during subsequent slow-wave sleep ([1994, Science](https://doi.org/10.1126/science.8036517)). Girardeau and colleagues found that selectively suppressing hippocampal sharp-wave ripples during post-training consolidation impaired later spatial-task performance in rats ([2009, Nature Neuroscience](https://doi.org/10.1038/nn.2384)).

**Cortex translation:** revisit a small sample of stored evidence together with the retrieval and outcome records that followed it. A replay candidate can be checked for continued usefulness, contradiction, missing dependencies, or pruning regret.

**Boundary:** Cortex re-reads database records. It does not replay neuronal firing sequences, reproduce sharp-wave ripples, or assume that every frequently accessed record should be strengthened. Retrieval count alone is not evidence of usefulness.

Deep Generative Replay showed an engineering approach for reducing catastrophic forgetting in sequentially trained neural networks by interleaving generated examples from earlier tasks with new training data ([Shin et al., NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html)).

**Cortex translation:** use replay as a continual-learning design pattern, while replaying traceable stored evidence and observed outcomes rather than generating substitute history.

**Boundary:** Cortex Sleep does not retrain Hermes’s base model, train a generator, or claim the guarantees of the NeurIPS method.

### Selective reactivation

Rudoy and colleagues associated sounds with object locations, replayed a subset of those sounds during sleep, and found more accurate later recall for the cued locations in their human experiment ([2009, Science](https://doi.org/10.1126/science.1179013)).

**Cortex translation:** do not reactivate everything equally. Give bounded review priority to explicit operator importance, recent correction, spaced successful use, unresolved contradiction, due prospective work, pruning regret, and evidence supporting an active derived memory.

**Boundary:** a software priority rule is not targeted memory reactivation in a person. Cortex does not deliver sensory cues, manipulate sleep, or infer that an item is important merely because it was called repeatedly in one burst.

### Relational integration

Ellenbogen and colleagues reported that offline time including sleep supported relational inference across separately learned premise pairs in a human study ([2007, PNAS](https://doi.org/10.1073/pnas.0700094104)).

**Cortex translation:** propose a relationship when separate memories repeatedly co-occur in successful retrievals, or propose a compact derived summary when every statement can cite active source-memory IDs. Candidate relations include `related`, `supports`, `contradicts`, and `supersedes`; they are not interchangeable.

**Boundary:** co-retrieval is only a clue. It does not prove a semantic relationship or a new fact. Sleep must preserve the original evidence, assign confidence, and route contradictions or unsupported inferences to review rather than merging them into “truth.”

### Homeostatic downscaling

Two 2017 mouse studies reported sleep-associated evidence relevant to synaptic downscaling: de Vivo and colleagues measured changes in axon-spine interface size across wake and sleep ([Science](https://doi.org/10.1126/science.aah5982)), while Diering and colleagues studied a Homer1a-dependent mechanism for scaling down excitatory synapses during sleep ([Science](https://doi.org/10.1126/science.aai8355)). Norimoto and colleagues reported that hippocampal sharp-wave ripples triggered pathway-selective synaptic depression in mice ([2018, Science](https://doi.org/10.1126/science.aao0702)).

**Cortex translation:** keep highly activated records and associations from monopolizing retrieval. Proposals may normalize or gently decay weak association weights, apply diminishing credit to repeated calls from one burst, and require successful, spaced use before utility rises substantially.

**Boundary:** a database score is not synaptic strength. Cortex does not alter biological synapses or language-model weights. Downscaling is not global deletion, and the cited experiments do not specify the correct decay function for software.

### Selective pruning and maintenance

Li and colleagues reported that REM sleep selectively pruned and maintained newly formed dendritic spines during development and motor learning in mice ([2017, Nature Neuroscience](https://doi.org/10.1038/nn.4479)). Yang and colleagues found branch-specific formation of dendritic spines after motor learning and sleep in mice ([2014, Science](https://doi.org/10.1126/science.1249098)). Together, these studies are a reminder that offline maintenance is not simply “delete the weak”; weakening, preservation, and formation can be selective and concurrent.

**Cortex translation:** produce separate proposals to preserve useful evidence, cool low-utility records, archive reversible candidates, consolidate near-duplicates, or create a sourced relationship. A candidate should be judged using age, confidence, currentness, importance, spaced helpful use, harmful outcomes, correction history, dependencies, and prior pruning regret—not age or call count alone.

**Boundary:** Cortex states and graph edges are not dendritic spines. Shadow mode only proposes lifecycle changes. Explicit apply mode can commit a bounded reversible transition, protected records cannot be pruned, and hard deletion is outside the cycle.

### Interference resistance

The replay studies above motivate preserving access to earlier experience, while the relational and selective-maintenance studies motivate distinguishing integration from indiscriminate strengthening. Shin and colleagues address catastrophic forgetting in a machine-learning setting; Girardeau and colleagues provide evidence that disrupting a particular post-training process can impair later memory in rats.

**Cortex translation:** search for memories that are highly similar but disagree on subject, predicate, value, validity interval, or provenance. Propose an explicit contradiction or supersession edge, mark dependent inferences for review, and test archive candidates through shadow retrieval before any later state change is approved. Replay samples should include older and low-frequency evidence so the newest or loudest memory cannot automatically erase competitors.

**Boundary:** these checks reduce a known software risk; they do not guarantee freedom from interference. Similar wording is not necessarily contradiction, different time periods can both be correct, and a language model can still misuse correctly retrieved evidence.

## Optional idle reflection tokens

Deterministic Sleep maintenance does not require a model call. The optional idle-reflection token budget therefore defaults to **0 (off)**.

When an operator configures a value above zero:

- it is a separate per-run capacity for proposing summaries, relationship explanations, or interference notes;
- it is not unused capacity from ordinary conversations;
- unused tokens are not banked, carried forward, borrowed, or converted into a future larger run;
- provider calls consume and are billed under the provider’s normal token accounting and prices;
- the configured ceiling is not a promise that the provider will use exactly that many tokens;
- private memory text leaves the host if the selected provider is remote;
- model output is treated as an untrusted proposal, never as evidence by itself.

Every reflection proposal must cite active memory IDs, fit a typed proposal schema, stay within the run’s candidate scope, and pass deterministic validation. A reflection call cannot directly change memory state, prune evidence, strengthen an edge, or create an authoritative fact. With the budget at zero, the cycle skips provider reflection entirely and still produces its deterministic maintenance report.

The Insights page plots daily stored-memory capacity with resolved helpful/harmful outcome labels and reports average recall context alongside it. That rate is an outcome-backed memory-helpfulness proxy. It is not general agent answer accuracy, and days without helpful or harmful labels remain visibly unlabeled. Any capacity/quality correlation is descriptive, not proof that growth or Sleep caused the change.

The included Linux timer reads `$HERMES_HOME/cortex/sleep.env`. The installer creates it with permission mode `0600` and only `CORTEX_SLEEP_TOKEN_BUDGET=0`. An operator who deliberately enables reflection supplies `CORTEX_SLEEP_ENDPOINT`, `CORTEX_SLEEP_MODEL`, `CORTEX_SLEEP_TOKEN_BUDGET`, `CORTEX_SLEEP_API_KEY_ENV`, and the corresponding key variable there. Run a manual shadow cycle and inspect the report before relying on the schedule.

## What a Sleep report should make clear

A useful report explains rather than anthropomorphizes:

- why each record entered the bounded candidate set;
- which deterministic signals were considered;
- which proposals came from deterministic rules and which came from optional model reflection;
- the evidence IDs and dependencies for every proposed summary or relation;
- why a proposal was rejected or protected;
- estimated context, provider tokens, local runtime, and provider cost when applicable;
- the exact operator action required to review, apply, or discard proposals;
- the exact before-and-after values for any applied edge or lifecycle transition, including whether it was later reversed;
- how many labeled recall outcomes occurred after the run, while stating that temporal sequence alone does not establish causation.

Until longitudinal and ablation tests show otherwise, Cortex Sleep should be described as an experimental maintenance design. The relevant success criteria are measurable software outcomes—retrieval quality, contradiction handling, context cost, pruning regret, restoration rate, and unsupported-inference rate—not resemblance to human sleep.

## Primary sources

- Wilson, M. A., & McNaughton, B. L. (1994). “Reactivation of hippocampal ensemble memories during sleep.” *Science*. [DOI: 10.1126/science.8036517](https://doi.org/10.1126/science.8036517)
- Rudoy, J. D., Voss, J. L., Westerberg, C. E., & Paller, K. A. (2009). “Strengthening individual memories by reactivating them during sleep.” *Science*. [DOI: 10.1126/science.1179013](https://doi.org/10.1126/science.1179013)
- de Vivo, L., et al. (2017). “Ultrastructural evidence for synaptic scaling across the wake/sleep cycle.” *Science*. [DOI: 10.1126/science.aah5982](https://doi.org/10.1126/science.aah5982)
- Diering, G. H., et al. (2017). “Homer1a drives homeostatic scaling-down of excitatory synapses during sleep.” *Science*. [DOI: 10.1126/science.aai8355](https://doi.org/10.1126/science.aai8355)
- Li, W., et al. (2017). “REM sleep selectively prunes and maintains new synapses in development and learning.” *Nature Neuroscience*. [DOI: 10.1038/nn.4479](https://doi.org/10.1038/nn.4479)
- Yang, G., et al. (2014). “Sleep promotes branch-specific formation of dendritic spines after learning.” *Science*. [DOI: 10.1126/science.1249098](https://doi.org/10.1126/science.1249098)
- Norimoto, H., et al. (2018). “Hippocampal ripples down-regulate synapses.” *Science*. [DOI: 10.1126/science.aao0702](https://doi.org/10.1126/science.aao0702)
- Ellenbogen, J. M., et al. (2007). “Human relational memory requires time and sleep.” *PNAS*. [DOI: 10.1073/pnas.0700094104](https://doi.org/10.1073/pnas.0700094104)
- Girardeau, G., et al. (2009). “Selective suppression of hippocampal ripples impairs spatial memory.” *Nature Neuroscience*. [DOI: 10.1038/nn.2384](https://doi.org/10.1038/nn.2384)
- Shin, H., Lee, J. K., Kim, J., & Kim, J. (2017). “Continual Learning with Deep Generative Replay.” *NeurIPS 30*. [Primary proceedings page](https://proceedings.neurips.cc/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html)
