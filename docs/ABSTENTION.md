# Abstention

**Status: mechanism shipped, policy disabled.** No signal available today can tell a good
recall from a bad one, so abstaining would cost more than it saves.

## The claim being tested

Recall always returns something. The best match on a bad day is the least-bad of a set of
irrelevant memories, and the agent then talks about the wrong thing with full confidence.
Abstention would rather return nothing and let the agent say so.

## What was measured — frozen set, 240 cases, 152 hits

| signal | AUC | hits mean | misses mean |
| --- | --- | --- | --- |
| top-1 score | 0.562 | 0.589 | 0.580 |
| query-term coverage (best result) | 0.480 | 0.506 | 0.520 |
| query-term coverage (top result) | 0.464 | 0.391 | 0.416 |
| score spread (top − last) | 0.559 | 0.142 | 0.126 |
| number of results | 0.617 | 5.099 | 4.602 |
| metacognitive confidence | 0.447 | 0.694 | 0.700 |

AUC 0.5 is a coin flip. Nothing here is meaningfully above it, and the metacognitive
confidence — the signal that claims to answer "should this be trusted" — is *below* chance.

Threshold sweep, ranked by top-1 score:

| threshold | cases silenced | of which hits | precision of silence |
| --- | --- | --- | --- |
| ≤ 0.45 | 0 | 0 | — |
| 0.50 | 4 | 3 | 0.25 |
| 0.55 & ≤ 3 results | 13 | 3 | 0.77 |

The one tolerable cell silences 13 cases to remove 10 misses, losing 3 hits. A memory that
answers "I don't know" to a question it would have answered correctly is not better than one
that tries; it is worse, because the user cannot tell the difference.

## What is implemented anyway

`abstention.should_abstain(results, threshold=..., min_results=...)` with protective guards
that a threshold can never override:

- a **pinned** top result is never silenced;
- a result with a **use boost** (it has informed an answer before) is never silenced;
- a result with **corroboration** is never silenced;
- a result already demoted by a **negative cue** is not evidence of a weak query.

`threshold=None` is the shipped default and means never abstain. `MemoryRetriever` accepts
`abstain_threshold` for exactly that purpose.

## What would make this real

- An **independent label slice** (backlog item #207) so precision can be measured on data
  the retriever has not been tuned against.
- A **learned decision rule** on per-case features rather than a hand-picked cut-off; the
  calibration scripts here (`scripts/abstention_features.py`, `scripts/abstention_metacog.py`,
  `scripts/calibrate_abstention.py`) are the harness for it.
- Feedback labels: a memory the user explicitly rejects ("that's not what I meant") is a
  free negative example, and none of it is recorded today.
