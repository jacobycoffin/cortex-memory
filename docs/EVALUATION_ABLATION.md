# Adaptive-feature ablation baseline

Local, deterministic, synthetic-only. No network, no LLM, no live database,
no private histories. Run: `python3 scripts/ablate_adaptive.py --reps 5`
(optional `--output report.json`). Test: `python3 -m unittest tests.test_ablation`.

## What was measured

Four conditions on identical seeded corpora (5 memories, 2 projects,
1 corrected fact) in fresh temp DBs: `baseline` (shadow defaults),
`sleep_apply` (one Sleep apply cycle first), `attention_policy`
(attention observations recorded + resolved first), `combined` (both).
Five cases per condition: exact query, paraphrase, corrected-fact wording,
wrong-project scope, no-memory greeting.

Metrics (aggregates only — no content, queries, or IDs leave the machine):
answer accuracy on the 3 answerable cases (expected memory in top-3),
irrelevant-recall rate (selection on the 2 unanswerable cases), mean
selected/rendered size, and p50/p95 `recall()` wall latency.

## Results (2026-09-09, reps=5, 25 recalls/condition)

| condition | accuracy | irrelevant | sel/recall | tok/recall | p50 ms | p95 ms |
|-----------|----------|------------|------------|------------|--------|--------|
| baseline | 1.0 | 0.2 | 2.0 | 91.8 | 5.347 | 9.200 |
| sleep_apply | 1.0 | 0.2 | 2.0 | 91.8 | 5.557 | 8.733 |
| attention_policy | 1.0 | 0.2 | 2.0 | 91.8 | 5.466 | 8.301 |
| combined | 1.0 | 0.2 | 2.0 | 91.8 | 5.036 | 6.691 |

Quality metrics are identical across conditions; latency deltas (~0.5 ms)
are noise on temp-DB millisecond-scale runs, not evidence.

## Limitations

- 5-memory synthetic corpus: Sleep apply has nothing worth consolidating
  (episodes also fall under the 12 h minimum age), so `sleep_apply`
  exercises the code path, not a real merge decision.
- No model in the loop: "accuracy" is retrieval placement, not answer
  quality; "irrelevant" is selection on unanswerable cases, not observed
  non-use.
- Latency is temp-DB wall time (~5 ms scale), incomparable with the live
  ~1 s tail the stage-latency work investigates.
- Metacognitive calibration and adaptive weight proposals have no
  retrieval-visible toggle at this scale (both record shadow evidence only),
  so they are measured as "no selection effect by design", not ablated off.

## Verdicts

- Sleep consolidation: **insufficient evidence** (for keep or removal) —
  needs real-history cases per `docs/EVALUATION.md` before any claim.
- Adaptive weight proposals: **insufficient evidence** — shadow-only at
  this scale; removal requires `evaluate_real_history.py` deltas.
- Attentional learning: **insufficient evidence** — governs budget policy,
  not selection; its cost/benefit needs live budget-outcome data.
- Metacognitive calibration: **insufficient evidence** — recorded but not
  selection-active; keep shadow until paired evaluation says otherwise.
- Local embeddings: out of scope — optional future experiment only after
  a real-history baseline exists. Do not add the dependency for this work.
