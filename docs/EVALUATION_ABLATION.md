# Adaptive-feature ablation baseline (tier: smoke-baseline)

Local, deterministic, synthetic-only. No network, no LLM, no live database,
no private histories. Run: `python3 scripts/ablate_adaptive.py --reps 5`
(optional `--output report.json`). Test: `python3 -m unittest tests.test_ablation`.

## Fixtures (each condition proves its mechanism ran)

- `sleep_apply` / `combined`: two resolved helpful co-use tasks over the
  same memory pair, then one Sleep apply cycle. The runner asserts usage
  replay processed tasks AND added association evidence (fail fast
  otherwise) and publishes both counts as `activation`.
- `attention_policy` / `combined`: four used attention samples on one
  topic/mode (two helpful). The runner asserts the weight row holds ≥4
  used samples AND the shadow recommendation for a lean live mode flips
  to procedural (fail fast otherwise).
- Measurement recalls never call `finish()`: eval batches stay pending in
  the discarded temp DB, so no auto-ignored labels pollute the training
  tables (or move the weights under test). Training signals come only
  from the explicit fixtures above.

## What was measured

Four conditions on identical seeded corpora (5 memories, 2 projects,
1 corrected fact) in fresh temp DBs: `baseline` (shadow defaults),
`sleep_apply` (one Sleep apply cycle first), `attention_policy`
(attention observations recorded + resolved first), `combined` (both).
Five cases per condition: exact query, paraphrase, corrected-fact wording,
wrong-project scope, no-memory greeting.

Metrics (aggregates only — no content, queries, or IDs leave the machine):
`retrieval_hit_at_3` (expected memory in selected top-3 — placement, not
answer accuracy), `rendered_evidence_hit_rate` (expected memory present anywhere in the
rendered context — all rendered IDs, not a top-3 cut), `false_positive_rate` (selection on no-memory cases,
with that 2-case subset as denominator), mean selected/rendered size, and
p50/p95 `recall()` wall latency.

## Results (2026-09-09, reps=5, 25 recalls/condition, tier smoke-baseline)

| condition | retr_hit@3 | rend_evid | fp_rate | sel/recall | tok/recall | p50 ms | p95 ms | activation |
|-----------|------------|------------|---------|------------|------------|--------|--------|------------|
| baseline | 1.0 | 1.0 | 0.5 | 2.0 | 91.8 | 5.935 | 8.748 | — |
| sleep_apply | 1.0 | 1.0 | 0.5 | 2.0 | 91.8 | 5.664 | 8.256 | replay ≥1 task, evidence ≥1 |
| attention_policy | 1.0 | 1.0 | 0.5 | 2.0 | 91.8 | 5.853 | 7.698 | 4 used samples, shadow → procedural |
| combined | 1.0 | 1.0 | 0.5 | 2.0 | 91.8 | 5.650 | 8.496 | both of the above |

Note: the old `irrelevant_recall_rate` (0.2) used all cases as denominator;
`false_positive_rate` (0.5) uses the no-memory subset — same underlying
selections (the greeting abstains, wrong-project selects), honest denominator.

Quality metrics are identical across conditions; latency deltas (~0.5 ms)
are noise on temp-DB millisecond-scale runs, not evidence.

## Limitations

- 5-memory synthetic corpus with ceiling hits (1.0): quality deltas between
  conditions cannot appear at this scale, so quality verdicts stay
  insufficient-evidence by construction. What this tier proves is mechanism
  engagement (activation assertions), training/eval separation, and metric
  honesty — not which feature wins.
- Sleep apply replays usage evidence; consolidation preview, lifecycle, and
  reflection paths are exercised but produce no eligible candidates here.
- No model in the loop: `retrieval_hit_at_3` is placement, not answer
  quality; `false_positive_rate` is selection on unanswerable cases, not
  observed non-use.
- Latency is temp-DB wall time (~6 ms scale), incomparable with the live
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
