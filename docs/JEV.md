# Jev engine (`cortex.jev`)

Cortex can route its automatic-judge decisions through **Jev** (TypeSafe
System One) — a fast model that answers *typed questions* (`noul` / `choice` /
`score`) over a supplied state and returns probabilities instead of prose.
Every threshold, permission, and side effect stays in code; the model only
supplies calibrated semantic judgments.

Two uses are shipped today:

| Use | Env switch | Status |
| --- | --- | --- |
| Admission (memory creation) | `CORTEX_AUTO_JUDGE_ENGINE=jev` | canary-ready (validated on the recorded corpus) |
| Link judgments (new memories + orphan linker) | `CORTEX_AUTO_JUDGE_JEV_LINKS=1` · `CORTEX_AUTO_JUDGE_LINK_ENGINE=jev` | validated on recorded edges (see below) |

The chat provider stays the fallback for admission
(`CORTEX_AUTO_JUDGE_JEV_FALLBACK=1`, default): retry → fallback provider →
defer-all. The engine never acts on a probability alone.

---

## Admission engine

### Question set (`cortex_admission.2026-09-17-v3`)

Seven nouls per candidate: `worth_saving`, `durable`, `standalone`,
`useful_again`, `scope_clear`, `curated`, `duplicate_of_related`. Only
`worth_saving` drives the decision; the rest are recorded as review
annotations in the decision log. `scope_clear` and `duplicate_of_related` are
never hard rules — measured compound rules collapsed agreement.

### Policy (`cortex_admission_policy.2026-09-18.1`)

| Path | Applies to | admit ≥ | reject ≤ | note |
| --- | --- | --- | --- | --- |
| `default_p1` | everything else | 0.60 | 0.40 | the calibrated knee |
| `semantic_strict` | kind `semantic` | 0.65 | 0.35 | weakest kind; flagged for audit |
| `builtin_memory_review` | source_type `builtin_memory` | — | — | never auto-decided; defers to review, flagged for audit |

Inside the band → `defer` (the harness contract's escalation). Positive
feedback shifts the effective value toward keeping, exactly like the chat
engine's confidence boost (strong 0.08 / ordinary 0.02, reused from the
auto-judge config). Guards (quarantine, redaction, contradictions, blocking
quality flags) run **after** the engine and can still force `needs_context`.
Feedback-adjusted decisions are used only for reasons — thresholds are fixed.

A config/env comment in `cortex/jev.py` documents why `builtin_memory` is
held: 72.7% agreement on the September slice with disagreements interleaved at
every confidence level — threshold tuning cannot separate them, so the canary
routes them to review until the operator sets a policy.

### Fallback and failure

- Per chunk: Jev failure → chat provider judges the same chunk
  (`fallback_chunks` counter).
- `CORTEX_AUTO_JUDGE_JEV_FALLBACK=0`: a Jev failure rotates the paid-for
  batch and raises; nothing is applied (fail closed).
- One bad candidate never sinks a batch: missing/invalid answers for a
  candidate resolve to `defer` for that candidate only.

---

## Link judgments

Pairwise questions (`cortex_links.2026-09-18-v1`): per related memory, one
`l{i}` noul gate ("should the candidate be linked?") plus one `r{i}` choice
relation (`supports`/`extends`/`refines`/`example_of`/`generalizes`/
`prerequisite`/`contradicts`/`none`). Edge creation requires gate ≥
`CORTEX_JEV_LINK_THRESHOLD` (default **0.65**) and a non-`none` relation, capped
at `CORTEX_JEV_MAX_LINKS_PER_ITEM` (default 3) per item. The model cannot
invent memory ids: answers map back onto the supplied related list by index.

Validation (2026-09-18, 400 sampled pairs over the live graph):

| pair class | ≥ 0.65 |
| --- | --- |
| operator-created edges | 91% |
| auto-judge edges | 84% |
| random pairs | ~3% |

---

## Configuration

All knobs are environment variables (the auto-judge service reads
`~/.hermes/cortex/auto-judge.env`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `CORTEX_AUTO_JUDGE_ENGINE` | `chat` | `chat` or `jev` (admission engine) |
| `CORTEX_AUTO_JUDGE_LINK_ENGINE` | `chat` | `chat` or `jev` (orphan-link pass) |
| `CORTEX_AUTO_JUDGE_JEV_LINKS` | `1` | link suggestions for admitted candidates when the jev engine is on |
| `CORTEX_AUTO_JUDGE_JEV_FALLBACK` | `1` | fall back to the chat provider on Jev failure |
| `CORTEX_JEV_ENDPOINT` | `https://api.typesafe.ai/v1/systemone` | 
| `CORTEX_JEV_MODEL` | `jev-latest` | pin the versioned id measured here: `jev-1.13.0` |
| `CORTEX_JEV_API_KEY_ENV` | `TYPESAFE_API_KEY` | key name resolved from the env or credential file |
| `CORTEX_JEV_CREDENTIAL_FILE` | inherits the auto-judge file | dotenv file holding the key |
| `CORTEX_JEV_TIMEOUT_SECONDS` | `30` | per attempt (1–120) |
| `CORTEX_JEV_BATCH_SIZE` | `1` | candidates per request; > 1 switches to labeled questions |
| `CORTEX_JEV_CONCURRENCY` | `8` | parallel calls (≤ 16) |
| `CORTEX_JEV_MAX_ATTEMPTS` | `3` | retries on 429/5xx/timeouts |
| `CORTEX_JEV_LINK_THRESHOLD` | `0.65` | link gate |
| `CORTEX_JEV_MAX_LINKS_PER_ITEM` | `3` | cap per item |
| `CORTEX_JEV_DECISION_LOG` | `~/.hermes/cortex/jev-decisions.jsonl` | append-only JSONL; empty string disables |

### Decision log

One line per judgment: `question_set`, `policy`, `run_ref`, `proposal_id`,
`kind`, `source_type`, `path`, the seven answers, `worth_adjusted`, `action`,
`audit`, `model`, `latency_ms`. Link judgments log separately (`event:
links`). No raw memory content ever enters the log.

---

## Canary procedure

1. Set `CORTEX_AUTO_JUDGE_ENGINE=jev`, `CORTEX_JEV_MODEL=jev-1.13.0` in
   `auto-judge.env`. Leave `CORTEX_AUTO_JUDGE_LINK_ENGINE=chat` until the link
   canary is requested; `CORTEX_AUTO_JUDGE_JEV_LINKS=1` covers admission-time
   links.
2. Watch for one week: defer rate by path, operator override rate on
   Jev-decided rows, weekly disagreement audit vs recorded expectations, cost
   per day (usage is summed into the run report under `jev.usage`), latency
   p50/p95 (`jev.latency_p50_ms`).
3. Rollback: revert the env lines. No data migration; decisions already made
   keep their actor (`cortex-auto-judge:<model>`) in the ledger.

### Evidence (2026-09-17/18, read-only)

- Save-side replay v1–v3: 89.4% → 93.6% binary agreement vs the recorded
  judge; the production engine (category paths + boosts) measures **96.3%**
  agreement on the current-model slice at 29% defer (662-row corpus replay,
  p50 333 ms, ~$0.03 per full corpus pass, 20 s wall at 12 workers).
- Test–retest stability: 59/60 verdict-band stable, `worth_saving` mean |Δ|
  0.013.
- Link replay: see the table above.

### Known limits

- The Jev engine does not emit `evidence_only` (admissions go to `remember`);
  this is deliberate in v1 and revisited with operator data.
- Batch sizes > 1 use labeled questions (`c1_worth_saving`, …) — supported
  and unit-tested, but the corpus numbers above were measured at batch size 1
  (the default).
- The consolidation question set (pairs → merge/keep/link) is **not** final:
  first-pass mapping over-links. Consolidation stays on the chat pass until a
  calibrated question set exists.
