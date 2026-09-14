# The Validity Judge

**Status:** design — grounded in live measurement, 2026-09-10
**Scope:** the confidence/validity brain behind memory retention, decay, and pruning
**Roadmap home:** 0.4 (Memory structure and calibration) → 0.6 (Lifecycle and self-repair)

---

## 1. What this is

A judge that answers three questions about every memory, continuously:

1. **Is it true?** (validity)
2. **Does it still matter?** (usefulness)
3. **Is it still current?** (freshness)

…and then, on that basis, decides whether a memory is kept active, cooled,
archived, or tombstoned.

The user's framing: *re-evaluate every time a memory is recalled; if it falls
below a relevance floor, prune or archive it — but if it was useful for a long
time, never prune, archive it for possible future recall.*

That framing is correct, and it maps onto ROADMAP 0.4 almost line for line.
The problem is not the shape of the idea. It is that **there is almost nothing
for the judge to reason with.**

---

## 2. Measured baseline (live, read-only)

Produced by `scripts/validity_probe.py` against `~/.hermes/cortex/cortex.db`.

### 2.1 The evidence supply

| Signal | Events | Memories |
|---|---|---|
| retrieved | 10,121 | 2,219 |
| used | 2,279 | 935 |
| helpful | 26 | 24 |
| harmful | 8 | 8 |
| confirmed | 10 | 10 |
| validated | 2 | 2 |
| corrected | 8 | 8 |
| false positive | 16 | 16 |
| **total truth events** | **70** | — |

**70 truth events across 3,590 memories.** Any judge trained on this will fit
noise. This is the binding constraint, not the algorithm.

### 2.2 `confidence` is a provenance lookup, not a measurement

`confidence` is set once at write time as a function of `source_category` and
never updated by anything:

| source_category | rows | avg confidence | range |
|---|---|---|---|
| DOCUMENT_EXTRACTED | 2,986 | 0.799 | 0.78–0.82 |
| TOOL_VERIFIED | 139 | 0.960 | 0.96–0.96 |
| AGENT_INFERENCE | 29 | 0.582 | 0.58–0.65 |

66 distinct values across 3,590 rows. So today, "how confident are you?"
answers *"where did this come from?"* — not *"is it true?"*

**Additional defect:** a memory the operator explicitly created
(`OPERATOR_APPROVED`, 0.76 in the retrieval trust table) is trusted **less**
than a vault chunk scraped off disk (`DOCUMENT_EXTRACTED`, 0.82). That
ordering is backwards and should be corrected.

### 2.3 The observation counters do not discriminate

```
retrieved = selected = injected = 10,121     (identical)
used                            = 2,279      (22.5%)
```

Three counters that should measure different things are one counter. The store
cannot currently distinguish *"Cortex offered this"* from *"the agent used
this."* Any precision or validity metric built on these is circular until they
are separated.

### 2.4 Signals that carried real information

**Opportunity-adjusted use.** Of 507 memories offered ≥5 times:

| bucket | count |
|---|---|
| offered but never used | 140 |
| used 1–24% | 180 |
| used 25–49% | 124 |
| used 50%+ | 63 |

This is free, zero-AI, and genuinely discriminative.

**Invisible rot.** 1,371 memories have never been retrieved: 434 still active,
932 already archived, 64 with importance ≥0.70. *A recall-triggered judge never
sees these.* This is the structural blind spot in the user's original framing.

**Source staleness.** 1,358 active vault chunks; **561 (41%)** have a source
file modified more than an hour after ingest (e.g. `Docker.md` read 2026-07-07,
modified 2026-09-08). Nothing tracks this today.

**Protection coverage.** 285 active memories match high-stakes keywords, but
274 of those are reference chunks that merely *mention* a protected word. Among
genuine personal memories the real exposure is **1** — an `OPERATOR_APPROVED`
memory at importance 0.90 that is neither `protected` nor `pinned`. The
protection system is working; it has one gap.

### 2.5 Signals that FAILED measurement

**Contradiction scan over stored triples: 0 conflicts found.**

3,206 memories carry `subject`/`predicate`/`object_value`, but 2,986 (98.5%) use
a single placeholder predicate — `documents` — with the subject being the
chunk's own source ref. These are ingest bookkeeping, not semantic claims:

```
subj=vault:Church.md#church-1-p1   pred=documents   obj=Church
```

Real semantic predicates exist but are rare: `works_for` 61, `fails_for` 17,
`execution_outcome` 139, `prefers` 1.

**Conclusion: a contradiction scanner is not worth building until extraction
produces comparable triples.** The bottleneck is extraction quality, not
conflict logic. `works_for` / `fails_for` is the one live surface where a
contradiction could meaningfully collide.

**Naive mtime staleness is also unsound as a content check.** Comparing file
mtime to `observed_at` flags same-second ingest writes as changes. The correct
check already exists in the ingest path — `vault.py` compares the stored chunk
digest against a freshly computed one — and should be surfaced rather than
reinvented.

---

## 3. Design

### 3.1 Three axes, never one score

| Axis | Nature | Can we act on it? |
|---|---|---|
| **Truth** | objective, near-binary | only with external evidence |
| **Usefulness** | subjective, time-bound | yes, from use-rate |
| **Freshness** | objective, time-bound | yes, from source digest |

Collapsing these into a single "relevancy score" produces the wrong action,
because they disagree in ways that matter:

- *true + useless* → archive (safe)
- *false + useful* → dangerous; this is exactly the case that must be surfaced
- *true + stale* → re-verify, do not delete
- *false + never used* → tombstone

### 3.2 Two speeds

**Speed 1 — on recall (microseconds, no AI, no network)**

- decay strength against elapsed time (exists: `_decayed_strength`, 0.995/day)
- opportunity-adjusted use rate (built: `scripts/validity_probe.py`)
- source digest comparison (exists: `vault.py` ingest idempotency check)
- protection / pin / immune-class check

Cheap arithmetic only. The recall path currently costs ~116 ms end to end and
must not regress.

**Speed 2 — offline, nightly, batched (Local Sleep)**

- re-observe the world where a claim is mechanically checkable
  (e.g. *"Plex is on 32400"* → probe the port; *"CT100 runs X"* → query Proxmox)
- resolve contradictions that survived extraction
- run provider-backed judgement only on what the cheap tier could not settle
- sweep memories that **no recall has ever touched** (§2.4)

The second bullet is the closest analogue to how biological memory actually
updates: **prediction error.** You act on a memory, the world disagrees, the
trace is revised. Expectation alone never confirms anything.

### 3.3 Evidence ladder (cheapest → most expensive)

| Tier | Evidence | Cost | Status |
|---|---|---|---|
| 0 | opportunity-adjusted use | free, local | **built** |
| 0 | source digest change | free, local | exists in ingest, needs surfacing |
| 0 | operator correction | free | wired (`correct`, −0.10 strength) |
| 1 | protection / immune class | free | schema exists, needs rules |
| 1 | mechanical re-observation (port, service, file) | one command | to build |
| 2 | contradiction on semantic triples | free but blocked | blocked on extraction |
| 3 | provider judgement on ambiguity | LLM call | exists, gated, barely runs (49 runs) |

### 3.4 Rules the judge must obey

1. **Confidence may never rise from retrieval alone.** Repetition can only stop
   decay; it cannot create validity. (Already a stated ROADMAP safety rule.
   Must be enforced in the formula, not just documented.)
2. **Never prune on frequency alone.** Rare-but-critical memories — credentials,
   medical, legal, emergency — are the exact things a usage-counting judge
   deletes first.
3. **Use only counts against opportunity.** `used / offered`, not raw `used`.
4. **Spacing beats count.** Ten reflections in one afternoon is one event, not
   ten confirmations.
5. **Severity beats frequency.** One costly failure must outweigh many mild
   successes (the burned-hand rule).
6. **Deletion is the only irreversible action.** It requires simulation and a
   measured regret ceiling first; archive is always the default.
7. **Overrides are training data.** An operator overturning a judgement must be
   recorded, or the judge cannot learn.

---

## 4. Open decisions (awaiting operator)

| # | Decision | Recommendation |
|---|---|---|
| 1 | Can repeated usefulness raise confidence that a memory is **true**? | **No** — only external evidence raises validity; repetition only resists decay |
| 2 | Which memories are immune to decay? | Fixed class (credentials, medical, legal, emergency, identity) **plus** manual pinning |
| 3 | What may the judge do unprompted? | Auto-**archive** (fully reversible); **ask** before any delete |
| 4 | Memory found false after already being acted on? | **Tell the operator**, with the date range it was in use |
| 5 | Re-judge memories that were never recalled? | **Yes** — scheduled sweep, cheapest checks first |

---

## 5. Non-goals

- No new embedding backend or vector index.
- No automatic rewrite or merge of memory content (governance change only).
- No deletion path in the first release; archive is reversible by construction.
- No training on retrieval counts (`§3.4.1`).
- No remote or LLM call on the recall path.

---

## 6. Verification

`scripts/validity_probe.py --json` is the instrumentation of record for this
workstream. It is read-only, dependency-free, and re-runnable:

```bash
python3 scripts/validity_probe.py          # human report
python3 scripts/validity_probe.py --json   # machine-readable
```

Any change to decay, retention, or protection rules must show a corresponding
movement in this report before it is considered to have worked.
