# Recall performance and background warming

Cortex should get faster without changing which evidence it trusts. Performance
changes preserve correction history, recall-set eligibility, context gates,
provenance, outcome attribution, and reversible maintenance. A speculative
memory is never evidence that the agent used it.

## Current implementation

- Recall scoring prepares query tokens, context maps, entity/system/version
  sets, and the specificity signal once per search.
- Context outcome feedback, graph-neighbor reads, and eligibility checks use
  bounded batches instead of one SQL query per candidate.
- Feature search precomputes query-feature frequency penalties. Schema 34
  upgrades the existing posting index to `(feature,memory_id,weight)`, allowing
  SQLite to read posting weights directly from the index. No memory or feature
  rows are rewritten. Opening an older database upgrades the index once.
- Core and Hermes recall ledger writes share an outer transaction. Every event
  remains recorded. The core's later `RecallBatch.context()` call records actual
  rendering in its own transaction; speculative warming cannot record it.
- Inactive research experiments use a read-only probe instead of an empty write
  transaction on every recall. Active experiments still assign under a writer
  transaction and recheck the experiment.
- The dashboard reuses summaries already returned by `dashboard_snapshot()`.
  Five summaries previously ran twice per full refresh.
- Exact recall caching is available to every Python harness. Query, complete
  context, retrieval options, semantic settings, TTL, and the store revision
  govern reuse. Corrections, outcome learning, eligibility changes, approved
  scoring profiles, and external database writes invalidate relevant results.
  Cache hits report lookup timing rather than replaying old retrieval timings.

The schema-34 index is readable by older Cortex versions. A rollback must still
follow the normal database backup procedure in [QUICKSTART.md](QUICKSTART.md);
opening the database with an old checkout is not a complete rollback protocol.

## Background worker

This is a local Python worker in the agent process. It does not install a browser
service worker, persist browser caches, upload data, or call a remote judge.
Normal local semantic fusion can run during warming if the operator has already
enabled it and installed the local model.

Core callers can provide an upcoming task hint while other preparation runs:

```python
from cortex import CortexMemory

memory = CortexMemory("/path/to/cortex.db")
memory.preload("What backup is required before deploying Juniper?",
               active_project="Juniper", task_type="deployment")
# Prepare tools or project state while the local worker warms recall.
batch = memory.recall("What backup is required before deploying Juniper?",
                      active_project="Juniper", task_type="deployment")
context = batch.context()
# Run inference, resolve attributed IDs and outcome, then close normally.
batch.finish()
print(memory.preload_stats())
memory.close()
```

`CortexHarnessAdapter.preload()` uses the same adaptive plan as `before_turn()`
and skips greeting-only turns. Supply the same task and context options to both.
A new project, version, query, budget, or database revision causes fresh recall.

Hermes warming is an experiment, disabled by default. To try it, set
`plugins.cortex.background_preload: true` in the Hermes configuration. Its
`queue_prefetch` hook warms an upcoming query. With the separate experimental
`background_preload_continuity: true` setting, after outcome sync it also warms
the most recent task as a simple continuity prediction using the new feedback
revision. This does not predict arbitrary future wording: a paraphrased next
query normally misses the exact cache and receives a fresh ranking.

The worker starts lazily. It has one active job and one pending job; a newer
hint replaces a pending one. Queries over 4,096 characters, limits above 20,
budgets above 4,000 tokens, graph depths above two, and oversized context maps
are declined. The cache holds at most 32 entries by default, with a 45-second TTL
and a maximum TTL of 300 seconds. It is in-memory only. `cache_ttl_seconds=0`
disables core caching and warming; Hermes uses `query_cache_ttl_seconds=0`.
Close cancels pending work and joins the active read before closing SQLite.

Warming writes no memories, usage, accesses, traces, episodes, outcomes, or
training labels. Foreground recall still records its own task, selection,
rendering, and eventual outcome. Failed warming jobs increment an aggregate
counter and leave normal recall available.

`preload_stats()` exposes aggregate cache hits, preload hits, revision discards,
completed/replaced/failed jobs, warming duration, and the last full foreground
preparation time including its commit. It exposes no query text or memory IDs.
The persisted `prepare_ms` is measured before the final trace and commit; use
full-call benchmark timing for end-to-end preparation claims.

A worker moves work earlier; it does not eliminate that work. It can compete
with foreground reads for CPU and the store lock. Enable it only when task hints
arrive early enough and useful hits justify the added work. Measure varied
traffic, misses, and normal outcome sync as well as repeated queries.

## Repeatable evaluation

Run from the checkout, with output and reusable fixtures outside the repository:

```bash
python3 scripts/benchmark_recall.py --size 2000 --repetitions 30 \
  --fixture /tmp/cortex-synthetic-fixture.db --output /tmp/cortex-candidate.json
python3 scripts/compare_retrieval_reports.py \
  --baseline /tmp/cortex-baseline.json --candidate /tmp/cortex-candidate.json
```

The fixture is accepted only when its memory content exactly matches the
specified synthetic corpus. The runner measures varied raw, core, and Hermes
recalls, includes normal outcome resolution outside the timed preparation, and
reports foreground latency, stage timing, durable commits, synthetic hit@k, and
warming cost. Warm conditions receive the upcoming query explicitly and finish
warming before recall: they measure an upper bound for an accurate early hint,
not real prediction hit rate or a model-inference speedup. Warm and uncached
conditions must select the same memories; a mismatch fails the runner. Hint-only
warming is measured; continuity prediction is not enabled in these conditions.

Compare on the same host, filesystem, corpus, seed, and settings, with at least
20 repetitions. The comparator rejects incompatible settings and fails on a
quality regression, selection mismatch, or a latency regression beyond the
chosen tolerance (20% by default). It is a review aid; timing noise still needs
investigation. CI runs correctness checks and prints an aggregate measurement report in its
logs without enforcing a cross-machine timing threshold.

For representative quality, label private real-history cases and run
[scripts/evaluate_real_history.py](../scripts/evaluate_real_history.py) as
explained in [EVALUATION.md](EVALUATION.md). The same comparator accepts two
sanitized reports from that runner and checks hit@k, recall, precision, MRR,
and latency. Use the same labeled cases and a fixed database snapshot in both
checkouts. At least eight cases are required; that floor is not sufficient for
a broad production claim. Never commit private labels, queries, or databases.

## Validate a live brain on its own host

For the operator-run CT117 procedure, including a separate Hermes preload trial
with useful hits, added process CPU, and p95, use
[cortex-ct117-validation.md](../cortex-ct117-validation.md). The development agent
does not connect to CT117; only the aggregate handoff report returns.

When the labels of record are active `evaluation_cases` rows, regenerate JSONL
on the brain host. Keep the database snapshot and labels outside every checkout,
with owner-only permissions. Do not copy either to a development machine or
paste their contents into agent context. The following procedure reads the live
SQLite database through a read-only connection and evaluates a consistent backup.

Set `BASELINE_CORTEX_ROOT` to the exact currently deployed package, including
any local adapter changes, and `CANDIDATE_CORTEX_ROOT` to the reviewed checkout.
The runner is copied into isolated import directories so both conditions use
the same evaluation code with their respective Cortex implementations. This
also works when the deployed plugin does not ship the real-history runner.

```bash
export HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
export BASELINE_CORTEX_ROOT="$HERMES_HOME/plugins/cortex"
export CANDIDATE_CORTEX_ROOT="/path/to/reviewed/cortex-memory"
umask 077
export RUN_DIR="$(mktemp -d "$HOME/cortex-private-validation.XXXXXXXX")"

python3 - <<'PY'
import json
import os
import sqlite3
from pathlib import Path

run = Path(os.environ["RUN_DIR"])
live = (Path(os.environ["HERMES_HOME"]) / "cortex/cortex.db").resolve()
source = sqlite3.connect(f"{live.as_uri()}?mode=ro", uri=True, timeout=5)
snapshot = sqlite3.connect(run / "brain-snapshot.db")
try:
    source.backup(snapshot)
    rows = snapshot.execute(
        "SELECT case_id,query,relevant_memory_ids,task_type "
        "FROM evaluation_cases WHERE active=1 ORDER BY case_id"
    ).fetchall()
    if len(rows) < 8:
        raise SystemExit("Refusing evaluation: fewer than eight active labels")
    labels = []
    for case_id, query, encoded_ids, task_type in rows:
        ids = json.loads(encoded_ids)
        if not isinstance(ids, list) or not ids or not all(
            isinstance(memory_id, str) and memory_id for memory_id in ids
        ):
            raise SystemExit("Refusing evaluation: invalid relevance labels")
        for memory_id in ids:
            if snapshot.execute("SELECT 1 FROM memories WHERE id=?", (memory_id,)).fetchone() is None:
                raise SystemExit("Refusing evaluation: a labeled memory is missing")
        labels.append(dict(schema_version=1, case_id=case_id, query=query,
                           relevant_memory_ids=ids, group=task_type or "general"))
    with (run / "retrieval-labels.jsonl").open("x", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label, ensure_ascii=False) + "\n")
    print(f"Prepared private snapshot and {len(labels)} labels on this host")
finally:
    snapshot.close()
    source.close()
PY

mkdir -p "$RUN_DIR/baseline/scripts" "$RUN_DIR/candidate/scripts"
ln -s "$BASELINE_CORTEX_ROOT" "$RUN_DIR/baseline/cortex"
ln -s "$CANDIDATE_CORTEX_ROOT" "$RUN_DIR/candidate/cortex"
for condition in baseline candidate; do
  cp "$CANDIDATE_CORTEX_ROOT/scripts/evaluate_real_history.py" \
     "$RUN_DIR/$condition/scripts/evaluate_real_history.py"
  PYTHONPATH="$RUN_DIR/$condition" python3 \
    "$RUN_DIR/$condition/scripts/evaluate_real_history.py" \
    --db "$RUN_DIR/brain-snapshot.db" --labels "$RUN_DIR/retrieval-labels.jsonl" \
    --output "$RUN_DIR/$condition.json" --policy adaptive --top-k 6 --token-budget 700
done
python3 "$CANDIDATE_CORTEX_ROOT/scripts/compare_retrieval_reports.py" \
  --baseline "$RUN_DIR/baseline.json" --candidate "$RUN_DIR/candidate.json" \
  --output "$RUN_DIR/comparison.json"
```

Run with the same Python environment and model settings as the deployed brain.
The evaluator makes a disposable copy for each condition, so schema migration
cannot change the frozen source snapshot or live brain. Run the comparison again
in reverse order if a latency result is close to the gate. Review only the
sanitized aggregates when deciding whether to install; 28 cases, for example,
are useful regression evidence but remain a small sample. This runner measures
retrieval, not model answer quality or predictive-preload hit rate.

Before installation, separately back up the deployed plugin and configuration;
the installer also backs up its previous plugin. Keep
`background_preload` and `background_preload_continuity` disabled for the initial
rollout. The installer never restarts Hermes: follow the operator's restart
procedure, then run the audit and behavior checks in [AGENTS.md](../AGENTS.md).
Retain the private snapshot and previous plugin for rollback. Trial warming
separately after foreground quality and latency pass.

## Experiments that can improve Cortex next

1. **Measure prediction usefulness before adding a predictor.** Compare explicit
   harness hints, task continuity, and a no-worker control. Track useful cache
   hits, CPU work, p95 foreground latency, and retrieval quality. A future
   predictor can learn from independently labeled successful tasks; speculation
   itself must never create positive training examples.
2. **Explore shared candidate preparation across paraphrases.** Cache reusable
   postings or decoded records, then score each actual query against its own
   context and current outcomes. Start with shadow comparisons; blindly reusing
   the previous task's selected memories can miss corrections or new evidence.
3. **Use context only when it helps.** Compare adaptive budgets and abstention
   with fixed recall on paired tasks. Fewer irrelevant tokens can save more
   total agent time than a faster local lookup, but requires model-facing tests.
4. **Detect drift and retain rollback.** Rerun fixed synthetic cases plus private
   representative cases before promoting retrieval, embedding, or Sleep policy
   changes. Keep latency and quality together. Continue shadow maintenance until
   pruning-regret and task-outcome evidence justify an apply experiment.

These experiments are not automatically enabled or promoted by a passing local
microbenchmark. See [ROADMAP.md](ROADMAP.md) for the broader release gates.
