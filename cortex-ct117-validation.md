# CT117 Cortex validation — Hermes

The preload implementation is landed on `testing` in merge `ec34bd1` (PR #4).
Use the current reviewed `testing` checkout for this procedure and its trial
runner. `main` is still the stable branch; it does not contain these changes yet.

**The operator runs this file on CT117. The development agent does not execute
it, connect by SSH, or receive a database, labels, queries, or memory IDs.**
Return only `handoff-aggregates.json`, reviewed locally before sharing.

This procedure does not install over the live plugin, restart Hermes, change its
configuration, enable timers, invoke a model/judge, or apply maintenance. Initial
retrieval evaluation has preloading off. The subsequent preload trial runs the
Hermes adapter against disposable database copies; live preloading stays off.

## 1. Set paths on CT117

Use the Python environment used by the deployed Cortex. Point the candidate at
the reviewed checkout and the baseline at the exact deployed package, including
any local changes. Keep that baseline unchanged throughout the comparison.

```bash
set -euo pipefail
umask 077
export HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
export CORTEX_DB="$HERMES_HOME/cortex/cortex.db"
export BASELINE_CORTEX_ROOT="$HERMES_HOME/plugins/cortex"
export CANDIDATE_CORTEX_ROOT="/path/to/reviewed/cortex-memory"
export CORTEX_VALIDATION_DIR="$(mktemp -d "$HOME/cortex-private-validation.XXXXXXXX")"
test -f "$CANDIDATE_CORTEX_ROOT/preload.py"
test -f "$CANDIDATE_CORTEX_ROOT/scripts/evaluate_preload.py"
```

On CT117, the root user's default `HERMES_HOME` resolves to the live brain's
location. Keep the validation directory outside all checkouts. No private file
or log from it belongs in this public repository or agent/model context.

## 2. Snapshot the brain and regenerate JSONL from `evaluation_cases`

The active database rows are the label source of record. No existing JSONL is
assumed. This reads the live database through a read-only connection and uses
SQLite backup to include committed WAL data consistently. Labels come from that
same snapshot. Both the snapshot and JSONL remain on CT117.

```bash
python3 - <<'PY'
import json
import os
import sqlite3
from pathlib import Path

run = Path(os.environ['CORTEX_VALIDATION_DIR'])
live = Path(os.environ['CORTEX_DB']).expanduser().resolve()
source = sqlite3.connect(live.as_uri() + '?mode=ro', uri=True, timeout=5)
snapshot = sqlite3.connect(run / 'brain-snapshot.db')
try:
    source.backup(snapshot)
    rows = snapshot.execute(
        'SELECT case_id,query,relevant_memory_ids,task_type '
        'FROM evaluation_cases WHERE active=1 ORDER BY created_at,case_id'
    ).fetchall()
    if len(rows) < 8:
        raise SystemExit('Refusing evaluation: fewer than eight active labels')
    labels = []
    seen = set()
    for case_id, query, encoded_ids, task_type in rows:
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise SystemExit('Refusing evaluation: invalid or duplicate case ID')
        if not isinstance(query, str) or not query.strip():
            raise SystemExit('Refusing evaluation: invalid query')
        seen.add(case_id)
        ids = json.loads(encoded_ids)
        if not isinstance(ids, list) or not ids or not all(
            isinstance(value, str) and value for value in ids
        ):
            raise SystemExit('Refusing evaluation: invalid relevance labels')
        ids = list(dict.fromkeys(ids))
        for memory_id in ids:
            if snapshot.execute('SELECT 1 FROM memories WHERE id=?', (memory_id,)).fetchone() is None:
                raise SystemExit('Refusing evaluation: a labelled memory is missing')
        # Omit operator-authored group names from this handoff.
        labels.append(dict(schema_version=1, case_id=case_id, query=query,
                           relevant_memory_ids=ids))
    with (run / 'retrieval-labels.jsonl').open('x', encoding='utf-8') as handle:
        for label in labels:
            handle.write(json.dumps(label, ensure_ascii=False) + '\n')
    print(f'Prepared private snapshot and {len(labels)} active labels')
finally:
    snapshot.close()
    source.close()
PY
```

Use the actual active case count. Stop below eight, on invalid labels, or if any
target is missing. Do not use `--allow-small-sample` or weaken the missing-ID
checks. The eight-case floor is a minimum gate, not representative evidence.

## 3. Retrieval quality and median/tail latency, preloading off

The real-history runner uses ordinary retrieval, not the preload worker. Both
conditions use the same evaluator code, labels and frozen snapshot, importing
their respective Cortex implementations. Disposable evaluator copies absorb
schema migrations; neither the frozen snapshot nor live brain is migrated.

```bash
mkdir -p "$CORTEX_VALIDATION_DIR/baseline/scripts" "$CORTEX_VALIDATION_DIR/candidate/scripts"
ln -s "$BASELINE_CORTEX_ROOT" "$CORTEX_VALIDATION_DIR/baseline/cortex"
ln -s "$CANDIDATE_CORTEX_ROOT" "$CORTEX_VALIDATION_DIR/candidate/cortex"
for condition in baseline candidate; do
  cp "$CANDIDATE_CORTEX_ROOT/scripts/evaluate_real_history.py" \
     "$CORTEX_VALIDATION_DIR/$condition/scripts/evaluate_real_history.py"
  PYTHONPATH="$CORTEX_VALIDATION_DIR/$condition" python3 \
    "$CORTEX_VALIDATION_DIR/$condition/scripts/evaluate_real_history.py" \
    --db "$CORTEX_VALIDATION_DIR/brain-snapshot.db" \
    --labels "$CORTEX_VALIDATION_DIR/retrieval-labels.jsonl" \
    --output "$CORTEX_VALIDATION_DIR/$condition.json" \
    --policy adaptive --top-k 6 --token-budget 700
done
python3 "$CANDIDATE_CORTEX_ROOT/scripts/compare_retrieval_reports.py" \
  --baseline "$CORTEX_VALIDATION_DIR/baseline.json" \
  --candidate "$CORTEX_VALIDATION_DIR/candidate.json" \
  --output "$CORTEX_VALIDATION_DIR/comparison.json"
```

For the requested direct live-brain check, run the same parameters against
`CORTEX_DB`. That evaluator also creates a disposable copy before opening Cortex:

```bash
PYTHONPATH="$CORTEX_VALIDATION_DIR/candidate" python3 \
  "$CORTEX_VALIDATION_DIR/candidate/scripts/evaluate_real_history.py" \
  --db "$CORTEX_DB" --labels "$CORTEX_VALIDATION_DIR/retrieval-labels.jsonl" \
  --output "$CORTEX_VALIDATION_DIR/live-candidate.json" \
  --policy adaptive --top-k 6 --token-budget 700
```

Concurrent live changes can invalidate labels after snapshot creation; a missing
target remains a stop condition. Use the frozen pair for baseline/candidate
claims. Review hit@k, recall, precision, MRR and p50/p95. If comparison fails,
stop before the preload trial. For timing near its gate, repeat the paired run
in reverse order with new report names; retain all measurements.

## 4. Separate preload trial: useful hits, added CPU, p95

After step 3 passes, replay the private labels on isolated copies with the
Hermes worker disabled and enabled. Ordinary exact caching is enabled equally
in both conditions. The default continuity predictor warms the preceding task
after normal outcome sync. It never receives the upcoming query as a hint.
A 50 ms pause occurs before each foreground request in both conditions. Active
warming may contend with foreground work; that contention affects measured p95.

```bash
for suffix in forward reverse; do
  order_flags=()
  if [[ "$suffix" == reverse ]]; then order_flags=(--reverse-order); fi
  if ! python3 "$CANDIDATE_CORTEX_ROOT/scripts/evaluate_preload.py" \
    --db "$CORTEX_VALIDATION_DIR/brain-snapshot.db" \
    --labels "$CORTEX_VALIDATION_DIR/retrieval-labels.jsonl" \
    --output "$CORTEX_VALIDATION_DIR/preload-$suffix.json" \
    --hint-mode continuity --lead-ms 50 --top-k 6 --token-budget 700 \
    "${order_flags[@]}"; then
    echo 'Trial failed; inspect its aggregate report locally before considering promotion.'
  fi
done
```

Each report contains aggregate-only:

- control and preload p50/p95 foreground preparation latency;
- cache/preload hits and **label-relevant preload hits** (a warm-cache hit that
  selected at least one labelled relevant memory, not credited model use);
- full-process CPU milliseconds in each condition, including normal foreground
  and outcome bookkeeping and active worker drain;
- **added CPU**, signed preload-minus-control milliseconds, total and per request;
- selected-memory parity, recall/precision/MRR, and worker failures.

Warming wall time is not CPU cost. A negative CPU delta can be measurement noise;
retain both condition orders rather than clipping it or choosing the best run.
Zero useful hits is evidence against enabling this predictor. Consider quality,
misses, added CPU and p95 together; the runner never activates a setting.

This is **offline label replay**, not live traffic. Active evaluation cases are
selected examples, and their creation order is not a complete conversation
stream. It tests the continuity mechanism but cannot establish production hit
rate or model task success. Semantic fusion and adaptive budget learning are
disabled identically for this controlled replay. Production settings need their
own later measurement. If accurate external task hints are available, optional
`--hint-mode oracle` measures an upper bound by revealing the upcoming query and
waiting for warming; never report oracle hit rate as prediction success.

## 5. Build and return only `handoff-aggregates.json`

Run this once after the initial evaluation, or after the separate trials if
step 3 passes. The private detailed evaluator reports remain on CT117. The
handoff strips per-case numeric rows as well as all private text and paths.
If you returned the initial retrieval handoff before running trials, retain it
locally under another filename before building the combined handoff; the
exporter refuses to overwrite an existing file.

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ['CORTEX_VALIDATION_DIR'])
aggregate = {'evaluation': 'cortex-host-validation-handoff', 'conditions': {}, 'preload_trials': {}}
for name in ('baseline', 'candidate', 'live-candidate'):
    report = json.loads((run / (name + '.json')).read_text())
    if report.get('privacy', {}).get('raw_private_text_omitted') is not True:
        raise SystemExit('Refusing handoff: retrieval privacy self-check failed')
    if report['summary']['cases'] < 8:
        raise SystemExit('Refusing handoff: fewer than eight evaluated cases')
    aggregate['conditions'][name] = {
        'settings': report['reproducibility'],
        'summary': report['summary'],
        'sample_size': report['sample_size'],
    }
aggregate['comparison'] = json.loads((run / 'comparison.json').read_text())
for order in ('forward', 'reverse'):
    path = run / ('preload-' + order + '.json')
    if not path.exists():
        continue
    report = json.loads(path.read_text())
    privacy = report.get('privacy', {})
    if privacy.get('raw_private_text_omitted') is not True or privacy.get('per_case_rows_omitted') is not True:
        raise SystemExit('Refusing handoff: preload privacy self-check failed')
    if report['settings']['case_count'] != aggregate['conditions']['candidate']['settings']['case_count']:
        raise SystemExit('Refusing handoff: preload case count differs')
    aggregate['preload_trials'][order] = report
with (run / 'handoff-aggregates.json').open('x', encoding='utf-8') as handle:
    json.dump(aggregate, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write('\n')
print('Aggregate handoff prepared; review locally before sharing')
PY
```

Hermes/operator reviews that one file locally and returns **only
`handoff-aggregates.json`**. If a stage fails before a report is written, return
only its stage and a sanitized reason/count. Do not return private labels,
databases, queries, IDs, group names, raw logs or stack traces. Retain private
inputs and detailed reports on CT117 for local audit. Installation, restart and
live rollout are separate operator steps after review.
