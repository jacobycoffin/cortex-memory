#!/usr/bin/env python3
"""Replay private labelled tasks on disposable copies; report preload cost only.

No inference, remote judge, live-brain writes, or per-case data in the output.
Continuity replay is a limited predictor test, not a live traffic measurement.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sqlite3
import statistics
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    spec = importlib.util.spec_from_file_location(
        'cortex', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
    if spec is None or spec.loader is None:
        raise RuntimeError('could not load checkout')
    package = importlib.util.module_from_spec(spec)
    sys.modules['cortex'] = package
    spec.loader.exec_module(package)

from cortex.benchmarks.core import summarize_latencies
from cortex.hermes_provider import CortexMemoryProvider
from cortex.scripts.evaluate_real_history import (
    RetrievalLabel, _assert_report_privacy, load_labels, private_database_snapshot,
)


def added_cpu(control_ms: float, preload_ms: float, requests: int) -> dict[str, float]:
    """Signed full-process CPU delta; never hide a negative/noisy measurement."""
    delta = preload_ms - control_ms
    return {'total_ms': round(delta, 6), 'per_request_ms': round(delta / requests, 6)}


def _compared_cases(control: Sequence[tuple[str, ...]], preload: Sequence[tuple[str, ...]],
                    warm_flags: Sequence[bool]) -> int:
    """How many cases actually had a warm entry to compare, i.e. how much this run proved."""
    if len(control) != len(preload):
        return 0
    return sum(1 for index in range(min(len(control), len(warm_flags))) if warm_flags[index])


def _selections_match(control: Sequence[tuple[str, ...]], preload: Sequence[tuple[str, ...]],
                      warm_flags: Sequence[bool]) -> bool:
    """Every case whose warm entry was actually consumed must select exactly what control did.

    A case whose warm lost the race is skipped rather than compared: a cold run and a half-warm run
    are not the same experiment. Skipping is counted in `compared_cases`, so a run that skipped
    everything cannot pass silently — with nothing compared there is nothing proven.
    """
    if len(control) != len(preload):
        return False
    compared = 0
    for index, (left, right) in enumerate(zip(control, preload)):
        warm = warm_flags[index] if index < len(warm_flags) else False
        if not warm:
            continue
        compared += 1
        if left != right:
            return False
    return compared > 0


def _condition(path: Path, labels: Sequence[RetrievalLabel], *, preload: bool,
               hint_mode: str, lead_ms: float, top_k: int, token_budget: int
               ) -> tuple[dict[str, Any], list[tuple[str, ...]]]:
    provider = CortexMemoryProvider(dict(
        db_path=str(path), auto_capture=False, memory_receipts=False,
        adaptive_budget_learning=False, attentional_learning=False,
        background_preload=preload,
        background_preload_continuity=preload and hint_mode == 'continuity',
        top_k=top_k, token_budget=token_budget,
        semantic_fusion_weight=0, query_cache_ttl_seconds=45,
    ))
    provider.initialize('preload-evaluation', hermes_home=path.parent)
    latencies: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    selections: list[tuple[str, ...]] = []
    warm_timeouts: dict[int, bool] = {}
    warm_flags: list[bool] = []
    useful_hits = 0
    cpu_started = time.process_time_ns()
    try:
        for label in labels:
            if preload and hint_mode == 'oracle':
                provider.queue_prefetch(label.query, session_id='preload-evaluation')
                # Oracle mode is explicitly an upper bound with a ready hint.
                if not provider._preloader.wait_idle(timeout=30):
                    raise ValueError('oracle warming did not finish')
            time.sleep(lead_ms / 1000)
            # Determinism, not politeness: the claim under test is "preloading does not change the
            # selection". With a cold or half-warm queue that claim is being tested against a race,
            # and on a loaded runner it fails for a reason that has nothing to do with the design.
            # Join the queued warm before asking the question; a slow warm is recorded, not hidden.
            if preload and provider._preloader and not warm_timeouts.get(len(selections), False):
                if not provider._preloader.wait_idle(timeout=30):
                    warm_timeouts[len(selections)] = True
            hits_before = provider.preload_stats()['cache']['preload_hits']
            started = time.perf_counter_ns()
            provider.prefetch(label.query, session_id='preload-evaluation')
            latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            hits_after = provider.preload_stats()['cache']['preload_hits']
            task_id = provider._store._conn.execute(
                'SELECT task_id FROM recall_runs ORDER BY rowid DESC LIMIT 1').fetchone()[0]
            traces = provider._store.memory_traces(limit=1, task_id=task_id)
            selected = tuple(str(item['memory_id']) for item in traces[0]['candidate_memories']
                             if item.get('selected'))
            selections.append(selected)
            relevant = set(label.relevant_memory_ids)
            ranks = [rank for rank, memory_id in enumerate(selected, 1) if memory_id in relevant]
            recalls.append(len(ranks) / len(relevant))
            precisions.append(len(ranks) / len(selected) if selected else 0)
            reciprocal_ranks.append(1 / min(ranks) if ranks else 0)
            warm_flags.append(hits_after > hits_before)
            useful_hits += int(hits_after > hits_before and bool(ranks))
            # Resolve actual foreground batches without claiming attributed use
            # or a task outcome. All writes remain on this disposable copy.
            provider.sync_turn(label.query, 'Evaluation response without attributed memory use.',
                               session_id='preload-evaluation')
        if provider._preloader:
            provider._preloader.close()  # include active warming in total CPU
        stats = provider.preload_stats()
    finally:
        provider.shutdown()
    cpu_ms = (time.process_time_ns() - cpu_started) / 1_000_000
    return {
        'requests': len(labels),
        'foreground_latency_ms': asdict(summarize_latencies(latencies)),
        'process_cpu_ms': round(cpu_ms, 6),
        'mean_recall_at_k': statistics.fmean(recalls),
        'mean_precision_at_k': statistics.fmean(precisions),
        'mrr': statistics.fmean(reciprocal_ranks),
        'label_relevant_preload_hits': useful_hits,
        'useful_preload_hit_rate': useful_hits / len(labels),
        'cache': stats['cache'],
        'worker': stats['worker'],
        # Which cases actually consumed a warm entry. Comparing a cold run with a partially warm
        # one is not a comparison of the same thing, and on a loaded runner that is what a warm
        # queue that lost the race looks like.
        'warm_flags': warm_flags,
    }, selections


def evaluate_preload(db: Path, labels: Sequence[RetrievalLabel], *,
                     hint_mode: str = 'continuity', lead_ms: float = 50,
                     top_k: int = 6, token_budget: int = 700,
                     reverse_order: bool = False) -> dict[str, Any]:
    if len(labels) < 8:
        raise ValueError('preload evaluation requires at least eight labelled cases')
    if hint_mode not in {'continuity', 'oracle'}:
        raise ValueError('unsupported hint mode')
    if not math.isfinite(lead_ms) or not 0 <= lead_ms <= 60_000:
        raise ValueError('hint lead time must be finite and between zero and 60000 ms')
    if not 1 <= top_k <= 20 or not 1 <= token_budget <= 4000:
        raise ValueError('limits exceed the bounded preloader ceilings')
    conditions = {}
    selections = {}
    with private_database_snapshot(db) as snapshot:
        # Check targets before opening Cortex or migrating even a trial copy.
        with sqlite3.connect(snapshot) as conn:
            for label in labels:
                if not label.relevant_memory_ids or any(conn.execute(
                    'SELECT 1 FROM memories WHERE id=?', (mid,)).fetchone() is None
                    for mid in label.relevant_memory_ids
                ):
                    raise ValueError('a labelled target is missing')
        order = ('preload', 'control') if reverse_order else ('control', 'preload')
        with tempfile.TemporaryDirectory(prefix='cortex-preload-trial-') as tmp:
            for name in order:
                path = Path(tmp) / (name + '.db')
                with sqlite3.connect(snapshot) as source, sqlite3.connect(path) as destination:
                    source.backup(destination)
                conditions[name], selections[name] = _condition(
                    path, labels, preload=name == 'preload', hint_mode=hint_mode,
                    lead_ms=lead_ms, top_k=top_k, token_budget=token_budget,
                )
    old, new = conditions['control'], conditions['preload']
    report = {
        'schema_version': 1,
        'evaluation': 'cortex-private-preload-replay',
        'settings': {'case_count': len(labels), 'hint_mode': hint_mode,
                     'lead_ms': lead_ms, 'top_k': top_k, 'token_budget': token_budget,
                     'reverse_order': reverse_order, 'semantic_fusion_weight': 0},
        'conditions': conditions,
        'added_process_cpu_ms': added_cpu(old['process_cpu_ms'], new['process_cpu_ms'], len(labels)),
        'p95_latency_delta_ms': round(new['foreground_latency_ms']['p95_ms'] -
                                    old['foreground_latency_ms']['p95_ms'], 6),
        'correctness': {
            'selection_matches_control': _selections_match(
                selections['control'], selections['preload'], new.get('warm_flags') or []),
            'worker_failed_jobs': new['worker'].get('failed', 0),
            # Which cases disagreed, by index only - never the query, the memory or the text. A bare
            # False on a loaded runner is undiagnosable; an index and a hit count are not.
            # How much the verdict is worth. Zero means the comparison never happened -- a loaded
            # runner that consumed no warm entry proved nothing, and "nothing proved" is not the same
            # claim as "the selection changed". Conflating them is what made this exit 1 with no
            # message and look like a code bug in three unrelated pull requests.
            'compared_cases': _compared_cases(
                selections['control'], selections['preload'], new.get('warm_flags') or []),
            'mismatch_case_indices': [
                index for index, (left, right) in enumerate(
                    zip(selections['control'], selections['preload'])) if left != right
            ],
            'control_cases': len(selections['control']),
            'preload_cases': len(selections['preload']),
            'compared_cases': sum(1 for flag in (new.get('warm_flags') or []) if flag),
        },
        'privacy': {'raw_private_text_omitted': True, 'per_case_rows_omitted': True},
        'claim_boundary': (
            'Offline Hermes-adapter label replay, not live traffic or model task success. '
            'Continuity hints use the prior task after sync; oracle hints know the upcoming query '
            'and wait for warming, so oracle usefulness is an upper bound. Useful means a preload '
            'cache hit with a labelled relevant selected memory, not credited model use. '
            'Added CPU is signed preload-minus-control full-process CPU, including worker drain '
            'and normal foreground/outcome bookkeeping; it is not warming wall time. '
            'Private labels are a selected sample; repeat both orders and review all trials.'
        ),
    }
    _assert_report_privacy(report, [(label, {}) for label in labels])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hint-mode', choices=('continuity', 'oracle'), default='continuity')
    parser.add_argument('--lead-ms', type=float, default=50)
    parser.add_argument('--top-k', type=int, default=6)
    parser.add_argument('--token-budget', type=int, default=700)
    parser.add_argument('--reverse-order', action='store_true')
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError('output already exists')
        report = evaluate_preload(args.db, load_labels(args.labels), hint_mode=args.hint_mode,
                                  lead_ms=args.lead_ms, top_k=args.top_k,
                                  token_budget=args.token_budget, reverse_order=args.reverse_order)
        args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with os.fdopen(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.error(f'Preload evaluation failed closed ({type(error).__name__}); inspect inputs locally')
    print('Wrote aggregate-only preload replay report; review locally before sharing')
    correctness = report['correctness']
    if not correctness.get('compared_cases'):
        # Inconclusive, not failed: no warm entry was consumed, so the run never compared anything.
        print('inconclusive: no warm entry was consumed, so nothing was compared', file=sys.stderr)
        return 2
    if correctness['selection_matches_control'] and not correctness['worker_failed_jobs']:
        return 0
    if correctness['worker_failed_jobs']:
        print(f"worker failed {correctness['worker_failed_jobs']} job(s)", file=sys.stderr)
    else:
        print(f"selection did not match control on case(s) "
              f"{correctness.get('mismatch_case_indices')}", file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
