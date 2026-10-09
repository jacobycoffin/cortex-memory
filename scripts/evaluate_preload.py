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
        'correctness': {'selection_matches_control': selections['control'] == selections['preload'],
                        'worker_failed_jobs': new['worker'].get('failed', 0)},
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
    return 0 if (report['correctness']['selection_matches_control'] and
                 not report['correctness']['worker_failed_jobs']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
