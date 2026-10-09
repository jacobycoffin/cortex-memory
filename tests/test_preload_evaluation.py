"""Host-local replay keeps private inputs local and measures added CPU honestly."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT
from cortex.scripts.evaluate_preload import evaluate_preload
from cortex.scripts.evaluate_real_history import RetrievalLabel
from cortex.store import CortexStore


class PreloadEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / 'synthetic.db'
        store = CortexStore(self.db)
        self.labels = []
        try:
            for index in range(8):
                mid, _ = store.add_memory(
                    f'Synthetic project juniper-{index}: the launch color is amber-{index}.',
                    kind='decision', source_category='TOOL_VERIFIED', confidence=.95, trust=.95,
                    currentness_confidence=.95, importance=.8)
                self.labels.append(RetrievalLabel(
                    f'synthetic-private-case-{index}',
                    f'What is the launch color for synthetic project juniper-{index}?', (mid,),
                    'synthetic-private-group'))
        finally:
            store.close()
        self.original = self.db.read_bytes()

    def tearDown(self):
        self.tmp.cleanup()

    def test_oracle_hits_preserve_source_and_omit_private_rows(self):
        report = evaluate_preload(self.db, self.labels, hint_mode='oracle', lead_ms=0)
        self.assertTrue(report['correctness']['selection_matches_control'])
        self.assertEqual(report['correctness']['worker_failed_jobs'], 0)
        self.assertGreater(report['conditions']['preload']['label_relevant_preload_hits'], 0)
        self.assertGreater(report['conditions']['preload']['process_cpu_ms'], 0)
        serialized = json.dumps(report)
        for label in self.labels:
            for private in (label.case_id, label.query, label.group, *label.relevant_memory_ids):
                self.assertNotIn(private, serialized)
        self.assertNotIn('cases', report)
        self.assertEqual(self.db.read_bytes(), self.original)

    def test_continuity_unique_tasks_does_not_claim_upcoming_query_hits(self):
        report = evaluate_preload(self.db, self.labels, lead_ms=0, reverse_order=True)
        self.assertEqual(report['conditions']['preload']['cache']['preload_hits'], 0)
        self.assertEqual(report['conditions']['preload']['label_relevant_preload_hits'], 0)
        self.assertGreater(report['conditions']['preload']['worker']['queued'], 0)
        self.assertEqual(self.db.read_bytes(), self.original)

    def test_missing_targets_and_small_samples_fail_without_source_changes(self):
        with self.assertRaisesRegex(ValueError, 'eight'):
            evaluate_preload(self.db, self.labels[:7])
        labels = self.labels[:]
        labels[0] = RetrievalLabel('synthetic-missing', 'Synthetic missing target?', ('missing-target',))
        with self.assertRaisesRegex(ValueError, 'missing'):
            evaluate_preload(self.db, labels)
        for options in ({'lead_ms': float('nan')}, {'top_k': 21}, {'token_budget': 4001}):
            with self.assertRaises(ValueError):
                evaluate_preload(self.db, self.labels, **options)
        self.assertEqual(self.db.read_bytes(), self.original)

    def test_signed_cpu_delta_is_not_clipped_when_order_noise_makes_it_negative(self):
        metrics = {'process_cpu_ms': 100, 'foreground_latency_ms': {'p95_ms': 10}, 'worker': {}}
        cheaper = copy.deepcopy(metrics)
        cheaper['process_cpu_ms'] = 80
        with patch('cortex.scripts.evaluate_preload._condition',
                   side_effect=[(metrics, []), (cheaper, [])]):
            report = evaluate_preload(self.db, self.labels, lead_ms=0)
        self.assertEqual(report['added_process_cpu_ms'], {'total_ms': -20, 'per_request_ms': -2.5})

    def test_cli_refuses_overwrite_without_leaking_private_paths(self):
        labels = self.root / 'labels.jsonl'
        labels.write_text('\n'.join(json.dumps(dict(case_id=row.case_id, query=row.query,
            relevant_memory_ids=list(row.relevant_memory_ids))) for row in self.labels))
        output = self.root / 'existing-report.json'
        output.write_text('keep the existing report')
        result = subprocess.run([sys.executable, str(ROOT/'scripts/evaluate_preload.py'),
            '--db', str(self.db), '--labels', str(labels), '--output', str(output)],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_text(), 'keep the existing report')
        self.assertNotIn(str(self.db), result.stderr)
        self.assertNotIn(str(labels), result.stderr)
        self.assertEqual(self.db.read_bytes(), self.original)

    def test_cli_exports_owner_only_aggregates(self):
        labels = self.root / 'labels.jsonl'
        labels.write_text('\n'.join(json.dumps(dict(case_id=row.case_id, query=row.query,
            relevant_memory_ids=list(row.relevant_memory_ids))) for row in self.labels))
        output = self.root / 'report.json'
        result = subprocess.run([sys.executable, str(ROOT/'scripts/evaluate_preload.py'),
            '--db', str(self.db), '--labels', str(labels), '--output', str(output),
            '--hint-mode', 'oracle', '--lead-ms', '0'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        report = json.loads(output.read_text())
        self.assertTrue(report['privacy']['per_case_rows_omitted'])
        self.assertGreater(report['conditions']['preload']['label_relevant_preload_hits'], 0)
        for row in self.labels:
            self.assertNotIn(row.query, output.read_text())
            self.assertNotIn(row.case_id, output.read_text())
        self.assertEqual(self.db.read_bytes(), self.original)

    def test_real_history_runner_does_not_choose_a_shadowing_legacy_package(self):
        legacy = self.root / 'Brain'
        legacy.mkdir()
        (legacy / '__init__.py').write_text("raise AssertionError('legacy package was selected')\n")
        result = subprocess.run([sys.executable, '-c',
            'import tests._bootstrap; from cortex.scripts import evaluate_real_history as runner; '
            'assert runner.MemoryRetriever.__module__ == "cortex.retrieval"'],
            cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(self.root)},
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_broken_selected_package_fails_instead_of_using_legacy_code(self):
        selected = self.root / 'cortex'
        selected.mkdir()
        (selected / '__init__.py').write_text('')
        legacy = self.root / 'Brain'
        legacy.mkdir()
        (legacy / '__init__.py').write_text("raise AssertionError('legacy fallback was selected')\n")
        script = self.root / 'runner/scripts/evaluate_real_history.py'
        script.parent.mkdir(parents=True)
        script.write_text((ROOT/'scripts/evaluate_real_history.py').read_text())
        command = 'import runpy; runpy.run_path(' + repr(str(script)) + ')'
        # Disable site import hooks so an editable install cannot fill in the
        # deliberately missing modules of this selected package.
        result = subprocess.run([sys.executable, '-S', '-c', command], cwd=self.root,
            env={**os.environ, 'PYTHONPATH': str(self.root)}, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ModuleNotFoundError', result.stderr)
        self.assertNotIn('legacy fallback was selected', result.stderr)
