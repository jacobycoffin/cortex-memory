from __future__ import annotations

import unittest

from tests._bootstrap import ROOT

from cortex.scripts.benchmark_cache import run_benchmark


class CacheBenchmarkTests(unittest.TestCase):
    def test_small_run_reports_correctness_and_claim_boundary(self) -> None:
        report = run_benchmark(size=40, repetitions=3, seed=11)

        self.assertEqual(report["evidence_type"], "synthetic_repeated_prefetch_preparation")
        self.assertIsInstance(report["correctness"]["cache_disabled_context_stable"], bool)
        self.assertTrue(report["correctness"]["cache_enabled_context_stable"])
        self.assertTrue(report["correctness"]["initial_conditions_returned_same_context"])
        self.assertIn("does not measure", report["claim_boundary"])
        self.assertEqual(report["conditions"]["cache_disabled"]["cache_ttl_seconds"], 0)
        self.assertEqual(report["conditions"]["cache_enabled"]["cache_ttl_seconds"], 300)


if __name__ == "__main__":
    unittest.main()
