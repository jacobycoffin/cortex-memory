"""The model-swap comparison must be exact, not forgiving (backlog item #198)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "model_swap_drill.py"


def load_module():
    spec = importlib.util.spec_from_file_location("model_swap_drill", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = {"hit_at_k": 0.516667, "mean_recall_at_k": 0.244792, "mrr": 0.292708}


class CompareTests(unittest.TestCase):
    def test_identical_metrics_hold(self) -> None:
        module = load_module()
        held, differences = module.compare(dict(BASE), dict(BASE))
        self.assertTrue(held)
        self.assertEqual(differences, [])

    def test_any_drift_is_a_failure(self) -> None:
        module = load_module()
        moved = dict(BASE, hit_at_k=BASE["hit_at_k"] + 0.001)
        held, differences = module.compare(BASE, moved)
        self.assertFalse(held)
        self.assertEqual(len(differences), 1)
        self.assertIn("hit_at_k", differences[0])

    def test_tiny_drift_still_fails(self) -> None:
        """Recall is deterministic: there is no acceptable amount of model dependence."""
        module = load_module()
        moved = dict(BASE, mrr=BASE["mrr"] + 1e-7)
        held, differences = module.compare(BASE, moved)
        self.assertFalse(held)

    def test_missing_metric_treated_as_zero(self) -> None:
        module = load_module()
        held, differences = module.compare(BASE, {"hit_at_k": 0.0})
        self.assertFalse(held)
        self.assertEqual(len(differences), 3)

    def test_float_representation_noise_ignored(self) -> None:
        module = load_module()
        same = {k: float(v) for k, v in BASE.items()}
        held, _ = module.compare(BASE, same)
        self.assertTrue(held)


if __name__ == "__main__":
    unittest.main()

class ReportParsingTests(unittest.TestCase):
    def test_a_report_without_metrics_is_an_error(self) -> None:
        """An empty metrics dict would make every swap 'pass' — refuse it instead."""
        module = load_module()
        import json, tempfile
        from pathlib import Path as P

        with tempfile.TemporaryDirectory() as tmp:
            runner = P(tmp) / "frozen_evalset.py"
            runner.write_text(
                "import json,sys\n"
                "out = sys.argv[sys.argv.index('--output') + 1]\n"
                "open(out,'w').write(json.dumps({'cases': []}))\n"
            )
            # point the module at the stub by monkeypatching its own path resolution
            module_path = P(module.__file__)
            stub_dir = P(tmp)
            (stub_dir / "frozen_evalset.py").write_text(runner.read_text())
            original_file = module.__file__
            module.__file__ = str(stub_dir / "model_swap_drill.py")
            try:
                with self.assertRaises(RuntimeError):
                    module.run_eval(P("/tmp/x.db"), stub_dir, P(tmp) / "out.json", None, None)
            finally:
                module.__file__ = original_file
