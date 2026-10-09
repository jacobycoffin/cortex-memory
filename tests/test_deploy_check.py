"""Deploy-closure classification: the check that guards a plugin deploy.

`scripts/deploy_check.py` exists because the live plugin and this repo are only
*assumed* to be ancestor-related; historically they diverged and a wholesale copy
would have deleted live-only work. These tests pin the pure classifier: what it
calls safe to copy, what it refuses to touch.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_deploy_check():
    """Load scripts/deploy_check.py without importing the Cortex package."""

    path = _REPO_ROOT / "scripts" / "deploy_check.py"
    spec = importlib.util.spec_from_file_location("deploy_check_under_test", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


deploy_check = _load_deploy_check()


class ClassifyTests(unittest.TestCase):
    def test_identical_file_is_at_head(self) -> None:
        closure = deploy_check.classify(
            {"store.py": b"current"}, {"store.py": b"current"}, lambda rel: []
        )
        self.assertEqual(closure.at_head, ["store.py"])
        self.assertTrue(closure.ok)
        self.assertEqual(closure.deployable, [])

    def test_file_matching_an_older_commit_is_behind_and_deployable(self) -> None:
        older = [(f"commit{index}", f"body{index}".encode()) for index in range(5)]
        closure = deploy_check.classify(
            {"store.py": b"body2"}, {"store.py": b"current"}, lambda rel: older
        )
        self.assertEqual(closure.behind, [("store.py", "commit2", 2)])
        self.assertEqual(closure.deployable, ["store.py"])
        self.assertTrue(closure.ok)

    def test_file_matching_no_commit_is_divergent_and_blocks(self) -> None:
        closure = deploy_check.classify(
            {"store.py": b"live-only work"},
            {"store.py": b"current"},
            lambda rel: [("commit0", b"body0")],
        )
        self.assertEqual(closure.divergent, ["store.py"])
        self.assertFalse(closure.ok)
        self.assertEqual(closure.deployable, [])

    def test_live_only_file_is_never_deployed_or_flagged(self) -> None:
        closure = deploy_check.classify(
            {"benchmark-results/run.json": b"{}"},
            {},
            lambda rel: [],
        )
        self.assertEqual(closure.live_only, ["benchmark-results/run.json"])
        self.assertTrue(closure.ok)
        self.assertEqual(closure.deployable, [])

    def test_classification_is_independent_per_file(self) -> None:
        lookup = {"a.py": [("c1", b"a-old")], "b.py": []}
        closure = deploy_check.classify(
            {"a.py": b"a-old", "b.py": b"b-live", "c.py": b"c-same"},
            {"a.py": b"a-new", "b.py": b"b-new", "c.py": b"c-same"},
            lambda rel: lookup.get(rel, []),
        )
        self.assertEqual(closure.at_head, ["c.py"])
        self.assertEqual(closure.behind, [("a.py", "c1", 0)])
        self.assertEqual(closure.divergent, ["b.py"])
        self.assertEqual(closure.deployable, ["a.py"])


if __name__ == "__main__":
    unittest.main()
