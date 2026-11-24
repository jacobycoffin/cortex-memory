"""Is the running plugin the code in the repository? (#221)"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import plugin_drift


class DriftTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.repo = self.base / "repo"
        self.deployed = self.base / "deployed"
        self.repo.mkdir()
        self.deployed.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.base, ignore_errors=True)
        self.tmp.cleanup()

    def write(self, root: Path, name: str, text: str) -> None:
        (root / name).write_text(text)

    def test_an_identical_copy_is_current(self) -> None:
        for name in ("retrieval.py", "store.py"):
            self.write(self.repo, name, f"# {name}\n")
            self.write(self.deployed, name, f"# {name}\n")
        found = plugin_drift.compare(self.repo, self.deployed)
        self.assertEqual(found.verdict, plugin_drift.CURRENT)
        self.assertIn("identical", found.describe())

    def test_a_missing_module_is_behind(self) -> None:
        self.write(self.repo, "retrieval.py", "# a\n")
        self.write(self.repo, "boost_caps.py", "# b\n")
        self.write(self.deployed, "retrieval.py", "# a\n")
        found = plugin_drift.compare(self.repo, self.deployed)
        self.assertEqual(found.verdict, plugin_drift.BEHIND)
        self.assertEqual(found.missing, ["boost_caps.py"])

    def test_a_changed_module_is_behind_and_named(self) -> None:
        self.write(self.repo, "retrieval.py", "# new\n")
        self.write(self.deployed, "retrieval.py", "# old\n")
        found = plugin_drift.compare(self.repo, self.deployed)
        self.assertEqual(found.verdict, plugin_drift.BEHIND)
        self.assertEqual(found.differing, ["retrieval.py"])

    def test_a_module_only_the_live_tree_has_is_ahead(self) -> None:
        """Someone patched the deployed copy. That is a finding, not a nothing."""
        self.write(self.repo, "retrieval.py", "# a\n")
        self.write(self.deployed, "retrieval.py", "# a\n")
        self.write(self.deployed, "hotfix.py", "# patched in place\n")
        found = plugin_drift.compare(self.repo, self.deployed)
        self.assertEqual(found.verdict, plugin_drift.AHEAD)
        self.assertEqual(found.extra, ["hotfix.py"])

    def test_a_missing_deployed_path_is_unreadable_not_an_error(self) -> None:
        found = plugin_drift.compare(self.repo, self.base / "nope")
        self.assertEqual(found.verdict, plugin_drift.UNREADABLE)
        self.assertIn("not a directory", found.reason)

    def test_an_empty_repo_is_unreadable(self) -> None:
        self.write(self.deployed, "retrieval.py", "# a\n")
        found = plugin_drift.compare(self.repo, self.deployed)
        self.assertEqual(found.verdict, plugin_drift.UNREADABLE)
        self.assertIn("no Python modules", found.reason)

    def test_a_deployed_copy_with_nothing_in_it_is_behind(self) -> None:
        self.write(self.repo, "retrieval.py", "# a\n")
        found = plugin_drift.compare(self.repo, self.deployed)
        self.assertEqual(found.verdict, plugin_drift.BEHIND)
        self.assertIn("no Python modules", found.reason)

    def test_non_python_files_are_not_modules(self) -> None:
        self.write(self.repo, "retrieval.py", "# a\n")
        self.write(self.repo, "README.md", "docs\n")
        self.write(self.deployed, "retrieval.py", "# a\n")
        found = plugin_drift.compare(self.repo, self.deployed)
        self.assertEqual(found.verdict, plugin_drift.CURRENT)
        self.assertEqual(found.repo_count, 1)

    def test_directories_are_not_modules(self) -> None:
        (self.repo / "tests").mkdir()
        self.write(self.repo, "retrieval.py", "# a\n")
        self.write(self.deployed, "retrieval.py", "# a\n")
        found = plugin_drift.compare(self.repo, self.deployed)
        self.assertEqual(found.verdict, plugin_drift.CURRENT)

    def test_counts_are_reported_on_both_sides(self) -> None:
        self.write(self.repo, "a.py", "1\n")
        self.write(self.repo, "b.py", "2\n")
        self.write(self.deployed, "a.py", "1\n")
        found = plugin_drift.compare(self.repo, self.deployed)
        self.assertEqual((found.repo_count, found.deployed_count), (2, 1))

    def test_the_reason_counts_both_kinds_of_drift(self) -> None:
        self.write(self.repo, "a.py", "1\n")
        self.write(self.repo, "b.py", "2\n")
        self.write(self.deployed, "a.py", "changed\n")
        found = plugin_drift.compare(self.repo, self.deployed)
        self.assertIn("does not have", found.reason)
        self.assertIn("different version", found.reason)

    def test_the_limit_caps_the_lists(self) -> None:
        for index in range(6):
            self.write(self.repo, f"m{index}.py", f"{index}\n")
        found = plugin_drift.compare(self.repo, self.deployed, limit=2)
        self.assertLessEqual(len(found.missing), 2)

    def test_samples_give_a_readable_head(self) -> None:
        for index in range(6):
            self.write(self.repo, f"m{index}.py", f"{index}\n")
        self.write(self.deployed, "retrieval.py", "# present\n")
        found = plugin_drift.compare(self.repo, self.deployed)
        heads = plugin_drift.samples(found, count=3)
        self.assertEqual(len(heads["missing"]), 3)

    def test_it_writes_nothing_into_either_tree(self) -> None:
        self.write(self.repo, "a.py", "1\n")
        self.write(self.deployed, "a.py", "1\n")
        before = {path.name: path.read_bytes() for path in self.deployed.iterdir()}
        plugin_drift.compare(self.repo, self.deployed)
        after = {path.name: path.read_bytes() for path in self.deployed.iterdir()}
        self.assertEqual(before, after)

    def test_the_real_repository_against_itself_is_current(self) -> None:
        """A sanity check on the comparison itself: a tree cannot drift from itself."""
        root = Path(plugin_drift.__file__).resolve().parent
        found = plugin_drift.compare(root, root)
        self.assertEqual(found.verdict, plugin_drift.CURRENT)


if __name__ == "__main__":
    unittest.main()
