from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT

from cortex.scripts.frozen_evalset import compare_to_baseline, retire_unretrievable
from cortex.scripts.evaluate_real_history import RetrievalLabel
from cortex.store import CortexStore

BUILD = ROOT / "scripts" / "build_frozen_evalset.py"
RUN = ROOT / "scripts" / "frozen_evalset.py"


def seed_database(path: Path, cases: int = 210) -> list[str]:
    store = CortexStore(path)
    memory_ids: list[str] = []
    try:
        for index in range(cases):
            memory_id, _ = store.add_memory(
                f"Deployment lane {index} uses private marker ORCHID-{index} for the release train.",
                kind="operational",
                source_ref=f"frozen-test-{index}",
            )
            memory_ids.append(memory_id)
            # Two used rows per query so every case has evidence and stable strength.
            for _ in range(2):
                store.log_access(
                    memory_id,
                    "used",
                    query=f"which private marker does deployment lane {index} use for the release train?",
                    session_id="frozen-test",
                )
    finally:
        store.close()
    return memory_ids


def build(db: Path, out: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BUILD), "--db", str(db), "--out", str(out), *extra],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def run_set(db: Path, set_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUN), "run", "--db", str(db), "--set-dir", str(set_dir), *extra],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


class FrozenEvalSetTests(unittest.TestCase):
    def test_build_is_deterministic_and_freezes_real_used_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "cortex.db"
            memory_ids = seed_database(db)
            first, second = root / "a" / "frozen-v1.jsonl", root / "b" / "frozen-v1.jsonl"
            for out in (first, second):
                result = build(db, out, "--set-id", "frozen-v1", "--limit", "210")
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                hashlib.sha256(first.read_bytes()).hexdigest(),
                hashlib.sha256(second.read_bytes()).hexdigest(),
                "the same database must freeze to byte-identical labels",
            )
            rows = [json.loads(line) for line in first.read_text().splitlines()]
            self.assertEqual(len(rows), 210)
            self.assertEqual(len({row["case_id"] for row in rows}), 210)
            for row in rows:
                self.assertEqual(row["schema_version"], 1)
                self.assertTrue(set(row["relevant_memory_ids"]) <= set(memory_ids))
                self.assertEqual(len(row["relevant_memory_ids"]), 1)
            manifest = json.loads((first.with_suffix(".manifest.json")).read_text())
            self.assertEqual(manifest["case_count"], 210)
            self.assertEqual(manifest["labels_sha256"], hashlib.sha256(first.read_bytes()).hexdigest())
            self.assertTrue(manifest["privacy"]["do_not_commit"])
            self.assertTrue(manifest["counts"]["eligible_queries"] >= 210)

    def test_build_refuses_small_corpus_and_repository_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "cortex.db"
            seed_database(db, cases=3)
            small = build(db, root / "small.jsonl")
            self.assertNotEqual(small.returncode, 0)
            self.assertIn("refusing to freeze below 200", small.stderr)

            refused = build(db, ROOT / "never-written.jsonl")
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("refusing to write private queries inside the repository", refused.stderr)
            self.assertFalse((ROOT / "never-written.jsonl").exists())

    def test_runner_pins_baseline_is_stable_and_flags_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "cortex.db"
            seed_database(db)
            set_dir = root / "frozen"
            built = build(db, set_dir / "frozen-v1.jsonl", "--set-id", "frozen-v1", "--limit", "205")
            self.assertEqual(built.returncode, 0, built.stderr)

            first = run_set(db, set_dir)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("baseline pinned", first.stdout)
            report_path = set_dir / "frozen-v1-report.json"
            report = json.loads(report_path.read_text())
            self.assertEqual(report["frozen_set"]["cases_scored"], 205)
            self.assertEqual(report["frozen_set"]["cases_retired"], 0)
            self.assertEqual(report["summary"]["cases"], 205)

            again = run_set(db, set_dir)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertIn("within tolerance", again.stdout)

            baseline_path = set_dir / "frozen-v1.baseline.json"
            baseline = json.loads(baseline_path.read_text())
            baseline["metrics"]["mean_recall_at_k"] = 1.0
            baseline["metrics"]["hit_at_k"] = 1.0
            baseline_path.write_text(json.dumps(baseline))
            regressed = run_set(db, set_dir)
            self.assertEqual(regressed.returncode, 1, regressed.stdout)
            self.assertIn("REGRESSION", regressed.stderr)

    def test_runner_rejects_tampered_set_and_retires_archived_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "cortex.db"
            memory_ids = seed_database(db)
            set_dir = root / "frozen"
            built = build(db, set_dir / "frozen-v1.jsonl", "--set-id", "frozen-v1", "--limit", "205")
            self.assertEqual(built.returncode, 0, built.stderr)

            labels_path = set_dir / "frozen-v1.jsonl"
            original = labels_path.read_text()
            rows = [json.loads(line) for line in original.splitlines()]
            rows[0]["relevant_memory_ids"] = ["00000000-0000-0000-0000-000000000000"]
            labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            tampered = run_set(db, set_dir)
            self.assertEqual(tampered.returncode, 2, tampered.stdout)
            self.assertIn("does not match its manifest", tampered.stderr)
            labels_path.write_text(original)

            store = CortexStore(db)
            try:
                store.set_state(memory_ids[0], "archived", reason="frozen-set-test")
            finally:
                store.close()
            retired_run = run_set(db, set_dir, "--min-cases", "200")
            self.assertEqual(retired_run.returncode, 0, retired_run.stderr)
            report = json.loads((set_dir / "frozen-v1-report.json").read_text())
            self.assertEqual(report["frozen_set"]["cases_retired"], 1)
            self.assertEqual(report["frozen_set"]["cases_scored"], 204)


class BaselineComparisonTests(unittest.TestCase):
    def test_tolerances_are_directional(self) -> None:
        baseline = {"metrics": {"hit_at_k": 0.50, "mean_recall_at_k": 0.30, "mrr": 0.40, "zero_result_rate": 0.02}}
        same = {"hit_at_k": 0.50, "mean_recall_at_k": 0.295, "mrr": 0.39, "zero_result_rate": 0.02}
        self.assertEqual(compare_to_baseline(same, baseline), [])
        worse = {"hit_at_k": 0.40, "mean_recall_at_k": 0.20, "mrr": 0.10, "zero_result_rate": 0.20}
        problems = compare_to_baseline(worse, baseline)
        self.assertEqual(len(problems), 4)
        self.assertTrue(any(problem.startswith("zero_result_rate") for problem in problems))

    def test_retirement_counts_missing_and_archived_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "cortex.db"
            store = CortexStore(db)
            try:
                active_id, _ = store.add_memory("Active memory for retirement test.", kind="semantic")
                archived_id, _ = store.add_memory("Archived memory for retirement test.", kind="semantic")
                store.set_state(archived_id, "archived", reason="retirement-test")
                labels = [
                    RetrievalLabel("case-active", "active question text here", (active_id,)),
                    RetrievalLabel("case-archived", "archived question text here", (archived_id,)),
                    RetrievalLabel("case-missing", "missing question text here", ("does-not-exist",)),
                ]
                kept, retired = retire_unretrievable(store, labels)
            finally:
                store.close()
            self.assertEqual([label.case_id for label in kept], ["case-active"])
            self.assertEqual(retired, 2)


if __name__ == "__main__":
    unittest.main()
