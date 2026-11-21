"""Public fixtures use synthetic data; operator history stays local."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_repository


class RepositoryPrivacyTests(unittest.TestCase):
    def check_fixture(self, content: str, *, relative: str = "tests/test_example.py") -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / relative
            fixture.parent.mkdir(parents=True)
            fixture.write_text(content)
            with patch.object(check_repository, "ROOT", root):
                return check_repository.check_public_files([fixture])

    def test_rejects_declared_history_provenance_in_fixture_locations(self) -> None:
        for relative in ("tests/test_example.py", "benchmarks/example.py", "fixtures/example.txt"):
            for source in ("REAL memories", "live corpus", "private operator history"):
                with self.subTest(relative=relative, source=source):
                    content = '"""These examples are trimmed from the ' + source + '."""'
                    failures = self.check_fixture(content, relative=relative)
                    self.assertEqual(
                        failures, [f"history-derived fixture found in tracked file: {relative}"]
                    )

    def test_accepts_synthetic_temporal_and_financial_shapes(self) -> None:
        self.assertEqual(
            self.check_fixture(
                '"""Invented examples, never copied from operator history."""\n'
                'EXAMPLE = "Example Card balance $125.00 as of 2026-08-07 at 192.0.2.20"\n'
            ),
            [],
        )

    def test_documentation_can_describe_private_evaluation(self) -> None:
        content = 'Evaluation uses cases extracted from the ' + 'private corpus locally.'
        self.assertEqual(self.check_fixture(content, relative="docs/EVALUATION.md"), [])


if __name__ == "__main__":
    unittest.main()
