"""The config reader must work when no YAML library can be imported.

The Hermes runtime ships neither PyYAML nor ruamel.yaml, so the old reader caught the ImportError and returned ``{}`` -
every ``plugins.cortex`` setting was silently ignored there, including ``db_path``. That is how a profile ended up opening
an empty store while its real corpus sat at the canonical path.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import hermes_config


SAMPLE = """
model:
  provider: something
memory:
  provider: cortex
plugins:
  enabled:
  - cortex
  cortex:
    db_path: /canonical/cortex.db
    shared_db_path: /canonical/shared.db
    background_prefetch: true
    memory_receipts: false
    semantic_fusion_pool: 20
    semantic_fusion_weight: 10
    shared_promotion_enabled: true
    note: "quoted value # not a comment"
  disabled: []
"""


class FallbackReaderTests(unittest.TestCase):
    def test_it_reads_the_cortex_block(self) -> None:
        block = hermes_config._read_cortex_block(SAMPLE)
        self.assertEqual(block["db_path"], "/canonical/cortex.db")
        self.assertEqual(block["shared_db_path"], "/canonical/shared.db")

    def test_it_coerces_scalar_types(self) -> None:
        block = hermes_config._read_cortex_block(SAMPLE)
        self.assertIs(block["background_prefetch"], True)
        self.assertIs(block["memory_receipts"], False)
        self.assertEqual(block["semantic_fusion_pool"], 20)
        self.assertEqual(block["semantic_fusion_weight"], 10)

    def test_it_keeps_a_hash_inside_quotes(self) -> None:
        block = hermes_config._read_cortex_block(SAMPLE)
        self.assertEqual(block["note"], "quoted value # not a comment")

    def test_it_stops_at_the_next_sibling(self) -> None:
        """``disabled: []`` belongs to plugins, not to cortex."""
        block = hermes_config._read_cortex_block(SAMPLE)
        self.assertNotIn("disabled", block)
        self.assertNotIn("provider", block)  # from the sibling ``memory:`` block

    def test_it_ignores_a_file_without_the_block(self) -> None:
        self.assertEqual(hermes_config._read_cortex_block("model:\n  provider: x\n"), {})

    def test_it_reads_a_real_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(SAMPLE, encoding="utf-8")
            block = hermes_config._read_config(Path(directory))
        self.assertEqual(block.get("db_path"), "/canonical/cortex.db")

    def test_a_missing_file_is_empty_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(hermes_config._read_config(Path(directory)), {})


class NoYamlLibraryTests(unittest.TestCase):
    def test_reading_survives_both_yaml_libraries_being_absent(self) -> None:
        """Simulate the runtime: neither PyYAML nor ruamel importable."""
        original = hermes_config._load_yaml

        def unavailable(_text: str):
            raise ImportError("no yaml library available")

        hermes_config._load_yaml = unavailable
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.yaml"
                path.write_text(SAMPLE, encoding="utf-8")
                block = hermes_config._read_config(Path(directory))
        finally:
            hermes_config._load_yaml = original
        self.assertEqual(block.get("db_path"), "/canonical/cortex.db")


if __name__ == "__main__":
    unittest.main()
