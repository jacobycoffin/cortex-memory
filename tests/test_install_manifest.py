"""Installer manifest must ship every top-level runtime module.

Regression guard: `serializers.py` was extracted from `store.py` but never
added to `scripts/install_local.sh` — the full unittest suite stayed green
(because it runs against the source checkout) while the installed plugin
failed at `import cortex.store`. This test parses the installer's copy list
and fails on any omission, before further module extractions land.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401


def _installer_copy_list() -> set[str]:
    text = (ROOT / "scripts" / "install_local.sh").read_text(encoding="utf-8")
    match = re.search(r"for file in (.*?); do", text, re.DOTALL)
    assert match is not None, "install_local.sh copy list not found"
    return set(match.group(1).split())


class InstallManifestTests(unittest.TestCase):
    def test_installer_ships_every_top_level_module(self) -> None:
        shipped = _installer_copy_list()
        modules = {path.name for path in ROOT.glob("*.py")}
        self.assertTrue(modules, "no top-level modules found")
        missing = sorted(modules - shipped)
        self.assertEqual(
            missing, [], f"install_local.sh omits runtime modules: {missing}"
        )

    def test_installer_list_has_no_stale_entries(self) -> None:
        shipped = _installer_copy_list()
        # .py entries must all exist in the source tree; non-code assets
        # (html, svg, ico, png, yaml, md) are exempt from this check.
        stale = sorted(
            name
            for name in shipped
            if name.endswith(".py") and not (ROOT / name).is_file()
        )
        self.assertEqual(
            stale, [], f"install_local.sh lists removed modules: {stale}"
        )


if __name__ == "__main__":
    unittest.main()
