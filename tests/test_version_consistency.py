"""Guard the four places the release version is written.

0.3.0 shipped with the version duplicated across ``pyproject.toml``,
``plugin.yaml``, ``CHANGELOG.md`` and ``cortex.__version__`` and nothing keeping
them in agreement, so a release could be tagged with one file disagreeing with
the others and no test would notice. This is a file-only check on purpose: it
reads the four sources as text rather than importing the package, so it runs
under any harness layout, with no installed package and no dependencies.

Only the numeric ``X.Y.Z`` base is compared. The three files legitimately spell
a pre-release differently (``0.3.0.dev1`` / ``0.3.0-dev.1`` / ``0.3.0``), so a
strict string compare would fail on every dev build and get deleted. Base
comparison still catches the drift that matters: one file left on an old
release.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = re.compile(r"\d+\.\d+\.\d+")


class VersionConsistencyTests(unittest.TestCase):
    def _sources(self) -> dict[str, str]:
        found: dict[str, str] = {}

        pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        project = pyproject.split("[project]", 1)[1] if "[project]" in pyproject else pyproject
        match = re.search(r'^version\s*=\s*"([^"]+)"', project, re.M)
        self.assertIsNotNone(match, "pyproject.toml has no [project] version")
        found["pyproject.toml"] = match.group(1)  # type: ignore[union-attr]

        plugin = (REPO / "plugin.yaml").read_text(encoding="utf-8")
        match = re.search(r"^version:\s*(\S+)", plugin, re.M)
        self.assertIsNotNone(match, "plugin.yaml has no top-level version")
        found["plugin.yaml"] = match.group(1)  # type: ignore[union-attr]

        changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
        match = re.search(r"^##\s*(\d+\.\d+\.\d+\S*)", changelog, re.M)
        self.assertIsNotNone(match, "CHANGELOG.md has no released version heading")
        found["CHANGELOG.md"] = match.group(1)  # type: ignore[union-attr]

        init = (REPO / "__init__.py").read_text(encoding="utf-8")
        match = re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.M)
        self.assertIsNotNone(match, "__init__.py has no __version__")
        found["cortex.__version__"] = match.group(1)  # type: ignore[union-attr]

        return found

    def test_all_release_sources_agree_on_the_base_version(self) -> None:
        sources = self._sources()
        bases: dict[str, str | None] = {
            name: (BASE.search(value).group(0) if BASE.search(value) else None)
            for name, value in sources.items()
        }
        self.assertNotIn(None, bases.values(), f"unparseable version in {sources}")
        self.assertEqual(
            1,
            len(set(bases.values())),
            f"version drift across release sources: {sources} -> {bases}",
        )

    def test_release_version_is_not_a_dev_placeholder(self) -> None:
        """A cut release must not still be marked unreleased."""
        changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
        heading = re.search(r"^##\s*(\d+\.\d+\.\d+\S*)\s*(.*)$", changelog, re.M)
        self.assertIsNotNone(heading, "CHANGELOG.md has no released version heading")
        assert heading is not None
        self.assertNotIn(
            "unreleased",
            heading.group(2).lower(),
            "CHANGELOG's top section is still marked Unreleased",
        )


if __name__ == "__main__":
    unittest.main()
