"""Make the repository's flat ``cortex`` package importable during pytest collection."""

from __future__ import annotations

import sys

from tests import _bootstrap  # noqa: F401


# The checkout directory may contain a hyphen, so pytest imports the package
# initializer as ``__init__`` while walking parent packages. Reuse the correctly
# bootstrapped ``cortex`` module instead of executing it without package context.
sys.modules.setdefault("__init__", sys.modules["cortex"])
