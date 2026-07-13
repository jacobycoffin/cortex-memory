"""Load the repository package as ``cortex`` regardless of checkout folder name."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

try:
    import cortex  # noqa: F401
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "cortex",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the Cortex package for tests")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cortex"] = module
    spec.loader.exec_module(module)
