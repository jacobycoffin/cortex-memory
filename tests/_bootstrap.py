"""Load the repository package as ``cortex`` regardless of checkout folder name.

An installed Cortex plugin also provides a top-level ``cortex`` module. When
that install sits on ``sys.path`` (it does on a machine running Hermes), a bare
``import cortex`` silently points the tests at the installed tree instead of the
checkout under test. So the checkout wins: if the imported module is not this
repository, it is evicted and reloaded from ``ROOT``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_from_checkout() -> None:
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


def _is_checkout(module: object) -> bool:
    location = getattr(module, "__file__", None)
    if not location:
        return False
    return Path(location).resolve() == (ROOT / "__init__.py").resolve()


try:
    import cortex  # noqa: F401

    if not _is_checkout(cortex):
        for name in [key for key in sys.modules if key == "cortex" or key.startswith("cortex.")]:
            del sys.modules[name]
        _load_from_checkout()
except ModuleNotFoundError:
    _load_from_checkout()
