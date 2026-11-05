"""Load the repository package as ``cortex`` regardless of checkout folder name.

The checkout uses a flat package layout (``cortex`` == repository root), so
``import cortex`` only works after this module registers it. A *different*
``cortex`` — the plugin copy installed under ``~/.hermes/installs`` — is
importable on machines that run Hermes, and if it wins the import the suite
silently exercises code that is not in this checkout. Only accept an existing
``cortex`` when it actually is this checkout.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _is_checkout(module: object) -> bool:
    origin = getattr(module, "__file__", None)
    if not origin:
        return False
    try:
        return Path(origin).resolve().parent == ROOT
    except OSError:
        return False


def _load_checkout() -> None:
    for name in [m for m in list(sys.modules) if m == "cortex" or m.startswith("cortex.")]:
        del sys.modules[name]
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


try:
    import cortex  # noqa: F401
except ModuleNotFoundError:
    _load_checkout()
else:
    if not _is_checkout(cortex):
        _load_checkout()
