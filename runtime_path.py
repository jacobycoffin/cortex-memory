"""Locate Cortex's optional embedding runtime when it lives beside Hermes.

Hermes rebuilds its managed dependency environment on ``hermes update`` and
``hermes pm repair``. Packages pip-installed straight into that environment
(numpy, onnxruntime, tokenizers) vanish when that happens, and declaring them
as plugin ``pip_dependencies`` makes Hermes refuse to admit the plugin because
onnxruntime has no wheels for every platform in its universal lock.

``scripts/install.sh`` therefore installs the runtime into a sidecar directory,
``$HERMES_HOME/cortex/pydeps/py<major>.<minor>``, that Hermes never touches.
This module appends that directory to the *end* of ``sys.path`` so packages
Hermes already ships always win. When the directory is absent (Claude Code /
MCP installs, or a Python upgrade) nothing changes and Cortex falls back to
lexical recall until the installer is re-run. ``CORTEX_PYDEPS`` overrides it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def sidecar_dir() -> Path | None:
    explicit = os.environ.get("CORTEX_PYDEPS", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HERMES_HOME", "").strip()
    base = Path(home).expanduser() if home else Path.home() / ".hermes"
    return base / "cortex" / "pydeps" / f"py{sys.version_info.major}.{sys.version_info.minor}"


def ensure_runtime_path() -> str | None:
    """Append the sidecar runtime directory to sys.path when it exists."""

    try:
        path = sidecar_dir()
    except Exception:  # pragma: no cover - defensive; never break plugin import
        return None
    if path is None or not path.is_dir():
        return None
    text = str(path)
    if text not in sys.path:
        sys.path.append(text)
    return text


ensure_runtime_path()
