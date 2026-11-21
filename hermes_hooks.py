"""Hermes output-hook bridge shared by active provider sessions."""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__package__)


_OUTPUT_HOOK_LOCK = threading.RLock()


_OUTPUT_PROVIDER_BY_SESSION: dict[str, Any] = {}


def _hermes_transform_llm_output_hook(
    response_text: str,
    *,
    session_id: str = "",
    **kwargs: Any,
) -> str | None:
    """Route Hermes's process-wide output hook to the session provider."""

    sid = session_id or "default"
    with _OUTPUT_HOOK_LOCK:
        provider = _OUTPUT_PROVIDER_BY_SESSION.get(sid)
    if provider is None:
        return None
    return provider.transform_llm_output(
        response_text,
        session_id=sid,
        **kwargs,
    )


def _install_hermes_output_hook(provider: Any, session_id: str) -> None:
    """Bridge Hermes memory loading to its general output-hook registry.

    Hermes's exclusive memory-provider collector intentionally treats
    ``register_hook`` as a no-op. Register one idempotent dispatcher directly
    with the process-wide manager so the supported ``transform_llm_output``
    lifecycle still works for the active memory provider.
    """

    sid = session_id or "default"
    with _OUTPUT_HOOK_LOCK:
        stale_sessions = [
            existing_session
            for existing_session, existing_provider in _OUTPUT_PROVIDER_BY_SESSION.items()
            if existing_provider is provider and existing_session != sid
        ]
        for existing_session in stale_sessions:
            _OUTPUT_PROVIDER_BY_SESSION.pop(existing_session, None)
        _OUTPUT_PROVIDER_BY_SESSION[sid] = provider
    try:
        from hermes_cli.plugins import get_plugin_manager

        manager = get_plugin_manager()
        hooks = getattr(manager, "_hooks", None)
        if not isinstance(hooks, dict):
            logger.warning(
                "Hermes plugin manager has no compatible hook registry; "
                "Cortex receipt enforcement is unavailable"
            )
            return
        callbacks = hooks.setdefault("transform_llm_output", [])
        if _hermes_transform_llm_output_hook not in callbacks:
            callbacks.append(_hermes_transform_llm_output_hook)
            logger.info("Cortex registered Hermes output receipt bridge")
    except (ImportError, AttributeError):
        # Standalone Cortex harnesses do not ship Hermes's plugin manager.
        return
