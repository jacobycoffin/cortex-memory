"""Public Cortex Memory API and Hermes plugin registration."""

from __future__ import annotations

__version__ = "0.3.0"

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional
from .attribution import (
    attribution_score,
    format_recall_trace_receipt,
    memory_receipt_prefixes,
    referenced_memory_prefixes,
    strip_memory_receipt,
)
from .client import CortexMemory, RecallBatch, _provenance_label
from .cognition import attention_topics, plan_recall
from .extraction import extract_candidates
from .harness import (
    CORTEX_BOOTSTRAP_POINTER,
    CortexHarnessAdapter,
    HarnessTurn,
    cortex_primary_system_prompt,
    harness_contract_manifest,
)
from .metacognition import MetacognitiveAssessment, assess_retrieval
from .retrieval import (
    LIVE_SCORING_POLICY_VERSION,
    SHADOW_SCORING_POLICY_VERSION,
    MemoryRetriever,
    RetrievalContext,
    RetrievalDiagnostics,
    RetrievalResult,
)
from .research import (
    assign_recall_condition,
    complete_agent_tasks,
    record_agent_task_start,
)
from .security import neutralize_role_tags, safe_prompt_text, sanitize_memory
from .store import CortexStore
from .tooling import build_tool_workflow, classify_task, extract_tool_executions, task_fingerprint

from .hermes_hooks import (  # noqa: F401 -- compatible root imports
    _OUTPUT_HOOK_LOCK,
    _OUTPUT_PROVIDER_BY_SESSION,
    _hermes_transform_llm_output_hook,
    _install_hermes_output_hook,
)

from .hermes_receipts import (  # noqa: F401 -- compatible root imports
    _AMBIGUOUS_FEEDBACK,
    _CREATION_POSITIVE_FEEDBACK,
    _GREETING_ONLY,
    _NEGATIVE_FEEDBACK,
    _POSITIVE_FEEDBACK,
    _RECEIPT_IRRELEVANT_FEEDBACK,
    _RECEIPT_WRONG_FEEDBACK,
    _STRONG_POSITIVE_FEEDBACK,
    _receipt_feedback_outcome,
    _resolve_allowed_prefixes,
)

from .hermes_config import (  # noqa: F401 -- compatible root imports
    DEFAULTS,
    _as_bool,
    _json_error,
    _json_ok,
    _read_config,
    _string_tuple,
)

from .hermes_provider import (  # noqa: F401 -- compatible root imports
    CORTEX_MEMORY_SCHEMA,
    CortexMemoryProvider,
    MemoryProvider,
    _RecallCacheEntry,
    logger,
)


def register(ctx) -> None:
    provider = CortexMemoryProvider()
    ctx.register_memory_provider(provider)
    ctx.register_hook("transform_llm_output", provider.transform_llm_output)


__all__ = [
    "CortexMemoryProvider",
    "CortexMemory",
    "RecallBatch",
    "RetrievalContext",
    "CortexHarnessAdapter",
    "HarnessTurn",
    "CORTEX_BOOTSTRAP_POINTER",
    "cortex_primary_system_prompt",
    "harness_contract_manifest",
    "register",
]
