"""Due-aware scheduler for opt-in Cortex brain-mechanics shadow passes."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .adaptive_reconsolidation import run_adaptive_reconsolidation
from .adaptive_weights import run_adaptive_weight_learning
from .autojudge import AutoJudgeConfig
from .relevance_pruning import run_adaptive_pruning
from .schema_formation import run_schema_formation
from .semantic_consolidation import run_semantic_consolidation
from .store import CortexStore, utc_now


def run_due_brain_mechanics(
    store: CortexStore,
    config: AutoJudgeConfig,
) -> dict[str, Any]:
    """Run only due, explicitly enabled, proposal-only mechanics passes."""

    config = brain_mechanics_config(config)
    report: dict[str, Any] = {
        "policy_version": "brain_mechanics_scheduler_v1",
        "mode": "shadow",
        "generated_at": utc_now(),
        "passes": {},
        "claim_boundary": (
            "Scheduler completion means a bounded proposal pass ran. It does not mean "
            "any memory, edge, lifecycle state, or scoring profile was promoted."
        ),
    }
    definitions: list[tuple[str, str, float, Callable[[], dict[str, Any]], bool]] = [
        (
            "reconsolidation",
            "CORTEX_AUTO_JUDGE_RECONSOLIDATE",
            5.0 / 60.0,
            lambda: run_adaptive_reconsolidation(
                store,
                config,
                lability_minutes=_env_int("CORTEX_LABILITY_WINDOW_MINUTES", 30, 1, 1440),
            ),
            False,
        ),
        (
            "pruning",
            "CORTEX_AUTO_JUDGE_PRUNE",
            24.0,
            lambda: run_adaptive_pruning(
                store,
                config,
                relevance_threshold=_env_float(
                    "CORTEX_AUTO_JUDGE_PRUNE_THRESHOLD", 0.25, 0.0, 1.0
                ),
                max_candidates=_env_int(
                    "CORTEX_AUTO_JUDGE_PRUNE_MAX_PER_RUN", 50, 1, 50
                ),
                apply=False,
            ),
            False,
        ),
        (
            "consolidation",
            "CORTEX_AUTO_JUDGE_CONSOLIDATE",
            _env_float(
                "CORTEX_AUTO_JUDGE_CONSOLIDATE_INTERVAL_HOURS",
                12.0,
                1.0,
                168.0,
            ),
            lambda: run_semantic_consolidation(store, config, apply=False),
            False,
        ),
        (
            "schemas",
            "CORTEX_AUTO_JUDGE_SCHEMAS",
            24.0 * 7.0,
            lambda: run_schema_formation(
                store,
                config,
                minimum_cluster=_env_int(
                    "CORTEX_AUTO_JUDGE_SCHEMA_MIN_CLUSTER", 3, 3, 10
                ),
            ),
            True,
        ),
        (
            "weights",
            "CORTEX_AUTO_JUDGE_TUNE_WEIGHTS",
            24.0 * 7.0,
            lambda: run_adaptive_weight_learning(
                store,
                config,
                lookback_days=_env_int(
                    "CORTEX_AUTO_JUDGE_WEIGHT_AUDIT_DAYS", 7, 1, 90
                ),
            ),
            False,
        ),
    ]
    for name, flag, interval_hours, callback, requires_sleep in definitions:
        if not _env_bool(flag, False):
            report["passes"][name] = {"status": "disabled", "flag": flag}
            continue
        due = _reserve_if_due(
            store,
            name,
            interval_hours=interval_hours,
            requires_completed_sleep=requires_sleep,
        )
        if not due["reserved"]:
            report["passes"][name] = due
            continue
        try:
            result = callback()
        except Exception as exc:
            _release_reservation(store, name)
            report["passes"][name] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:400]}",
            }
            continue
        _complete_reservation(store, name)
        report["passes"][name] = {"status": "completed", "result": result}
    return report


def brain_mechanics_config(config: AutoJudgeConfig) -> AutoJudgeConfig:
    """Apply optional provider overrides shared by every mechanics pass."""

    model = os.environ.get("CORTEX_BRAIN_MECHANICS_MODEL", "").strip()
    timeout_raw = os.environ.get("CORTEX_BRAIN_MECHANICS_TIMEOUT_SECONDS", "").strip()
    if not model and not timeout_raw:
        return config
    timeout_seconds = config.timeout_seconds
    if timeout_raw:
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise ValueError(
                "CORTEX_BRAIN_MECHANICS_TIMEOUT_SECONDS must be numeric"
            ) from exc
    overridden = replace(
        config,
        model=model or config.model,
        timeout_seconds=timeout_seconds,
    )
    overridden.validate()
    return overridden


def _reserve_if_due(
    store: CortexStore,
    name: str,
    *,
    interval_hours: float,
    requires_completed_sleep: bool,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    completed_key = f"brain_mechanics:{name}:completed_at"
    claim_key = f"brain_mechanics:{name}:claimed_at"
    with store.transaction() as conn:
        completed_row = conn.execute(
            "SELECT value FROM meta WHERE key=?", (completed_key,)
        ).fetchone()
        claim_row = conn.execute("SELECT value FROM meta WHERE key=?", (claim_key,)).fetchone()
        completed_at = _parse_time(completed_row["value"] if completed_row else None)
        claimed_at = _parse_time(claim_row["value"] if claim_row else None)
        if claimed_at and now - claimed_at < timedelta(minutes=30):
            return {"status": "already_running", "reserved": False}
        if completed_at and now - completed_at < timedelta(hours=interval_hours):
            return {
                "status": "not_due",
                "reserved": False,
                "last_completed_at": completed_at.isoformat(),
            }
        if requires_completed_sleep:
            sleep = conn.execute(
                """SELECT completed_at FROM sleep_runs
                   WHERE status='completed' ORDER BY completed_at DESC LIMIT 1"""
            ).fetchone()
            sleep_at = _parse_time(sleep["completed_at"] if sleep else None)
            if not sleep_at or (completed_at and sleep_at <= completed_at):
                return {
                    "status": "awaiting_sleep",
                    "reserved": False,
                    "last_completed_at": completed_at.isoformat() if completed_at else None,
                }
        conn.execute(
            """INSERT INTO meta(key,value) VALUES(?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (claim_key, now.isoformat()),
        )
    return {"status": "reserved", "reserved": True}


def _complete_reservation(store: CortexStore, name: str) -> None:
    with store.transaction() as conn:
        conn.execute(
            """INSERT INTO meta(key,value) VALUES(?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (f"brain_mechanics:{name}:completed_at", utc_now()),
        )
        conn.execute(
            "DELETE FROM meta WHERE key=?", (f"brain_mechanics:{name}:claimed_at",)
        )


def _release_reservation(store: CortexStore, name: str) -> None:
    with store.transaction() as conn:
        conn.execute(
            "DELETE FROM meta WHERE key=?", (f"brain_mechanics:{name}:claimed_at",)
        )


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or ""))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))
