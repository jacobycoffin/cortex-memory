"""Local Cortex dashboard with authenticated, auditable memory reviews."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import os
import shutil
import subprocess
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .benchmarking import (
    DASHBOARD_BENCHMARK_QUERIES,
    DASHBOARD_BENCHMARK_SIZES,
    DASHBOARD_BENCHMARK_SUITE,
    DASHBOARD_BENCHMARK_VERSION,
    run_dashboard_benchmark,
)
from .dashboard_auth import DashboardAuth, SESSION_COOKIE
from .evaluation import (
    REAL_HISTORY_MIN_CASES,
    REAL_HISTORY_SUITE,
    REAL_HISTORY_VERSION,
    HistoryCase,
    compare_real_history,
)
from .research import (
    create_prospective_item,
    generate_summary_candidates,
    review_summary_candidate,
    set_recall_experiment,
    start_sleep_apply_trial,
    undo_sleep_apply_trial,
    update_prospective_item,
)
from .review_copilot import (
    ReviewCopilot,
    ReviewCopilotConfig,
    ReviewCopilotError,
    ReviewCopilotUnavailable,
)
from .retrieval import MemoryRetriever
from .sleep import SleepConfig, run_sleep
from .store import CortexStore, utc_now


DEFAULT_SLEEP_SCHEDULE = "Nightly · 03:00–05:00 local window"


def _sleep_schedule_status() -> dict[str, object]:
    """Read the installed systemd timer without making the dashboard depend on it."""

    status: dict[str, object] = {
        "source": "bundled systemd timer",
        "schedule": os.environ.get("CORTEX_SLEEP_SCHEDULE_LABEL", DEFAULT_SLEEP_SCHEDULE),
        "active": None,
        "enabled": None,
        "next_run": os.environ.get("CORTEX_SLEEP_NEXT_RUN") or None,
        "last_trigger": None,
        "detail": "Starts at 03:00 local time with up to two hours of randomized delay; missed runs catch up.",
    }
    systemctl = shutil.which("systemctl")
    if not systemctl:
        status["source"] = "bundled schedule"
        status["detail"] += " Live timer state is unavailable on this host."
        return status
    try:
        result = subprocess.run(
            [
                systemctl,
                "--user",
                "show",
                "cortex-sleep.timer",
                "--property=ActiveState",
                "--property=UnitFileState",
                "--property=NextElapseUSecRealtime",
                "--property=LastTriggerUSec",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        status["detail"] += " The installed timer could not be queried."
        return status
    if result.returncode != 0:
        status["detail"] += " The timer is not installed for this dashboard user."
        return status
    values = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value.strip()
    status.update(
        {
            "source": "live systemd timer",
            "active": values.get("ActiveState") == "active",
            "enabled": values.get("UnitFileState") == "enabled",
            "next_run": values.get("NextElapseUSecRealtime") or status["next_run"],
            "last_trigger": values.get("LastTriggerUSec") or None,
        }
    )
    return status


def _basic_auth_valid(header: str | None, username: str, password: str) -> bool:
    if not header or not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
        supplied_user, supplied_password = decoded.split(":", 1)
    except (ValueError, UnicodeDecodeError):
        return False
    return hmac.compare_digest(supplied_user, username) and hmac.compare_digest(supplied_password, password)


def _basic_credentials(header: str | None) -> tuple[str, str] | None:
    if not header or not header.startswith("Basic "):
        return None
    try:
        decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
        if ":" not in decoded:
            return None
        username, password = decoded.split(":", 1)
        return username, password
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return None


def serve_dashboard(db_path: str | Path, *, port: int = 8765, open_browser: bool = True) -> None:
    """Serve the dashboard on localhost until interrupted.

    Authentication is stored as a PBKDF2 password hash beside the Cortex
    database. Legacy CORTEX_DASHBOARD_USER/PASSWORD values seed the file once;
    subsequent password changes invalidate older signed sessions.
    """
    store = CortexStore(db_path)
    store.abandon_active_benchmarks()
    store.abandon_active_evaluations()
    html_path = Path(__file__).with_name("dashboard.html")
    html = html_path.read_bytes()
    public_assets = {
        "/favicon.svg": ("image/svg+xml", Path(__file__).with_name("favicon.svg").read_bytes()),
        "/favicon.ico": ("image/x-icon", Path(__file__).with_name("favicon.ico").read_bytes()),
        "/apple-touch-icon.png": ("image/png", Path(__file__).with_name("apple-touch-icon.png").read_bytes()),
    }
    auth_user = os.environ.get("CORTEX_DASHBOARD_USER", "")
    auth_password = os.environ.get("CORTEX_DASHBOARD_PASSWORD", "")
    auth_path = Path(
        os.environ.get("CORTEX_DASHBOARD_AUTH_FILE", str(Path(db_path).expanduser().parent / "dashboard-auth.json"))
    ).expanduser()
    auth = DashboardAuth(auth_path)
    if auth_user and auth_password:
        auth.ensure(auth_user, auth_password)
    auth_enabled = auth.configured
    reviews_enabled = os.environ.get("CORTEX_DASHBOARD_REVIEWS", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    review_copilot = ReviewCopilot(ReviewCopilotConfig.from_env())
    failed_logins: dict[str, list[float]] = {}
    failed_logins_lock = threading.RLock()
    # Client IPs supplied by a reverse proxy (X-Forwarded-For / CF-Connecting-IP)
    # are only trusted when the direct peer is one of these configured proxies;
    # otherwise those headers are attacker-spoofable and would defeat the login
    # rate limiter. Empty by default -> always key on the real peer address.
    trusted_proxies = {
        ip.strip()
        for ip in os.environ.get("CORTEX_DASHBOARD_TRUSTED_PROXIES", "").split(",")
        if ip.strip()
    }
    sleep_lock = threading.RLock()
    sleep_runtime: dict[str, object] = {
        "status": "idle",
        "phase": "waiting",
        "progress": 0,
        "message": "Ready for a user-started shadow review.",
        "started_at": None,
        "completed_at": None,
        "run_id": None,
        "steps": [],
        "report": None,
        "error": None,
    }
    sleep_job: dict[str, threading.Thread | None] = {"thread": None}
    benchmark_lock = threading.RLock()
    benchmark_job: dict[str, threading.Thread | None] = {"thread": None}
    evaluation_lock = threading.RLock()
    evaluation_job: dict[str, threading.Thread | None] = {"thread": None}
    schedule_cache: dict[str, object] = {"checked_at": 0.0, "value": None}
    refinery_lock = threading.RLock()
    refinery_job: dict[str, threading.Thread | None] = {"thread": None}
    refinery_runtime: dict[str, object] = {
        "status": "idle",
        "phase": "waiting",
        "progress": 0,
        "message": "No presentation rebuild is running.",
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result": None,
    }

    def refinery_rebuild_status() -> dict[str, object]:
        with refinery_lock:
            return json.loads(json.dumps(refinery_runtime, default=str))

    def refinery_rebuild_progress(update: dict[str, object]) -> None:
        with refinery_lock:
            refinery_runtime.update(
                {
                    "status": "running",
                    "phase": str(update.get("phase") or "rebuilding"),
                    "progress": int(update.get("progress") or 0),
                    "message": str(update.get("message") or "Rebuilding presentations."),
                }
            )

    def run_refinery_rebuild() -> None:
        try:
            result = store.rebuild_presentations(progress_callback=refinery_rebuild_progress)
            with refinery_lock:
                refinery_runtime.update(
                    {
                        "status": "completed",
                        "phase": "completed",
                        "progress": 100,
                        "message": (
                            f"Rebuilt {result.get('rebuilt', 0)} presentations. "
                            "Raw memory content was not changed."
                        ),
                        "completed_at": utc_now(),
                        "result": result,
                    }
                )
        except Exception as error:
            with refinery_lock:
                refinery_runtime.update(
                    {
                        "status": "failed",
                        "phase": "failed",
                        "progress": 100,
                        "message": "Presentation rebuild stopped before completing.",
                        "completed_at": utc_now(),
                        "error": str(error)[:300],
                    }
                )

    def sleep_schedule() -> dict[str, object]:
        now = time.monotonic()
        cached = schedule_cache.get("value")
        if isinstance(cached, dict) and now - float(schedule_cache.get("checked_at") or 0) < 60:
            return dict(cached)
        value = _sleep_schedule_status()
        schedule_cache.update({"checked_at": now, "value": value})
        return dict(value)

    def sleep_status() -> dict[str, object]:
        with sleep_lock:
            return json.loads(json.dumps(sleep_runtime, default=str))

    def sleep_progress(update: dict[str, object]) -> None:
        phase = str(update.get("phase") or "working")
        with sleep_lock:
            sleep_runtime.update(
                {
                    "status": "failed" if phase == "failed" else "completed" if phase == "completed" else "running",
                    "phase": phase,
                    "progress": int(update.get("progress") or 0),
                    "message": str(update.get("message") or "Sleep review is running."),
                    "run_id": update.get("run_id") or sleep_runtime.get("run_id"),
                }
            )
            steps = sleep_runtime.setdefault("steps", [])
            if isinstance(steps, list) and (not steps or steps[-1].get("phase") != phase):
                steps.append(
                    {
                        "phase": phase,
                        "progress": int(update.get("progress") or 0),
                        "message": str(update.get("message") or ""),
                        "at": utc_now(),
                    }
                )
                del steps[:-8]
            if update.get("report") is not None:
                sleep_runtime["report"] = update["report"]
            if update.get("error") is not None:
                sleep_runtime["error"] = str(update["error"])

    def run_dashboard_sleep() -> None:
        try:
            report = run_sleep(
                store,
                SleepConfig(mode="shadow", reflection_token_budget=0),
                progress_callback=sleep_progress,
            )
            with sleep_lock:
                sleep_runtime.update(
                    {
                        "status": "completed",
                        "phase": "completed",
                        "progress": 100,
                        "message": "Sleep review complete. Proposals are ready; no memory was changed.",
                        "completed_at": utc_now(),
                        "run_id": report.get("run_id"),
                        "report": report,
                    }
                )
        except Exception as error:
            with sleep_lock:
                sleep_runtime.update(
                    {
                        "status": "failed",
                        "phase": "failed",
                        "progress": 100,
                        "message": "Sleep stopped before the review completed.",
                        "completed_at": utc_now(),
                        "error": str(error)[:300],
                    }
                )

    def run_dashboard_benchmark_job(run_id: str) -> None:
        try:
            report = run_dashboard_benchmark(
                progress_callback=lambda update: store.update_benchmark_progress(
                    run_id,
                    phase=str(update.get("phase") or "running"),
                    progress=int(update.get("progress") or 0),
                    message=str(update.get("message") or "Benchmark is running."),
                )
            )
            store.complete_benchmark_run(run_id, report)
        except Exception as error:
            store.fail_benchmark_run(run_id, str(error))

    def run_private_evaluation_job(run_id: str, cases: list[HistoryCase]) -> None:
        try:
            report = compare_real_history(
                store.path,
                cases,
                progress_callback=lambda update: store.update_evaluation_progress(
                    run_id,
                    phase=str(update.get("phase") or "running"),
                    progress=int(update.get("progress") or 0),
                    message=str(update.get("message") or "Private evaluation is running."),
                ),
            )
            store.complete_evaluation_run(run_id, report)
        except Exception as error:
            store.fail_evaluation_run(run_id, str(error))

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._headers_only(HTTPStatus.OK, "text/html; charset=utf-8", len(html))
                return
            if parsed.path in public_assets:
                content_type, payload = public_assets[parsed.path]
                self._headers_only(HTTPStatus.OK, content_type, len(payload))
                return
            if parsed.path == "/api/auth/status":
                self._headers_only(HTTPStatus.OK, "application/json; charset=utf-8", 0)
                return
            if parsed.path in {
                "/api/snapshot", "/api/memory", "/api/sleep/status",
                "/api/benchmark/status", "/api/evaluation/status",
                "/api/refinery/summary", "/api/refinery/items", "/api/refinery/shadow",
                "/api/recall-sets",
            }:
                if not self._authorized(complete=True):
                    self._headers_only(HTTPStatus.UNAUTHORIZED, "application/json; charset=utf-8", 0)
                else:
                    self._headers_only(HTTPStatus.OK, "application/json; charset=utf-8", 0)
                return
            self._headers_only(HTTPStatus.NOT_FOUND, "application/json; charset=utf-8", 0)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", html)
                return
            if parsed.path in public_assets:
                content_type, payload = public_assets[parsed.path]
                self._send(HTTPStatus.OK, content_type, payload)
                return
            if parsed.path == "/api/auth/status":
                authenticated = self._authorized(complete=False)
                self._json(
                    HTTPStatus.OK,
                    {
                        "auth_enabled": auth_enabled,
                        "authenticated": authenticated,
                        "must_change_password": auth.must_change_password() if authenticated and auth_enabled else False,
                        "username": auth.username() if authenticated and auth_enabled else "",
                    },
                )
                return
            if not self._require_auth(complete=True):
                return
            if parsed.path == "/api/snapshot":
                snapshot = store.dashboard_snapshot()
                snapshot["review_writes_enabled"] = reviews_enabled
                snapshot["review_copilot"] = review_copilot.status()
                snapshot["sleep_runtime"] = sleep_status()
                snapshot["sleep_schedule"] = sleep_schedule()
                self._json(HTTPStatus.OK, snapshot)
                return
            if parsed.path == "/api/sleep/status":
                self._json(HTTPStatus.OK, sleep_status())
                return
            if parsed.path == "/api/benchmark/status":
                self._json(HTTPStatus.OK, store.benchmark_snapshot())
                return
            if parsed.path == "/api/evaluation/status":
                self._json(HTTPStatus.OK, store.evaluation_snapshot())
                return
            if parsed.path == "/api/memory":
                raw_id = parse_qs(parsed.query).get("id", [""])[0]
                memory_id = store.resolve_id(raw_id)
                if not memory_id:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "memory not found"})
                    return
                self._json(HTTPStatus.OK, store.explain(memory_id))
                return
            if parsed.path == "/api/refinery/summary":
                summary = store.refinery_summary()
                summary["rebuild"] = refinery_rebuild_status()
                summary["review_writes_enabled"] = reviews_enabled
                self._json(HTTPStatus.OK, summary)
                return
            if parsed.path == "/api/refinery/items":
                params = parse_qs(parsed.query)
                try:
                    items = store.refinery_items(
                        view=params.get("view", ["readable"])[0],
                        limit=int(params.get("limit", ["60"])[0]),
                        offset=int(params.get("offset", ["0"])[0]),
                    )
                except ValueError as error:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                self._json(HTTPStatus.OK, items)
                return
            if parsed.path == "/api/refinery/shadow":
                params = parse_qs(parsed.query)
                query_text = params.get("q", [""])[0][:500]
                if not query_text.strip():
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "q is required"})
                    return
                retriever = MemoryRetriever(store)
                self._json(HTTPStatus.OK, retriever.shadow_tiered_comparison(query_text))
                return
            if parsed.path == "/api/recall-sets":
                self._json(HTTPStatus.OK, {"recall_sets": store.recall_set_snapshot()})
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            allowed_paths = {
                "/api/auth/login",
                "/api/auth/change-password",
                "/api/auth/logout",
                "/api/review/conflict",
                "/api/review/inference",
                "/api/review/copilot",
                "/api/review/proposal",
                "/api/review/creation",
                "/api/review/undo",
                "/api/policy/action",
                "/api/sleep/start",
                "/api/benchmark/start",
                "/api/evaluation/start",
                "/api/outcome/label",
                "/api/outcome/undo",
                "/api/experiment/control",
                "/api/sleep/trial/start",
                "/api/sleep/trial/undo",
                "/api/summary/generate",
                "/api/summary/review",
                "/api/prospective/create",
                "/api/prospective/update",
                "/api/refinery/preview",
                "/api/refinery/action",
                "/api/refinery/undo",
                "/api/refinery/rebuild-presentations",
                "/api/recall-sets/preview",
                "/api/recall-sets/activate",
                "/api/recall-sets/switch",
            }
            if parsed.path not in allowed_paths:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if self.headers.get("X-Cortex-Request") != "1":
                self._json(HTTPStatus.FORBIDDEN, {"error": "request verification failed"})
                return
            if parsed.path == "/api/auth/login":
                self._login()
                return
            needs_complete_auth = (
                parsed.path.startswith(
                    ("/api/review/", "/api/policy/", "/api/refinery/", "/api/recall-sets/")
                )
                or parsed.path in {
                    "/api/sleep/start", "/api/benchmark/start", "/api/evaluation/start",
                    "/api/outcome/label", "/api/outcome/undo", "/api/experiment/control",
                    "/api/sleep/trial/start", "/api/sleep/trial/undo", "/api/summary/generate",
                    "/api/summary/review", "/api/prospective/create", "/api/prospective/update",
                }
            )
            if not self._require_auth(complete=needs_complete_auth):
                return
            if parsed.path == "/api/auth/logout":
                self._json(HTTPStatus.OK, {"success": True}, set_cookie=self._expired_cookie())
                return
            if parsed.path == "/api/auth/change-password":
                self._change_password()
                return
            if parsed.path == "/api/sleep/start":
                self._start_sleep()
                return
            if parsed.path == "/api/benchmark/start":
                self._start_benchmark()
                return
            if parsed.path == "/api/evaluation/start":
                self._start_evaluation()
                return
            if parsed.path in {"/api/outcome/label", "/api/outcome/undo"}:
                self._outcome_feedback(undo=parsed.path.endswith("/undo"))
                return
            if parsed.path.startswith(("/api/experiment/", "/api/sleep/trial/", "/api/summary/", "/api/prospective/")):
                self._research_action(parsed.path)
                return
            if parsed.path == "/api/refinery/preview":
                preview_payload = self._read_json()
                if preview_payload is None:
                    return
                try:
                    preview = store.refinery_preview(
                        str(preview_payload.get("kind") or ""),
                        str(preview_payload.get("memory_id") or ""),
                        target_role=(
                            str(preview_payload.get("target_role"))
                            if preview_payload.get("target_role")
                            else None
                        ),
                    )
                except ValueError as error:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                self._json(HTTPStatus.OK, {"success": True, "preview": preview})
                return
            if not reviews_enabled:
                self._json(HTTPStatus.FORBIDDEN, {"error": "guided review changes are disabled on this dashboard"})
                return
            payload = self._read_json()
            if payload is None:
                return
            try:
                actor = auth.username() if auth_enabled else "local-operator"
                if parsed.path == "/api/review/copilot":
                    proposal_id = str(payload.get("proposal_id") or "")
                    raw_conversation = payload.get("conversation")
                    conversation = raw_conversation if isinstance(raw_conversation, list) else []
                    item = next(
                        (
                            row
                            for row in store.review_inbox_snapshot(limit=1000)["items"]
                            if row.get("proposal_id") == proposal_id
                        ),
                        None,
                    )
                    if not item:
                        raise ValueError("this proposal is no longer waiting for review")
                    try:
                        result = review_copilot.interpret(
                            item,
                            str(payload.get("operator_text") or ""),
                            conversation=conversation,
                        )
                    except ReviewCopilotUnavailable as error:
                        self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
                        return
                    except ReviewCopilotError as error:
                        self._json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
                        return
                    audit_response = {
                        "mode": result["mode"],
                        "message": result.get("message"),
                        "question": result.get("question"),
                        "recommendation": result.get("recommendation"),
                    }
                    interpretation_id = store.record_review_copilot_interpretation(
                        proposal_id=proposal_id,
                        operator_text=str(result["operator_text"]),
                        conversation=list(result["conversation"]),
                        response_mode=str(result["mode"]),
                        response=audit_response,
                        provider=str(result["provider"]),
                        model=str(result["model"]),
                        usage=dict(result.get("usage") or {}),
                    )
                    self._json(
                        HTTPStatus.OK,
                        {
                            "success": True,
                            **audit_response,
                            "interpretation_id": interpretation_id,
                            "provider": result["provider"],
                            "model": result["model"],
                        },
                    )
                    return
                if parsed.path == "/api/policy/action":
                    self._policy_action(payload, actor=actor)
                    return
                if parsed.path == "/api/review/creation":
                    result = store.review_memory_creation(
                        str(payload.get("proposal_id") or ""),
                        str(payload.get("action") or ""),
                        edited_content=(
                            str(payload.get("content")) if payload.get("content") is not None else None
                        ),
                        reason_text=str(payload.get("reason_text") or ""),
                        actor=actor,
                        decision_scope=str(payload.get("decision_scope") or "item_only"),
                    )
                    self._json(HTTPStatus.OK, {"success": True, "result": result})
                    return
                if parsed.path == "/api/recall-sets/preview":
                    self._json(
                        HTTPStatus.OK,
                        {"success": True, "preview": store.preview_trained_recall_set()},
                    )
                    return
                if parsed.path == "/api/recall-sets/activate":
                    result = store.start_trained_recall_set(
                        actor=actor,
                        expected_preview_id=str(payload.get("preview_id") or ""),
                    )
                    self._json(HTTPStatus.OK, {"success": True, "result": result})
                    return
                if parsed.path == "/api/recall-sets/switch":
                    target = str(payload.get("target") or "")
                    snapshot = store.recall_set_snapshot()
                    if target in {"legacy", "trained"}:
                        target_id = str(
                            (snapshot.get("sets", {}).get(target) or {}).get("recall_set_id") or ""
                        )
                    else:
                        target_id = str(payload.get("recall_set_id") or target)
                    result = store.activate_recall_set(target_id, actor=actor)
                    self._json(HTTPStatus.OK, {"success": True, "result": result})
                    return
                if parsed.path == "/api/refinery/action":
                    proposed = payload.get("proposed_records")
                    result = store.apply_refinery_action(
                        str(payload.get("action") or ""),
                        str(payload.get("memory_id") or ""),
                        proposed_records=proposed if isinstance(proposed, list) else None,
                        reason_code=str(payload.get("reason_code") or "unspecified"),
                        reason_text=str(payload.get("reason_text") or ""),
                        actor=actor,
                        decision_scope=str(payload.get("decision_scope") or "item_only"),
                    )
                    self._json(HTTPStatus.OK, {"success": True, "result": result})
                    return
                if parsed.path == "/api/refinery/undo":
                    changed = store.undo_refinery_action(
                        str(payload.get("review_id") or ""), actor=actor
                    )
                    if not changed:
                        self._json(HTTPStatus.CONFLICT, {"error": "This refinery decision cannot be reversed."})
                        return
                    self._json(HTTPStatus.OK, {"success": True})
                    return
                if parsed.path == "/api/refinery/rebuild-presentations":
                    with refinery_lock:
                        worker = refinery_job.get("thread")
                        if worker and worker.is_alive():
                            self._json(HTTPStatus.CONFLICT, {"error": "a presentation rebuild is already running"})
                            return
                        refinery_runtime.update(
                            {
                                "status": "running",
                                "phase": "starting",
                                "progress": 1,
                                "message": "Starting a bounded presentation rebuild.",
                                "started_at": utc_now(),
                                "completed_at": None,
                                "error": None,
                                "result": None,
                            }
                        )
                        worker = threading.Thread(
                            target=run_refinery_rebuild,
                            name="cortex-refinery-rebuild",
                            daemon=True,
                        )
                        refinery_job["thread"] = worker
                        worker.start()
                    self._json(HTTPStatus.ACCEPTED, refinery_rebuild_status())
                    return
                if parsed.path == "/api/review/proposal":
                    result = store.decide_review_proposal(
                        str(payload.get("proposal_id") or ""),
                        str(payload.get("action") or ""),
                        reason_code=str(payload.get("reason_code") or "unspecified"),
                        reason_text=str(payload.get("reason_text") or ""),
                        actor=actor,
                        decision_scope=str(payload.get("decision_scope") or "item_only"),
                        copilot_interpretation_id=(
                            str(payload.get("copilot_interpretation_id"))
                            if payload.get("copilot_interpretation_id")
                            else None
                        ),
                    )
                    self._json(HTTPStatus.OK, {"success": True, "result": result})
                    return
                if parsed.path == "/api/review/undo":
                    review_id = str(payload.get("review_id") or "")
                    changed = store.undo_review_decision(review_id, actor=actor)
                    if not changed:
                        # Refinery decisions live in their own reversible ledger.
                        changed = store.undo_refinery_action(review_id, actor=actor)
                    if not changed:
                        self._json(HTTPStatus.CONFLICT, {"error": "This review decision cannot be reversed."})
                        return
                    self._json(HTTPStatus.OK, {"success": True})
                    return
                if parsed.path == "/api/review/conflict":
                    if str(payload.get("decision_scope") or "item_only") == "exact_duplicates":
                        raise ValueError("exact-duplicate reach is not available for a conflict review")
                    first_id = str(payload.get("first_id") or "")
                    second_id = str(payload.get("second_id") or "")
                    resolution = str(payload.get("resolution") or "")
                    changed = store.resolve_contradiction(
                        first_id,
                        second_id,
                        resolution,
                    )
                else:
                    memory_id = str(payload.get("memory_id") or "")
                    resolution = str(payload.get("resolution") or "")
                    changed = store.review_inference_with_scope(
                        memory_id,
                        resolution,
                        decision_scope=str(payload.get("decision_scope") or "item_only"),
                    )
            except ValueError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            if not changed:
                self._json(HTTPStatus.CONFLICT, {"error": "This review item is no longer active. Refresh and try again."})
                return
            reason_code = str(payload.get("reason_code") or "unspecified")
            reason_text = str(payload.get("reason_text") or "")
            decision_scope = str(payload.get("decision_scope") or "item_only")
            if parsed.path == "/api/review/conflict":
                store.record_operator_review(
                    item_type="conflict",
                    item_key=f"conflict:{':'.join(sorted((first_id, second_id)))}",
                    action=resolution,
                    reason_code=reason_code,
                    reason_text=reason_text,
                    actor=actor,
                    src_id=first_id,
                    dst_id=second_id,
                    decision_scope=decision_scope,
                )
            else:
                store.record_operator_review(
                    item_type="inference",
                    item_key=f"inference:{memory_id}",
                    action=resolution,
                    reason_code=reason_code,
                    reason_text=reason_text,
                    actor=actor,
                    src_id=memory_id,
                    effect={"memory_ids": list(changed)},
                    decision_scope=decision_scope,
                )
            self._json(
                HTTPStatus.OK,
                {
                    "success": True,
                    "result": {
                        "decision_scope": decision_scope,
                        "affected_memory_ids": (
                            list(changed)
                            if parsed.path != "/api/review/conflict"
                            else [first_id, second_id]
                        ),
                    },
                },
            )

        def _policy_action(self, payload: dict[str, object], *, actor: str) -> None:
            action = str(payload.get("action") or "").casefold()
            candidate_id = str(payload.get("candidate_id") or "")
            if action == "compile":
                result = store.compile_policy_candidates()
            elif action == "replay":
                result = store.evaluate_policy_candidate(candidate_id, actor=actor)
            elif action == "start_shadow":
                result = store.start_policy_shadow(candidate_id, actor=actor)
            elif action == "promote":
                result = store.promote_policy_candidate(
                    candidate_id,
                    activation_scope=str(payload.get("activation_scope") or "scoped"),
                    actor=actor,
                )
            elif action == "reject":
                changed = store.reject_policy_candidate(
                    candidate_id,
                    reason=str(payload.get("reason") or "operator rejected"),
                    actor=actor,
                )
                if not changed:
                    raise ValueError("this policy candidate cannot be rejected")
                result = {"changed": True, "candidate_id": candidate_id}
            elif action == "rollback":
                version_id = str(payload.get("version_id") or "")
                reason = str(payload.get("reason") or "operator requested rollback")
                changed = store.rollback_policy_version(version_id, reason=reason, actor=actor)
                if not changed:
                    raise ValueError("this policy version is not active")
                result = {"changed": True, "version_id": version_id}
            else:
                raise ValueError("unsupported policy action")
            self._json(
                HTTPStatus.OK,
                {"success": True, "result": result, "policy_training": store.policy_training_snapshot()},
            )

        def _start_sleep(self) -> None:
            payload = self._read_json()
            if payload is None:
                return
            if str(payload.get("mode") or "shadow").casefold() != "shadow":
                self._json(HTTPStatus.BAD_REQUEST, {"error": "dashboard Sleep sessions are shadow-only"})
                return
            with sleep_lock:
                if sleep_runtime.get("status") in {"starting", "running"}:
                    self._json(HTTPStatus.CONFLICT, {"error": "a Sleep session is already running"})
                    return
                sleep_runtime.update(
                    {
                        "status": "starting",
                        "phase": "starting",
                        "progress": 1,
                        "message": "Opening a bounded shadow review.",
                        "started_at": utc_now(),
                        "completed_at": None,
                        "run_id": None,
                        "steps": [],
                        "report": None,
                        "error": None,
                    }
                )
                worker = threading.Thread(target=run_dashboard_sleep, name="cortex-dashboard-sleep", daemon=True)
                sleep_job["thread"] = worker
                worker.start()
            self._json(HTTPStatus.ACCEPTED, sleep_status())

        def _start_benchmark(self) -> None:
            payload = self._read_json()
            if payload is None:
                return
            if str(payload.get("suite") or "standard").casefold() != "standard":
                self._json(HTTPStatus.BAD_REQUEST, {"error": "only the fixed standard benchmark is available"})
                return
            with benchmark_lock:
                worker = benchmark_job.get("thread")
                active = store.benchmark_snapshot(limit=5).get("active")
                if (worker and worker.is_alive()) or active:
                    self._json(HTTPStatus.CONFLICT, {"error": "a benchmark is already running"})
                    return
                run_id = store.begin_benchmark_run(
                    suite=DASHBOARD_BENCHMARK_SUITE,
                    suite_version=DASHBOARD_BENCHMARK_VERSION,
                    corpus_memories=max(DASHBOARD_BENCHMARK_SIZES),
                    queries=DASHBOARD_BENCHMARK_QUERIES,
                )
                worker = threading.Thread(
                    target=run_dashboard_benchmark_job,
                    args=(run_id,),
                    name="cortex-dashboard-benchmark",
                    daemon=True,
                )
                benchmark_job["thread"] = worker
                worker.start()
            self._json(HTTPStatus.ACCEPTED, store.benchmark_snapshot())

        def _start_evaluation(self) -> None:
            payload = self._read_json()
            if payload is None:
                return
            if str(payload.get("suite") or "real_history").casefold() != "real_history":
                self._json(HTTPStatus.BAD_REQUEST, {"error": "only the private real-history suite is available"})
                return
            case_rows = store.active_evaluation_cases()
            if len(case_rows) < REAL_HISTORY_MIN_CASES:
                self._json(
                    HTTPStatus.CONFLICT,
                    {"error": f"label at least {REAL_HISTORY_MIN_CASES} helpful or validated tasks first"},
                )
                return
            cases = [
                HistoryCase(
                    query=str(row["query"]),
                    relevant_memory_ids=tuple(row["relevant_memory_ids"]),
                    task_type=str(row.get("task_type") or "general"),
                )
                for row in case_rows
            ]
            with evaluation_lock:
                worker = evaluation_job.get("thread")
                active = store.evaluation_snapshot(limit=5).get("active")
                if (worker and worker.is_alive()) or active:
                    self._json(HTTPStatus.CONFLICT, {"error": "a private evaluation is already running"})
                    return
                run_id = store.begin_evaluation_run(
                    suite=REAL_HISTORY_SUITE,
                    suite_version=REAL_HISTORY_VERSION,
                    case_count=len(cases),
                )
                worker = threading.Thread(
                    target=run_private_evaluation_job,
                    args=(run_id, cases),
                    name="cortex-dashboard-private-evaluation",
                    daemon=True,
                )
                evaluation_job["thread"] = worker
                worker.start()
            self._json(HTTPStatus.ACCEPTED, store.evaluation_snapshot())

        def _outcome_feedback(self, *, undo: bool) -> None:
            if not reviews_enabled:
                self._json(HTTPStatus.FORBIDDEN, {"error": "outcome feedback is disabled on this dashboard"})
                return
            payload = self._read_json()
            if payload is None:
                return
            task_id = str(payload.get("task_id") or "")
            if not task_id:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "task_id is required"})
                return
            try:
                actor = auth.username() if auth_enabled else "local-operator"
                if undo:
                    changed = store.undo_task_outcome_label(task_id, actor=actor)
                    if not changed:
                        self._json(HTTPStatus.CONFLICT, {"error": "this task has no active dashboard label"})
                        return
                    result: object = {"changed": True, "task_id": task_id, "outcome": None}
                else:
                    decision_scope = str(payload.get("decision_scope") or "item_only")
                    if decision_scope == "exact_duplicates":
                        raise ValueError("exact-duplicate reach is not available for an answer review")
                    result = store.label_task_outcome(
                        task_id,
                        str(payload.get("outcome") or "").casefold(),
                        actor=actor,
                    )
                    store.record_operator_review(
                        item_type="outcome",
                        item_key=f"outcome:{task_id}",
                        action=str(payload.get("outcome") or "").casefold(),
                        reason_code=str(payload.get("reason_code") or "direct_assessment"),
                        reason_text=str(payload.get("reason_text") or ""),
                        actor=actor,
                        effect={"memory_ids": result.get("memory_ids", [])},
                        decision_scope=decision_scope,
                    )
            except ValueError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self._json(HTTPStatus.OK, {"result": result, "outcome_lab": store.outcome_lab_snapshot()})

        def _research_action(self, path: str) -> None:
            if not reviews_enabled:
                self._json(HTTPStatus.FORBIDDEN, {"error": "research and memory changes are disabled on this dashboard"})
                return
            payload = self._read_json()
            if payload is None:
                return
            actor = auth.username() if auth_enabled else "local-operator"
            try:
                if path == "/api/experiment/control":
                    action = str(payload.get("action") or "start").casefold()
                    if action not in {"start", "stop"}:
                        raise ValueError("experiment action must be start or stop")
                    result = set_recall_experiment(store, active=action == "start")
                elif path == "/api/sleep/trial/start":
                    result = start_sleep_apply_trial(store, pair_limit=int(payload.get("pair_limit") or 8))
                elif path == "/api/sleep/trial/undo":
                    result = undo_sleep_apply_trial(store, str(payload.get("trial_id") or ""))
                elif path == "/api/summary/generate":
                    result = generate_summary_candidates(store, limit=int(payload.get("limit") or 6))
                elif path == "/api/summary/review":
                    result = review_summary_candidate(
                        store,
                        str(payload.get("candidate_id") or ""),
                        action=str(payload.get("action") or "").casefold(),
                        actor=actor,
                    )
                elif path == "/api/prospective/create":
                    result = create_prospective_item(
                        store,
                        content=str(payload.get("content") or ""),
                        due_at=str(payload.get("due_at") or "") or None,
                        actor=actor,
                    )
                else:
                    memory_id = store.resolve_id(str(payload.get("memory_id") or ""))
                    if not memory_id:
                        raise ValueError("prospective memory not found")
                    result = update_prospective_item(
                        store,
                        memory_id,
                        status=str(payload.get("status") or "open").casefold(),
                        due_at=(str(payload.get("due_at")) if "due_at" in payload else None),
                    )
            except (TypeError, ValueError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self._json(HTTPStatus.OK, {"result": result})

        def _login(self) -> None:
            if not auth_enabled:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "authentication is disabled"})
                return
            if not self._throttle_ok():
                self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "Too many attempts. Try again shortly."})
                return
            payload = self._read_json()
            if payload is None:
                return
            username = str(payload.get("username") or "")
            password = str(payload.get("password") or "")
            if len(username) > 128 or len(password) > 256 or not auth.verify_password(username, password):
                self._throttle_record()
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "Username or password is incorrect."})
                return
            self._throttle_clear()
            self._json(
                HTTPStatus.OK,
                {
                    "authenticated": True,
                    "must_change_password": auth.must_change_password(),
                    "username": auth.username(),
                },
                set_cookie=self._session_cookie(auth.issue_session()),
            )

        def _change_password(self) -> None:
            payload = self._read_json()
            if payload is None:
                return
            try:
                token = auth.change_password(
                    auth.username(),
                    str(payload.get("current_password") or ""),
                    str(payload.get("new_password") or ""),
                )
            except ValueError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self._json(
                HTTPStatus.OK,
                {"success": True, "must_change_password": False, "username": auth.username()},
                set_cookie=self._session_cookie(token),
            )

        def _authorized(self, *, complete: bool) -> bool:
            if not auth_enabled:
                return True
            authenticated = auth.session_from_cookie(self.headers.get("Cookie")) is not None
            if not authenticated:
                credentials = _basic_credentials(self.headers.get("Authorization"))
                # Throttle Basic-auth verification: each attempt runs an expensive
                # PBKDF2, so an un-limited stream of them is a CPU-exhaustion vector.
                if credentials and self._throttle_ok():
                    if auth.verify_password(*credentials):
                        authenticated = True
                        self._throttle_clear()
                    else:
                        self._throttle_record()
            return authenticated and (not complete or not auth.must_change_password())

        def _require_auth(self, *, complete: bool) -> bool:
            if self._authorized(complete=complete):
                return True
            authenticated = self._authorized(complete=False)
            if authenticated and complete:
                self._json(HTTPStatus.FORBIDDEN, {"error": "password change required", "must_change_password": True})
            else:
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "authentication required"})
            return False

        def _read_json(self) -> dict[str, object] | None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16_384:
                    raise ValueError
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError
                return payload
            except (ValueError, json.JSONDecodeError):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON request"})
                return None

        def _client_key(self) -> str:
            peer = self.client_address[0]
            # Only honor proxy-forwarded client IPs from a configured trusted
            # proxy; otherwise the header is spoofable and defeats throttling.
            if peer in trusted_proxies:
                forwarded = self.headers.get("CF-Connecting-IP") or self.headers.get("X-Forwarded-For") or ""
                client = forwarded.split(",", 1)[0].strip()
                if client:
                    return client
            return peer

        def _throttle_ok(self) -> bool:
            """True if this client is under the failed-attempt limit (no record kept)."""
            client = self._client_key()
            now = time.monotonic()
            with failed_logins_lock:
                recent = [stamp for stamp in failed_logins.get(client, []) if now - stamp < 300]
                failed_logins[client] = recent
                return len(recent) < 8

        def _throttle_record(self) -> None:
            client = self._client_key()
            with failed_logins_lock:
                failed_logins.setdefault(client, []).append(time.monotonic())

        def _throttle_clear(self) -> None:
            with failed_logins_lock:
                failed_logins.pop(self._client_key(), None)

        def _session_cookie(self, token: str) -> str:
            secure = self.headers.get("X-Forwarded-Proto", "").casefold() == "https"
            attributes = [f"{SESSION_COOKIE}={token}", "Path=/", "HttpOnly", "SameSite=Strict", "Max-Age=43200"]
            if secure:
                attributes.append("Secure")
            return "; ".join(attributes)

        @staticmethod
        def _expired_cookie() -> str:
            return f"{SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"

        def _json(self, status: HTTPStatus, payload: object, *, set_cookie: str | None = None) -> None:
            self._send(
                status,
                "application/json; charset=utf-8",
                json.dumps(payload, default=str).encode(),
                set_cookie=set_cookie,
            )

        def _send(
            self,
            status: HTTPStatus,
            content_type: str,
            payload: bytes,
            *,
            set_cookie: str | None = None,
        ) -> None:
            self._headers_only(status, content_type, len(payload), set_cookie=set_cookie)
            self.wfile.write(payload)

        def _headers_only(
            self,
            status: HTTPStatus,
            content_type: str,
            content_length: int,
            *,
            set_cookie: str | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(content_length))
            if set_cookie:
                self.send_header("Set-Cookie", set_cookie)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data:; worker-src 'self' blob:; form-action 'self'; "
                "base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Cortex Brain dashboard: {url}")
    print("Bound to localhost; review changes require an authenticated session. Press Ctrl-C to stop.")
    print(f"Browser authentication: {'form + signed session' if auth_enabled else 'disabled'}")
    print(f"Guided review changes: {'enabled' if reviews_enabled else 'disabled (read-only)'}")
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker = sleep_job.get("thread")
        if worker and worker.is_alive():
            worker.join()
        benchmark_worker = benchmark_job.get("thread")
        if benchmark_worker and benchmark_worker.is_alive():
            benchmark_worker.join()
        evaluation_worker = evaluation_job.get("thread")
        if evaluation_worker and evaluation_worker.is_alive():
            evaluation_worker.join()
        refinery_worker = refinery_job.get("thread")
        if refinery_worker and refinery_worker.is_alive():
            refinery_worker.join()
        store.close()
