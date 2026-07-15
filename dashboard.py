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

from .dashboard_auth import DashboardAuth, SESSION_COOKIE
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
    failed_logins: dict[str, list[float]] = {}
    failed_logins_lock = threading.RLock()
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
    schedule_cache: dict[str, object] = {"checked_at": 0.0, "value": None}

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
            if parsed.path in {"/api/snapshot", "/api/memory", "/api/sleep/status"}:
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
                        "username": auth.username() if auth_enabled else "",
                    },
                )
                return
            if not self._require_auth(complete=True):
                return
            if parsed.path == "/api/snapshot":
                snapshot = store.dashboard_snapshot()
                snapshot["review_writes_enabled"] = reviews_enabled
                snapshot["sleep_runtime"] = sleep_status()
                snapshot["sleep_schedule"] = sleep_schedule()
                self._json(HTTPStatus.OK, snapshot)
                return
            if parsed.path == "/api/sleep/status":
                self._json(HTTPStatus.OK, sleep_status())
                return
            if parsed.path == "/api/memory":
                raw_id = parse_qs(parsed.query).get("id", [""])[0]
                memory_id = store.resolve_id(raw_id)
                if not memory_id:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "memory not found"})
                    return
                self._json(HTTPStatus.OK, store.explain(memory_id))
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
                "/api/sleep/start",
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
            needs_complete_auth = parsed.path.startswith("/api/review/") or parsed.path == "/api/sleep/start"
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
            if not reviews_enabled:
                self._json(HTTPStatus.FORBIDDEN, {"error": "guided review changes are disabled on this dashboard"})
                return
            payload = self._read_json()
            if payload is None:
                return
            try:
                if parsed.path == "/api/review/conflict":
                    changed = store.resolve_contradiction(
                        str(payload.get("first_id") or ""),
                        str(payload.get("second_id") or ""),
                        str(payload.get("resolution") or ""),
                    )
                else:
                    changed = store.review_inference(
                        str(payload.get("memory_id") or ""),
                        str(payload.get("resolution") or ""),
                    )
            except ValueError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            if not changed:
                self._json(HTTPStatus.CONFLICT, {"error": "This review item is no longer active. Refresh and try again."})
                return
            self._json(HTTPStatus.OK, {"success": True})

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

        def _login(self) -> None:
            if not auth_enabled:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "authentication is disabled"})
                return
            client = self._client_key()
            now = time.monotonic()
            with failed_logins_lock:
                recent = [stamp for stamp in failed_logins.get(client, []) if now - stamp < 300]
                failed_logins[client] = recent
                if len(recent) >= 8:
                    self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "Too many attempts. Try again shortly."})
                    return
            payload = self._read_json()
            if payload is None:
                return
            username = str(payload.get("username") or "")
            password = str(payload.get("password") or "")
            if len(username) > 128 or len(password) > 256 or not auth.verify_password(username, password):
                with failed_logins_lock:
                    failed_logins.setdefault(client, []).append(now)
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "Username or password is incorrect."})
                return
            with failed_logins_lock:
                failed_logins.pop(client, None)
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
                authenticated = bool(credentials and auth.verify_password(*credentials))
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
            forwarded = self.headers.get("CF-Connecting-IP") or self.headers.get("X-Forwarded-For") or ""
            return forwarded.split(",", 1)[0].strip() or self.client_address[0]

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
        store.close()
