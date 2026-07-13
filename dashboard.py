"""Read-only local web dashboard for Cortex memory data."""

from __future__ import annotations

import base64
import hmac
import json
import os
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .store import CortexStore


def _basic_auth_valid(header: str | None, username: str, password: str) -> bool:
    if not header or not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
        supplied_user, supplied_password = decoded.split(":", 1)
    except (ValueError, UnicodeDecodeError):
        return False
    return hmac.compare_digest(supplied_user, username) and hmac.compare_digest(supplied_password, password)


def serve_dashboard(db_path: str | Path, *, port: int = 8765, open_browser: bool = True) -> None:
    """Serve the dashboard on localhost until interrupted.

    Set both CORTEX_DASHBOARD_USER and CORTEX_DASHBOARD_PASSWORD to require
    HTTP Basic authentication. This is intended for a TLS-terminating private
    reverse proxy such as Cloudflare Tunnel; credentials are never logged.
    """
    store = CortexStore(db_path)
    html_path = Path(__file__).with_name("dashboard.html")
    html = html_path.read_bytes()
    auth_user = os.environ.get("CORTEX_DASHBOARD_USER", "")
    auth_password = os.environ.get("CORTEX_DASHBOARD_PASSWORD", "")
    auth_enabled = bool(auth_user and auth_password)

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
            if auth_enabled and not _basic_auth_valid(self.headers.get("Authorization"), auth_user, auth_password):
                self._unauthorized(head_only=True)
                return
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._headers_only(HTTPStatus.OK, "text/html; charset=utf-8", len(html))
                return
            if parsed.path in {"/api/snapshot", "/api/memory"}:
                self._headers_only(HTTPStatus.OK, "application/json; charset=utf-8", 0)
                return
            self._headers_only(HTTPStatus.NOT_FOUND, "application/json; charset=utf-8", 0)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if auth_enabled and not _basic_auth_valid(self.headers.get("Authorization"), auth_user, auth_password):
                self._unauthorized()
                return
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", html)
                return
            if parsed.path == "/api/snapshot":
                self._json(HTTPStatus.OK, store.dashboard_snapshot())
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

        def _unauthorized(self, *, head_only: bool = False) -> None:
            payload = b"Authentication required"
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("WWW-Authenticate", 'Basic realm="Cortex Brain", charset="UTF-8"')
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if not head_only:
                self.wfile.write(payload)

        def _json(self, status: HTTPStatus, payload: object) -> None:
            self._send(status, "application/json; charset=utf-8", json.dumps(payload, default=str).encode())

        def _send(self, status: HTTPStatus, content_type: str, payload: bytes) -> None:
            self._headers_only(status, content_type, len(payload))
            self.wfile.write(payload)

        def _headers_only(self, status: HTTPStatus, content_type: str, content_length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(content_length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data:; worker-src 'self' blob:",
            )
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Cortex Brain dashboard: {url}")
    print("Read-only and bound to localhost. Press Ctrl-C to stop.")
    print(f"Browser authentication: {'enabled' if auth_enabled else 'disabled'}")
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()
