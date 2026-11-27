"""Deep-dive dashboard split: phone layout, first paint, and cached assets.

UI-005 pulls the stylesheet and script out of ``dashboard.html``, vendors the
graph libraries, and serves them from the Python dashboard. The byte budgets
are the gzip bodies a phone downloads. The files on disk stay readable so a
later task can still edit them.
"""

from __future__ import annotations

import gzip
import http.client
import os
import re
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests._bootstrap import ROOT

from cortex.dashboard import serve_dashboard


HTML_BUDGET = 60 * 1024
CSS_BUDGET = 80 * 1024
JS_BUDGET = 200 * 1024
_FEW_LINES = 6

_VIEWS = (
    "overview",
    "graph",
    "timeline",
    "review",
    "auto-judge",
    "insights",
    "cognition",
    "trust",
    "tools",
    "learning",
    "health",
    "explorer",
    "settings",
)

_VENDOR = (
    "graphology.umd.min.js",
    "sigma.min.js",
    "force-graph.min.js",
    "cytoscape.min.js",
    "3d-force-graph.min.js",
)


def _gzip_size(path: Path) -> int:
    return len(gzip.compress(path.read_bytes(), compresslevel=9))


class DashboardSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (ROOT / "dashboard.html").read_text(encoding="utf-8")
        cls.css = (ROOT / "dashboard.css").read_text(encoding="utf-8")
        cls.js = (ROOT / "dashboard.js").read_text(encoding="utf-8")

    def test_transferred_bytes_stay_inside_the_budget(self) -> None:
        html = ROOT / "dashboard.html"
        css = ROOT / "dashboard.css"
        js = ROOT / "dashboard.js"
        self.assertLess(
            _gzip_size(html),
            HTML_BUDGET,
            f"html gzip {_gzip_size(html)} raw {html.stat().st_size}",
        )
        self.assertLess(
            _gzip_size(css),
            CSS_BUDGET,
            f"css gzip {_gzip_size(css)} raw {css.stat().st_size}",
        )
        self.assertLess(
            _gzip_size(js),
            JS_BUDGET,
            f"js gzip {_gzip_size(js)} raw {js.stat().st_size}",
        )

    def test_page_has_no_large_inline_style_or_script(self) -> None:
        styles = re.findall(r"<style\b[^>]*>(.*?)</style>", self.html, flags=re.I | re.S)
        scripts = re.findall(r"<script\b(?![^>]*\bsrc=)[^>]*>(.*?)</script>", self.html, flags=re.I | re.S)
        for block in styles + scripts:
            lines = [line for line in block.splitlines() if line.strip()]
            self.assertLessEqual(len(lines), _FEW_LINES)
        self.assertIn('<link rel="stylesheet" href="/dashboard.css">', self.html)
        self.assertIn('<script src="/dashboard.js" defer></script>', self.html)
        self.assertNotIn("cdn.jsdelivr.net", self.html)
        self.assertNotIn("cdn.jsdelivr.net", self.css)
        self.assertNotIn("cdn.jsdelivr.net", self.js)

    def test_first_paint_states_the_number_and_shows_skeleton_rows(self) -> None:
        self.assertIn('id="hero-memory-count">no data yet</p>', self.html)
        self.assertGreaterEqual(self.html.count('class="skeleton-row"'), 12)
        self.assertIn('id="recent-activity" aria-busy="true"', self.html)
        self.assertIn('id="signal-list" aria-busy="true"', self.html)
        self.assertIn('id="memory-list" aria-busy="true"', self.html)
        self.assertIn('class="loading-screen hidden" id="loading-screen" hidden', self.html)
        self.assertIn('fetch("/api/snapshot"', self.js)
        self.assertNotIn("setInterval(()=>load(true),60000)", self.js)
        self.assertIn("function pollIfVisible()", self.js)

    def test_javascript_disabled_fallback_is_readable(self) -> None:
        self.assertIn("<noscript>", self.html)
        self.assertIn("JavaScript is off", self.html)
        for view in _VIEWS:
            self.assertIn(f'id="view-{view}"', self.html)
        self.assertIn('id="auth-gate"', self.html)
        self.assertIn('id="auth-login-form"', self.html)

    def test_phone_rules_wrap_the_tab_bar_and_keep_card_rows(self) -> None:
        self.assertIn("@media (max-width: 430px)", self.css)
        self.assertIn("overflow-x: clip", self.css)
        self.assertIn("grid-template-columns: repeat(4, minmax(0, 1fr))", self.css)
        self.assertIn("table, tbody, tr, td { display: block", self.css)
        self.assertIn(":root { --sidebar: 172px; }", self.css)
        self.assertIn("@media (max-width: 1220px)", self.css)
        self.assertIn("@media (max-width: 760px)", self.css)

    def test_graph_libraries_are_vendored_with_their_licenses(self) -> None:
        for name in _VENDOR:
            path = ROOT / "vendor" / name
            text = path.read_text(encoding="utf-8", errors="replace")
            self.assertIn(f"/vendor/{name}", self.js)
            self.assertIn("Permission is hereby granted", text)
        self.assertIn("Version 1.51.4 force-graph", (ROOT / "vendor" / "force-graph.min.js").read_text(encoding="utf-8", errors="replace"))
        self.assertIn("Version 1.80.0 3d-force-graph", (ROOT / "vendor" / "3d-force-graph.min.js").read_text(encoding="utf-8", errors="replace"))
        package = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"dashboard.css"', package)
        self.assertIn('"dashboard.js"', package)
        self.assertIn('"vendor/*.js"', package)

    def test_existing_views_and_auth_handlers_stay_in_the_script(self) -> None:
        for name in (
            "function renderInsights()",
            "function renderCognition()",
            "function renderSleep",
            "async function selectMemory(",
            "async function bootAuth()",
            "function renderReviewInbox()",
            "function renderHealth()",
        ):
            self.assertIn(name, self.js)
        server = (ROOT / "dashboard.py").read_text(encoding="utf-8")
        self.assertIn("def _dashboard_static", server)
        self.assertNotIn('"/api/pulse"', server)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class DashboardStaticServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls._saved = {
            key: os.environ.get(key)
            for key in (
                "CORTEX_DASHBOARD_ALLOWED_HOSTS",
                "CORTEX_DASHBOARD_USER",
                "CORTEX_DASHBOARD_PASSWORD",
                "CORTEX_DASHBOARD_AUTH_FILE",
                "CORTEX_DASHBOARD_REVIEWS",
            )
        }
        os.environ["CORTEX_DASHBOARD_ALLOWED_HOSTS"] = ""
        os.environ["CORTEX_DASHBOARD_USER"] = "viewer"
        os.environ["CORTEX_DASHBOARD_PASSWORD"] = "correct-horse-battery"
        os.environ["CORTEX_DASHBOARD_AUTH_FILE"] = str(Path(cls.tmp.name) / "dashboard-auth.json")
        os.environ["CORTEX_DASHBOARD_REVIEWS"] = ""
        cls.port = _free_port()
        cls.errors: list[BaseException] = []

        def run() -> None:
            try:
                serve_dashboard(Path(cls.tmp.name) / "cortex.db", port=cls.port, open_browser=False)
            except BaseException as exc:  # pragma: no cover - surfaced below
                cls.errors.append(exc)

        cls.thread = threading.Thread(target=run, daemon=True)
        cls.thread.start()
        deadline = time.time() + 30
        while time.time() < deadline:
            if cls.errors:
                raise cls.errors[0]
            try:
                with socket.create_connection(("127.0.0.1", cls.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise TimeoutError("dashboard did not accept connections")

    @classmethod
    def tearDownClass(cls) -> None:
        for key, value in cls._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        cls.tmp.cleanup()

    def _request(self, method: str, path: str, *, accept_encoding: str | None = None) -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        headers = {"Host": "127.0.0.1", "X-Cortex-Request": "1"}
        if accept_encoding:
            headers["Accept-Encoding"] = accept_encoding
        connection.request(method, path, headers=headers)
        response = connection.getresponse()
        try:
            body = response.read()
            headers_out = {key.lower(): value for key, value in response.getheaders()}
            return response.status, headers_out, body
        finally:
            connection.close()

    def test_css_and_js_are_cached_public_assets(self) -> None:
        for path, content_type, file_name in (
            ("/dashboard.css", "text/css; charset=utf-8", "dashboard.css"),
            ("/dashboard.js", "text/javascript; charset=utf-8", "dashboard.js"),
            ("/vendor/graphology.umd.min.js", "text/javascript; charset=utf-8", "vendor/graphology.umd.min.js"),
        ):
            with self.subTest(path=path):
                status, headers, body = self._request("GET", path)
                disk = (ROOT / file_name).read_bytes()
                stat = (ROOT / file_name).stat()
                self.assertEqual(status, 200)
                self.assertEqual(body, disk)
                self.assertEqual(headers["content-type"], content_type)
                self.assertEqual(headers["cache-control"], "public, max-age=86400")
                self.assertEqual(headers["etag"], f'"{stat.st_mtime_ns}-{stat.st_size}"')
                self.assertNotIn("content-encoding", headers)
                head_status, head_headers, head_body = self._request("HEAD", path)
                self.assertEqual(head_status, 200)
                self.assertEqual(head_body, b"")
                self.assertEqual(head_headers["content-length"], headers["content-length"])
                self.assertEqual(head_headers["cache-control"], "public, max-age=86400")

    def test_gzip_transfer_meets_the_same_budget(self) -> None:
        expectations = (
            ("/", "text/html; charset=utf-8", ROOT / "dashboard.html", HTML_BUDGET, "no-store"),
            ("/dashboard.css", "text/css; charset=utf-8", ROOT / "dashboard.css", CSS_BUDGET, "public, max-age=86400"),
            ("/dashboard.js", "text/javascript; charset=utf-8", ROOT / "dashboard.js", JS_BUDGET, "public, max-age=86400"),
        )
        for path, content_type, file_path, budget, cache in expectations:
            with self.subTest(path=path):
                status, headers, body = self._request("GET", path, accept_encoding="gzip")
                self.assertEqual(status, 200)
                self.assertEqual(headers["content-encoding"], "gzip")
                self.assertEqual(headers["content-type"], content_type)
                self.assertEqual(headers["cache-control"], cache)
                self.assertLess(len(body), budget)
                self.assertEqual(gzip.decompress(body), file_path.read_bytes())

    def test_html_and_api_stay_unstored_and_auth_still_gates_data(self) -> None:
        status, headers, body = self._request("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertIn(b"no data yet", body)
        self.assertIn(b'src="/dashboard.js"', body)

        status, headers, body = self._request("GET", "/api/auth/status")
        self.assertEqual(status, 200)
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertIn(b'"auth_enabled": true', body)

        status, headers, body = self._request("GET", "/api/snapshot")
        self.assertEqual(status, 401)
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertNotIn(b'"memories"', body)

        status, headers, _body = self._request("GET", "/dashboard.css")
        self.assertEqual(status, 200)
        self.assertEqual(headers["cache-control"], "public, max-age=86400")

    def test_vendor_paths_cannot_escape_and_missing_files_are_plain_404(self) -> None:
        for path in (
            "/vendor/../dashboard.py",
            "/vendor/not-real.js",
            "/vendor/notes.txt",
            "/vendor/nested/file.js",
            "/vendor/%2e%2e/dashboard.py",
        ):
            with self.subTest(path=path):
                status, headers, body = self._request("GET", path)
                self.assertEqual(status, 404)
                self.assertEqual(body, b"not found")
                self.assertTrue(headers["content-type"].startswith("text/plain"))


if __name__ == "__main__":
    unittest.main()
