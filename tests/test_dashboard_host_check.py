"""Host-allowlist enforcement on the dashboard's HTTP surface.

DNS rebinding is the one attack the existing `X-Cortex-Request` header does not
stop: the attacker's page is same-origin with the rebound server, so the browser
will happily send that header, and the request arrives with
`Host: <attacker domain>`. Validating `Host` is the defense, and it has to be
opt-in because a reverse-proxied or tunnelled dashboard legitimately receives its
public hostname.

These tests drive the real handler over a socket so the header handling is
exercised end to end, not simulated.
"""

from __future__ import annotations

import ast
import http.client
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path

from tests._bootstrap import ROOT

from cortex import dashboard as dashboard_module
from cortex.dashboard import _normalize_allowed_hosts
from cortex.dashboard_auth import DashboardAuth


def _handler_bound_to(auth: DashboardAuth, *, allowed_hosts: set[str]) -> type:
    """Extract ``serve_dashboard``'s Handler with its bindings overridden.

    Mirrors the harness in ``tests/test_dashboard_auth_enforcement.py``: the class
    body is compiled against the real dashboard module namespace with only the
    names the test needs replaced.
    """

    tree = ast.parse((ROOT / "dashboard.py").read_text(encoding="utf-8"))
    serve = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "serve_dashboard"
    )
    handler_def = next(
        node for node in serve.body if isinstance(node, ast.ClassDef) and node.name == "Handler"
    )
    namespace = dict(vars(dashboard_module))
    namespace.update(
        auth=auth,
        failed_logins={},
        failed_logins_lock=threading.RLock(),
        trusted_proxies=set(),
        reviews_enabled=False,
        allowed_hosts=allowed_hosts,
        # handler attributes normally bound by serve_dashboard's closure
        public_assets={},
        html="",
    )
    exec(
        compile(
            ast.Module(body=[handler_def], type_ignores=[]),
            str(ROOT / "dashboard.py"),
            "exec",
        ),
        namespace,
    )
    return namespace["Handler"]


class _ServerFixture:
    def __init__(self, handler: type) -> None:
        self.server = HTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def request(self, method: str, path: str, host: str) -> tuple[int, str]:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            # skip_host + explicit putheader is the only way to send a Host the
            # client would otherwise refuse to forge.
            connection.putrequest(method, path, skip_host=True)
            connection.putheader("Host", host)
            connection.putheader("X-Cortex-Request", "1")
            connection.endheaders()
            response = connection.getresponse()
            body = response.read().decode("utf-8", "replace")
            return response.status, body
        finally:
            connection.close()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class AllowedHostParsingTests(unittest.TestCase):
    def test_parsing_drops_ports_and_case(self) -> None:
        self.assertEqual(
            _normalize_allowed_hosts(" 127.0.0.1 , LocalHost:8100 , [::1]:9 ,, Brain.Example.COM "),
            {"127.0.0.1", "localhost", "[::1]", "brain.example.com"},
        )

    def test_empty_setting_disables_enforcement(self) -> None:
        self.assertEqual(_normalize_allowed_hosts(""), set())
        self.assertEqual(_normalize_allowed_hosts("   "), set())


class HostAllowlistEnforcementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.auth = DashboardAuth(Path(self.tmp.name) / "dashboard-auth.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _serve(self, allowed_hosts: set[str]) -> _ServerFixture:
        return _ServerFixture(_handler_bound_to(self.auth, allowed_hosts=allowed_hosts))

    def test_configured_allowlist_rejects_a_foreign_host(self) -> None:
        fixture = self._serve({"127.0.0.1", "localhost"})
        try:
            status, body = fixture.request("GET", "/api/auth/status", "evil.example.com")
            self.assertEqual(status, 403)
            self.assertIn("host not allowed", body)

            # The rebinding shape: the attacker's domain, with the port the
            # browser actually dialled.
            status, _body = fixture.request("GET", "/api/snapshot", "evil.example.com:8100")
            self.assertEqual(status, 403)

            # State changes are rejected too, before any handler logic runs.
            status, _body = fixture.request("POST", "/api/auth/login", "evil.example.com")
            self.assertEqual(status, 403)
        finally:
            fixture.stop()

    def test_configured_allowlist_admits_listed_hosts_with_or_without_port(self) -> None:
        fixture = self._serve({"127.0.0.1", "brain.example.com"})
        try:
            for host in ("127.0.0.1", "127.0.0.1:8100", "brain.example.com", "brainexample.com"):
                with self.subTest(host=host):
                    status, _body = fixture.request("GET", "/api/auth/status", host)
                    expected = 200 if host != "brainexample.com" else 403
                    self.assertEqual(status, expected)
        finally:
            fixture.stop()

    def test_unconfigured_allowlist_changes_nothing(self) -> None:
        """No allowlist configured -> behaviour is exactly as before."""
        fixture = self._serve(set())
        try:
            status, _body = fixture.request("GET", "/api/auth/status", "any.example.com")
            self.assertEqual(status, 200)
        finally:
            fixture.stop()


if __name__ == "__main__":
    unittest.main()
