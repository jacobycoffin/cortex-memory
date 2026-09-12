"""Reflection calls must not forward credentials across an origin change.

``urllib`` follows 3xx responses automatically and carries request headers
along, so a provider that redirected the reflection call to another host would
receive the ``Authorization`` bearer token.  Same-origin redirects stay allowed.
"""

from __future__ import annotations

import http.server
import json
import threading
import unittest

from tests._bootstrap import ROOT  # noqa: F401 - loads the flat ``cortex`` package

from cortex.sleep import _post_chat

TOKEN = "synthetic-bearer-token-1234567890"


class _TargetHandler(http.server.BaseHTTPRequestHandler):
    """Second origin: records whatever credentials it is handed."""

    received_authorization: list[str | None] = []
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        type(self).received_authorization.append(self.headers.get("Authorization"))
        body = json.dumps({"ok": True, "host": "target"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # silence test output
        return


class _ProviderHandler(http.server.BaseHTTPRequestHandler):
    """Provider origin: redirects either cross-origin or within itself."""

    cross_origin_url = ""
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path == "/redirect-cross-origin":
            location = self.cross_origin_url
            status = 302
        elif self.path == "/redirect-same-origin":
            location = "/ok"
            status = 302
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/ok":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.dumps({"ok": True, "host": "provider"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # silence test output
        return


class ReflectionRedirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.target = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _TargetHandler)
        cls.provider = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ProviderHandler)
        _ProviderHandler.cross_origin_url = f"http://127.0.0.1:{cls.target.server_port}/collect"
        for server in (cls.target, cls.provider):
            threading.Thread(target=server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        for server in (cls.target, cls.provider):
            server.shutdown()
            server.server_close()

    @classmethod
    def _provider_url(cls, path: str) -> str:
        return f"http://127.0.0.1:{cls.provider.server_port}{path}"

    def setUp(self) -> None:
        _TargetHandler.received_authorization.clear()

    def test_cross_origin_redirect_is_refused_and_leaks_nothing(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            _post_chat(
                self._provider_url("/redirect-cross-origin"),
                TOKEN,
                {"messages": []},
                timeout=5,
            )
        self.assertIn("302", str(caught.exception))
        self.assertEqual(_TargetHandler.received_authorization, [])

    def test_same_origin_redirect_still_succeeds(self) -> None:
        response = _post_chat(
            self._provider_url("/redirect-same-origin"),
            TOKEN,
            {"messages": []},
            timeout=5,
        )
        self.assertEqual(response, {"ok": True, "host": "provider"})


if __name__ == "__main__":
    unittest.main()
