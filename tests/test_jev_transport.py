"""Real loopback HTTP probes; never contact a provider."""
from __future__ import annotations

import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from tests._bootstrap import ROOT  # noqa: F401
from cortex import jev


class JevTransportTests(unittest.TestCase):
    def test_redirect_cannot_forward_authorization_to_another_target(self):
        target_requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", "/redirected")
                self.end_headers()
            def do_GET(self):
                target_requests.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"answers": {}}')
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            settings = jev.JevSettings(endpoint=f"http://127.0.0.1:{server.server_port}/", max_attempts=1)
            with patch.object(jev.JevSettings, "api_key", return_value="synthetic-not-a-secret"):
                try:
                    jev.systemone_call(settings, {"synthetic": True}, {})
                except jev.JevError:
                    pass
            self.assertEqual(target_requests, [], "redirects must never receive the request or credentials")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_direct_transport_rejects_credential_bearing_endpoints(self):
        for endpoint in ("https://user:pass@example.com/v1/systemone", "https://example.com/#fragment"):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(jev.JevError):
                    jev.systemone_call(jev.JevSettings(endpoint=endpoint), {}, {}, call=lambda *args: {"answers": {}})


if __name__ == "__main__":
    unittest.main()
