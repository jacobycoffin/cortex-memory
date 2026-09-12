"""Interface robustness fixes from the audit.

* DashboardAuth.verify_password must not raise on non-ASCII usernames.
* Tooling failure classification must treat nonzero exit codes as failure.
* Dashboard JSON reads must not crash the handler on deeply nested payloads.
* Login throttling must reserve slots atomically (no burst over the limit).
"""

from __future__ import annotations

import base64
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401 - loads the flat ``cortex`` package
from tests.test_dashboard_auth_enforcement import _handler_bound_to

from cortex.dashboard_auth import DashboardAuth
from cortex.tooling import _looks_failed


class UnicodeUsernameTests(unittest.TestCase):
    def test_non_ascii_username_fails_cleanly_not_with_type_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            auth = DashboardAuth(Path(tmp) / "dashboard-auth.json")
            password = auth.reset(username="viewer")
            self.assertFalse(auth.verify_password("é", "anything"))
            self.assertTrue(auth.verify_password("viewer", password))


class ExitCodeClassificationTests(unittest.TestCase):
    def test_nonzero_exit_codes_are_failures_even_with_success_markers(self) -> None:
        self.assertTrue(_looks_failed(json.dumps({"exit_code": 2, "output": "assertion mismatch", "error": None})))
        self.assertTrue(_looks_failed(json.dumps({"returncode": 1, "stdout": "", "stderr": ""})))
        self.assertTrue(_looks_failed(json.dumps({"status": "success", "exit_code": 9, "output": ""})))

    def test_zero_exit_code_still_uses_existing_success_heuristics(self) -> None:
        self.assertFalse(_looks_failed(json.dumps({"status": "success", "exit_code": 0, "output": "done"})))


class DeepJsonGuardTests(unittest.TestCase):
    def test_deep_json_returns_400_instead_of_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            auth = DashboardAuth(Path(tmp) / "dashboard-auth.json")
            handler = _handler_bound_to(auth)
            request = handler.__new__(handler)
            body = b'{"x":' + b"[" * 1100 + b"0" + b"]" * 1100 + b"}"
            request.headers = {"Content-Length": str(len(body))}
            request.rfile = io.BytesIO(body)
            request.responses: list[tuple[int, object]] = []
            request._json = lambda status, payload, **kw: request.responses.append((int(status), payload))

            self.assertIsNone(request._read_json())
            self.assertEqual(request.responses[0][0], 400)


class ThrottleReservationTests(unittest.TestCase):
    def test_burst_cannot_exceed_the_attempt_limit(self) -> None:
        class CountingAuth:
            configured = True
            attempts = 0

            def session_from_cookie(self, header):  # type: ignore[no-untyped-def]
                return None

            def verify_password(self, username, password):  # type: ignore[no-untyped-def]
                type(self).attempts += 1
                return False

        with tempfile.TemporaryDirectory() as tmp:
            auth = DashboardAuth(Path(tmp) / "dashboard-auth.json")
            handler = _handler_bound_to(CountingAuth())
            token = base64.b64encode(b"viewer:wrong-password").decode()

            def worker() -> None:
                request = handler.__new__(handler)
                request.headers = {"Authorization": "Basic " + token}
                request.client_address = ("127.0.0.1", 9999)
                try:
                    request._authorized(complete=False)
                except Exception:
                    pass

            threads = [threading.Thread(target=worker) for _ in range(12)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=6)

            self.assertEqual(CountingAuth.attempts, 8)  # the configured limit, no more


if __name__ == "__main__":
    unittest.main()