"""Dashboard authorization must follow live configuration, not a startup snapshot.

Two regression tests for audit findings on the dashboard auth path:

* ``DashboardAuth._load`` used to validate its cache *after* reading, so a
  password reset landing between the read and the stat cached stale
  credentials as current and kept accepting the old password.
* ``serve_dashboard`` captured ``auth_enabled`` once at startup, so a dashboard
  that started before credentials existed stayed unauthenticated forever.
"""

from __future__ import annotations

import ast
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT

from cortex import dashboard as dashboard_module
from cortex.dashboard_auth import DashboardAuth

OLD_PASSWORD = "old-secure-password-1"
NEW_PASSWORD = "new-secure-password-2"


def _handler_bound_to(auth: DashboardAuth) -> type:
    """Extract ``serve_dashboard``'s Handler and bind it to ``auth``.

    The handler lives inside the serve function, so the class body is compiled
    against the real dashboard module namespace with just the few names it
    needs overridden.
    """

    tree = ast.parse((ROOT / "dashboard.py").read_text(encoding="utf-8"))
    serve = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "serve_dashboard"
    )
    handler_def = next(node for node in serve.body if isinstance(node, ast.ClassDef) and node.name == "Handler")
    snapshot = next(
        node
        for node in serve.body
        if isinstance(node, ast.FunctionDef) and node.name == "_auto_judge_snapshot"
    )
    namespace = dict(vars(dashboard_module))
    namespace.update(
        auth=auth,
        failed_logins={},
        failed_logins_lock=threading.RLock(),
        trusted_proxies=set(),
        reviews_enabled=False,
    )
    exec(
        compile(
            ast.Module(body=[handler_def, snapshot], type_ignores=[]),
            str(ROOT / "dashboard.py"),
            "exec",
        ),
        namespace,
    )
    return namespace["Handler"]


class DashboardAuthRaceTests(unittest.TestCase):
    def test_reset_racing_a_read_does_not_cache_stale_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dashboard-auth.json"
            writer = DashboardAuth(path)
            writer.reset(username="viewer", password=OLD_PASSWORD, must_change=False)

            loader = DashboardAuth(path)
            original_read_text = Path.read_text
            triggered: list[bool] = []

            def racing_read(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                # Read the old bytes, then have a separate process replace the
                # file before the caller takes its closing stat — exactly the
                # interleave that used to be cached as "current".
                content = original_read_text(self, *args, **kwargs)
                if self == path and not triggered:
                    triggered.append(True)
                    writer.reset(username="viewer", password=NEW_PASSWORD, must_change=False)
                return content

            with patch.object(Path, "read_text", racing_read):
                loader.username()

            self.assertTrue(triggered)
            self.assertFalse(loader.verify_password("viewer", OLD_PASSWORD))
            self.assertTrue(loader.verify_password("viewer", NEW_PASSWORD))


class DashboardAuthorizationLatchTests(unittest.TestCase):
    def test_credentials_created_after_startup_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            auth = DashboardAuth(Path(tmp) / "dashboard-auth.json")
            handler = _handler_bound_to(auth)
            request = handler.__new__(handler)
            request.headers = {}
            request.client_address = ("127.0.0.1", 9999)

            # Nothing configured yet: a local dashboard stays reachable.
            self.assertTrue(request._authorized(complete=True))

            auth.reset(username="viewer", password=NEW_PASSWORD, must_change=False)

            # Credentials now exist: the same running handler must require them.
            self.assertFalse(request._authorized(complete=True))


if __name__ == "__main__":
    unittest.main()
