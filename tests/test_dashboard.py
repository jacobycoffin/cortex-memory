from __future__ import annotations

import base64
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path


from tests._bootstrap import ROOT

from cortex.dashboard import _basic_auth_valid
from cortex.dashboard_auth import DashboardAuth, SESSION_COOKIE


class _IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.ids.extend(value for name, value in attrs if name == "id" and value is not None)


class DashboardAuthTests(unittest.TestCase):
    def test_basic_auth_validation(self) -> None:
        token = base64.b64encode(b"viewer:correct-horse").decode()
        self.assertTrue(_basic_auth_valid(f"Basic {token}", "viewer", "correct-horse"))
        self.assertFalse(_basic_auth_valid(f"Basic {token}", "viewer", "wrong"))
        self.assertFalse(_basic_auth_valid(None, "viewer", "correct-horse"))
        self.assertFalse(_basic_auth_valid("Bearer token", "viewer", "correct-horse"))
        self.assertFalse(_basic_auth_valid("Basic not-base64", "viewer", "correct-horse"))

    def test_temporary_password_is_hashed_and_forces_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dashboard-auth.json"
            auth = DashboardAuth(path)
            temporary = auth.reset(username="viewer")
            self.assertTrue(auth.verify_password("viewer", temporary))
            self.assertTrue(auth.must_change_password())
            self.assertNotIn(temporary, path.read_text())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_password_change_revokes_old_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            auth = DashboardAuth(Path(tmp) / "dashboard-auth.json")
            temporary = auth.reset(username="viewer")
            old_session = auth.issue_session(now=100)
            self.assertIsNotNone(auth.verify_session(old_session, now=101))
            new_session = auth.change_password("viewer", temporary, "a-new-secure-password")
            self.assertIsNone(auth.verify_session(old_session, now=101))
            self.assertIsNotNone(auth.verify_session(new_session))
            self.assertFalse(auth.must_change_password())
            self.assertFalse(auth.verify_password("viewer", temporary))
            self.assertTrue(auth.verify_password("viewer", "a-new-secure-password"))
            self.assertIsNotNone(auth.session_from_cookie(f"other=1; {SESSION_COOKIE}={new_session}"))


class DashboardInterfaceTests(unittest.TestCase):
    def test_dashboard_has_unique_ids_and_primary_brain_views(self) -> None:
        html = (ROOT / "dashboard.html").read_text()
        parser = _IdCollector()
        parser.feed(html)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        required = {
            "graph-mode-2d",
            "graph-mode-3d",
            "graph-cluster",
            "timeline-svg",
            "timeline-trail",
            "view-insights",
            "recall-funnel",
            "view-cognition",
            "cognition-metrics",
            "recall-mode-list",
            "budget-learning-list",
            "budget-learning-total",
            "lifecycle-feed",
            "workflow-feed",
            "health-actions",
            "health-areas",
            "auth-gate",
            "auth-login-form",
            "auth-forced-change-form",
            "account-open",
            "account-change-form",
            "account-logout",
        }
        self.assertTrue(required.issubset(set(parser.ids)))

    def test_dashboard_defaults_to_physics_and_keeps_constellation(self) -> None:
        html = (ROOT / "dashboard.html").read_text()
        self.assertIn('localStorage.getItem("cortex-graph-renderer") || "physics"', html)
        self.assertIn('data-renderer-mode="physics"', html)
        self.assertIn('data-renderer-mode="constellation"', html)
        self.assertIn("function renderInsights()", html)
        self.assertIn("function renderCognition()", html)
        self.assertIn("function renderBudgetLearning", html)
        self.assertIn("function renderHealthActions", html)


if __name__ == "__main__":
    unittest.main()
