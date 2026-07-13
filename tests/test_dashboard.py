from __future__ import annotations

import base64
import unittest
from html.parser import HTMLParser
from pathlib import Path


from tests._bootstrap import ROOT

from cortex.dashboard import _basic_auth_valid


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
            "lifecycle-feed",
            "workflow-feed",
            "health-actions",
            "health-areas",
        }
        self.assertTrue(required.issubset(set(parser.ids)))

    def test_dashboard_defaults_to_physics_and_keeps_constellation(self) -> None:
        html = (ROOT / "dashboard.html").read_text()
        self.assertIn('localStorage.getItem("cortex-graph-renderer") || "physics"', html)
        self.assertIn('data-renderer-mode="physics"', html)
        self.assertIn('data-renderer-mode="constellation"', html)
        self.assertIn("function renderInsights()", html)
        self.assertIn("function renderCognition()", html)
        self.assertIn("function renderHealthActions", html)


if __name__ == "__main__":
    unittest.main()
