from __future__ import annotations

import base64
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree


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
            "cortex-sleep",
            "sleep-status",
            "sleep-cycle-summary",
            "sleep-metrics",
            "sleep-safety",
            "sleep-proposal-total",
            "sleep-proposal-list",
            "sleep-start",
            "sleep-progress",
            "sleep-next-run",
            "sleep-run-stats",
            "sleep-process",
            "sleep-run-history",
            "sleep-effects",
            "sleep-ledger-filters",
            "trend-chart",
            "trend-legend",
            "capacity-impact-chart",
            "capacity-impact-legend",
            "capacity-impact-summary",
            "capacity-impact-range",
            "cortex-benchmark",
            "benchmark-start",
            "benchmark-status",
            "benchmark-progress",
            "benchmark-summary",
            "benchmark-history-chart",
            "benchmark-recommendations",
            "benchmark-run-list",
            "benchmark-method",
            "view-trust",
            "metacognition-mode",
            "metacognition-metrics",
            "calibration-chart",
            "calibration-samples",
            "metacognition-decision-mix",
            "metacognition-sources",
            "metacognition-filter",
            "metacognition-predictions",
            "kind-guide",
            "view-settings",
            "settings-theme-slot",
            "settings-account-slot",
            "lifecycle-feed",
            "workflow-feed",
            "health-actions",
            "health-action-summary",
            "health-reviewer",
            "health-review-body",
            "health-review-progress",
            "health-review-close",
            "health-areas",
            "health-maintenance",
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
        self.assertIn("function renderSleep", html)
        self.assertIn("function startSleepSession", html)
        self.assertIn("/api/sleep/start", html)
        self.assertIn("function renderTrendChart", html)
        self.assertIn("app.data.memory_timeline_by_day_kind", html)
        self.assertIn("renderTimelineBars(timelineRows)", html)
        self.assertIn("function renderCapacityImpactChart", html)
        self.assertIn("function renderBenchmark", html)
        self.assertIn("function renderBenchmarkHistory", html)
        self.assertIn("function startBenchmark", html)
        self.assertIn("/api/benchmark/start", html)
        self.assertIn("Test Cortex on this system", html)
        self.assertIn("What it does not measure", html)
        self.assertIn("outcome-backed memory accuracy proxy", html)
        self.assertIn("function renderMetacognition", html)
        self.assertIn("function renderCalibrationChart", html)
        self.assertIn("Does Cortex know when to doubt itself?", html)
        self.assertIn("Observe the judgments before enforcing them.", html)
        self.assertIn("used, verified, or withheld", html)
        self.assertIn("function renderKindGuide", html)
        self.assertIn("Cortex Sleep", html)
        self.assertIn("Proposed is not applied.", html)
        self.assertIn("Exactly what one cycle does", html)
        self.assertIn("function renderSleepLedger", html)
        self.assertIn("Observed after this run", html)
        self.assertIn("function renderHealthActions", html)
        self.assertIn("function copyHealthInstructions", html)
        self.assertIn("Simple review", html)
        self.assertIn("learning automatically", html)
        self.assertIn("Resolved when:", html)
        self.assertIn("Four independent health layers", html)
        self.assertIn("function sleepHealthState", html)
        self.assertIn("function renderHealthReviewer", html)
        self.assertIn("Which statement should Cortex use now?", html)
        self.assertIn("Both are valid in different situations", html)
        self.assertIn("/api/review/conflict", html)
        self.assertIn("/api/review/inference", html)

    def test_dashboard_keeps_ipad_navigation_and_index_content_readable(self) -> None:
        html = (ROOT / "dashboard.html").read_text()
        self.assertIn(":root { --sidebar: 172px; }", html)
        self.assertIn(".nav-btn .nav-text { display: block", html)
        self.assertIn("-webkit-line-clamp: 4", html)
        self.assertIn("tap for full memory", html)

    def test_dashboard_theme_catalog_stays_complete(self) -> None:
        html = (ROOT / "dashboard.html").read_text()
        themes = {
            "field", "ink", "dusk", "tide", "moss", "mono",
            "midnight", "ember", "orchid", "glacier", "sepia", "terminal",
        }
        self.assertEqual(html.count('class="theme-option"'), len(themes))
        for theme in themes:
            self.assertIn(f'data-theme-choice="{theme}"', html)
            self.assertIn(f'html[data-theme="{theme}"]', html)
        self.assertIn('localStorage.setItem("cortex-theme", theme)', html)
        self.assertIn('Choose a palette · 12', html)

    def test_dashboard_has_a_complete_favicon_set(self) -> None:
        html = (ROOT / "dashboard.html").read_text()
        self.assertIn('rel="icon" href="/favicon.svg" type="image/svg+xml"', html)
        self.assertIn('rel="icon" href="/favicon.ico" sizes="any"', html)
        self.assertIn('rel="apple-touch-icon" href="/apple-touch-icon.png"', html)
        svg = ElementTree.parse(ROOT / "favicon.svg").getroot()
        self.assertEqual(svg.attrib["viewBox"], "0 0 64 64")
        self.assertGreater((ROOT / "favicon.ico").stat().st_size, 100)
        self.assertGreater((ROOT / "apple-touch-icon.png").stat().st_size, 100)

    def test_guided_review_is_documented_as_opt_in(self) -> None:
        server = (ROOT / "dashboard.py").read_text()
        readme = (ROOT / "README.md").read_text()
        self.assertIn('os.environ.get("CORTEX_DASHBOARD_REVIEWS"', server)
        self.assertIn("guided review changes are disabled", server)
        self.assertIn("CORTEX_DASHBOARD_REVIEWS=1", readme)
        self.assertIn('parsed.path == "/api/sleep/start"', server)
        self.assertIn('SleepConfig(mode="shadow", reflection_token_budget=0)', server)
        self.assertIn('snapshot["sleep_schedule"] = sleep_schedule()', server)
        self.assertIn('parsed.path == "/api/benchmark/start"', server)
        self.assertIn("run_dashboard_benchmark", server)
        self.assertIn("dashboard Sleep is fixed to deterministic shadow mode", readme)


if __name__ == "__main__":
    unittest.main()
