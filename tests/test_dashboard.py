from __future__ import annotations

import base64
import inspect
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree


from tests._bootstrap import ROOT

from cortex.dashboard import _basic_auth_valid, serve_dashboard
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
    def test_dashboard_server_factory_reaches_request_handler_and_serve_loop(self) -> None:
        source = inspect.getsource(serve_dashboard)
        self.assertIn("class Handler(BaseHTTPRequestHandler)", source)
        self.assertIn("server.serve_forever", source)

    def test_dashboard_has_unique_ids_and_primary_brain_views(self) -> None:
        html = (ROOT / "dashboard.html").read_text()
        parser = _IdCollector()
        parser.feed(html)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        required = {
            "graph-mode-2d",
            "graph-mode-3d",
            "graph-cluster",
            "graph-selection-links",
            "graph-return-review",
            "timeline-svg",
            "timeline-trail",
            "view-review",
            "review-nav-count",
            "review-filters",
            "review-focus",
            "training-next-action",
            "policy-compile",
            "view-insights",
            "view-auto-judge",
            "aj-summary-metrics",
            "aj-funnel",
            "aj-decision-funnel",
            "aj-consolidation-metrics",
            "aj-consolidation-list",
            "aj-recent-list",
            "aj-edge-types",
            "recall-funnel",
            "view-cognition",
            "cognition-metrics",
            "recall-mode-list",
            "budget-learning-list",
            "budget-learning-total",
            "scoring-health-metrics",
            "scoring-policy-status",
            "scoring-health-chart",
            "scoring-health-legend",
            "scoring-signal-list",
            "scoring-health-boundary",
            "attention-learning-metrics",
            "attention-learning-status",
            "attention-learning-topics",
            "attention-learning-gate",
            "attention-learning-boundary",
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
            "outcome-lab",
            "outcome-summary",
            "outcome-task-list",
            "evaluation-start",
            "evaluation-summary",
            "evaluation-runs",
            "hierarchy-flow",
            "supported-claim-list",
            "sleep-hypotheses",
            "metacognition-gate",
            "metacognition-gate-checks",
            "tool-evaluation-summary",
            "guidance-ledger",
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
            "view-learning",
            "experiment-toggle",
            "experiment-conditions",
            "experiment-task-list",
            "agent-accuracy-chart",
            "failure-list",
            "sleep-trial-start",
            "sleep-trial-list",
            "summary-candidate-list",
            "prospective-form",
            "prospective-list",
            "reconsolidation-list",
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
            "memory-feedback-metrics",
            "memory-feedback-storage",
            "memory-feedback-boundary",
            "auth-gate",
            "auth-login-form",
            "auth-forced-change-form",
            "account-open",
            "account-change-form",
            "account-logout",
            "role-view-bar",
            "role-count-readable",
            "role-count-reference",
            "role-count-clarity",
            "role-count-all",
            "role-view-note",
            "role-strip",
            "graph-reference",
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
        self.assertIn("function renderAttentionLearning", html)
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
        self.assertIn("function renderOutcomeLab", html)
        self.assertIn("function startPrivateEvaluation", html)
        self.assertIn("/api/evaluation/start", html)
        self.assertIn("/api/outcome/label", html)
        self.assertIn("function renderEvidenceHierarchy", html)
        self.assertIn("function renderSleepHypotheses", html)
        self.assertIn("function renderMetacognitionGate", html)
        self.assertIn("function renderToolEvaluation", html)
        self.assertIn("Test Cortex on this system", html)
        self.assertIn("What it does not measure", html)
        self.assertIn("outcome-backed memory accuracy proxy", html)
        self.assertIn("function renderMetacognition", html)
        self.assertIn("function renderCalibrationChart", html)
        self.assertIn("Label real answers; Cortex handles the calibration.", html)
        self.assertIn("Use, verify, or abstain", html)
        self.assertIn("function renderKindGuide", html)
        self.assertIn("History and connections", html)
        self.assertIn("edge.explanation", html)
        self.assertIn("filter(edge=>edge.explainable!==false)", html)
        self.assertIn("Hygiene queue", html)
        self.assertIn("app.data?.stats?.kinds", html)
        self.assertIn("d.stats?.states?.active", html)
        self.assertIn("d.stats?.states?.quarantine", html)
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
        self.assertIn("/api/review/proposal", html)
        self.assertIn("/api/review/copilot", html)
        self.assertIn("/api/review/undo", html)
        self.assertIn("function renderReviewInbox", html)
        self.assertIn("function renderConnectionBrief", html)
        self.assertIn("function renderConnectionReviewGuide", html)
        self.assertIn("function showConnectionOnMap", html)
        self.assertIn("Same durable subject", html)
        self.assertIn("Memory A supports B", html)
        self.assertIn("Teach this connection pattern", html)
        self.assertIn("Explain it naturally. Kaya will translate.", html)
        self.assertIn("function renderReviewCopilot", html)
        self.assertIn("function matchingCopilotInterpretation", html)
        self.assertIn("Use this recommendation", html)
        self.assertIn("It cannot apply anything.", html)
        self.assertIn("edgeMatchesHighlight", html)
        self.assertIn('typedPairs.has(pairKey(edge))', html)
        self.assertIn("Give this replay-supported pair a meaningful type", html)
        self.assertIn("function renderPolicyTraining", html)
        self.assertIn("function runPolicyAction", html)
        self.assertIn("/api/policy/action", html)
        self.assertIn("What should happen?", html)
        self.assertIn("Where should this apply?", html)
        self.assertIn("Only this memory", html)
        self.assertIn("exact_duplicates", html)
        self.assertIn("Teach Kaya from this", html)
        self.assertIn("Want to say why?", html)
        self.assertIn("Optional. If none fit, leave this blank.", html)
        self.assertIn('reasonText?"own_words":"operator_choice"', html)
        self.assertIn("function renderLearning", html)
        self.assertIn("function renderExperimentCenter", html)
        self.assertIn("function renderAgentAccuracyChart", html)
        self.assertIn("function renderFailureExplorer", html)
        self.assertIn("function renderSleepTrials", html)
        self.assertIn("function renderSummaryCandidates", html)
        self.assertIn("function renderProspective", html)
        self.assertIn("function renderReconsolidation", html)
        self.assertIn("Semantic consolidation", html)
        self.assertIn("d.semantic_consolidation", html)
        self.assertIn("/api/semantic-consolidation", inspect.getsource(serve_dashboard))
        self.assertIn("/api/experiment/control", html)
        self.assertIn("/api/sleep/trial/start", html)
        self.assertIn("/api/summary/review", html)
        self.assertIn("/api/prospective/update", html)

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

    def test_refinery_views_and_clarity_review_are_wired(self) -> None:
        html = (ROOT / "dashboard.html").read_text()
        self.assertIn('explorerRole: "readable"', html)
        self.assertIn('data-role-view="readable"', html)
        self.assertIn('data-role-view="reference"', html)
        self.assertIn('data-role-view="clarity"', html)
        self.assertIn('data-role-view="all"', html)
        self.assertIn("Readable memories", html)
        self.assertIn("Reference evidence", html)
        self.assertIn("Needs clarity", html)
        self.assertIn("All records", html)
        self.assertIn("View raw evidence", html)
        self.assertIn('raw.className="raw-evidence"', html)
        self.assertIn("What Kaya remembers", html)
        self.assertIn("When this applies", html)
        self.assertIn("Why it is retained", html)
        self.assertIn("Source and evidence", html)
        self.assertIn("Raw record", html)
        self.assertIn("History and connections", html)
        self.assertIn('"[data-review-filter]"', html)
        self.assertIn("Keep as readable memory", html)
        self.assertIn("Keep only as reference", html)
        self.assertIn("Rewrite clearly", html)
        self.assertIn("Split into separate memories", html)
        self.assertIn("function loadClarityPreview", html)
        self.assertIn("function renderClarityPreview", html)
        self.assertIn("/api/refinery/preview", html)
        self.assertIn("/api/refinery/action", html)
        self.assertIn("app.graphShowReference", html)
        self.assertIn("function renderRoleStrip", html)
        self.assertIn("tap for full memory", html)

    def test_clarity_draft_survives_refresh_and_is_captured_before_rerender(self) -> None:
        html = (ROOT / "dashboard.html").read_text()
        self.assertIn("function reviewDecisionInProgress()", html)
        self.assertIn("load(false, { preserveReviewDraft: reviewDecisionInProgress() })", html)
        self.assertIn("renderAll({ preserveReviewDraft })", html)
        self.assertIn("if (!preserveReviewDraft) renderReviewInbox()", html)
        capture_index = html.index('const proposedRecords=item.item_type==="clarity"')
        rerender_index = html.index("app.review.busy=true;renderReviewInbox();", capture_index)
        self.assertLess(capture_index, rerender_index)
        self.assertIn("payload.proposed_records=proposedRecords", html)

    def test_refinery_routes_follow_dashboard_security_boundaries(self) -> None:
        server = (ROOT / "dashboard.py").read_text()
        for route in (
            "/api/refinery/summary",
            "/api/refinery/items",
            "/api/refinery/shadow",
            "/api/refinery/preview",
            "/api/refinery/action",
            "/api/refinery/undo",
            "/api/refinery/rebuild-presentations",
        ):
            self.assertIn(route, server)
        self.assertIn('"/api/recall-sets/"', server)
        self.assertIn("apply_refinery_action", server)
        self.assertIn("undo_refinery_action", server)
        self.assertIn("rebuild_presentations", server)
        # Mutating refinery routes sit behind the same reviews_enabled gate;
        # the read-only preview is dispatched before it.
        preview_index = server.index('parsed.path == "/api/refinery/preview"')
        gate_index = server.index("guided review changes are disabled on this dashboard")
        action_index = server.index('parsed.path == "/api/refinery/action"')
        self.assertLess(preview_index, gate_index)
        self.assertGreater(action_index, gate_index)

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
        self.assertIn('path == "/api/experiment/control"', server)
        self.assertIn('path == "/api/sleep/trial/start"', server)
        self.assertIn('path == "/api/summary/review"', server)
        self.assertIn('path == "/api/prospective/create"', server)
        self.assertIn('parsed.path == "/api/policy/action"', server)
        self.assertIn("promote_policy_candidate", server)
        self.assertIn("rollback_policy_version", server)
        self.assertIn('payload.get("decision_scope")', server)
        self.assertIn("dashboard Sleep is fixed to deterministic shadow mode", readme)

    def test_review_copilot_is_authenticated_confirmed_and_provider_transparent(self) -> None:
        server = (ROOT / "dashboard.py").read_text()
        html = (ROOT / "dashboard.html").read_text()
        readme = (ROOT / "README.md").read_text()
        self.assertIn('"/api/review/copilot"', server)
        self.assertIn("ReviewCopilotConfig.from_env()", server)
        self.assertIn('snapshot["review_copilot"] = review_copilot.status()', server)
        self.assertIn("record_review_copilot_interpretation", server)
        self.assertIn("copilot_interpretation_id", server)
        self.assertIn("Provider: ${config.provider} · Model: ${config.model}", html)
        self.assertIn("You still confirm the final action below.", html)


if __name__ == "__main__":
    unittest.main()
