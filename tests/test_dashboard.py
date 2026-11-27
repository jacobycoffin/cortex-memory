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


def _dashboard_source(*names: str) -> str:
    """Read one or more deep-dive assets. The shell, stylesheet, and script are separate files."""

    return "\n".join((ROOT / name).read_text() for name in names)


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

    def test_reset_by_separate_instance_revokes_old_credentials(self) -> None:
        """A password reset from another process must take effect here.

        Regression test: DashboardAuth used to cache its state indefinitely,
        so a running dashboard kept accepting the old password/sessions and
        rejected the new password after an external reset.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dashboard-auth.json"
            dashboard = DashboardAuth(path)
            old_password = dashboard.reset(username="viewer")
            old_session = dashboard.issue_session(now=100)
            self.assertIsNotNone(dashboard.verify_session(old_session, now=101))

            cli = DashboardAuth(path)
            new_password = cli.reset(username="viewer")

            self.assertFalse(dashboard.verify_password("viewer", old_password))
            self.assertTrue(dashboard.verify_password("viewer", new_password))
            self.assertIsNone(dashboard.verify_session(old_session, now=101))
            self.assertTrue(dashboard.must_change_password())


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
        html = _dashboard_source("dashboard.html")
        script = _dashboard_source("dashboard.js")
        self.assertIn('localStorage.getItem("cortex-graph-renderer") || "physics"', script)
        self.assertIn('data-renderer-mode="physics"', html)
        self.assertIn('data-renderer-mode="constellation"', html)
        self.assertIn("function renderInsights()", script)
        self.assertIn("function renderCognition()", script)
        self.assertIn("function renderBudgetLearning", script)
        self.assertIn("function renderAttentionLearning", script)
        self.assertIn("function renderSleep", script)
        self.assertIn("function startSleepSession", script)

    def test_memory_receipt_deep_link_opens_and_tracks_the_detail_drawer(self) -> None:
        html = _dashboard_source("dashboard.html")
        script = _dashboard_source("dashboard.js")
        self.assertIn(
            'deepLinkMemory: new URLSearchParams(window.location.search).get("memory")', script,
        )
        self.assertIn("await selectMemory(memoryId, { updateUrl: false })", script)
        self.assertIn(
            "async function selectMemory(id, { updateUrl = true } = {})", script,
        )
        self.assertIn('url.searchParams.set("memory", id)', script)
        self.assertIn('url.searchParams.delete("memory")', script)
        self.assertIn(
            'deepLinkTrace: new URLSearchParams(window.location.search).get("trace")', script,
        )
        self.assertIn("await selectTrace(taskId, { updateUrl: false })", script)
        self.assertIn(
            "async function selectTrace(taskId, { updateUrl = true } = {})", script,
        )
        self.assertIn('url.searchParams.set("trace", taskId)', script)
        self.assertIn("/api/trace?task_id=", script)
        self.assertIn("/api/memory/feedback", script)
        self.assertIn("Memory-by-memory evidence", script)
        self.assertIn("Attributed to answer", script)
        self.assertIn("Recalled; no use evidence", script)
        self.assertIn("Considered; not recalled", script)
        self.assertIn("Helpful", script)
        self.assertIn("Irrelevant", script)
        self.assertIn("Outdated", script)
        self.assertIn("Clear label", script)
        self.assertIn("reverse the old signal", script)
        self.assertIn("/api/sleep/start", script)
        self.assertIn("function renderTrendChart", script)
        self.assertIn("app.data.memory_timeline_by_day_kind", script)
        self.assertIn("renderTimelineBars(timelineRows)", script)
        self.assertIn("function renderCapacityImpactChart", script)
        self.assertIn("function renderBenchmark", script)
        self.assertIn("function renderBenchmarkHistory", script)
        self.assertIn("function startBenchmark", script)
        self.assertIn("/api/benchmark/start", script)
        self.assertIn("function renderOutcomeLab", script)
        self.assertIn("function startPrivateEvaluation", script)
        self.assertIn("/api/evaluation/start", script)
        self.assertIn("/api/outcome/label", script)
        self.assertIn("function renderEvidenceHierarchy", script)
        self.assertIn("function renderSleepHypotheses", script)
        self.assertIn("function renderMetacognitionGate", script)
        self.assertIn("function renderToolEvaluation", script)
        self.assertIn("Test Cortex on this system", html)
        self.assertIn("What it does not measure", html)
        self.assertIn("outcome-backed memory accuracy proxy", html)
        self.assertIn("function renderMetacognition", script)
        self.assertIn("function renderCalibrationChart", script)
        self.assertIn("Label real answers; Cortex handles the calibration.", html)
        self.assertIn("Use, verify, or abstain", html)
        self.assertIn("function renderKindGuide", script)
        self.assertIn("History and connections", script)
        self.assertIn("edge.explanation", script)
        self.assertIn("filter(edge=>edge.explainable!==false)", script)
        self.assertIn("Hygiene queue", script)
        self.assertIn("app.data?.stats?.kinds", script)
        self.assertIn("d.stats?.states?.active", script)
        self.assertIn("d.stats?.states?.quarantine", script)
        self.assertIn("Cortex Sleep", html)
        self.assertIn("Proposed is not applied.", html)
        self.assertIn("Exactly what one cycle does", html)
        self.assertIn("function renderSleepLedger", script)
        self.assertIn("Observed after this run", script)
        self.assertIn("function renderHealthActions", script)
        self.assertIn("function copyHealthInstructions", script)
        self.assertIn("Simple review", script)
        self.assertIn("learning automatically", script)
        self.assertIn("Resolved when:", script)
        self.assertIn("Four independent health layers", html)
        self.assertIn("function sleepHealthState", script)
        self.assertIn("function renderHealthReviewer", script)
        self.assertIn("Which statement should Cortex use now?", script)
        self.assertIn("Both are valid in different situations", script)
        self.assertIn("/api/review/conflict", script)
        self.assertIn("/api/review/inference", script)
        self.assertIn("/api/review/proposal", script)
        self.assertIn("/api/review/copilot", script)
        self.assertIn("/api/review/undo", script)
        self.assertIn("function renderReviewInbox", script)
        self.assertIn("function renderConnectionBrief", script)
        self.assertIn("function renderConnectionReviewGuide", script)
        self.assertIn("function showConnectionOnMap", script)
        self.assertIn("Same durable subject", script)
        self.assertIn("Memory A supports B", script)
        self.assertIn("Teach this connection pattern", script)
        self.assertIn("Explain it naturally. The copilot will translate.", script)
        self.assertIn("function renderReviewCopilot", script)
        self.assertIn("function matchingCopilotInterpretation", script)
        self.assertIn("Use this recommendation", script)
        self.assertIn("It cannot apply anything.", script)
        self.assertIn("edgeMatchesHighlight", script)
        self.assertIn('typedPairs.has(pairKey(edge))', script)
        self.assertIn("Give this replay-supported pair a meaningful type", script)
        self.assertIn("function renderPolicyTraining", script)
        self.assertIn("function runPolicyAction", script)
        self.assertIn("/api/policy/action", script)
        self.assertIn("What should happen?", script)
        self.assertIn("Where should this apply?", script)
        self.assertIn("Only this memory", script)
        self.assertIn("exact_duplicates", script)
        self.assertIn("Teach Cortex from this", script)
        self.assertIn("Want to say why?", script)
        self.assertIn("Optional. If none fit, leave this blank.", script)
        self.assertIn('reasonText?"own_words":"operator_choice"', script)
        self.assertIn("function renderLearning", script)
        self.assertIn("function renderExperimentCenter", script)
        self.assertIn("function renderAgentAccuracyChart", script)
        self.assertIn("function renderFailureExplorer", script)
        self.assertIn("function renderSleepTrials", script)
        self.assertIn("function renderSummaryCandidates", script)
        self.assertIn("function renderProspective", script)
        self.assertIn("function renderReconsolidation", script)
        self.assertIn("Semantic consolidation", script)
        self.assertIn("d.semantic_consolidation", script)
        self.assertIn("/api/semantic-consolidation", inspect.getsource(serve_dashboard))
        self.assertIn("/api/experiment/control", script)
        self.assertIn("/api/sleep/trial/start", script)
        self.assertIn("/api/summary/review", script)
        self.assertIn("/api/prospective/update", script)

    def test_dashboard_keeps_ipad_navigation_and_index_content_readable(self) -> None:
        style = _dashboard_source("dashboard.css")
        script = _dashboard_source("dashboard.js")
        self.assertIn(":root { --sidebar: 172px; }", style)
        self.assertIn(".nav-btn .nav-text { display: block", style)
        self.assertIn("-webkit-line-clamp: 4", style)
        self.assertIn("tap for full memory", script)

    def test_dashboard_theme_catalog_stays_complete(self) -> None:
        html = _dashboard_source("dashboard.html")
        style = _dashboard_source("dashboard.css")
        script = _dashboard_source("dashboard.js")
        themes = {
            "field", "ink", "dusk", "tide", "moss", "mono",
            "midnight", "ember", "orchid", "glacier", "sepia", "terminal",
        }
        self.assertEqual(html.count('class="theme-option"'), len(themes))
        for theme in themes:
            self.assertIn(f'data-theme-choice="{theme}"', html)
            self.assertIn(f'html[data-theme="{theme}"]', style)
        self.assertIn('localStorage.setItem("cortex-theme", theme)', script)
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
        html = _dashboard_source("dashboard.html")
        script = _dashboard_source("dashboard.js")
        self.assertIn('explorerRole: "readable"', script)
        self.assertIn('data-role-view="readable"', html)
        self.assertIn('data-role-view="reference"', html)
        self.assertIn('data-role-view="clarity"', html)
        self.assertIn('data-role-view="all"', html)
        self.assertIn("Readable memories", html)
        self.assertIn("Reference evidence", html)
        self.assertIn("Needs clarity", html)
        self.assertIn("All records", html)
        self.assertIn("View raw evidence", script)
        self.assertIn('raw.className="raw-evidence"', script)
        self.assertIn("What the agent remembers", script)
        self.assertIn("When this applies", script)
        self.assertIn("Why it is retained", script)
        self.assertIn("Source and evidence", script)
        self.assertIn("Raw record", script)
        self.assertIn("History and connections", script)
        self.assertIn('"[data-review-filter]"', script)
        self.assertIn("Keep as readable memory", script)
        self.assertIn("Keep only as reference", script)
        self.assertIn("Rewrite clearly", script)
        self.assertIn("Split into separate memories", script)
        self.assertIn("function loadClarityPreview", script)
        self.assertIn("function renderClarityPreview", script)
        self.assertIn("/api/refinery/preview", script)
        self.assertIn("/api/refinery/action", script)
        self.assertIn("app.graphShowReference", script)
        self.assertIn("function renderRoleStrip", script)
        self.assertIn("tap for full memory", script)

    def test_clarity_draft_survives_refresh_and_is_captured_before_rerender(self) -> None:
        script = _dashboard_source("dashboard.js")
        self.assertIn("function reviewDecisionInProgress()", script)
        self.assertIn("load(false, { preserveReviewDraft: reviewDecisionInProgress() })", script)
        self.assertIn("renderAll({ preserveReviewDraft })", script)
        self.assertIn("if (!preserveReviewDraft) renderReviewInbox()", script)
        capture_index = script.index('const proposedRecords=item.item_type==="clarity"')
        rerender_index = script.index("app.review.busy=true;renderReviewInbox();", capture_index)
        self.assertLess(capture_index, rerender_index)
        self.assertIn("payload.proposed_records=proposedRecords", script)

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
        self.assertIn('parsed.path == "/api/trace"', server)
        self.assertIn('"/api/memory/feedback"', server)
        self.assertIn("trace_memory_feedback", server)
        self.assertIn("set_trace_memory_feedback", server)
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
        script = _dashboard_source("dashboard.js")
        readme = (ROOT / "README.md").read_text()
        self.assertIn("Review Copilot", readme)
        self.assertIn('"/api/review/copilot"', server)
        self.assertIn("ReviewCopilotConfig.from_env()", server)
        self.assertIn('snapshot["review_copilot"] = review_copilot.status()', server)
        self.assertIn("record_review_copilot_interpretation", server)
        self.assertIn("copilot_interpretation_id", server)
        self.assertIn("Provider: ${config.provider} · Model: ${config.model}", script)
        self.assertIn("You still confirm the final action below.", script)


    def test_dashboard_polls_only_while_the_tab_is_visible(self) -> None:
        """A background tab must not fetch a full snapshot every minute forever."""
        script = _dashboard_source("dashboard.js")
        self.assertNotIn("setInterval(()=>load(true),60000)", script)
        self.assertIn("function pollIfVisible()", script)
        self.assertIn('if (document.visibilityState !== "visible") return;', script)
        self.assertIn("setInterval(pollIfVisible, 60000)", script)
        self.assertIn('document.addEventListener("visibilitychange", pollIfVisible)', script)
        # Coming back to the tab refreshes immediately rather than waiting out an
        # interval, and the in-flight guard stops overlapping snapshots.
        self.assertIn('if (document.getElementById("refresh").classList.contains("refreshing")) return;', script)


if __name__ == "__main__":
    unittest.main()
