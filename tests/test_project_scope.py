"""Project scope at capture time, and the backfill (#217)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.project_scope import (
    apply_backfill,
    backfill_plan,
    infer_project,
    paths_in,
    project_coverage,
    scope_for_capture,
    with_project,
)
from cortex.store import CortexStore

CONFIGURED = ["temple-ios", "cortex-memory", "website", "duelist-road"]


class PathTests(unittest.TestCase):
    def test_paths_are_extracted_from_a_vault_memory(self) -> None:
        content = "Vault note: Duo Tabletop. Path: Temple/Temple/Duo/Tabletop/View.swift"
        self.assertIn("Temple/Temple/Duo/Tabletop/View.swift", paths_in(content))

    def test_a_memory_without_paths_returns_nothing(self) -> None:
        self.assertEqual(paths_in("Jacoby prefers the V2 nightly channel."), [])


class InferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_named_path_places_the_memory(self) -> None:
        evidence = infer_project(
            self.store,
            "Vault note: Duo. Path: Temple/Temple/Duo/Tabletop/View.swift",
            configured=CONFIGURED,
        )
        self.assertEqual(evidence.project, "temple-ios")

    def test_a_query_mention_places_the_memory(self) -> None:
        evidence = infer_project(
            self.store, "The dashboard build is green.", query="cortex-memory roadmap", configured=CONFIGURED
        )
        self.assertEqual(evidence.project, "cortex-memory")

    def test_split_evidence_returns_no_project(self) -> None:
        evidence = infer_project(
            self.store,
            "Temple and cortex-memory both need the token budget reworked.",
            configured=CONFIGURED,
        )
        self.assertIsNone(evidence.project)

    def test_vault_folders_are_not_projects(self) -> None:
        """The first version inferred `references`, `research` and `plans` as projects.

        A path is evidence *about* a candidate, never a candidate by itself.
        """
        evidence = infer_project(
            self.store, "Path: references/2026/sources.md", configured=CONFIGURED
        )
        self.assertIsNone(evidence.project)

    def test_no_configuration_means_no_projects(self) -> None:
        evidence = infer_project(self.store, "Path: Temple/Temple/Duo/View.swift")
        self.assertIsNone(evidence.project)

    def test_an_explicit_project_is_never_overwritten(self) -> None:
        scope, evidence = scope_for_capture(
            self.store,
            "Path: Temple/Temple/Duo/View.swift",
            scope={"active_project": "website", "task_type": "filesystem"},
            configured=CONFIGURED,
        )
        self.assertEqual(scope["active_project"], "website")
        self.assertEqual(scope["task_type"], "filesystem")
        self.assertEqual(evidence.project, "temple-ios")

    def test_with_project_keeps_other_keys(self) -> None:
        self.assertEqual(
            with_project({"task_type": "shell"}, "website"),
            {"task_type": "shell", "active_project": "website"},
        )


class StrictBackfillTests(unittest.TestCase):
    """A stored body mentions many things; only a path says where a note belongs."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_path_places_the_note(self) -> None:
        from cortex.project_scope import strict_project

        self.assertEqual(
            strict_project(
                self.store,
                "Vault note: Duo. Path: Temple/Temple/Duo/Tabletop/View.swift",
                configured=CONFIGURED,
            ),
            "temple-ios",
        )

    def test_a_passing_mention_does_not_place_the_note(self) -> None:
        """Measured false positive: a CobyBot note filed under temple-ios."""
        from cortex.project_scope import strict_project

        self.assertIsNone(
            strict_project(
                self.store,
                "Vault note: CobyBot - Minecraft agent Path: projects/CobyBot.md. "
                "Discussed in a planning session with temple and cortex work.",
                configured=CONFIGURED,
            )
        )

    def test_two_paths_are_ambiguous(self) -> None:
        from cortex.project_scope import strict_project

        self.assertIsNone(
            strict_project(
                self.store,
                "Path: Temple/Temple/Duo/View.swift and Path: cortex-memory/retrieval.py",
                configured=CONFIGURED,
            )
        )


class BackfillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        for content in (
            "Vault note: Duo tabletop. Path: Temple/Temple/Duo/Tabletop/View.swift",
            "Cortex recall latency is 125ms on the worktree.",
            "Just an unrelated fact about coffee.",
        ):
            self.store.add_memory(
                content=content, kind="semantic", source_type="user_turn",
                source_category="USER_EXPLICIT",
            )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_the_plan_only_contains_confident_matches(self) -> None:
        """Path evidence places two; the prose-only memory is left alone.

        "Cortex recall latency is 125ms" is obviously about cortex-memory to a human. The
        backfill will not say so, because prose attribution filed a Portfolio Website note
        under cortex-memory during measurement — the note mentioned cortex twice.
        """
        plan = backfill_plan(self.store, configured=CONFIGURED)
        projects = {project for _memory_id, project, _why in plan}
        self.assertEqual(projects, {"temple-ios"})

    def test_applying_writes_scope_and_is_idempotent(self) -> None:
        plan = backfill_plan(self.store, configured=CONFIGURED)
        before = project_coverage(self.store)
        self.assertEqual(before["scoped"], 0)
        changed = apply_backfill(self.store, plan)
        after = project_coverage(self.store)
        self.assertEqual(changed, len(plan))
        self.assertEqual(after["scoped"], len(plan))
        # Second run: nothing left to do, and no duplicate writes.
        self.assertEqual(apply_backfill(self.store, backfill_plan(self.store, configured=CONFIGURED)), 0)

    def test_the_scoped_json_stays_valid(self) -> None:
        apply_backfill(self.store, backfill_plan(self.store, configured=CONFIGURED))
        for row in self.store._conn.execute(
            "SELECT scope_json FROM memories WHERE scope_json IS NOT NULL"
        ).fetchall():
            self.assertIsInstance(json.loads(row["scope_json"]), dict)

    def test_a_backfilled_project_matches_what_capture_would_write(self) -> None:
        """The backfill and the capture path must agree, or old and new memories diverge."""
        content = "Vault note: Duo tabletop. Path: Temple/Temple/Duo/Tabletop/View.swift"
        capture_scope, _ = scope_for_capture(self.store, content, configured=CONFIGURED)
        plan = backfill_plan(self.store, configured=CONFIGURED)
        temple = [project for _id, project, _why in plan if project == "temple-ios"]
        self.assertEqual(capture_scope["active_project"], temple[0])


class ProviderCaptureTests(unittest.TestCase):
    def test_remember_records_the_project(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            provider = __import__(
                "cortex.hermes_provider", fromlist=["CortexMemoryProvider"]
            ).CortexMemoryProvider(
                {
                    "token_budget": 700,
                    "top_k": 6,
                    "projects": CONFIGURED,
                    "database_path": str(Path(tmp.name) / "p.db"),
                }
            )
            provider.initialize("proj-session", hermes_home=tmp.name, agent_context="primary")
            store = provider._store
            provider.handle_tool_call(
                "cortex_memory",
                {
                    "action": "remember",
                    "content": "Duo tabletop view lives at Temple/Temple/Duo/Tabletop/View.swift",
                },
            )
            rows = store._conn.execute(
                "SELECT candidate_json FROM memory_creation_proposals ORDER BY rowid DESC"
            ).fetchall()
            self.assertTrue(rows, "no proposal was staged")
            candidate = json.loads(rows[0]["candidate_json"] or "{}")
            scope = candidate.get("scope") or {}
            self.assertEqual(scope.get("active_project"), "temple-ios")
            store.close()
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
