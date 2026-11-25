"""The active project is written at capture time, so scoped memory can exist at all (#217)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import scope_capture
from cortex.store import CortexStore


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, session_id: str = "s-1",
               scope: dict | None = None) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_STATED", session_id=session_id, scope=scope or {},
        )[0]

    def scope_of(self, memory_id: str) -> dict:
        row = self.store._conn.execute(
            "SELECT scope_json FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return json.loads(row["scope_json"] or "{}")


class DecisionTests(unittest.TestCase):
    CANDIDATES = ("duelist-road", "arttools", "cortex-memory")

    def test_an_explicit_project_wins_and_detection_does_not_vote(self) -> None:
        decision = scope_capture.for_turn(
            query="arttools needs a rebuild", explicit="cortex-memory", candidates=self.CANDIDATES)
        self.assertEqual(decision.project, "cortex-memory")
        self.assertEqual(decision.how, scope_capture.EXPLICIT)

    def test_the_working_directory_identifies_the_project(self) -> None:
        decision = scope_capture.for_turn(cwd="/home/jacoby/work/cortex-memory", candidates=self.CANDIDATES)
        self.assertEqual(decision.project, "cortex-memory")
        self.assertEqual(decision.how, scope_capture.DETECTED)

    def test_a_query_mention_identifies_the_project(self) -> None:
        decision = scope_capture.for_turn(query="why does duelist-road fail to boot",
                                          candidates=self.CANDIDATES)
        self.assertEqual(decision.project, "duelist-road")

    def test_no_evidence_records_no_project(self) -> None:
        decision = scope_capture.for_turn(query="what time is it", candidates=self.CANDIDATES)
        self.assertIsNone(decision.project, "no default project and no most-likely fallback")
        self.assertEqual(decision.how, scope_capture.NONE)
        self.assertEqual(decision.scope, {})

    def test_a_tie_records_nothing(self) -> None:
        decision = scope_capture.for_turn(query="arttools and duelist-road both fail",
                                          candidates=self.CANDIDATES)
        self.assertIsNone(decision.project, "a tie is inherited from the detector: guessing is worse than none")

    def test_the_decision_names_how_it_was_made(self) -> None:
        decision = scope_capture.for_turn(cwd="/home/jacoby/work/arttools", candidates=self.CANDIDATES)
        self.assertEqual(decision.scope["project_source"], scope_capture.DETECTED)
        self.assertIn("arttools", decision.render())


class CaptureTests(Base):
    CANDIDATES = ("duelist-road", "arttools", "cortex-memory")

    def test_the_turn_project_is_written_into_scope(self) -> None:
        memory_id = self.memory("the nginx timeout is 60s")
        decision = scope_capture.for_turn(cwd="/home/jacoby/work/arttools", candidates=self.CANDIDATES)
        merged = scope_capture.capture_turn(self.store, memory_id, decision)
        self.assertEqual(merged["project"], "arttools")
        self.assertEqual(self.scope_of(memory_id)["project"], "arttools")

    def test_an_existing_project_is_never_overwritten(self) -> None:
        memory_id = self.memory("the nginx timeout is 60s", scope={"project": "duelist-road"})
        decision = scope_capture.for_turn(cwd="/home/jacoby/work/arttools", candidates=self.CANDIDATES)
        scope_capture.capture_turn(self.store, memory_id, decision)
        self.assertEqual(self.scope_of(memory_id)["project"], "duelist-road")

    def test_no_decision_leaves_scope_alone(self) -> None:
        memory_id = self.memory("what time is it")
        decision = scope_capture.for_turn(query="what time is it", candidates=self.CANDIDATES)
        before = self.scope_of(memory_id)
        scope_capture.capture_turn(self.store, memory_id, decision)
        self.assertEqual(self.scope_of(memory_id), before)

    def test_capturing_twice_is_idempotent(self) -> None:
        memory_id = self.memory("the nginx timeout is 60s")
        decision = scope_capture.for_turn(cwd="/home/jacoby/work/arttools", candidates=self.CANDIDATES)
        scope_capture.capture_turn(self.store, memory_id, decision)
        first = self.scope_of(memory_id)
        scope_capture.capture_turn(self.store, memory_id, decision)
        self.assertEqual(self.scope_of(memory_id), first)

    def test_the_other_scope_dimensions_survive(self) -> None:
        memory_id = self.memory("jacoby prefers the amber theme", scope={"who": ["jacoby"]})
        decision = scope_capture.for_turn(cwd="/home/jacoby/work/arttools", candidates=self.CANDIDATES)
        merged = scope_capture.capture_turn(self.store, memory_id, decision)
        # The store normalises scope values on write, so the assertion is about survival, not about the exact shape.
        self.assertIn("jacoby", str(merged["who"]))
        self.assertEqual(merged["project"], "arttools")

    def test_a_project_makes_a_scope_non_global(self) -> None:
        self.assertFalse(scope_capture.is_global({"project": "arttools"}))
        self.assertTrue(scope_capture.is_global({}))


class BackfillTests(Base):
    CANDIDATES = ("duelist-road", "arttools")

    def test_the_plan_finds_only_unscoped_memories(self) -> None:
        bare = self.memory("the nginx timeout is 60s")
        scoped = self.memory("the cache is warm", scope={"project": "duelist-road"})
        plan = scope_capture.plan_session_backfill(self.store, session_id="s-1", project="arttools")
        self.assertEqual(plan.memory_ids, [bare])
        self.assertEqual(plan.skipped_already_scoped, 1)
        self.assertNotIn(scoped, plan.memory_ids)

    def test_the_default_is_a_dry_run(self) -> None:
        memory_id = self.memory("the nginx timeout is 60s")
        plan = scope_capture.plan_session_backfill(self.store, session_id="s-1", project="arttools")
        result = scope_capture.apply_session_backfill(self.store, plan)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["would_apply"], 1)
        self.assertEqual(self.scope_of(memory_id), {}, "a dry run changes nothing")

    def test_applying_needs_consent(self) -> None:
        self.memory("the nginx timeout is 60s")
        plan = scope_capture.plan_session_backfill(self.store, session_id="s-1", project="arttools")
        result = scope_capture.apply_session_backfill(self.store, plan, apply=True)
        self.assertFalse(result["ok"])
        self.assertIn("consent", result["reason"])

    def test_applying_writes_the_project_and_marks_it_a_backfill(self) -> None:
        memory_id = self.memory("the nginx timeout is 60s")
        plan = scope_capture.plan_session_backfill(self.store, session_id="s-1", project="arttools")
        result = scope_capture.apply_session_backfill(self.store, plan, apply=True, consented_by="teknium")
        self.assertTrue(result["ok"])
        self.assertEqual(result["applied"], 1)
        scope = self.scope_of(memory_id)
        self.assertEqual(scope["project"], "arttools")
        self.assertEqual(scope["project_source"], "backfill", "an inferred scope says where it came from")

    def test_a_backfill_does_not_touch_another_session(self) -> None:
        mine = self.memory("mine", session_id="s-1")
        other = self.memory("theirs", session_id="s-2")
        plan = scope_capture.plan_session_backfill(self.store, session_id="s-1", project="arttools")
        self.assertEqual(plan.memory_ids, [mine])
        self.assertNotIn(other, plan.memory_ids)

    def test_a_backfill_needs_a_project(self) -> None:
        plan = scope_capture.plan_session_backfill(self.store, session_id="s-1", project="")
        self.assertFalse(plan.readable)
        self.assertIn("needs a session and a project", plan.reason)


if __name__ == "__main__":
    unittest.main()
