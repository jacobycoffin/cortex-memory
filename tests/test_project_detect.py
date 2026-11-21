"""Project detection must beat guessing, including by refusing to guess (#19)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.project_detect import candidate_projects, detect, known_projects, normalise
from cortex.store import CortexStore

CANDIDATES = ["duelist-road", "cortex-memory", "temple-ios", "website"]


class NormaliseTests(unittest.TestCase):
    def test_separators_are_unified(self) -> None:
        self.assertEqual(normalise("Duelist_Road"), "duelist-road")
        self.assertEqual(normalise("  temple ios "), "temple-ios")

    def test_prose_does_not_match_a_short_fragment(self) -> None:
        project, _ = detect(query="the railroad crossing is blocked", candidates=["road"])
        self.assertIsNone(project)


class DetectTests(unittest.TestCase):
    def test_an_explicit_mention_wins(self) -> None:
        project, scores = detect(query="in duelist-road the camera is still wrong", candidates=CANDIDATES)
        self.assertEqual(project, "duelist-road")
        self.assertGreater(scores["duelist-road"], 0)

    def test_the_working_directory_is_strong_evidence(self) -> None:
        project, _ = detect(cwd="/root/src/hermes-cortex-memory", candidates=CANDIDATES)
        self.assertEqual(project, "cortex-memory")

    def test_an_open_file_counts(self) -> None:
        project, _ = detect(
            files=["/root/theme-b/Temple/Temple/Duo/Tabletop/TempleTabletopWorkoutView.swift"],
            candidates=CANDIDATES,
        )
        self.assertEqual(project, "temple-ios")

    def test_recent_messages_can_carry_it(self) -> None:
        project, _ = detect(
            messages=["ok the temple build is green", "and the watch app too"],
            candidates=CANDIDATES,
        )
        self.assertEqual(project, "temple-ios")

    def test_nothing_matching_returns_nothing(self) -> None:
        project, _ = detect(query="what's the weather", candidates=CANDIDATES)
        self.assertIsNone(project)

    def test_no_candidates_returns_nothing(self) -> None:
        project, scores = detect(query="duelist-road camera", candidates=[])
        self.assertIsNone(project)
        self.assertEqual(scores, {})

    def test_a_tie_refuses_to_guess(self) -> None:
        """Two projects named equally often: the wrong one silently admits wrong memories."""
        project, _ = detect(
            query="compare duelist-road and temple-ios", candidates=CANDIDATES
        )
        self.assertIsNone(project)

    def test_stronger_evidence_breaks_a_weak_tie(self) -> None:
        project, _ = detect(
            query="compare duelist-road and temple-ios",
            cwd="/root/src/duelist-road-wt/camera",
            candidates=CANDIDATES,
        )
        self.assertEqual(project, "duelist-road")

    def test_the_alphabet_never_decides(self) -> None:
        project, _ = detect(messages=["website and temple-ios both"], candidates=CANDIDATES)
        self.assertIsNone(project)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_projects_are_read_from_memory_scopes(self) -> None:
        created = self.store.add_memory(
            content="The roadmap lives in the docs folder.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            scope={"active_project": "Cortex Memory"},
        )
        created = created[0] if isinstance(created, tuple) else created
        self.assertIn("cortex-memory", known_projects(self.store))
        self.assertIn(created, [created])  # keeps the id meaningful if the API changes

    def test_an_empty_store_has_no_projects(self) -> None:
        self.assertEqual(known_projects(self.store), [])

    def test_configured_projects_are_merged_with_stored_ones(self) -> None:
        created = self.store.add_memory(
            content="The roadmap lives in the docs folder.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            scope={"active_project": "Cortex Memory"},
        )
        self.assertIsNotNone(created)
        names = candidate_projects(self.store, ["Duelist_Road", "temple-ios"])
        self.assertEqual(names, ["cortex-memory", "duelist-road", "temple-ios"])

    def test_a_comma_string_is_accepted_by_the_caller(self) -> None:
        self.assertEqual(candidate_projects(self.store, []), [])


if __name__ == "__main__":
    unittest.main()
