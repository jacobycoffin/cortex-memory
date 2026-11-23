"""A user correcting themselves supersedes immediately (#33)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import self_correction
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class DetectionTests(unittest.TestCase):
    def test_i_meant_is_a_strong_correction(self) -> None:
        found = self_correction.detect("The port is 8200. I meant 8300.")
        self.assertTrue(found.is_correction)
        self.assertEqual(found.strength, "strong")
        self.assertIn("8300", found.statement)

    def test_no_its_is_a_strong_correction(self) -> None:
        found = self_correction.detect("No, it's on the r630 box.")
        self.assertTrue(found.is_correction)
        self.assertEqual(found.strength, "strong")
        self.assertIn("r630", found.statement)

    def test_a_message_opening_with_no_is_a_correction(self) -> None:
        found = self_correction.detect("No, the dashboard listens on 8200.")
        self.assertEqual(found.strength, "strong")
        self.assertIn("8200", found.statement)

    def test_a_no_inside_a_sentence_is_not_a_correction(self) -> None:
        self.assertFalse(self_correction.detect("There is no backup on that host.").is_correction)

    def test_scratch_that_is_a_strong_correction(self) -> None:
        self.assertEqual(
            self_correction.detect("Scratch that — backups run at 4am.").strength, "strong"
        )

    def test_that_is_wrong_is_a_strong_correction(self) -> None:
        self.assertEqual(
            self_correction.detect("That's wrong, the vault is on the VPS.").strength, "strong"
        )

    def test_not_x_but_y_is_a_strong_correction(self) -> None:
        found = self_correction.detect("Not the Proxmox box, but the r630.")
        self.assertTrue(found.is_correction)

    def test_instead_of_is_a_preference_not_a_correction(self) -> None:
        message = "I want it concise, instead of a technical execution log."
        self.assertFalse(self_correction.detect(message).is_correction)

    def test_a_bare_actually_is_weak(self) -> None:
        self.assertEqual(self_correction.detect("Actually the dashboard is fine.").strength, "weak")

    def test_no_i_want_is_a_preference_not_a_correction(self) -> None:
        self.assertFalse(self_correction.detect("no i want opencode").is_correction)

    def test_a_long_passage_is_never_a_correction(self) -> None:
        passage = ("Not remove but pause is the setting. " * 12).strip()
        self.assertFalse(self_correction.detect(passage).is_correction)

    def test_an_ordinary_sentence_is_not_a_correction(self) -> None:
        self.assertFalse(self_correction.detect("The dashboard runs on port 8100.").is_correction)

    def test_an_empty_message_is_not_a_correction(self) -> None:
        self.assertFalse(self_correction.detect("   ").is_correction)

    def test_the_statement_excludes_the_old_value(self) -> None:
        found = self_correction.detect("Actually, no — the dashboard listens on 8200 now.")
        self.assertNotIn("actually", found.statement.lower())
        self.assertIn("8200", found.statement)

    def test_detection_round_trips(self) -> None:
        payload = self_correction.detect("Correction: the sweep runs on weekdays.").as_dict()
        for key in ("is_correction", "strength", "marker", "statement"):
            self.assertIn(key, payload)

    def test_describe_reads_as_a_sentence(self) -> None:
        self.assertIn("strong", self_correction.detect("I meant 8300.").describe())
        self.assertEqual(self_correction.detect("nothing here").describe(), "not a correction")


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.original = self.store.add_memory(
            content="The dashboard runs on port 8100 behind nginx.",
            kind="semantic", source_type="user_turn", source_category="USER_EXPLICIT",
        )[0]

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def state(self, memory_id: str) -> str:
        return str(self.store._conn.execute(
            "SELECT state FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()[0])

    def supersedes(self, memory_id: str) -> str | None:
        return self.store._conn.execute(
            "SELECT supersedes_id FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()[0]

    def test_a_correction_supersedes_immediately(self) -> None:
        result = self_correction.intercept(
            self.store, "No, it's port 8200 on the dashboard.", at=NOW
        )
        self.assertTrue(result["applied"])
        self.assertEqual(self.state(self.original), "superseded")

    def test_the_winner_carries_supersedes_id(self) -> None:
        result = self_correction.intercept(
            self.store, "No, it's port 8200 on the dashboard.", at=NOW
        )
        self.assertEqual(self.supersedes(result["memory_id"]), self.original)

    def test_the_loser_keeps_its_content(self) -> None:
        self_correction.intercept(self.store, "No, it's port 8200 on the dashboard.", at=NOW)
        content = self.store._conn.execute(
            "SELECT content FROM memories WHERE id = ?", (self.original,)
        ).fetchone()[0]
        self.assertIn("8100", content)

    def test_the_correction_is_written_as_a_user_correction(self) -> None:
        result = self_correction.intercept(
            self.store, "No, it's port 8200 on the dashboard.", at=NOW
        )
        row = self.store._conn.execute(
            "SELECT kind, source_category FROM memories WHERE id = ?", (result["memory_id"],)
        ).fetchone()
        self.assertEqual(row["kind"], "correction")
        self.assertEqual(row["source_category"], "USER_CORRECTION")

    def test_an_explicit_target_is_used(self) -> None:
        result = self_correction.intercept(
            self.store, "I meant 8300.", target_id=self.original, at=NOW
        )
        self.assertEqual(result["superseded_id"], self.original)

    def test_a_weak_marker_with_no_target_does_nothing(self) -> None:
        empty = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            result = self_correction.intercept(
                empty, "Actually the dashboard is fine.", allow_weak=True
            )
            self.assertFalse(result["applied"])
            self.assertIn("nothing to correct", result["reason"])
        finally:
            empty.close()

    def test_a_weak_marker_is_off_by_default(self) -> None:
        """Every weak hit on the live corpus was a false positive, so it ships off."""
        result = self_correction.intercept(
            self.store, "Actually the dashboard uses 8200.", target_id=self.original, at=NOW
        )
        self.assertFalse(result["applied"])
        self.assertIn("weak marker ignored", result["reason"])
        self.assertEqual(self.state(self.original), "active")

    def test_a_weak_marker_with_a_target_acts_when_enabled(self) -> None:
        result = self_correction.intercept(
            self.store, "Actually the dashboard uses 8200.", target_id=self.original, at=NOW,
            allow_weak=True,
        )
        self.assertTrue(result["applied"])

    def test_the_weak_switch_reads_the_environment(self) -> None:
        self.assertFalse(self_correction.weak_enabled({}))
        self.assertTrue(self_correction.weak_enabled({"CORTEX_WEAK_CORRECTION": "1"}))

    def test_an_ordinary_message_writes_nothing(self) -> None:
        before = self.store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        result = self_correction.intercept(self.store, "The dashboard runs on port 8100.")
        self.assertFalse(result["applied"])
        after = self.store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        self.assertEqual(before, after)

    def test_an_equally_matching_pair_is_ambiguous_and_left_alone(self) -> None:
        """Superseding the wrong twin retires a fact nobody corrected."""
        twin = self.store.add_memory(
            content="The dashboard is served on port 8100 by nginx.",
            kind="semantic", source_type="user_turn", source_category="USER_EXPLICIT",
        )[0]
        self.assertNotEqual(twin, self.original)
        result = self_correction.intercept(
            self.store, "No, the dashboard listens on 8200.", at=NOW
        )
        self.assertIsNone(result["superseded_id"])
        self.assertNotIn(self.state(self.original), {"superseded"})
        self.assertNotIn(self.state(twin), {"superseded"})

    def test_the_history_records_the_marker(self) -> None:
        self_correction.intercept(self.store, "I meant 8300.", target_id=self.original, at=NOW)
        entry = self_correction.history(self.store)[0]
        self.assertEqual(entry["superseded_id"], self.original)
        self.assertTrue(entry["marker"])

    def test_undo_restores_the_corrected_fact(self) -> None:
        self_correction.intercept(self.store, "I meant 8300.", target_id=self.original, at=NOW)
        result = self_correction.undo(self.store)
        self.assertTrue(result["undone"])
        self.assertEqual(self.state(self.original), "active")

    def test_a_correction_is_not_undone_twice(self) -> None:
        self_correction.intercept(self.store, "I meant 8300.", target_id=self.original, at=NOW)
        self_correction.undo(self.store)
        self.assertFalse(self_correction.undo(self.store)["undone"])

    def test_the_report_counts_what_it_has(self) -> None:
        self_correction.intercept(self.store, "I meant 8300.", target_id=self.original, at=NOW)
        described = self_correction.report(self.store)
        self.assertEqual(described["corrections"], 1)
        self.assertEqual(described["with_target"], 1)

    def test_an_unrelated_neighbour_is_not_the_target(self) -> None:
        self.store.add_memory(
            content="Nginx serves the status page on the r630 host.",
            kind="semantic", source_type="user_turn", source_category="USER_EXPLICIT",
        )
        result = self_correction.intercept(
            self.store, "No, the dashboard listens on 8200.", at=NOW
        )
        # the neighbour shares "dashboard" but the port is the distinctive part of the claim,
        # so the correction must not silently retire an unrelated host fact
        self.assertEqual(result["superseded_id"], self.original)


if __name__ == "__main__":
    unittest.main()
