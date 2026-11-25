"""Argument memory (#134): failures are kept, secrets are refused, a bad argv is never recommended."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import argument_memory
from cortex.store import CortexStore


class ArgumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def remember(self, argv, *, outcome="success", note="", command="pytest"):
        return argument_memory.remember(self.store, command, argv, outcome=outcome, note=note)

    # -- what is kept ------------------------------------------------------------------

    def test_a_working_invocation_is_remembered_and_recalled(self) -> None:
        self.remember(["tests/", "-q", "-n", "4"])
        recall = argument_memory.recall(self.store, "pytest")
        self.assertEqual(recall.best.argv, ["tests/", "-q", "-n", "4"])
        self.assertIn("for pytest: tests/ -q -n 4", recall.statement())

    def test_a_failing_invocation_is_kept_too(self) -> None:
        self.remember(["-n", "8"], outcome="failure", note="no faster than -n 4")
        recall = argument_memory.recall(self.store, "pytest")
        self.assertIsNone(recall.best, "a known-bad argv must never be the recommendation")
        self.assertEqual(recall.known_bad[0].argv, ["-n", "8"])
        self.assertIn("every recorded attempt failed", recall.statement())
        self.assertIn("no faster than -n 4", recall.statement())

    def test_a_good_and_a_bad_argv_coexist(self) -> None:
        self.remember(["-n", "4"])
        self.remember(["-n", "8"], outcome="failure", note="same wall time")
        recall = argument_memory.recall(self.store, "pytest")
        self.assertEqual(recall.best.argv, ["-n", "4"])
        self.assertEqual(recall.known_bad[0].argv, ["-n", "8"])

    def test_the_same_argv_with_both_outcomes_is_reported_as_mixed(self) -> None:
        """Context-dependent results must not be settled by whichever record was written last."""
        self.remember(["--mode=block"], command="systemd-inhibit")
        self.remember(["--mode=block"], command="systemd-inhibit", outcome="failure",
                      note="polkit refused over ssh")
        recall = argument_memory.recall(self.store, "systemd-inhibit")
        self.assertEqual(recall.mixed[0].argv, ["--mode=block"])
        self.assertIn("works in some contexts", recall.statement())

    def test_recording_twice_increments_instead_of_duplicating(self) -> None:
        self.remember(["-q"])
        second = self.remember(["-q"])
        self.assertTrue(second["deduplicated"])
        self.assertEqual(second["times"], 2)
        recall = argument_memory.recall(self.store, "pytest")
        self.assertEqual(len(recall.known_good), 1)
        self.assertEqual(recall.known_good[0].times, 2)

    def test_the_most_used_working_argv_is_preferred(self) -> None:
        self.remember(["-x"])
        for _ in range(3):
            self.remember(["-q", "-n", "4"])
        self.assertEqual(argument_memory.recall(self.store, "pytest").best.argv, ["-q", "-n", "4"])

    # -- secrets never land ------------------------------------------------------------

    def test_a_provider_key_is_refused_and_not_echoed(self) -> None:
        result = self.remember(["--key", "sk-abcdefghijklmnopqrstuvwx"])
        self.assertFalse(result["ok"])
        self.assertIn("provider key", result["reason"])
        self.assertNotIn("abcdefghijkl", str(result))
        self.assertEqual(argument_memory.recall(self.store, "pytest").known_good, [])

    def test_a_bearer_token_is_refused(self) -> None:
        result = self.remember(["-H", "Authorization: Bearer abc123def456"])
        self.assertFalse(result["ok"])
        self.assertIn("bearer token", result["reason"])

    def test_a_named_secret_flag_is_refused(self) -> None:
        for flag in ("--api-key", "--token", "--password", "--auth"):
            result = self.remember([flag, "value"])
            self.assertFalse(result["ok"], flag)

    def test_a_long_opaque_value_is_refused(self) -> None:
        result = self.remember(["-h", "A" * 44])
        self.assertFalse(result["ok"])
        self.assertIn("long opaque value", result["reason"])

    def test_an_ordinary_argument_is_not_mistaken_for_a_secret(self) -> None:
        self.assertTrue(self.remember(["tests/", "-q", "-n", "4", "--maxfail=1"])["ok"])

    # -- inputs ------------------------------------------------------------------------

    def test_an_unknown_outcome_is_refused(self) -> None:
        result = self.remember(["-q"], outcome="maybe")
        self.assertFalse(result["ok"])
        self.assertIn("outcome must be", result["reason"])

    def test_flags_belong_in_argv_not_in_the_command(self) -> None:
        result = argument_memory.remember(self.store, "pytest tests/", ["-q"])
        self.assertFalse(result["ok"])
        self.assertIn("program name", result["reason"])

    def test_an_empty_command_is_refused(self) -> None:
        self.assertFalse(argument_memory.remember(self.store, "  ", ["-q"])["ok"])

    def test_an_empty_argv_is_allowed_and_rendered_plainly(self) -> None:
        self.remember([])
        recall = argument_memory.recall(self.store, "pytest")
        self.assertEqual(recall.best.argv, [])
        self.assertIn("(no arguments)", "\n".join(recall.render()))

    # -- reading -----------------------------------------------------------------------

    def test_an_unknown_command_says_nothing_is_recorded(self) -> None:
        self.assertEqual(argument_memory.recall(self.store, "nope").statement(), "nothing recorded for nope")

    def test_known_commands_are_listed_by_use(self) -> None:
        self.remember(["-q"])
        self.remember(["-q"])
        argument_memory.remember(self.store, "git", ["status"])
        self.assertEqual(argument_memory.known_commands(self.store), ["pytest", "git"])

    def test_a_broken_store_says_it_could_not_read(self) -> None:
        class Broken:
            _conn = None

        recall = argument_memory.recall(Broken(), "pytest")
        self.assertFalse(recall.readable)
        self.assertIn("could not be read", recall.render()[0])

    def test_recall_writes_nothing(self) -> None:
        self.remember(["-q"])
        before = self.store._conn.execute(
            f"SELECT COUNT(*), SUM(times) FROM {argument_memory.ARGUMENT_TABLE}").fetchone()
        argument_memory.recall(self.store, "pytest")
        after = self.store._conn.execute(
            f"SELECT COUNT(*), SUM(times) FROM {argument_memory.ARGUMENT_TABLE}").fetchone()
        self.assertEqual(tuple(before), tuple(after))


if __name__ == "__main__":
    unittest.main()
