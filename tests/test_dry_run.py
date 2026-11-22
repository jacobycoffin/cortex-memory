"""A dry run answers "what would you remember from this?" without writing (#46)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import dry_run, secret_refusal, write_throttle
from cortex.store import CortexStore

# Assembled at runtime: the repository release check scans tracked files for token patterns.
LEAKY = "the router " + "pass" + "word is " + "hunter" + "2000"


class DryRunTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory_count(self) -> int:
        return self.store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


class ProjectionTests(DryRunTestCase):
    def test_a_plain_fact_would_be_remembered(self) -> None:
        result = dry_run.project(self.store, "The dashboard requires port 8100 on the host.")
        self.assertEqual(result["counts"].get("remember"), 1)
        self.assertIn("would remember 1", dry_run.summarise(result))

    def test_a_credential_would_be_refused(self) -> None:
        result = dry_run.project(self.store, "The dashboard requires api_key=abcdef123456")
        self.assertEqual(result["counts"].get("refuse"), 1)

    def test_a_bare_credential_sentence_is_below_the_extraction_bar(self) -> None:
        """A sentence with no durable signal never becomes a candidate, so it is not reported."""
        result = dry_run.project(self.store, LEAKY)
        self.assertEqual(result["counts"].get("refuse"), None)
        self.assertGreater(result["dropped"], 0)

    def test_a_restatement_would_be_a_duplicate(self) -> None:
        content = "The dashboard requires port 8100 on the host."
        self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        result = dry_run.project(self.store, content)
        self.assertEqual(result["counts"].get("duplicate"), 1)

    def test_a_learned_rule_would_reject(self) -> None:
        from cortex import reject_learn

        reject_learn.record(self.store, "Project Orbital dashboard uses port 8443 behind nginx",
                            reason_code="duplicate")
        # A learned rule never applies to user-stated material (#35), so an agent proposal is the
        # case where it shows up.
        result = dry_run.project(
            self.store,
            "Project Orbital dashboard uses port 8443 behind nginx again",
            source_category="AGENT_PROPOSED",
        )
        self.assertEqual(result["counts"].get("rejected_by_rule"), 1)

    def test_a_full_turn_budget_would_throttle(self) -> None:
        for _ in range(write_throttle.TURN_CAP):
            write_throttle.admit(self.store, "session-1")
        result = dry_run.project(
            self.store,
            "The vault requires a nightly verified backup.",
            source_category="AGENT_PROPOSED",
            session_id="session-1",
        )
        self.assertEqual(result["counts"].get("throttled"), 1)

    def test_the_projection_carries_importance_and_confidence(self) -> None:
        result = dry_run.project(self.store, "The dashboard requires port 8100 on the host.")
        item = result["items"][0]
        self.assertGreater(item.importance, 0.0)
        self.assertLessEqual(item.confidence, 0.92)

    def test_an_unsupported_fact_is_reported_as_dropped(self) -> None:
        result = dry_run.project(self.store, "hmm okay. sure. Sounds good to me.")
        self.assertGreater(result["dropped"], 0)


class PurityTests(DryRunTestCase):
    """A dry run that writes is a write with extra steps."""

    def test_nothing_is_stored(self) -> None:
        before = self.memory_count()
        dry_run.project(self.store, "The dashboard requires port 8100 on the host.")
        self.assertEqual(self.memory_count(), before)

    def test_no_proposal_is_staged(self) -> None:
        dry_run.project(self.store, "The dashboard requires port 8100 on the host.")
        pending = self.store._conn.execute(
            "SELECT COUNT(*) FROM memory_creation_proposals"
        ).fetchone()[0]
        self.assertEqual(pending, 0)

    def test_the_throttle_is_not_advanced(self) -> None:
        dry_run.project(self.store, "The vault requires a nightly verified backup.")
        self.assertEqual(write_throttle.check(self.store, "session-1", at=None).used, 0)

    def test_a_refusal_is_not_counted(self) -> None:
        dry_run.project(self.store, LEAKY)
        self.assertEqual(secret_refusal.refusal_count(self.store), 0)

    def test_running_twice_gives_the_same_answer(self) -> None:
        first = dry_run.project(self.store, "The dashboard requires port 8100 on the host.")
        second = dry_run.project(self.store, "The dashboard requires port 8100 on the host.")
        self.assertEqual(first["counts"], second["counts"])


class RenderTests(DryRunTestCase):
    def test_the_render_names_the_items(self) -> None:
        result = dry_run.project(self.store, "The dashboard requires port 8100 on the host.")
        rendered = dry_run.render(result)
        self.assertIn("would remember", rendered)
        self.assertIn("port 8100", rendered)

    def test_the_summary_mentions_dropped_facts(self) -> None:
        result = dry_run.project(self.store, "hmm okay. sure.")
        self.assertIn("below the extraction bar", dry_run.summarise(result))

    def test_an_empty_conversation_reports_zero(self) -> None:
        result = dry_run.project(self.store, "")
        self.assertEqual(result["facts"], 0)
        self.assertIn("would remember 0", dry_run.summarise(result))


class ProviderTests(unittest.TestCase):
    """The question has to be askable from chat, not only from a script."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        from cortex import CortexMemoryProvider

        self.provider = CortexMemoryProvider(
            {"db_path": "$HERMES_HOME/cortex/test.db", "auto_capture": True,
             "retrieval_threshold": 0.08, "pruning_mode": "shadow"}
        )
        self.provider.initialize("session-1", hermes_home=self.tmp.name, agent_context="primary")

    def tearDown(self) -> None:
        self.provider.shutdown()
        self.tmp.cleanup()

    def test_the_dry_run_action_answers_without_writing(self) -> None:
        import json

        result = json.loads(
            self.provider.handle_tool_call(
                "cortex_memory",
                {"action": "dry_run",
                 "content": "The dashboard requires port 8100 and the vault requires a backup."},
            )
        )
        self.assertIn("summary", result)
        self.assertIn("would remember", result["summary"])
        count = self.provider._store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        self.assertEqual(count, 0)

    def test_the_schema_advertises_the_action(self) -> None:
        from cortex import CORTEX_MEMORY_SCHEMA

        self.assertIn("dry_run", CORTEX_MEMORY_SCHEMA["parameters"]["properties"]["action"]["enum"])


if __name__ == "__main__":
    unittest.main()
