"""A credential is refused, not scrubbed (#43)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import secret_refusal
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


# Sample credentials are assembled at runtime on purpose: a test that pastes a real-looking token
# into a tracked file trips the repository's own release check, which scans for exactly the patterns
# this module detects. What is under test is the detector, not the fixture.
BANNER = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"
SAMPLES = (
    "AKIA" + "IOSFODNN7EXAMPLE",
    "gh" + "p_" + "a" * 36,
    "xo" + "xb-1234567890-abcdefghij",
    "AI" + "za" + "b" * 35,
    "ey" + "JhbGciOiJIUzI1NiJ9." + "eyJzdWIiOiIxMjM0In0." + "SflKxwRJSMeKKF2QT4",
    "Bea" + "rer abcdefghijklmnopqrstuvwxyz",
    "s" + "k-" + "c" * 30,
)


class RefusalTests(unittest.TestCase):
    def test_a_private_key_block_is_refused(self) -> None:
        verdict = secret_refusal.assess(BANNER + "\nabc\n-----END " + BANNER[11:])
        self.assertTrue(verdict.refused)
        self.assertIn("private_key", verdict.reasons)

    def test_provider_tokens_are_refused(self) -> None:
        for sample in SAMPLES:
            self.assertTrue(secret_refusal.assess(f"here it is {sample}").refused, sample)

    def test_a_labelled_password_with_a_value_is_refused(self) -> None:
        for text in (
            "the router password is hunter2000",
            "password: hunter2000",
            "api_key=abc123def456",
            "the client secret is s3cr3tvalue",
        ):
            verdict = secret_refusal.assess(text)
            self.assertTrue(verdict.refused, text)

    def test_the_matched_value_is_not_echoed_in_the_verdict(self) -> None:
        """A refusal record must not become the leak it is refusing."""
        verdict = secret_refusal.assess("the router password is hunter2000")
        self.assertNotIn("hunter2000", " ".join(verdict.matched))


class RedactTests(unittest.TestCase):
    def test_a_location_is_a_mention_not_a_secret(self) -> None:
        """Where the password lives is useful and carries no value, so it is allowed."""
        self.assertEqual(secret_refusal.assess("the password is stored in the vault").action,
                         secret_refusal.ALLOW)

    def test_a_scrubbed_value_is_neutralised(self) -> None:
        verdict = secret_refusal.assess("password: ***")
        self.assertEqual(verdict.action, secret_refusal.REDACT)

    def test_an_already_redacted_value_is_only_redacted(self) -> None:
        self.assertEqual(secret_refusal.assess("password = [REDACTED]").action,
                         secret_refusal.REDACT)

    def test_a_rotated_credential_is_a_mention(self) -> None:
        self.assertEqual(secret_refusal.assess("the api key was rotated on friday").action,
                         secret_refusal.ALLOW)


class AllowTests(unittest.TestCase):
    def test_ordinary_text_is_allowed(self) -> None:
        for text in (
            "The dashboard runs on port 8100.",
            "Jacoby prefers short answers.",
            "The vault lives at ~/.hermes/obsidian-vault.",
            "",
        ):
            self.assertEqual(secret_refusal.assess(text).action, secret_refusal.ALLOW, text)

    def test_a_bare_mention_is_allowed(self) -> None:
        verdict = secret_refusal.assess("I keep the router password in the vault")
        self.assertEqual(verdict.action, secret_refusal.ALLOW)
        self.assertIn("without a value", " ".join(verdict.reasons))

    def test_a_short_token_like_word_is_not_a_secret(self) -> None:
        self.assertEqual(secret_refusal.assess("the token bucket refills").action,
                         secret_refusal.ALLOW)


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_refusal_is_counted(self) -> None:
        verdict = secret_refusal.assess("password: hunter2000")
        self.assertEqual(secret_refusal.note_refusal(self.store, verdict, at=NOW), 1)
        self.assertEqual(secret_refusal.refusal_count(self.store), 1)

    def test_repeated_refusals_accumulate(self) -> None:
        verdict = secret_refusal.assess("password: hunter2000")
        for _ in range(3):
            secret_refusal.note_refusal(self.store, verdict, at=NOW)
        described = secret_refusal.report(self.store)
        self.assertEqual(described["refusals"], 3)
        self.assertTrue(described["reasons"])

    def test_an_empty_counter_reports_zero(self) -> None:
        self.assertEqual(secret_refusal.report(self.store)["refusals"], 0)

    def test_the_verdict_describes_itself(self) -> None:
        self.assertIn("refused", secret_refusal.assess(SAMPLES[0]).describe())
        self.assertIn("no credential", secret_refusal.assess("a plain fact").describe())


class ProviderTests(unittest.TestCase):
    """The refusal has to hold at the tool boundary too, not only in the helper."""

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

    def call(self, **args):
        import json

        return json.loads(self.provider.handle_tool_call("cortex_memory", args))

    def test_a_credential_is_refused_and_not_stored(self) -> None:
        result = self.call(action="remember", content="the router password is hunter2000")
        self.assertTrue(result.get("refused"))
        self.assertIn("credential", result.get("message", "").lower() or result.get("message"))
        count = self.provider._store._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content LIKE '%hunter2000%'"
        ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_the_refusal_is_counted(self) -> None:
        self.call(action="remember", content="the router password is hunter2000")
        self.assertEqual(secret_refusal.refusal_count(self.provider._store), 1)

    def test_an_ordinary_fact_still_stores(self) -> None:
        result = self.call(action="remember", content="The dashboard runs on port 8100.")
        self.assertFalse(result.get("refused"))
        self.assertTrue(result.get("proposal_id") or result.get("memory_id"))


class CaptureTests(unittest.TestCase):
    """The real path: a credential must never reach the store."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_the_redaction_layer_still_runs_for_everything_else(self) -> None:
        """#43 adds refusal; it does not remove the existing redaction."""
        from cortex.security import sanitize_memory

        sanitized = sanitize_memory("api key: abcdef123456")
        self.assertTrue(sanitized.redacted)
        self.assertNotIn("abcdef123456", sanitized.text)


if __name__ == "__main__":
    unittest.main()
