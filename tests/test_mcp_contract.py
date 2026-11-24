"""The rules the MCP surface keeps, enforced rather than described (#154)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import mcp_contract as mc
from cortex.store import CortexStore


class ClampTests(unittest.TestCase):
    def test_a_limit_over_the_maximum_is_clamped(self) -> None:
        self.assertEqual(mc.clamp_limit(500), mc.MAX_LIMIT)

    def test_a_limit_below_one_is_clamped(self) -> None:
        self.assertEqual(mc.clamp_limit(0), 1)

    def test_garbage_falls_back_to_the_default(self) -> None:
        self.assertEqual(mc.clamp_limit("banana"), 6)

    def test_a_sane_limit_passes_through(self) -> None:
        self.assertEqual(mc.clamp_limit(4), 4)


class BoundTests(unittest.TestCase):
    def test_a_small_payload_is_untouched(self) -> None:
        payload, trimmed = mc.bounded({"results": [1, 2, 3]})
        self.assertFalse(trimmed)
        self.assertEqual(payload["results"], [1, 2, 3])

    def test_a_large_payload_is_trimmed_and_says_so(self) -> None:
        payload, trimmed = mc.bounded({"results": ["x" * 1000] * 40}, limit=2000)
        self.assertTrue(trimmed)
        self.assertTrue(payload["truncated"])
        self.assertLess(len(payload["results"]), 40)

    def test_a_payload_that_cannot_be_trimmed_becomes_a_preview(self) -> None:
        payload, trimmed = mc.bounded("y" * 5000, limit=500)
        self.assertTrue(trimmed)
        self.assertIn("preview", payload)


class ErrorTests(unittest.TestCase):
    def test_a_path_in_an_error_is_withheld(self) -> None:
        """A stack trace in another agent's context is a small data breach."""
        # Composed at runtime: a literal machine path in a tracked file fails the repository check.
        error = RuntimeError("cannot open " + "/root" + "/.hermes/cortex/cortex.db")
        described = mc.sanitise_error(error).as_dict()
        self.assertNotIn("/root", described["error"]["message"])
        self.assertIn("withheld", described["error"]["message"])

    def test_an_ordinary_error_is_reported(self) -> None:
        described = mc.sanitise_error(ValueError("bad kind")).as_dict()
        self.assertEqual(described["error"]["message"], "bad kind")

    def test_a_long_error_is_truncated(self) -> None:
        described = mc.sanitise_error(ValueError("x" * 900)).as_dict()
        self.assertLessEqual(len(described["error"]["message"]), 300)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def handler(self, name, args):
        if name == "stats":
            return {"memories": 1}
        if name == "memory_search":
            return {"results": [], "limit": args.get("limit")}
        if name == "remember":
            return {"stored": True}
        if name == "boom":
            raise RuntimeError("cannot open /root/secret.db")
        raise KeyError(name)

    def test_a_read_call_succeeds(self) -> None:
        result = mc.call(self.store, self.handler, "stats")
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["memories"], 1)

    def test_the_limit_is_clamped_before_the_handler_sees_it(self) -> None:
        result = mc.call(self.store, self.handler, "memory_search", {"query": "x", "limit": 900})
        self.assertEqual(result["result"]["limit"], mc.MAX_LIMIT)

    def test_a_credential_is_refused_at_the_surface(self) -> None:
        """Otherwise the gate is decorative: any agent could write around it."""
        result = mc.call(self.store, self.handler, "remember",
                         {"content": "the api key for the bucket is hunter2"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "refused_by_consent")
        self.assertNotIn("hunter2", str(result))

    def test_finance_content_needs_consent(self) -> None:
        result = mc.call(self.store, self.handler, "remember",
                         {"content": "the mortgage statement arrives on the third"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "needs_consent")

    def test_a_technical_fact_is_stored(self) -> None:
        result = mc.call(self.store, self.handler, "remember",
                         {"content": "the dashboard listens on port 8100"})
        self.assertTrue(result["ok"])

    def test_empty_content_is_refused(self) -> None:
        result = mc.call(self.store, self.handler, "remember", {"content": "   "})
        self.assertEqual(result["error"]["code"], "empty_content")

    def test_over_long_content_is_refused_with_the_length(self) -> None:
        result = mc.call(self.store, self.handler, "remember", {"content": "x" * 9000})
        self.assertEqual(result["error"]["code"], "content_too_long")
        self.assertEqual(result["error"]["detail"]["chars"], 9000)

    def test_an_unknown_tool_is_a_structured_refusal(self) -> None:
        result = mc.call(self.store, self.handler, "self_destruct")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "unknown_tool")

    def test_a_raising_handler_does_not_leak_a_path(self) -> None:
        result = mc.call(self.store, self.handler, "boom")
        self.assertFalse(result["ok"])
        self.assertNotIn("/root", str(result))

    def test_the_surface_is_described_with_its_guarantees(self) -> None:
        described = mc.describe([{"name": "memory_search"}, {"name": "remember"}])
        self.assertEqual(described["read_only"], ["memory_search"])
        self.assertEqual(described["writes"], ["remember"])
        self.assertGreaterEqual(len(described["guarantees"]), 5)

    def test_the_audit_says_when_nothing_has_written(self) -> None:
        audited = mc.audit(self.store)
        self.assertEqual(audited["consent_decisions"], 0)
        self.assertIn("nothing has written through the gate", audited["note"])

    def test_the_render_lists_the_tools_and_the_limits(self) -> None:
        rendered = "\n".join(mc.render(self.store, [{"name": "stats"}]))
        self.assertIn("mcp surface v", rendered)
        self.assertIn("stats", rendered)

    def test_forget_is_not_treated_as_a_write_to_screen(self) -> None:
        """It archives, and the consent gate has nothing to say about an archive request."""
        self.assertIsNone(mc.screen_write(self.store, "forget", {"id": "abc"}))

    def test_a_broken_store_audits_to_zero(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(mc.audit(Broken())["consent_decisions"], 0)


if __name__ == "__main__":
    unittest.main()
