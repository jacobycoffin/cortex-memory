"""MCP server (stdio JSON-RPC) and Claude Code hook behaviour."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import mcp_server
from cortex.mcp_server import CortexService, handle_message


class McpServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"CORTEX_HOME": str(self.home), "CORTEX_PRELOAD": "0"})
        self.env.start()
        self.service = CortexService(self.home, warm=False)
        self.state: dict = {}

    def tearDown(self) -> None:
        self.service.close()
        self.env.stop()
        self.tmp.cleanup()

    def call(self, method: str, params: dict | None = None, msg_id: int | None = 1):
        message = {"jsonrpc": "2.0", "method": method}
        if msg_id is not None:
            message["id"] = msg_id
        if params is not None:
            message["params"] = params
        return handle_message(lambda: self.service, message, self.state)

    def tool(self, name: str, **arguments):
        response = self.call("tools/call", {"name": name, "arguments": arguments})
        self.assertIn("result", response)
        return response["result"]

    def test_initialize_echoes_protocol_and_lists_tools(self) -> None:
        init = self.call("initialize", {"protocolVersion": "2025-06-18"})["result"]
        self.assertEqual(init["protocolVersion"], "2025-06-18")
        self.assertEqual(init["serverInfo"]["name"], "cortex")
        self.assertIsNone(self.call("notifications/initialized", msg_id=None))
        names = [tool["name"] for tool in self.call("tools/list")["result"]["tools"]]
        self.assertEqual(names, ["memory_search", "context_brief", "remember", "forget", "stats"])

    def test_remember_search_brief_forget_round_trip(self) -> None:
        stored = json.loads(self.tool("remember", content="Deploys to the zebra cluster need a canary first.",
                                      kind="procedure")["content"][0]["text"])
        self.assertTrue(stored["created"])
        self.assertEqual(stored["state"], "active")
        found = json.loads(self.tool("memory_search", query="zebra cluster canary")["content"][0]["text"])
        self.assertIn(stored["id"], [item["id"] for item in found["results"]])
        brief = self.tool("context_brief", task="deploy to zebra")["content"][0]["text"]
        self.assertIn("zebra", brief)
        forgotten = json.loads(self.tool("forget", id=stored["id"][:8])["content"][0]["text"])
        self.assertTrue(forgotten["archived"])
        after = json.loads(self.tool("memory_search", query="zebra cluster canary")["content"][0]["text"])
        self.assertNotIn(stored["id"], [item["id"] for item in after["results"]])

    def test_bad_tool_input_is_a_tool_error_not_a_crash(self) -> None:
        result = self.tool("remember", content="")
        self.assertTrue(result.get("isError"))
        unknown = self.call("tools/call", {"name": "nope", "arguments": {}})
        self.assertTrue("error" in unknown or unknown["result"].get("isError"))
        self.assertIn("error", self.call("no/such/method"))

    def test_stats_reports_home_and_paths(self) -> None:
        stats = json.loads(self.tool("stats")["content"][0]["text"])
        self.assertEqual(Path(stats["cortex_home"]), self.home)
        self.assertEqual(stats["memories"], 0)

    def test_backup_is_private_sqlite_copy(self) -> None:
        self.tool("remember", content="The backup test memory mentions walrus.")
        path = self.service.backup()
        self.assertTrue(path.exists())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)


class ModelDirTests(unittest.TestCase):
    def test_cortex_model_dir_overrides_hermes_home(self) -> None:
        from cortex.embeddings import default_model_dir

        with patch.dict(os.environ, {"CORTEX_MODEL_DIR": "/opt/m", "HERMES_HOME": "/h"}):
            self.assertEqual(default_model_dir(), Path("/opt/m"))
        env = {k: v for k, v in os.environ.items() if k != "CORTEX_MODEL_DIR"}
        env["HERMES_HOME"] = "/h"
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(default_model_dir(), Path("/h/models/bge-small-en-v1.5"))


class HookTests(unittest.TestCase):
    def test_hooks_never_fail_and_prompt_hook_recalls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {"CORTEX_HOME": tmp, "CORTEX_PRELOAD": "0", "CORTEX_RECORD_EPISODES": "1"}
            with patch.dict(os.environ, env):
                service = CortexService(Path(tmp), warm=False)
                service.remember("The walrus service listens behind the blue proxy.", kind="operational")
                service.close()
                for event, payload in (
                    ("session-start", {"session_id": "s", "source": "startup"}),
                    ("prompt", {"session_id": "s", "prompt": "where does the walrus service listen?"}),
                    ("stop", {"session_id": "s", "last_assistant_message": "Behind the blue proxy."}),
                ):
                    out = io.StringIO()
                    with patch("sys.stdin", io.StringIO(json.dumps(payload))), redirect_stdout(out):
                        code = mcp_server.main(["hook", event])
                    self.assertEqual(code, 0, event)
                    if event == "prompt":
                        self.assertIn("walrus", out.getvalue())


if __name__ == "__main__":
    unittest.main()
