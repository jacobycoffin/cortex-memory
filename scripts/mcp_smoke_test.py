#!/usr/bin/env python3
"""End-to-end JSON-RPC smoke test for the Cortex stdio MCP server.

Usage: python scripts/mcp_smoke_test.py [--python PY] [--keep]

Starts ``PY -m cortex.mcp_server`` with ``CORTEX_HOME`` pointing at the current
value (or a temporary directory unless --use-home is given), then calls
initialize, tools/list and every tool. Exits non-zero on the first failure.
Writes one clearly-labelled test memory and archives it again with ``forget``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--use-home", action="store_true",
                        help="test against the real CORTEX_HOME instead of a temporary one")
    args = parser.parse_args()
    env = dict(os.environ)
    temp = None
    if not args.use_home:
        temp = tempfile.TemporaryDirectory(prefix="cortex-mcp-smoke-")
        env["CORTEX_HOME"] = temp.name
        real_model = os.path.join(os.path.expanduser(os.environ.get("CORTEX_HOME", "~/.cortex")),
                                  "models", "bge-small-en-v1.5")
        if "CORTEX_MODEL_DIR" not in env and os.path.isdir(real_model):
            env["CORTEX_MODEL_DIR"] = real_model
    proc = subprocess.Popen([args.python, "-m", "cortex.mcp_server"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    next_id = 0

    def rpc(method, params=None, notify=False):
        nonlocal next_id
        message = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            next_id += 1
            message["id"] = next_id
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()
        if notify:
            return None
        line = proc.stdout.readline()
        if not line:
            raise SystemExit("server exited: " + proc.stderr.read())
        response = json.loads(line)
        if "error" in response:
            raise SystemExit(f"{method} failed: {response['error']}")
        return response["result"]

    def tool(name, arguments=None):
        started = time.perf_counter()
        result = rpc("tools/call", {"name": name, "arguments": arguments or {}})
        ms = (time.perf_counter() - started) * 1000
        text = result["content"][0]["text"]
        if result.get("isError"):
            raise SystemExit(f"tool {name} returned an error: {text}")
        print(f"ok  {name:<14} {ms:7.1f} ms  {text[:110].replace(chr(10), ' | ')}")
        return text

    try:
        init = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                  "clientInfo": {"name": "smoke", "version": "0"}})
        rpc("notifications/initialized", notify=True)
        print("ok  initialize     server", init["serverInfo"])
        names = [t["name"] for t in rpc("tools/list")["tools"]]
        print("ok  tools/list    ", names)
        for required in ("memory_search", "context_brief", "remember", "forget", "stats"):
            assert required in names, required
        stored = json.loads(tool("remember", {"content": "Cortex smoke test: the deploy script lives in scripts/deploy.sh",
                                              "kind": "operational"}))
        assert stored["id"], stored
        found = json.loads(tool("memory_search", {"query": "where is the deploy script?"}))
        assert any(r["id"] == stored["id"] for r in found["results"]), found
        brief = tool("context_brief", {"task": "deploy the app"})
        assert stored["id"][:8] in brief, brief
        tool("stats")
        forgotten = json.loads(tool("forget", {"id": stored["id"], "reason": "smoke test cleanup"}))
        assert forgotten["archived"], forgotten
        again = json.loads(tool("memory_search", {"query": "where is the deploy script?"}))
        assert all(r["id"] != stored["id"] for r in again["results"]), again
        print("PASS: all Cortex MCP tools answered over stdio JSON-RPC")
        return 0
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
