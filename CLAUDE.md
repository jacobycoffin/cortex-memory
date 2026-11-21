# CLAUDE.md: set up Cortex Memory for Claude Code

If the user said "set this up", do exactly this. Do not ask them anything:
the defaults are local, free, and need no API keys.

```bash
bash scripts/install.sh            # detects Claude Code via CLAUDECODE=1; same as --agent claude
```

It is done when the output contains `PASS: all Cortex MCP tools answered over
stdio JSON-RPC` and `claude mcp get cortex` shows `Status: ✔ Connected`, and the
script exits 0. Then tell the user to start a new Claude Code session (or run
`/mcp`) so the `cortex` tools and hooks load.

What it installs (all under `~/.cortex` unless `CORTEX_HOME` is set):

- a private venv with `cortex-memory[embeddings]` and the pinned local
  embedding model (`bge-small-en-v1.5`, ~67 MB, SHA-256 verified);
- the MCP server `cortex` at user scope, with the tools `memory_search`, `context_brief`,
  `remember`, `forget`, `stats`;
- hooks in `~/.claude/settings.json` (SessionStart brief, UserPromptSubmit
  recall, Stop episode log). Pass `--no-hooks` to skip them;
- the memory-usage block from `docs/claude-memory-block.md` in `~/.claude/CLAUDE.md`. Pass `--no-claude-md` to skip it.

Re-running is safe and is how you upgrade (`git pull && bash scripts/install.sh`).
Optional paid components (the auto-judge and Jev) stay off and are Hermes-only; leave them alone
unless the user explicitly asks.

If something fails, follow `docs/TROUBLESHOOTING.md`. The full guide, including
manual steps and the Hermes path, is in `AGENTS.md`.

## Working on this repository (contributors)

```bash
python3 -m venv .venv && .venv/bin/pip install '.[embeddings]' pytest
.venv/bin/python -m pytest tests/ -q
python3 scripts/check_repository.py      # privacy + packaging gates
```

Never commit real memories, databases, prompts, hostnames, or credentials.
