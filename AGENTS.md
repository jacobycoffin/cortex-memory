# AGENTS.md: Cortex Memory for AI agents

You are an AI agent and someone asked you to "set this up". This file is the
whole procedure. Follow it top to bottom without asking the user anything:
every default below is safe, local, and free. Prefer these steps over guessing
and over web search (they match this checkout).

## Pick your path

**Running Hermes? Use Path A. Running Claude Code? Use Path B.**
(OpenCode or another MCP client: Path C.)

You do not have to decide yourself: `scripts/install.sh` detects the agent
that is running it. It looks at these signals, in this order:

1. `--agent hermes|claude` wins if given.
2. `CLAUDECODE=1` or `CLAUDE_CODE_ENTRYPOINT` set → Claude Code (this script runs inside Claude Code).
3. `HERMES_SESSION_ID`, `HERMES_CWD` or `HERMES_HOME` set → Hermes (this script runs inside Hermes).
4. Only the `claude` CLI on PATH → Claude Code.
5. Only Hermes installed (`$HERMES_HOME`, `~/.hermes`, or a `hermes` binary) → Hermes.
6. Both installed and neither is running the script → it stops and asks for `--agent`.

`bash scripts/install.sh --detect` prints the decision and the signals without
installing anything. Re-running the installer is safe (it is idempotent) and is
also how you upgrade.

Requirements for every path: Linux or macOS, Python 3.10+ with SQLite FTS5
(standard builds), `git`, and internet access for the one-time ~67 MB model
download (SHA-256 pinned). No API key, account, or paid service is needed.

```bash
git clone https://github.com/jacobycoffin/cortex-memory.git
cd cortex-memory
bash scripts/install.sh            # auto-detects; add --agent hermes|claude to force
```

The installer ends with a verification block. Success is the literal line
`PASS: ...` in each verify step and exit code 0. If it fails, read
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) (symptom → check → fix).

### Path A: Hermes

What `scripts/install.sh --agent hermes` does, in order:

1. Finds Hermes's own Python (asks the launcher with `hermes --run-module site`,
   falls back to older venv layouts; override with `HERMES_PYTHON=/path/to/python`).
2. Installs the local embedding runtime (numpy, onnxruntime, tokenizers) into a
   sidecar directory, `$HERMES_HOME/cortex/pydeps/py3.X`. It is deliberately
   outside Hermes's managed environment, so `hermes update` and `hermes pm repair`
   do not wipe it.
3. Copies the plugin to `$HERMES_HOME/plugins/cortex` (`scripts/install_local.sh`,
   which backs up any previous plugin and never restarts Hermes).
4. Downloads and verifies `bge-small-en-v1.5` (ONNX, quantized) into
   `$HERMES_HOME/models/bge-small-en-v1.5`.
5. Backs up `config.yaml` and then sets `memory.provider: cortex` plus
   `plugins.cortex.background_preload: true`, `query_cache_ttl_seconds: 45` and
   `semantic_fusion_weight: 10`. Comments are preserved, and keys you already set are never overwritten.
6. Runs `hermes plugins enable cortex` (Hermes lists newly found plugins as disabled).
7. Embeds any existing memories, then runs `scripts/verify_hermes_install.py --write-test`
   through Hermes's own plugin loader. That checks the config, the loader, semantic fusion (384-dim vectors),
   the background preloader, non-blocking `queue_prefetch`, the prefetch latency
   budget, and a real capture → review → recall round trip (then archives the test memory).

Then restart Hermes (new session, or `hermes gateway restart` if you run the gateway).

**How Hermes memory admission works (important):** in Hermes, Cortex captures
memory candidates from conversation automatically. It *stages* them as review
proposals instead of trusting them blindly. Staged proposals are not recalled
until they are approved. With the default (no remote services), approve them locally:

```bash
PYTHONPATH="$HERMES_HOME/plugins" python3 -m cortex review list          # pending proposals
PYTHONPATH="$HERMES_HOME/plugins" python3 -m cortex review approve <id>  # or: reject <id>, approve-all
```

(Use Hermes's Python if `python3` lacks PyYAML; the installer prints it.) The
local dashboard (`scripts/install_dashboard_service.sh`) offers the same queue
in a browser. Automatic review is the optional remote auto-judge, which is off by default (see
"Optional paid component" below).

### Path B: Claude Code

What `scripts/install.sh --agent claude` does, in order:

1. Creates a private venv at `$CORTEX_HOME/venv` (`CORTEX_HOME` defaults to
   `~/.cortex`, mode 700) and runs `pip install '.[embeddings]'` there.
2. Downloads and verifies the embedding model into `$CORTEX_HOME/models/bge-small-en-v1.5`.
3. Registers the stdio MCP server for all projects:
   `claude mcp add -s user cortex -e CORTEX_HOME=$CORTEX_HOME -- $CORTEX_HOME/venv/bin/python -m cortex.mcp_server`.
   Tools: `memory_search`, `context_brief`, `remember`, `forget` (reversible archive), `stats`.
4. Merges three hooks into `~/.claude/settings.json` (after a backup; skip them with `--no-hooks`):
   `SessionStart` injects a short brief; `UserPromptSubmit` injects memories relevant to the prompt;
   `Stop` logs an episode (set `CORTEX_RECORD_EPISODES=0` to turn that off).
5. Adds the memory-usage block from [docs/claude-memory-block.md](docs/claude-memory-block.md)
   to `~/.claude/CLAUDE.md` between `<!-- cortex:begin -->` markers (replaced on re-run; skip with `--no-claude-md`).
6. Verifies: `scripts/mcp_smoke_test.py` drives every tool over JSON-RPC against a
   temporary database, then `python -m cortex.mcp_server doctor` checks the real store, then `claude mcp get cortex` must show `✔ Connected`.

Then start a new Claude Code session (or run `/mcp`).

In Claude Code, `remember` writes active memories directly (the agent is acting on
your instructions in that session); `forget` archives reversibly. Data lives in
`$CORTEX_HOME/cortex.db`. Back it up with `$CORTEX_HOME/venv/bin/python -m cortex.mcp_server backup`.

Manual equivalent, if you cannot run the script:

```bash
python3 -m venv ~/.cortex/venv
~/.cortex/venv/bin/pip install '.[embeddings]'
~/.cortex/venv/bin/python scripts/download_embedding_model.py --output ~/.cortex/models/bge-small-en-v1.5
claude mcp add -s user cortex -e CORTEX_HOME="$HOME/.cortex" -- ~/.cortex/venv/bin/python -m cortex.mcp_server
~/.cortex/venv/bin/python scripts/mcp_smoke_test.py      # must print PASS
```

### Path C: OpenCode or any other MCP client

Run Path B steps 1–2 (`bash scripts/install.sh --agent claude --no-hooks --no-claude-md`
does them; it skips registration if `claude` is absent and writes `~/.claude.json` instead,
which other clients ignore), then register this stdio server in your client.
For OpenCode, `opencode.json`:

```json
{ "mcp": { "cortex": { "type": "local", "enabled": true,
  "command": ["/home/YOU/.cortex/venv/bin/python", "-m", "cortex.mcp_server"],
  "environment": { "CORTEX_HOME": "/home/YOU/.cortex" } } } }
```

Paste [docs/claude-memory-block.md](docs/claude-memory-block.md) into your client's instructions file (`AGENTS.md`).

## Optional paid component: auto-judge and Jev (OFF by default, always)

Everything above works with no remote service. Two optional pieces call one:

- **Auto-judge** (Hermes only): a five-minute timer that sends staged proposals to an
  LLM (OpenRouter by default) to approve or reject them automatically. Needs an API key and sends memory text off the machine.
- **Jev**: TypeSafe's paid "System One" judgment API. It is an alternative engine for the auto-judge and needs a `TYPESAFE_API_KEY`.

Both stay off even when credentials exist on the machine. They turn on only when you explicitly ask:

| When | Turn on | Turn off |
|---|---|---|
| During setup | `bash scripts/install.sh --enable-jev` (implies the auto-judge) or `--enable-auto-judge` | `--no-jev` / `--no-auto-judge` |
| After setup | `CORTEX_JEV_ENABLED=1` / `CORTEX_AUTO_JUDGE_ENABLED=1` in `$HERMES_HOME/cortex/auto-judge.env` | set them to `0` |

`CORTEX_JEV_ENABLED=0` forces the chat engine even if `CORTEX_AUTO_JUDGE_ENGINE=jev`
is set. Details and rollback: [docs/AUTO_JUDGE.md](docs/AUTO_JUDGE.md) and [docs/JEV.md](docs/JEV.md).

## Maintenance and upgrades

- **Upgrade Cortex:** `git pull && bash scripts/install.sh` (same flags as before). It is idempotent and keeps your data and settings.
- **After `hermes update` / `hermes pm repair`:** nothing is required (the embedding runtime lives in the sidecar).
  If Hermes moved to a new Python minor version, re-run `bash scripts/install.sh`. If you installed systemd timers, re-run their installers, because they embed Hermes's Python path.
- **Backups:** Claude Code: `~/.cortex/venv/bin/python -m cortex.mcp_server backup [PATH]`
  (to `~/.cortex/backups/` by default). Hermes: copy `$HERMES_HOME/cortex/cortex.db` while Hermes is stopped,
  or `python3 -c "import sqlite3;s=sqlite3.connect('$HERMES_HOME/cortex/cortex.db');s.backup(sqlite3.connect('cortex-backup.db'))"`.
  Restore steps: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md#restore-from-a-backup).
- **Health:** `python -m cortex.mcp_server doctor` (Claude Code) or
  `PYTHONPATH="$HERMES_HOME/plugins" <hermes-python> scripts/verify_hermes_install.py` (Hermes).
- **Uninstall:** Claude Code: `claude mcp remove -s user cortex`, delete the `cortex` hook entries and the
  `cortex:begin/end` block, then remove `~/.cortex` (that deletes your memories). Hermes: `scripts/uninstall_local.sh`.

## What this is

Local-first durable memory for agent harnesses: bounded recall, outcome
learning, reversible consolidation, offline Sleep, and a Brain dashboard.
The core (`CortexMemory`, `RecallBatch`) is harness-neutral; Hermes and the MCP server are adapters.
Psychology inspires hypotheses; benchmarks decide.

## Reference: advanced Hermes configuration

The installer already covers a standard setup. The rest of this file is for operators.


1. Cortex-primary deployment: set `memory.nudge_interval: 0` in Hermes config
   so the legacy background writer stops; Cortex `sync_turn` owns proposals.
2. Vault import (read-only toward the vault; preview before applying):

```bash
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex vault-index "$HOME/.hermes/obsidian-vault"
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex vault-index "$HOME/.hermes/obsidian-vault" --apply
```

3. Nightly timers (all shadow/off by default; enable explicitly):

```bash
HERMES_HOME="$HOME/.hermes" VAULT_PATH="$HOME/.hermes/obsidian-vault" ./scripts/install_vault_timer.sh
HERMES_HOME="$HOME/.hermes" ./scripts/install_sleep_timer.sh
```

4. Automatic judge (remote LLM review of staged proposals — privacy-sensitive,
   normally billed; read [docs/AUTO_JUDGE.md](docs/AUTO_JUDGE.md) first, and
   [docs/JEV.md](docs/JEV.md) for the Jev decision engine and its rollback):

```bash
bash scripts/install.sh --agent hermes --enable-auto-judge     # or --enable-jev
```

5. Dashboard (localhost only; put Caddy/Nginx/Cloudflare Tunnel in front):

```bash
HERMES_HOME="$HOME/.hermes" PORT=8100 ./scripts/install_dashboard_service.sh
```

Behind a proxy or tunnel, list every hostname the dashboard answers on so it can
reject DNS-rebinding requests that arrive with a foreign `Host` header:

```bash
CORTEX_DASHBOARD_ALLOWED_HOSTS="127.0.0.1,localhost,dashboard.example.com"
```

Unset means no `Host` filtering (the pre-existing behaviour), which is what a
loopback-only dashboard needs; set it as soon as the dashboard is reachable
under a name a browser can be tricked into resolving.

## Verify (acceptance test)

```bash
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex audit
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex recall-stats
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex recall-set-health
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex sleep --mode shadow --reflection-token-budget 0
```

`audit.ok` must be `true`. Keep pruning/consolidation in `shadow` until you
have real pruning-regret and retrieval evidence. Behavior checks: store a
decision in session A, paraphrase-ask in session B, explain provenance,
correct it (old version must survive), and confirm a greeting injects ~zero
memory tokens.

## Integrate a new harness (five-event contract)

```bash
# The plugin path, as install_local.sh arranges it (no console script exists there):
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex harness-contract --tool-name your_memory_tool
# Or, with an installed package:
cortex-memory harness-contract --tool-name your_memory_tool
```

Wire the printed lifecycle in code (recall before inference, inject as
fallible evidence, resolve used IDs + outcome after, record episodes).
Cortex is the durable store of record; keep harness-native memory to a
bootstrap pointer plus session scratch. Never store secrets, credentials, or
authorizations as memories — remember a safe location reference instead.
Full contract: [docs/INTEGRATION.md](docs/INTEGRATION.md).

## Evaluate before claiming anything

- Real-history retrieval: label your own cases locally (never commit them),
  then `scripts/evaluate_real_history.py` (refuses < `--min-cases`, default 8;
  fails closed on privacy self-check). Needs a `cortex`-importable package:
  `pip install .` first, or run from a checkout whose parent dir exposes it.
- Paired tool-calling: `scripts/benchmark_tool_calling.py recorded|live`.
- Claim boundaries (what each runner does and does not prove):
  [docs/EVALUATION.md](docs/EVALUATION.md). Small samples are exploratory.

## Build / contribute

```bash
python3 -m unittest discover -s tests      # full suite, must stay green (use a venv with the ONNX deps; without them the embedding tests skip)
python3 -m pytest tests/ -q                # same suite via pytest (needs pytest installed)
python3 scripts/check_repository.py # privacy + packaging gates before push
```

`main` is stable; measurement work lands on `testing` (see
[docs/ROADMAP.md](docs/ROADMAP.md) for what is implemented vs deferred).
Sanitized aggregate reports only — raw queries, memory IDs, prompts, and tool
results never enter the repo. Sleep proposals stay `shadow` until evidence
justifies `--apply`, and every apply path must keep its undo.
