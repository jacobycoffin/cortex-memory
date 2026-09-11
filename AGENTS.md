# AGENTS.md — Cortex Memory for AI agents

You are an AI agent. This file tells you how to install, configure, verify,
and build on `cortex-memory`. Prefer these steps over guessing; prefer the
docs linked here over web search (they match this checkout).

## What this is

Local-first durable memory for agent harnesses: bounded recall, outcome
learning, reversible consolidation, offline Sleep, and a Brain dashboard.
Hermes is the included reference adapter, not a requirement — the core
(`CortexMemory`, `RecallBatch`) is harness-neutral. Psychology inspires
hypotheses; benchmarks decide.

## Install

Requirements: Python 3.10+, SQLite with FTS5 (standard Python builds).

Agent-neutral core (any harness):

```bash
git clone https://github.com/jacobycoffin/cortex-memory.git
cd cortex-memory
python3 -m pip install .
```

Hermes adapter (Hermes hosts only):

```bash
git clone https://github.com/jacobycoffin/cortex-memory.git
cd cortex-memory
HERMES_HOME="$HOME/.hermes" ./scripts/install_local.sh
hermes memory setup   # choose `cortex`, then restart Hermes yourself
```

The installer backs up any previous plugin, never restarts Hermes, and never
enables the automatic-judge timer without explicit egress consent (below).
Full migration/backup/rollback: [docs/QUICKSTART.md](docs/QUICKSTART.md).

## Configure (Hermes host)

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
   normally billed; read [docs/AUTO_JUDGE.md](docs/AUTO_JUDGE.md) first):

```bash
CORTEX_INSTALL_AUTO_JUDGE_TIMER=1 CORTEX_AUTO_JUDGE_DATA_EGRESS_CONSENT=1 \
  HERMES_HOME="$HOME/.hermes" ./scripts/install_local.sh
```

5. Dashboard (localhost only; put Caddy/Nginx/Cloudflare Tunnel in front):

```bash
HERMES_HOME="$HOME/.hermes" PORT=8100 ./scripts/install_dashboard_service.sh
```

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
