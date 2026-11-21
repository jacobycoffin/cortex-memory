# Troubleshooting

Each entry is **symptom → check → fix**. `CORTEX_HOME` defaults to `~/.cortex`
(Claude Code / MCP) and `HERMES_HOME` to `~/.hermes`. "Hermes Python" is the
interpreter the installer prints as `Hermes Python:`. Re-running
`bash scripts/install.sh` is always safe and fixes most problems.

## Installer

**`ERROR: both Claude Code and Hermes are installed and neither is running this script`**
- Check: `bash scripts/install.sh --detect` shows the signals it saw.
- Fix: re-run with `--agent claude` or `--agent hermes`, or run it from inside the agent you want.

**`ERROR: found neither Claude Code ... nor Hermes`**
- Fix: install the agent first, or pass `--agent claude` (this also covers OpenCode and other MCP clients; see AGENTS.md Path C).

**`python venv failed`** (Debian/Ubuntu)
- Fix: `sudo apt install python3-venv`, or set `CORTEX_PYTHON=/path/to/python3.10+`.

**Model download fails or the checksum does not match**
- Check: network/proxy access to huggingface.co.
- Fix: re-run (the download resumes and verifies). Offline machines: copy
  `models/bge-small-en-v1.5` from another machine into `$CORTEX_HOME/models/` or
  `$HERMES_HOME/models/`. Until the model is there Cortex still works with lexical (keyword) recall.

## Claude Code

**The `cortex` tools are not in Claude Code**
- Check: `claude mcp get cortex` should show `✔ Connected`.
- Fix: start a new session or run `/mcp`. If it shows "failed", run the
  command it lists by hand (`~/.cortex/venv/bin/python -m cortex.mcp_server doctor`) to see the error, then re-run the installer.

**Hooks do nothing / an error on every prompt**
- Check: `~/.claude/settings.json` has the `cortex.mcp_server hook` commands, and
  `echo '{"session_id":"t","prompt":"test prompt here"}' | ~/.cortex/venv/bin/python -m cortex.mcp_server hook prompt` exits 0.
- Fix: re-run the installer (it rewrites the hook paths), or `--no-hooks` to remove them. The tools work without hooks.

**Search only finds exact words (no semantic matches)**
- Check: `~/.cortex/venv/bin/python -m cortex.mcp_server doctor` → `semantic_recall: true`, `model_files_present: true`, `embedder_load_error: null`.
- Fix: re-run the installer (reinstalls `[embeddings]` and the model). Missing embeddings for old memories are backfilled automatically when the server starts.

## Hermes

**`hermes memory status` does not show cortex as active**
- Check: `config.yaml` has `memory.provider: cortex`, and `plugins.disabled` does not list `cortex`.
- Fix: `hermes plugins enable cortex`, then re-run the installer, then restart Hermes.

**"Memory provider 'cortex' is disabled via plugins.disabled"**
- Fix: `hermes plugins enable cortex`. Hermes marks newly found plugins as disabled; the installer does this for you.

**Memories are captured but never recalled**
- Check: `PYTHONPATH="$HERMES_HOME/plugins" python3 -m cortex review list`. If proposals are pending, this is by design:
  Hermes captures are staged for review.
- Fix: `... -m cortex review approve <id>` (or `approve-all`), use the dashboard's review queue, or opt into the auto-judge (AGENTS.md, "Optional paid component").

**Semantic recall stopped after `hermes update` or a Python upgrade**
- Check: `PYTHONPATH="$HERMES_HOME/plugins" <hermes-python> scripts/verify_hermes_install.py` → `semantic_fusion_active`.
  The runtime lives in `$HERMES_HOME/cortex/pydeps/py3.X`; a new Python minor version needs a new directory.
- Fix: re-run `bash scripts/install.sh --agent hermes`.

**A systemd timer (sleep, auto-judge, vault, dashboard) fails with "No such file" for python**
- Cause: PM-managed Hermes rebuilds its environment on update/repair, which changes the Python path embedded in the unit.
- Fix: re-run that timer's installer script (they now find Hermes's current Python automatically).

**`hermes plugins enable cortex` fails with a dependency conflict about onnxruntime**
- Cause: an older checkout declared `pip_dependencies` in `plugin.yaml`; Hermes resolves those for every platform, and onnxruntime has no wheels for some of them.
- Fix: `git pull && bash scripts/install.sh` (the current plugin declares none and uses the sidecar runtime).

## Optional remote components

**The auto-judge or Jev runs when I did not ask for it**
- Check: `$HERMES_HOME/cortex/auto-judge.env` for `CORTEX_AUTO_JUDGE_ENABLED=1` / `CORTEX_JEV_ENABLED=1`; `systemctl --user status cortex-auto-judge.timer`.
- Fix: `bash scripts/install.sh --no-auto-judge --no-jev`. Neither is ever turned on by detecting an API key.

## Data

**`database is locked`**
- Cause: two writers at the same moment (for example, a backup while Hermes is writing). Cortex retries; persistent locks mean a stuck process.
- Fix: stop the agent, check with `fuser $CORTEX_HOME/cortex.db` (or `$HERMES_HOME/cortex/cortex.db`), then retry.

**`audit_ok: false` in `doctor` / `python -m cortex audit`**
- Fix: take a backup first (below), then `python -m cortex maintenance` (preview) and `--apply`. If that does not clear it, restore the last good backup.

### Restore from a backup

Backups are plain SQLite files. Claude Code backups are in `$CORTEX_HOME/backups/`
(`python -m cortex.mcp_server backup`). Hermes backups are whatever copies you made, plus the installer's `config.yaml.bak-cortex-*` files for the config.

1. Stop the agent (quit Claude Code sessions; stop Hermes / `hermes gateway stop`).
2. Keep the current file aside: `mv cortex.db cortex.db.broken-$(date +%Y%m%d%H%M%S)`. Also move any `cortex.db-wal` and `cortex.db-shm` files next to it.
3. Copy the backup in place: `cp backups/cortex-YYYYMMDDTHHMMSS.db cortex.db && chmod 600 cortex.db`.
4. Check it: `~/.cortex/venv/bin/python -m cortex.mcp_server doctor` (`audit_ok: true`), or for Hermes
   `PYTHONPATH="$HERMES_HOME/plugins" python3 -m cortex --db "$HERMES_HOME/cortex/cortex.db" audit`.
5. Start the agent again. To restore a Hermes config, copy the chosen `config.yaml.bak-cortex-*` over `config.yaml`.

## Still stuck

Run the verifier with JSON output and include it (it contains no memory text) in an issue:
`python scripts/mcp_smoke_test.py` (Claude Code) or `scripts/verify_hermes_install.py --json` (Hermes).
