# Cortex Memory Hermes adapter quickstart

For a framework-neutral install and the five-event harness contract, start with the main README and [integration guide](INTEGRATION.md). This guide covers the included Hermes adapter, vault migration, Linux services, and rollback.

## 1. Back up existing Hermes memory

The installer backs up an existing Cortex plugin directory. Back up the Hermes files and Cortex database explicitly before a VPS migration:

```bash
mkdir -p "$HOME/.hermes/backups/manual"
cp -a "$HOME/.hermes/memories" "$HOME/.hermes/backups/manual/" 2>/dev/null || true
cp -a "$HOME/.hermes/cortex" "$HOME/.hermes/backups/manual/" 2>/dev/null || true
```

Cortex does not require discarding `MEMORY.md`, `USER.md`, or an Obsidian vault.

## 2. Install and activate

```bash
git clone https://github.com/jacobycoffin/cortex-memory.git
cd cortex-memory
HERMES_HOME="$HOME/.hermes" ./scripts/install_local.sh
hermes memory setup
```

Choose `cortex` and restart the Hermes agent process.

## 3. Import an Obsidian vault

Preview first:

```bash
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex vault-index "$HOME/.hermes/obsidian-vault"
```

Apply the reviewed plan:

```bash
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex vault-index "$HOME/.hermes/obsidian-vault" --apply
```

The indexer is read-only toward the vault. It chunks notes by heading, records source paths and hashes, connects wiki links, supersedes changed sections, and archives removed sections instead of deleting them.

On an always-on Linux host:

```bash
HERMES_HOME="$HOME/.hermes" VAULT_PATH="$HOME/.hermes/obsidian-vault" \
  ./scripts/install_vault_timer.sh
```

## 4. Enable offline Cortex Sleep

Preview one bounded deterministic cycle:

```bash
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex sleep --mode shadow --reflection-token-budget 0
```

On an always-on Linux host, install the low-priority nightly timer:

```bash
HERMES_HOME="$HOME/.hermes" ./scripts/install_sleep_timer.sh
systemctl --user list-timers cortex-sleep.timer
```

The installed job runs in shadow mode and uses no provider tokens. It records replay evidence and proposals without changing memories or graph weights. Optional provider settings live in `$HERMES_HOME/cortex/sleep.env`, which the installer creates with mode `0600` and a zero budget. See [Cortex Sleep](SLEEP.md) before enabling remote reflection or explicit apply mode.

## 5. Run the dashboard

```bash
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex dashboard --no-open --port 8765
```

For systemd, install the authenticated localhost service:

```bash
HERMES_HOME="$HOME/.hermes" PORT=8100 ./scripts/install_dashboard_service.sh
systemctl --user start cortex-dashboard
```

The installer prints a one-time password. At first sign-in the dashboard requires a new password of at least 12 characters. Use the **Account** control in the top bar to change it again or sign out.

If the password is lost, reset it from the host and restart the service:

```bash
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex dashboard-password --username cortex
systemctl --user restart cortex-dashboard
```

Place Caddy, Nginx, or Cloudflare Tunnel in front of `127.0.0.1:8100`; do not bind the Python server directly to the public internet. Follow the [dashboard self-hosting guide](DASHBOARD_HOSTING.md) to connect a hostname you control without publishing personal deployment details.

## 6. Acceptance test

1. In session A, store a specific durable decision.
2. In session B, ask a paraphrased question about it.
3. Ask Cortex to explain the memory and provenance.
4. Correct it and confirm the old version remains.
5. Run two related tool tasks and inspect the Tool notes and Cognition pages.
6. Run the audit.
7. Run one shadow Sleep cycle and inspect its proposal counts in Cognition.

```bash
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex audit
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex recall-stats
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex sleep --mode shadow --reflection-token-budget 0
```

`audit.ok` should be `true`. Keep pruning and consolidation in `shadow` until you have real pruning-regret and retrieval evidence.

## Upgrade

Pull the repository and run `install_local.sh` again. It creates a timestamped backup of the previous plugin. Opening the database migrates it in place; make a database copy before major upgrades.

## Roll back or uninstall

The safe default removes only plugin code and keeps `$HERMES_HOME/cortex/cortex.db`:

```bash
HERMES_HOME="$HOME/.hermes" ./scripts/uninstall_local.sh
```

Then select another provider with `hermes memory setup` and restart Hermes. To restore a plugin backup, copy the chosen directory from `$HERMES_HOME/backups/` back to `$HERMES_HOME/plugins/cortex`.
