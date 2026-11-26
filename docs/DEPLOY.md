# Deploying the Cortex plugin

The running plugin (a directory under the Hermes home, `plugins/cortex`) and this repository are two different
trees, and they drift. This is how to move one onto the other without losing anything that only exists on the
box.

## The rule that makes it safe

**A file whose deployed content exists nowhere in git is a candidate local patch. A file whose content *is* in git is
merely old and may be replaced.**

That is only the first half, and the second half is the one that bites. "Not in git" does not mean "newer": on this box
11 files were absent from history, and 9 of them were **older** than `main` — the box's `cortex_records.py` raised
`ValueError: invalid state: suppressed` against modules that already used that state. Keeping every uncommitted file
produced a mixed tree that failed **79 tests**.

So the composition is not decided by a rule. It is **measured**, one file at a time:

| case | action |
|---|---|
| module only the repository has | add it |
| module only the box has | keep it (it cannot be regenerated) |
| file on both, deployed content **is** in git history | take the repository's version |
| file on both, deployed content **is not** in git | **run the suite both ways** and keep whichever passes; then diff the two versions to see what the box's copy actually adds |

For a file that survives that test, the diff tells you whether it is a feature (`dashboard.py`: 588 lines of JEV panel
the repo lacks) or a stale copy (7 lines). Replace the stale ones, port the features.

## Procedure

1. **Back up first, and not with `cp`.**
   ```bash
   python3 scripts/deploy_plugin.py backup --plugin <plugin-dir> --out <backup-dir> --db <db-file>
   ```
   That writes the plugin tree, a sha256 manifest of it, and a database copy made with the SQLite backup API.
   The database is WAL-mode. A plain copy of `cortex.db` contains **none** of the recent data — the writes are in the
   `-wal` file. Use `sqlite3.Connection.backup` (what the script does) or `sqlite3 cortex.db ".backup out.db"`.

2. **Audit the divergence.**
   ```bash
   python3 scripts/deploy_plugin.py audit --plugin <plugin-dir> --repo <repo-dir>
   ```
   For each deployed file it reports whether the content is reachable from any git ref. Anything that is not is a local
   patch and must survive the deploy.

3. **Stage beside the live tree**, applying the four rules above:
   ```bash
   python3 scripts/deploy_plugin.py stage --plugin <plugin-dir> --repo <repo-dir> --stage <staging-dir>
   ```

4. **Verify the staged tree, twice.**
   - *Import* every module as a package. A module that fails to import is a module that fails at first recall.
   - *Run* the new modules against a **copy** of the database (`--db`). Importing proves a module loads; only calling it
     proves it works against the older core files the box kept.

   ```bash
   python3 scripts/deploy_plugin.py verify --stage <staging-dir> --db <db-file>
   ```

5. **Swap by rename, and verify again after the swap.**
   ```bash
   python3 scripts/deploy_plugin.py swap --plugin <plugin-dir> --stage <staging-dir> --db <db-file>
   ```
   The tool re-runs the verification at the live path and renames the previous tree back if it fails. A deploy that
   cannot roll itself back is a deploy you perform once.

6. **Restart last, and separately.** The running process keeps the modules it loaded. Deploying from inside a Hermes
   session cannot restart that session, so the swap and the restart are two steps:
   ```bash
   systemctl restart cortex-dashboard.service    # brain website
   systemctl restart hermes-dashboard.service    # dashboard
   ```

## Writing the rollback note

Every deploy leaves a note beside the plugin itself — outside the repository, because it names the machine's paths and
this repository may not. It must contain the exact undo commands, the backup paths, what was *not* verified, and a
symptom table, because the person reading it is usually looking at a broken site and has no context.

## What this procedure does not solve

Verifying a composition is not the same as merging it. A box file that survives the suite still has to be diffed against
`main` to see whether it *adds* anything; if it does, that work has to be ported rather than carried, or it will drift
again at the next deploy. On this box that meant carrying `dashboard.py` and `jev.py`, and accepting that the fleet tool
(`_fleet_tool` in the old provider) stopped being wired until it is ported.
