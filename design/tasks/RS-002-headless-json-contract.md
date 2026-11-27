---
task_id: RS-002
assignee: cursor-cloud
status: unassigned
layer: headless
wave: 2
superseded_by: null
owner_files:
  - rust/Cargo.lock
  - rust/crates/cortex-cli/
depends_on:
  - RS-001
verify:
  - cargo test --manifest-path rust/Cargo.toml -p cortex-cli
  - python3 scripts/check_repository.py
---

# Headless `--json` contract

> **Read first.** `design/RUST-SERVICE.md` sections **Headless JSON contract** and **The one write**. This task supersedes UI-001. Do not edit `cli.py`.

## Why

An agent with no screen needs five answers — recall, status, remember, last dream, health — as one JSON object and a stable exit code. UI-001 asked for that on the Python CLI. The Python CLI opens `CortexStore`, which migrates on open. The contract moves to `cortex-cli` so the read stays read-only.

## What to build

The `cortex-cli` binary speaks the argv and the payloads in `design/RUST-SERVICE.md`:

| command | exit | payload |
|---|---|---|
| `recall QUERY --json` | 0 with hits, 1 with none, 3 if `memory_fts` is missing or the file will not open | `schema_version`, `query`, `ranker`, `memories`, `count`, `note` |
| `status --json` | 0 when the file opened, 3 otherwise | counts, `procedures`, `health.store_schema_version` |
| `sleep-status --json` | 0 even when no dream has run, 3 when the file will not open | `last_run`, `next_scheduled: null` |
| `remember TEXT --json` | Python's result mapped as specified, 3 when `cortex-memory` cannot be run | `proposal: false` |

`--json` is mandatory on these four. Missing flag, missing query, or an unknown subcommand exits 2. Stdout is a single JSON object. `why` keys are the plain words from `plain_summary.COMPONENT_WORDS`. A zero is only legal for an empty `memories` table (with the empty-store note) and for `recall` `count: 0` on exit 1.

`remember` is a subprocess of `cortex-memory`. This crate does not insert into `memories`.

Replace the RS-001 stub. You may edit this crate's `Cargo.toml`. You may not edit `rust/Cargo.toml`.

## Acceptance

- Subprocess tests (Rust `std::process`) for each command: required keys, exit codes, no ANSI bytes in stdout.
- A fixture memory whose content contains a non-ASCII character survives inside the JSON string. The bytes outside string values are ASCII.
- After `status --json` against a fixture whose `meta.schema_version` is `34`, the value is still `34`.
- `recall` of a query that produces no FTS tokens exits 2.
- `sleep-status` with an empty `sleep_runs` returns `last_run: null` and a note, not zeros.
- A running sleep row returns `applied_changes: null`.

## Out of scope

Serving HTTP, drawing the dot grid, changing recall ranking, editing `cli.py`.
