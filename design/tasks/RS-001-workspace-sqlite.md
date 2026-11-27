---
task_id: RS-001
assignee: cursor-cloud
status: unassigned
layer: service
wave: 1
superseded_by: null
owner_files:
  - rust/Cargo.toml
  - rust/Cargo.lock
  - rust/crates/cortex-store/
  - rust/crates/cortex-cli/Cargo.toml
  - rust/crates/cortex-cli/src/main.rs
  - rust/crates/cortex-http/Cargo.toml
  - rust/crates/cortex-http/src/main.rs
  - rust/crates/cortex-view/Cargo.toml
  - rust/crates/cortex-view/src/main.rs
depends_on: []
verify:
  - cargo test --manifest-path rust/Cargo.toml -p cortex-store
  - cargo check --manifest-path rust/Cargo.toml
  - python3 scripts/check_repository.py
---

# Rust workspace and read-only SQLite layer

> **Read first.** `design/RUST-SERVICE.md` sections **Workspace** and **SQLite coexistence** are the spec. This brief says what "done" looks like for the crate. Do not open a Python `CortexStore`.

## Why

Every later Rust task needs one library that can read `cortex.db` without migrating it, locking it, or creating it. The workspace has to exist before those tasks can compile, and they must not take turns editing `rust/Cargo.toml`.

## What to build

A Cargo workspace exactly as `design/RUST-SERVICE.md` describes, plus the `cortex-store` library:

- Open an existing file `mode=ro`, `PRAGMA query_only=ON`, `PRAGMA busy_timeout=5000`. Refuse to create a missing file. Refuse to fall back to a read-write open.
- Read `meta.schema_version` as text. Expose it. No function in this crate writes.
- Typed reads for the queries in that doc: memory counts, recalled-7d, graded answers, procedures, latest sleep run, child-row counts, usage buckets, recall-run sparkline buckets, lexical recall, score breakdown, journal mode.
- A missing required table is an error. A missing optional table is a distinct "not present" result, not a zero and not a panic.
- Bundled SQLite, FTS5 available. Prove it by creating `memory_fts` inside a temporary database in the test.

Stubs: `cortex-cli`, `cortex-http`, and `cortex-view` each have a `main.rs` that writes `not implemented` to stderr and exits 2. Their `Cargo.toml` files compile. Do not put real command logic in the stubs. Do not add the `crates/cortex` member; RS-005 does that.

No `__init__.py` anywhere under `rust/`. `rust/target/` stays untracked.

## Acceptance

- A test builds a temporary database with the required tables and a `meta` row `schema_version='34'`, runs every read, and asserts the row is still `'34'` and that no table was created.
- A test points the opener at a missing path and asserts it returns the unreadable error without creating a file.
- A test holds a write transaction on a second connection for longer than a single statement and asserts the reader either waits within the busy timeout or returns the unavailable error, and that the writer's commit still succeeds.
- `cargo check --manifest-path rust/Cargo.toml` builds the stubs.

## Out of scope

HTTP, JSON argv, terminal rendering, systemd, any edit under `*.py`.
