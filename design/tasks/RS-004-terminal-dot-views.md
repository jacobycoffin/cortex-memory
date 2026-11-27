---
task_id: RS-004
assignee: cursor-cloud
status: unassigned
layer: glanceable
wave: 2
superseded_by: null
owner_files:
  - rust/Cargo.lock
  - rust/crates/cortex-view/
depends_on:
  - RS-001
verify:
  - cargo test --manifest-path rust/Cargo.toml -p cortex-view
  - python3 scripts/check_repository.py
---

# Terminal dot views: `pulse` and `dream`

> **Read first.** `design/RUST-SERVICE.md` section **Terminal views** and `design/dot-glyphs.json`. This task supersedes UI-002 and UI-004. Do not edit `sleep.py` or `dashboard.html`.

## Why

The primary screen is one status line and one character grid, readable the way a hex-dictation display is readable. The same language covers an ordinary minute (`pulse`) and one dream cycle (`dream`). Both are read-only views of SQLite.

## What to build

The `cortex-view` library exposes pure functions `render_pulse(frame) -> String` and `render_dream(frame) -> String`. The bin subcommands `pulse` and `dream` call them. Argv, the status line, the dream verdict table, and the running-versus-finished denominators are in the spec.

- Glyphs come from `design/dot-glyphs.json`. `--ascii` selects the ASCII column. Tests load that file and fail if a rendered glyph is not in it.
- Default grid 40×6. `--cols` / `--rows` override.
- In-place redraw: cursor home, no blanking newline after the first paint. `--interval` defaults to 1 second. `Ctrl-C` exits 0 and leaves the terminal usable.
- No colour when stdout is not a TTY or when `NO_COLOR` is set to a non-empty value.
- A store error inside the loop prints `store unavailable: <reason>` and keeps the loop alive. `--once` exits 3 in that case.
- While a dream is `running`, counts come from child tables. Do not display `sleep_runs.episodes_replayed` (it is still the insert default). A shadow run's verdict does not mention `sleep-undo`.
- No dream row prints `no dream has run yet`.

Replace the RS-001 stub. You may edit this crate's `Cargo.toml`. You may not edit `rust/Cargo.toml`.

## Acceptance

- Exact-string tests for an empty grid, an all-zero grid, a saturated grid, and one spike, both glyph sets.
- A running-run fixture with `episodes_replayed = 0` on the row and 3 rows in `sleep_episode_state` renders `3`, not `0`.
- A finished shadow run's verdict contains `shadow only` and does not contain `sleep-undo`.
- An apply run with `applied_changes = 4` names that run id on the undo line.
- No ANSI bytes when stdout is not a terminal. `--ascii` output is ASCII-only.
- The redraw test feeds two frames and asserts the second write contains no extra trailing newline beyond the frame.

## Out of scope

The browser tile (UI-003), changing what a dream writes, HTTP.
