---
task_id: UI-006
assignee: cursor-cloud
status: unassigned
layer: glanceable + deep-dive
wave: 2
superseded_by: null
owner_files:
  - pulse.html
  - dashboard.html
  - dashboard.js
  - tests/test_ui_states.py
depends_on:
  - UI-003
  - UI-005
verify:
  - python3 -m pytest tests/test_ui_states.py -q
  - python3 scripts/check_repository.py
---

# States that tell the truth

> **Amendment in force.** Where this brief disagrees with the Amendment at the bottom, follow the Amendment. Rust surfaces implement the same four states inside RS-002 and RS-004. This task does not edit Rust.

## Why

Every surface today renders a zero when data is missing, and a zero looks like a healthy store. The whole point of this
project's measurement work was that an unfed mechanism must never look like a working one. Apply that to the UI.

## What to build

For **every** number on both surfaces, four distinct states, visually different:

| state | shows | example |
|---|---|---|
| **no data yet** | the mechanism has never run | `dreams: not run yet` |
| **running** | in progress, with progress | `dreams: running…` |
| **stale** | last update older than its cadence | `dreams: 3 d ago ⚠` |
| **fresh** | current | `dreams: 04:15 ✓` |

Also:

- **Empty store**: a first-run screen that explains what will appear here and why it is empty — not three zeros.
- **Server unreachable**: the page says so in one line and keeps the last-known values greyed with their timestamp.
- **Loading**: skeleton rows that match final layout (no spinner, no layout jump).

## Acceptance

- A test enumerates every numeric id/element in `dashboard.html` and `pulse.html` and asserts each has a state-mapping
  function attached — no element may render a bare `0` without a state.
- Tests for all four states on each surface using fixtures.
- Screenshots of: empty store, running dream, stale data (24 h old), server down.

## Out of scope

Do not change what the server computes, only how absence is communicated.

## Amendment

Active, wave 2. Depends on UI-003 and UI-005, both of which have to be `done` before this is dispatched. Rust already implements the same four states in RS-002 and RS-004; do not edit `rust/`.

After UI-005, the deep-dive script lives in `dashboard.js`. `owner_files` therefore includes `dashboard.js` and `dashboard.html`. `dashboard.css` stays with UI-005.

Every number this task touches has the four states in the table above. A null or missing instrument renders `not measurable yet` (or `not run yet` for a dream that has no row). The only bare `0` allowed is a `memories` table that exists and has no rows, and it is accompanied by `the store is empty`. `recalled_7d: null` is not `0`.

The test enumerates numeric elements in `pulse.html`, `dashboard.html`, and `dashboard.js`. Server-down and empty-store screenshots stay in scope. Do not change `/api/snapshot` or `/api/pulse`.
