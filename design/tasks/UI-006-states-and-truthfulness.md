---
task_id: UI-006
assignee: cursor-cloud
status: unassigned
layer: glanceable + deep-dive
owner_files: [pulse.html, dashboard.html, tests/test_ui_states.py]
depends_on: [UI-003, UI-005]
verify:
  - python3 -m pytest tests/test_ui_states.py -q
  - python3 scripts/check_repository.py
---

# States that tell the truth

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
