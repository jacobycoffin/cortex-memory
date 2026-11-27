---
task_id: UI-002
assignee: cursor-cloud
status: unassigned
layer: glanceable
owner_files: [pulse_tui.py, tests/test_pulse_tui.py]
depends_on: [UI-001]
verify:
  - python3 -m pytest tests/test_pulse_tui.py -q
  - python3 scripts/check_repository.py
---

# The dot view: a terminal live pane

## Why

Jacoby wants to *see the brain working* the way hex dictation shows sound — dots, instantly readable, no chrome. This is
the primary screen visual. It runs next to the agent, not in a browser.

## What to build

`cortex-memory pulse` — a live, in-place terminal view. One status line, one dot grid, nothing else.

```
cortex  3843 mem  ▁▂▅▇▅▂▁ 4.2/s   last dream 04:15 ✓ 23 changes
········································
·░░▒▒▓▓▒░··············░░▒▓▒░░···········
········································
```

- **Grid**: 40 columns × 6 rows by default, `--cols/--rows` overrides. Each cell is one recent turn bucket; glyph level is
  how much that turn actually used what was recalled (`·` none → `█` heavily used).
- **Status line**: memory count, a sparkline of recall rate (last 60 buckets), current rate/s, last dream result, health dot.
- **In-place update**: redraw with cursor home, never clear-and-scroll. `--interval` default 1 s.
- **Degrade**: `--ascii` swaps glyphs; no colour when not a TTY or when `NO_COLOR`; works over SSH at 80×24; `Ctrl-C` exits
  cleanly leaving the terminal usable.
- **Silence when idle**: an agent doing nothing shows a flat calm grid, not a busy one.

## Data

Read-only. Prefer a single cheap query per tick (counts + last N usage rows), not a full snapshot. If the store is locked
or unreadable, print one line — `store unavailable: <reason>` — and keep the pane alive.

## Acceptance

- Deterministic render function: given a list of buckets, return the exact string block. Test it with fixture buckets
  (empty, all-zero, all-full, one spike) and assert exact output.
- Test that `--ascii` contains only ASCII bytes and default mode contains no ANSI when not a TTY.
- Test that a store error prints the fallback line and exits 0.
- A test asserts the update loop does not reprint the whole screen (no newlines after the first paint).

## Out of scope

No browser, no images, no animation beyond redrawing. Do not touch `dashboard.html`.
