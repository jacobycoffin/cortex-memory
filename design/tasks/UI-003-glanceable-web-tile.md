---
task_id: UI-003
assignee: cursor-cloud
status: unassigned
layer: glanceable
owner_files: [pulse.html, dashboard.py, tests/test_pulse_page.py]
depends_on: [UI-002]
verify:
  - python3 -m pytest tests/test_pulse_page.py -q
  - python3 scripts/check_repository.py
---

# The dot view: a phone tile

## Why

The same glance, on a phone, without opening the 622 KB dashboard. One screen, no scrolling, readable in a second.

## What to build

A **single self-contained file** `pulse.html`, under **20 KB uncompressed**, served by `dashboard.py` at `/pulse`. It must
also work opened directly from disk (`file://`) by reading `/api/snapshot` and degrading gracefully when that fails.

- Character grid exactly like the terminal view (same glyphs, same meaning). Render as text inside a `<pre>`; do **not**
  build a canvas or SVG dot matrix — characters are the design.
- One status line, one grid, one recent-events list of at most five rows.
- Auto-refresh every 2 s via `fetch`, `AbortController` on navigation, no polling storm when the tab is hidden
  (`document.visibilityState`).
- Sizes with `clamp()` so it fills a phone screen without scroll at 390×844 and stays sane on desktop.
- Dark by default using the existing CSS custom properties; light only if `prefers-color-scheme: light`.
- No dependency, no build, no CDN, no web font. Monospace system stack only.

## Server change (the only allowed one)

`dashboard.py` gains one route: `/pulse` serving `pulse.html` from disk, and `/api/pulse` returning
`{buckets:[0..4 × N], status:{memories, recalled_7d, last_dream}, schema_version}` — read-only, auth-gated exactly like
`/api/snapshot`. Keep it to a handful of aggregate queries; it is polled every 2 s.

## Acceptance

- Test that `/api/pulse` returns the documented keys and is 401 without a session.
- Test that `pulse.html` is under 20 KB and contains no `http://`/`https://` reference.
- Test the bucket→glyph mapping in a headless way (the same function the terminal view uses — share it, do not duplicate).
- Screenshots at 390×844 and 1440×900, plus one with the API failing.

## Out of scope

Do not restyle or extend `dashboard.html`. Do not add a second data source; `/api/pulse` is the only new one.
