---
task_id: UI-003
assignee: cursor-cloud
status: unassigned
layer: glanceable
wave: 1
superseded_by: null
owner_files:
  - pulse.html
  - tests/test_pulse_page.py
depends_on: []
verify:
  - python3 -m pytest tests/test_pulse_page.py -q
  - python3 scripts/check_repository.py
---

# The dot view: a phone tile

> **Amendment in force.** Where this brief disagrees with the Amendment at the bottom, follow the Amendment. Server routes are RS-003, not this task.

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

## Amendment

Active, wave 1, no dependencies. Dispatch this task as written here.

`dashboard.py` is no longer an owner file. RS-003 serves `GET /pulse` and `GET /api/pulse`. This task only writes `pulse.html` and `tests/test_pulse_page.py`.

- The page is still one self-contained file under 20 KB, no `http://` or `https://` URL, monospace system stack, `<pre>` grid, poll every 2 seconds, hidden-tab pause, `clamp()` layout, dark by default.
- Glyphs and the 0..4 mapping are `design/dot-glyphs.json`. The test loads that file and asserts the page's mapping matches it. Do not import a Python or Rust renderer.
- The page calls `/api/pulse` and also works from `file://`. When `fetch` fails, show one line `server unreachable` and keep the last good frame greyed, with its timestamp. When there has never been a frame, show `no data yet`. A successful payload whose buckets are all `0` is a calm idle grid, not an error and not `no data yet`.
- Buckets in the payload are integers 0..4. Render them with the glyph file. The status line uses the four words `not run yet`, `running`, `stale`, `fresh` (and `failed` when `last_dream.state` is `failed`). A null `recalled_7d` renders `not measurable yet`.
- The 401 test moved to RS-003, because this task does not own the server. This task's test checks file size, the absence of remote URLs, the glyph mapping, and that the unreachable branch's text is present in the file.

`/api/pulse` shape is fixed in `design/RUST-SERVICE.md`. Do not invent a second payload.
