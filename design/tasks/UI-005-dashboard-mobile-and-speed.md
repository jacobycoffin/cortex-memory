---
task_id: UI-005
assignee: cursor-cloud
status: unassigned
layer: deep-dive
wave: 1
superseded_by: null
owner_files:
  - dashboard.html
  - dashboard.css
  - dashboard.js
  - dashboard.py
  - pyproject.toml
  - vendor/
  - tests/test_dashboard_mobile.py
depends_on: []
verify:
  - python3 -m pytest tests/test_dashboard_mobile.py -q
  - python3 scripts/check_repository.py
---

# Make the deep-dive dashboard usable on a phone, and faster to open

> **Amendment in force.** Where this brief disagrees with the Amendment at the bottom, follow the Amendment. `dashboard.py` is in `owner_files` only for the static routes the split needs.

## Why

`dashboard.html` is 622 KB in a single file with one `<style>` and one `<script>`. It is good on a laptop and painful on a
phone: everything is stacked desktop-first, and the whole file must arrive before anything renders.

## What to build — in this order, stopping when the budget is met

1. **Split without a build step**: extract CSS to `dashboard.css` and JS to `dashboard.js`, referenced normally, and have
   `dashboard.py` serve both. Target: initial HTML under 60 KB, CSS under 80 KB, JS under 200 KB, all still cached by the
   browser on reload.
2. **Mobile pass at 390 px**: the sidebar becomes a bottom tab bar, panels stack, tables become card rows, no horizontal
   scroll anywhere. Test it at 390×844.
3. **First paint**: state the number before it; render skeleton rows, then hydrate from `/api/snapshot`. No full-page
   blocking fetch.
4. **Nothing regresses on desktop**: the same views, the same data, the same auth gate.

## Acceptance

- Byte-size assertions in the test: HTML < 60 KB, CSS < 80 KB, JS < 200 KB (fail the test if exceeded).
- A test asserts the page contains no inline `<style>` or `<script>` blocks larger than a few lines.
- A test asserts `/dashboard.css` and `/dashboard.js` are served with correct content types and cached headers.
- Screenshots: 390×844 (each view), 1440×900 (overview), and one with JS disabled showing a readable fallback.

## Out of scope

No framework, no bundler, no npm. Do not change the API surface. Do not remove any existing view.

## Amendment

Active, wave 1, no dependencies. Dispatch this task as written here. It runs in parallel with RS-001 and UI-003. It does not wait for the Rust server.

The original owner list omitted `dashboard.py` while the body required it to serve the split files. `owner_files` now includes `dashboard.py`, `dashboard.css`, `dashboard.js`, `pyproject.toml`, and `vendor/`. Touch `dashboard.py` only to serve those static files. Do not add `/api/pulse` and do not change an existing API route. Touch `pyproject.toml` only to add the new files to `[tool.setuptools.package-data]` `cortex`, or a pip install will not ship them.

`dashboard.html` today loads these scripts from `cdn.jsdelivr.net`: graphology 0.26.0, sigma 3.0.3, force-graph 1.51.4, cytoscape 3.34.0, 3d-force-graph 1.80.0. Commit those exact versions under `vendor/` (keep their license headers) and point the page at `/vendor/<file>`. `dashboard.py` serves `/vendor/*` as JavaScript with the same cache headers as `/dashboard.js`. No build step. The views stay. RS-003 will serve the same directory later; until that cutover, the Python server must keep working, which is why this task edits `dashboard.py`.

Byte budgets, the 390 px pass, skeleton first paint, and the desktop regression stay as written above. Cached headers apply to the CSS, the JS, and `vendor/`, not to HTML or `/api/*` (those stay `no-store`, matching the server today).
