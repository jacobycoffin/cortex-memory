---
task_id: UI-005
assignee: cursor-cloud
status: unassigned
layer: deep-dive
owner_files: [dashboard.html, tests/test_dashboard_mobile.py]
depends_on: []
verify:
  - python3 -m pytest tests/test_dashboard_mobile.py -q
  - python3 scripts/check_repository.py
---

# Make the deep-dive dashboard usable on a phone, and faster to open

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
