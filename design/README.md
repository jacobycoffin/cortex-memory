# Design briefs — read this before writing any task

Instructions for **external design agents** (Cursor cloud). Two audiences must be served, and they are not the same thing:

| audience | what it needs | where it lives |
|---|---|---|
| **headless agent** | a stable, machine-readable contract — no colours, no prose, no browser | CLI `--json` + a small HTTP API |
| **agent with a machine and a screen** | something glanceable in under a second | a **dot / ASCII live view** |

The existing 622 KB `dashboard.html` stays as the **deep-dive surface**. It is not the primary interface and must not grow.

## The visual language: dots, not dashboards

Jacoby's reference is *hex dictation visuals* — the display is dots, characters and one number, and it reads instantly
without being looked at properly. Rules for anything at the "glanceable" layer:

- **Monospace, character-grid only.** No images, no icon fonts, no web fonts. One cell is one unit of meaning.
- **Five levels, one glyph each:** `·` empty, `░` low, `▒` medium, `▓` high, `█` saturated. Plain-ASCII fallback:
  `.`, `:`, `+`, `*`, `#`.
- **Colour is optional and never the only signal** — it must be legible in one colour.
- **One glance = one line + one grid.** A status line, then a grid of dots. Nothing else on the glanceable view.
- **Fast beats rich.** Target: first paint under 100 ms locally, updates without layout shift, no animation that costs
  frames, no spinner that costs a second.
- **Silence when nothing is happening.** An idle agent shows a calm grid, not a busy one.

## Hard constraints (they break the deploy if ignored)

1. **No build step.** `dashboard.py` is a stdlib `HTTPServer` that reads `dashboard.html` from disk. Vanilla HTML/CSS/JS, or
   assets committed already-compiled. A bundler that must run before serving is a broken deploy.
2. **No CDN, no third-party host.** Works offline from a fresh clone.
3. **Terminal output must degrade**: colour only when `isatty()`, honours `NO_COLOR`, and pipes cleanly (`| jq`, `| grep`).
4. **Read-only by default.** No task may add a write path to the dashboard unless its `owner_files` include the server and
   the brief explicitly says so.

## Assignment convention

Every task file opens with YAML front-matter:

```yaml
task_id: UI-002
assignee: cursor-cloud
status: unassigned          # unassigned | in-progress | review | done | blocked
layer: glanceable           # glanceable | headless | deep-dive
owner_files: [...]          # the ONLY files it may modify
depends_on: [UI-001]
verify: [...]               # commands that must pass before review
```

`design/index.json` mirrors all of it and is what a dispatcher reads.

**Rules for an assigned agent**

1. Touch only `owner_files`. Anything outside is rejected on review.
2. Run the `verify` commands and paste their real output into the PR. A claim without output is not a result.
3. One task per PR, titled `<task_id>: <short description>`.
4. Three screenshots or transcripts per visual task: idle, active, and one degraded state (empty store or server down).
5. Never invent numbers. If data is missing, the view says "no data yet" — it does not show a zero that looks healthy.

## Definition of done

- Renders correctly at 80×24 terminal and at 390 px wide, with no horizontal scroll.
- Correct with an empty store **and** a full one.
- `python3 -m pytest tests/test_dashboard.py -q` and `python3 scripts/check_repository.py` both pass.
- No new runtime dependency.
