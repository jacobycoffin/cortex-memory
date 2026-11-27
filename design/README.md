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
  `.`, `:`, `+`, `*`, `#`. The canonical code points, sparkline, and thresholds are `design/dot-glyphs.json`.
- **Colour is optional and never the only signal** — it must be legible in one colour.
- **One glance = one line + one grid.** A status line, then a grid of dots. Nothing else on the glanceable view.
- **Fast beats rich.** Target: first paint under 100 ms locally, updates without layout shift, no animation that costs
  frames, no spinner that costs a second.
- **Silence when nothing is happening.** An idle agent shows a calm grid, not a busy one.

## Hard constraints (they break the deploy if ignored)

1. **No JavaScript build step.** `dashboard.py` is a stdlib `HTTPServer` that reads `dashboard.html` from disk. Vanilla
   HTML/CSS/JS, or assets committed already-compiled. A bundler that must run before serving is a broken deploy. The Rust
   service is a separate Cargo binary (`design/RUST-SERVICE.md`); compiling it is not a dashboard deploy step, and it does
   not add npm.
2. **No CDN, no third-party host.** Works offline from a fresh clone.
3. **Terminal output must degrade**: colour only when `isatty()`, honours `NO_COLOR`, and pipes cleanly (`| jq`, `| grep`).
4. **Read-only by default.** No task may add a write path to the dashboard unless its `owner_files` include the server and
   the brief explicitly says so.

## Assignment convention

Every task file opens with YAML front-matter:

```yaml
task_id: UI-002
assignee: cursor-cloud
status: unassigned          # unassigned | in-progress | review | done | blocked | superseded | reference
layer: glanceable           # glanceable | headless | deep-dive | service | routing
wave: 1                     # earliest wave; null on superseded and reference tasks
superseded_by: null         # task id that replaced this brief, or null
owner_files: [...]          # the ONLY files it may modify; a trailing slash means that directory
depends_on: [UI-001]
verify: [...]               # commands that must pass before review
```

`design/index.json` mirrors all of it and is what a dispatcher reads. An `## Amendment` at the bottom of a brief wins
over the paragraphs above it. `design/RUST-SERVICE.md` wins over an `RS-*` brief where they disagree.

## Rust service track

Phase 1 is a read-only Rust binary beside the Python writer. It does not bump `meta.schema_version` and it does not
replace the Hermes plugin or the nightly dream. The task graph, the SQLite contract, auth, and the unit file are
`design/RUST-SERVICE.md`. Wave 1 (no shared files, nothing else has to be `done` first): **RS-001**, **UI-003**,
**UI-005**. UI-001, UI-002, and UI-004 are superseded and are not dispatched.

**Rules for an assigned agent**

1. Touch only `owner_files`. Anything outside is rejected on review.
2. Run the `verify` commands and paste their real output into the PR. A claim without output is not a result.
3. One task per PR, titled `<task_id>: <short description>`.
4. Three screenshots or transcripts per visual task: idle, active, and one degraded state (empty store or server down).
5. Never invent numbers. If data is missing, the view says "no data yet" — it does not show a zero that looks healthy.

## Definition of done

- Renders correctly at 80×24 terminal and at 390 px wide, with no horizontal scroll.
- Correct with an empty store **and** a full one.
- The task's own `verify` commands pass, and their output is pasted into the PR. Python tasks that touch the dashboard
  also pass `python3 -m pytest tests/test_dashboard.py -q`. Rust tasks pass the `cargo` commands in their brief.
  `python3 scripts/check_repository.py` passes for every task.
- No new Python runtime dependency. The Rust binary's crates are build-time dependencies of `rust/`, not of the Python
  package.
