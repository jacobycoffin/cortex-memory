---
task_id: UI-004
assignee: cursor-cloud
status: unassigned
layer: glanceable
owner_files: [pulse_dream.py, tests/test_pulse_dream.py]
depends_on: [UI-002]
verify:
  - python3 -m pytest tests/test_pulse_dream.py -q
  - python3 scripts/check_repository.py
---

# Watch a dream happen

## Why

Cortex now runs a nightly reconsolidation cycle (04:15, `sleep --mode apply`). Its first run demoted 212 memories
active → cold and proposed 96 new associations. Today you can only learn that from a log. It should be visible as it
happens, in the same dot language.

## What to build

`cortex-memory dream --watch` — a terminal view for one dream cycle:

```
dream  run 566befae  apply  started 04:15:02  ████████░░ 8/10
episodes   ████████████████████ 400/400
demoted    ░░░░░░░░░░▒▒▒▒▒▒▒▒░░ 212  active → cold
linked     ······▒▒▓▓▓▓▓▓▓▓···· 96 new associations, 86 reinforced
edges      ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░ 250 decay proposals
```

- Progress bars are character cells, same five levels as the dot grid.
- Reads `sleep_runs` + `sleep_state_changes` + `sleep_proposals` for the current run id.
- `--follow` tails an in-progress run; without it, prints the last run once and exits.
- Ends with a one-line verdict: `213 changes applied · undo: cortex-memory sleep-undo <run_id>`.

## Acceptance

- Pure render function tested against fixture rows: in-progress, completed, failed, and no-runs-at-all.
- No-runs case prints `no dream has run yet` rather than a zeroed dashboard.
- Test that the undo line names the correct run id.
- Snapshots at 100 and 80 columns.

## Out of scope

Do not change the dream itself, its schedule, or what it applies.
