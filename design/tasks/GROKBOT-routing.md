---
task_id: META-000
assignee: grokbot
status: reference
layer: routing
owner_files: []
depends_on: []
verify: []
---

# GROKBOT — routing instructions

You are reading **task briefs for Cursor cloud design agents**, not docs and not a changelog.

## How to tell what is what

| file | what it is | what you do |
|---|---|---|
| `design/index.json` | machine-readable list of all tasks | read first; source of truth for status |
| `design/tasks/UI-*.md` | one brief each, YAML front-matter | hand the whole file to the assigned agent |
| `design/README.md` | constraints + definition of done | include as context with every brief |

A task is **available** when its front-matter says `status: unassigned` and every `depends_on` id is `done`.

## Handing one out

1. Pick the lowest-numbered `unassigned` task whose dependencies are `done`.
2. Give the agent: the task file verbatim, `design/README.md` as context, the repo URL, and branch `item/<task_id>-<slug>`.
3. Claim it: set `status: in-progress` in the task file **and** `index.json`, commit `design: claim <task_id>`.
4. On PR open: `status: review` plus the PR number in `index.json`.
5. On merge, after its `verify` commands pass on `main`: `status: done`.

## Never

- Never edit a brief's body to match what was built; append an `## Amendment` section instead.
- Never allow edits outside a task's `owner_files`.
- Never mark `done` without pasting the verify output.
- Never dispatch a task whose dependencies are open.
