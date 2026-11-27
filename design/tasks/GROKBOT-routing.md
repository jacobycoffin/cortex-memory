---
task_id: META-000
assignee: grokbot
status: reference
layer: routing
wave: null
superseded_by: null
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
| `design/tasks/*.md` | one brief each, YAML front-matter | hand the whole file to the assigned agent |
| `design/README.md` | constraints + definition of done | include as context with every brief |
| `design/RUST-SERVICE.md` | normative Rust service plan | include with every `RS-*` brief; it wins over a brief |

A task is **available** when its front-matter says `status: unassigned`, every `depends_on` id is `done`, and the id is in
`dispatch_order`. `superseded` and `reference` are never available. `depends_on` is the rule; `wave` is only the earliest
wave that can happen. Eligible tasks in one wave go out together. UI-001, UI-002, and UI-004 are superseded by RS-002 and
RS-004.

An `## Amendment` at the bottom of a brief wins over the paragraphs above it.

## Handing one out

1. Walk `dispatch_order`. A task is eligible when it is `unassigned` and its dependencies are `done`. Hand out every
   eligible task. Do not hold a ready wave-2 task for an unrelated blocked sibling.
2. Give the agent: the task file verbatim, `design/README.md`, `design/RUST-SERVICE.md` for an `RS-*` id, the repo URL,
   and branch `item/<task_id>-<slug>`.
3. Claim it: set `status: in-progress` in the task file **and** `index.json`, commit `design: claim <task_id>`.
4. On PR open: `status: review` plus the PR number in `index.json`.
5. On merge, after its `verify` commands pass on `main`: `status: done`.

## Never

- Never edit a brief's body to match what was built; append an `## Amendment` section instead. An `owner_files` entry
  that ends with `/` allows any file under that directory.
- Never allow edits outside a task's `owner_files`.
- Never mark `done` without pasting the verify output.
- Never dispatch a task whose dependencies are open, whose status is `superseded`, or which is absent from
  `dispatch_order`.
