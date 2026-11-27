# GROKBOT — routing instructions for this directory

You are reading a directory of **task briefs for Cursor cloud design agents**, not a changelog and not feature docs.
Everything here is an instruction to be handed out, tracked, and verified.

## How to tell what is what

| file | what it is | what you do with it |
|---|---|---|
| `design/index.json` | machine-readable list of every task | read this first; it is the source of truth for status |
| `design/tasks/*.md` | one brief per task, with front-matter | hand the whole file to the assigned agent |
| `design/README.md` | constraints and definition of done | include it as context with every brief |
| `design/RUST-SERVICE.md` | normative Rust service plan | include it with every `RS-*` brief; it wins over a brief |

A task is **available** when its front-matter says `status: unassigned`, every id in `depends_on` is `done`, and its id is
in `dispatch_order`. `superseded` and `reference` are never available. Tasks in the same wave whose dependencies are
`done` may be handed out together; they do not share files.

`depends_on` is the rule. The `wave` field is only the earliest wave in which that can happen. UI-001, UI-002, and UI-004
are superseded by RS-002 and RS-004. Do not hand them out.

An `## Amendment` at the bottom of a brief wins over the original paragraphs. A blockquote under the title that says the
amendment is in force means the same thing.

## How to hand one out

1. Read `design/index.json`. Walk `dispatch_order`. A task is eligible when its `status` is `unassigned` and every
   `depends_on` is `done`. Hand out every eligible task in the current wave together; do not wait for wave 2 while wave
   1 still has eligible work, and do not skip an eligible wave-2 task just because another wave-2 task is still blocked.
2. Give the agent: the task file verbatim, `design/README.md`, `design/RUST-SERVICE.md` when the id starts with `RS-`,
   the repo URL, and the branch name `item/<task_id>-<slug>`.
3. Record the claim: set that task's `status` to `in-progress` in both the task file front-matter and `index.json`, in a
   commit titled `design: claim <task_id>`.
4. When the agent opens a PR, set `status: review` and note the PR number in `index.json`.
5. When it merges and the `verify` commands pass on `main`, set `status: done`.

## What you must not do

- Do not edit a task's body to match what an agent actually built. If the brief was wrong, add a `## Amendment` section
  at the bottom explaining what changed and why. An entry in `owner_files` that ends with `/` allows any file under that
  directory.
- Do not let an agent touch files outside its `owner_files` list.
- Do not mark a task `done` without the verify commands having been run on `main` and their output recorded.
- Do not hand out a task whose dependencies are not `done`, a task whose status is `superseded`, or a task missing from
  `dispatch_order`.

## If something is blocked

Add the blocker to `index.json` under `"blockers"` with the task id and one sentence. A blocked task is not a failed task.
