# GROKBOT — routing instructions for this directory

You are reading a directory of **task briefs for Cursor cloud design agents**, not a changelog and not feature docs.
Everything here is an instruction to be handed out, tracked, and verified.

## How to tell what is what

| file | what it is | what you do with it |
|---|---|---|
| `design/index.json` | machine-readable list of every task | read this first; it is the source of truth for status |
| `design/tasks/UI-*.md` | one brief per task, with front-matter | hand the whole file to the assigned agent |
| `design/README.md` | constraints and definition of done | include it as context with every brief |

A task is **available** when its front-matter says `status: unassigned` and every id in `depends_on` is `done`.

## How to hand one out

1. Read `design/index.json`, pick the lowest-numbered task whose `status` is `unassigned` and whose `depends_on` are all
   `done`.
2. Give the agent: the task file verbatim, plus `design/README.md` as context, plus the repo URL and the branch name
   `item/<task_id>-<slug>`.
3. Record the claim: set that task's `status` to `in-progress` in both the task file front-matter and `index.json`, in a
   commit titled `design: claim <task_id>`.
4. When the agent opens a PR, set `status: review` and note the PR number in `index.json`.
5. When it merges and the `verify` commands pass on `main`, set `status: done`.

## What you must not do

- Do not edit a task's body to match what an agent actually built. If the brief was wrong, add a `## Amendment` section
  at the bottom explaining what changed and why.
- Do not let an agent touch files outside its `owner_files` list.
- Do not mark a task `done` without the verify commands having been run on `main` and their output recorded.
- Do not hand out a task whose dependencies are not `done`.

## If something is blocked

Add the blocker to `index.json` under `"blockers"` with the task id and one sentence. A blocked task is not a failed task.
