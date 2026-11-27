---
task_id: UI-001
assignee: cursor-cloud
status: superseded
layer: headless
wave: null
superseded_by: RS-002
owner_files:
  - cli.py
  - tests/test_cli_agent_contract.py
depends_on: []
verify:
  - python3 -m pytest tests/test_cli_agent_contract.py -q
  - python3 scripts/check_repository.py
---

# Give a headless agent a stable contract

> **Do not implement this brief.** It is superseded by [RS-002](RS-002-headless-json-contract.md). The original text stays so the intent is not lost. The Amendment at the bottom is the routing decision.

## Why

An agent with no screen needs five things from Cortex and nothing else: what do I remember, does anything match this,
save this, tell me what you did last night, and is it healthy. Today those exist as commands meant for a person — prose,
colour, and shapes that change. An agent cannot depend on prose.

## What to build

A `--json` mode for the commands an agent actually uses. Non-interactive, one object per invocation, stable field names,
`schema_version` in every payload.

| command | must return |
|---|---|
| `cortex-memory --db X recall "<query>" --json` | `{schema_version, query, memories:[{id, kind, content, score, why}], count}` |
| `cortex-memory --db X status --json` | `{schema_version, memories:{active,cold,archived}, recalled_7d, graded, procedures:{total,graded}, health:{...}}` |
| `cortex-memory --db X remember "<text>" --json` | `{schema_version, memory_id, kind, proposal:bool}` |
| `cortex-memory --db X sleep-status --json` | `{schema_version, last_run:{id,mode,at,applied_changes}, next_scheduled}` |

Requirements:

- **Exit codes are part of the contract**: `0` success, `1` no match (not an error), `2` bad usage, `3` store unreadable.
  Document them in `--help` and in the test.
- **No colour, no box drawing, no locale-dependent numbers** when `--json` is passed. Pure `json.dumps`.
- **`why` must be plain words**, not component keys: `"why": {"wording": 0.62, "used_before": 0.31, "found_by_meaning": 0.18}`.
- Missing data is `null` plus a `"note"` string explaining it — never `0`.
- Every payload must be stable enough that `jq -r .memories[0].content` keeps working across releases.

## Acceptance

- A test drives each command through `subprocess`, parses the JSON, and asserts the schema keys and exit codes.
- A second test asserts `--json` output contains no ANSI escape and no non-ASCII glyph.
- `cortex-memory --db X recall "x" --json | jq -e .schema_version` works in the test.

## Out of scope

Do not change recall behaviour, scoring, or what the store records. This task adds a contract, not a capability.

## Amendment

Superseded by RS-002 on 2026-11-27. Do not dispatch this task and do not edit `cli.py`.

The contract above is still the intent, with these corrections from the real code (`design/RUST-SERVICE.md` is the spec):

- Python `cortex-memory` has `search`, `status`, and `remember`. It has no `recall` subcommand and no `sleep-status` subcommand. `status` today is prose and exits 0 when the database file is missing. Adding `--json` there would go through `CortexStore`, which rewrites `meta.schema_version` on open.
- The JSON field `schema_version` is contract version `1`. The database version is the TEXT column `meta.value` for key `schema_version`, reported as `health.store_schema_version`. Rust must not write it.
- `why` keys are the plain words in `plain_summary.COMPONENT_WORDS` (`wording match`, `found by meaning`, `past usefulness`), not the snake_case keys in the example table above.
- Memory text may contain non-ASCII. The "no non-ASCII" rule applies to CLI chrome, not to `content` strings.
- `remember` stays a Python write, invoked as a subprocess. `proposal` is `false` for that CLI path.
- Phase-1 recall is the lexical FTS read model (`ranker: "lexical"`). Semantic fusion stays in the Python retriever.
