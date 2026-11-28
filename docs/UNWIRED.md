# Modules that shipped without a call path

Cortex had a habit of shipping a module, listing it in `scripts/install_local.sh`, and never importing it from
anything but its own tests. Fourteen of those were deleted on 2026-11-27. This note records what went, what was
deliberately kept, and how to re-check — because the failure mode was never "the code is wrong", it was "the code
looks present".

## The rule used

A module is dead when **nothing in the shipped package or the operator scripts imports it** and **no test asserts
anything about it except its own**.

The check that matters — the one that finds all of them — is the `from cortex import <name>` form, not just
`from .<name> import`:

```bash
cd <repo>
git grep -ln 'import <name>\|import .*\<name>\>' HEAD -- '*.py' | grep -v "HEAD:<name>.py" | grep -v HEAD:tests/
```

An empty result means dead. **Run it against every module in `scripts/install_local.sh`** — that list is the claim
about what ships.

## Deleted

| module | why it was dead |
|---|---|
| `tool_deprecation.py` | never imported; `tool_procedures` / `tool_shapes` tables never created |
| `undo_window.py` | never imported; `undo_tokens` table never created, so no action could be undone through it |
| `used_resolution.py` | never imported; `used_resolutions` table never created |
| `compaction_survival.py` | never imported, and it read `used_resolutions` inside a bare `except`, so a missing table silently became "no memories were used" |
| `cross_domain.py` | installer entry only |
| `delete_proof.py` | installer entry only |
| `failure_modes.py` | installer entry only |
| `fleet_share.py` | installer entry only |
| `handoff_notes.py` | installer entry only |
| `layout_cache.py` | installer entry only |
| `node_actions.py` | installer entry only |
| `portability.py` | installer entry only |
| `privacy_modes.py` | installer entry only |
| `sleep_budget.py` | installer entry only |

## Kept, and why — these *look* unwired but are not

The first pass flagged these three as dead. They are loaded by **operator scripts**, through the
`from cortex import <name>` form:

- `edge_decay.py` ← `scripts/edge_decay.py`
- `recall_events.py` ← `scripts/recall_events.py`
- `vault_header.py` ← `scripts/vault_header.py`

That is exactly the mistake this note exists to prevent: a `grep` for `from .<name> import` returns nothing for all
three, and deleting them would have destroyed working maintenance tooling. **Always check the script form.**

## Not deleted, but still not earning their place

- **`relevance_pruning.py`** runs every night and has recorded **70 runs with zero decisions**: candidates need a
  relevance score at or below `0.25` and the live distribution bottoms out at `0.212`, clearing the cut for 7 of 3,535
  rows. It was *not* deleted, because pruning is the one pass that removes memory — deleting it is a data-lifecycle
  decision, not a dead-code one. What it needs is honesty: a zero-candidate streak surfaced as a warning instead of a
  clean `completed`.
- **The `memory_creation_feedback` table** has a live call site and no readers. `sum(positive_feedback_count)` is 0
  across 1,452 proposals, so the insert has never executed. Wire or drop; either is honest, leaving it is not.
