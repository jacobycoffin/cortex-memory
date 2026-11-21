# Retention policy

What Cortex keeps, for how long, and what actually happens when something is pruned.

This document is the policy. `scripts/retention_check.py` is the same policy expressed
as code: it reads a live database and reports exactly what a sweep would do, without
doing it. If the two disagree, the script is wrong — the policy here is the intent.

## Principles

1. **Nothing is destroyed by age alone.** A memory that stops being used is *cooled*
   (excluded from preload) and then *archived* (excluded from recall, still findable by
   explicit search). Deletion requires an explicit request or a manifest.
2. **Losing access is not losing the fact.** "Forgotten" means deprioritised. The row,
   its versions and its evidence stay readable.
3. **Use outranks recency.** A memory is retained because it was *acted on*, not because
   it was recently retrieved. `access_log.event = 'used'` is the signal; `retrieved` is not.
4. **A wrong prune is measurable.** If an archived memory is requested again inside the
   regret window, that is a miss and the window widens automatically.

## Windows

| Kind | Cool after | Archive after | Auto-delete |
|---|---|---|---|
| `episode` | 30 days unused | 90 days unused | never |
| `semantic` | 90 days unused | 180 days unused | never |
| `operational` | 90 days unused | 180 days unused | never |
| `decision`, `preference`, `identity`, `procedure`, `prospective` | 365 days unused | never | never |

"Unused" means no `used` event in the window. Retrieval alone does not reset the clock.

## Protected regardless of age

- `pinned = 1`
- has ever been `used`, and the use was confirmed (`helpful`/`confirmed` feedback)
- carries a `wrong`, `corrected` or `useful` feedback event
- is an endpoint of an edge that has evidence
- is an open `prospective_item`
- is the last surviving memory of its `subject`/`subject`+`predicate` pair
- is a version or a source of a current memory (history is never pruned)

## Deletion

Deletion happens in exactly three ways:

1. **Explicit user request** — "forget X". Soft delete (`state = 'tombstoned'`) plus a
   manifest of what was removed, so the removal can be proved and audited.
2. **Right-to-be-forgotten purge for a person** — hard delete with a manifest listing
   every removed row id and content hash.
3. **Legal hold reversal** — out of scope; legal-hold rows are never pruned.

There is no scheduled hard delete. No cron job deletes rows. If that changes, this
document changes first.

## Regret

`pruning_regret` records archived memories that were requested again. Policy: one regret
inside 30 days widens that memory's window by 30 days; three regrets inside 90 days
restore it to active and protect it permanently. The counter is never reset by hand.

## What a sweep reports

Each run prints, per kind: cooled, archived, protected, regretted. It writes a receipt
line ("archived 40, none requested since") so a change can be attributed to a run rather
than to mystery.

## Retention of the logs themselves

- `access_log`: 12 months, then aggregate to per-memory counts and drop the query text.
- `memory_traces`: 90 days.
- `recall_runs`: 90 days.
- `usage_records`: 12 months.
- Judge and consolidation transcripts: 30 days.

Log retention is separate from memory retention: pruning a log never touches a memory.
