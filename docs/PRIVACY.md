# Privacy and security

Cortex is local-first, but a memory database is sensitive. It can contain preferences, infrastructure facts, paths, summarized conversations, vault excerpts, and redacted tool outcomes.

## Defaults

- storage stays in `$HERMES_HOME/cortex/cortex.db`;
- the dashboard binds to `127.0.0.1`;
- the dashboard exposes read-only memory endpoints and keeps guided review writes disabled by default;
- dashboard passwords are stored as salted PBKDF2 hashes, not readable credentials;
- browser sessions are signed, expire after 12 hours, and are revoked by a password change;
- the first generated password must be replaced after sign-in;
- likely secrets are redacted at capture;
- instruction-like memory text is quarantined from recall;
- vault notes are read but never modified;
- tool guidance keeps argument keys, not values;
- no hard-delete API exists.
- opt-in guided review can only resolve one conflict or unsupported inference per confirmed request, and preserves lifecycle/version history plus an audit record;
- scheduled Cortex Sleep uses deterministic local analysis and zero provider tokens by default;
- optional remote Sleep reflection sends a bounded selection of sanitized memory text to the configured provider and is therefore an explicit privacy boundary.

## Metacognitive monitoring data

The Trust monitor stores one `metacognitive_predictions` row for each memory considered during recall. A row contains the memory identifier, broad task and source categories, retrieval and reliability signals, the use/verify/abstain decision, an inspectable reason, and any later helpful or harmful outcome. It does not duplicate the memory text or the user's raw query.

These rows are sensitive behavioral metadata. They remain in the local Cortex database, follow the database's existing backup and retention policy, disappear when their referenced memory is hard-deleted, and are only exposed through the authenticated dashboard snapshot. Shadow mode records them without changing Kaya's prompt; enforcement is opt-in.

Dashboard benchmark runs use generated synthetic facts in an isolated temporary database. Cortex stores the run configuration, host Python/platform description, aggregate retrieval metrics, score, progress, and error state in `benchmark_runs`; it does not copy production memory content into a benchmark record and does not call a model provider. Benchmark history is authenticated dashboard data and follows the main database's backup and retention policy.

Outcome Lab labels store a task ID, outcome, dashboard actor, timestamps, and reversal state. Positive labels maintain a private `evaluation_cases` row containing the original query and relevant memory IDs inside the same Cortex database. Those private case fields are exposed only through the authenticated Outcome Lab and are never included in a completed evaluation report. The paired evaluator uses a disposable SQLite snapshot and persists aggregate quality, context, and local latency metrics in `evaluation_runs`; it omits queries, memory IDs and text, source references, and database paths. Undoing or replacing a label deactivates its evaluation case without deleting the audit history.

Tool-guidance exposure rows store task/session identifiers, broad task type, the recommended tool name or workflow key, predicted reliability, whether later execution matched it, and any explicit task outcome. Argument values, tool results, and memory text are not duplicated. These rows are behavioral metadata and follow the Cortex database's backup and retention policy.

## Operator responsibilities

- protect the Hermes home directory with host-level permissions and backups;
- use TLS plus strong authentication before routing a dashboard through a hostname;
- never commit `cortex.db`, dashboard credentials, vault content, or raw private benchmark traces;
- review quarantine and unsupported-inference counts;
- rotate credentials if they are displayed or copied into a public place;
- keep `dashboard-auth.json` private and mode `0600`;
- treat archived records as retained data, not deleted data.
- keep `CORTEX_SLEEP_TOKEN_BUDGET=0` unless the provider's privacy terms, model, key handling, and expected cost are acceptable;
- remember that a reflection budget is normal billed provider usage, not free or banked tokens.

## Reporting a vulnerability

Do not open a public issue containing a secret, private memory, or exploitable credential. Follow [SECURITY.md](../SECURITY.md).
