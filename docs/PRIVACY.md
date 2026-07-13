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
