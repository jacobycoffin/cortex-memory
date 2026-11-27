# Rust service layer

Normative plan for a **phase-1 Rust service** in front of the existing SQLite file `cortex.db`. Python keeps writing. Rust owns the hot read path. This file wins over a task brief when they disagree; a brief that was wrong gets an `## Amendment`, not a silent rewrite.

Phase 1 is a strangler fig, not a port:

| stays Python | moves to the Rust binary |
|---|---|
| Hermes plugin, recall ranking, embeddings, nightly dream (`sleep.py`), review writes, `CortexStore` migrations | one binary, read-only SQLite, headless `--json`, terminal dot views, HTTP for `/api/pulse` and static files, auth parity, reverse proxy for everything else |

Nothing in phase 1 changes `cortex_schema.py`, bumps `meta.schema_version`, or adds a table.

## Workspace

Cargo workspace at `rust/`. It is not a Python package. Do not add `__init__.py` under `rust/`. `pyproject.toml` maps the `cortex` package onto the repo root (`package-dir = {cortex = "."}`); a Rust tree without `__init__.py` stays out of that package. `scripts/check_repository.py` scans every tracked file for private pathnames, secret shapes, and tracked `*.db` / `dashboard-auth.json`. It has no opinion about Cargo. Do not commit `rust/target/` (already gitignored), a database, or an auth file.

```
rust/
  Cargo.toml                 # workspace; members listed below
  Cargo.lock                 # commit it
  crates/
    cortex-store/            # RS-001 library
    cortex-cli/              # RS-002 library + bin `cortex-cli`
    cortex-http/             # RS-003 library + bin `cortex-http`
    cortex-view/             # RS-004 library + bin `cortex-view`
    cortex/                  # RS-005 bin `cortex-rs` only
```

`rust/Cargo.toml` members, in this order:

```toml
[workspace]
resolver = "2"
members = [
  "crates/cortex-store",
  "crates/cortex-cli",
  "crates/cortex-http",
  "crates/cortex-view",
]
```

RS-001 creates that file and compiling stubs for the three bins (`fn main` prints `not implemented` to stderr and exits 2). RS-005 is the only later task allowed to edit `rust/Cargo.toml`, and only to append `"crates/cortex"`. Nobody else edits the workspace manifest. Each later task edits only its own crate, plus `rust/Cargo.lock` when Cargo rewrites it. Parallel wave-2 branches will both touch that lockfile; the later merge rebases it.

Crates: `rusqlite` with the `bundled` feature (FTS5 must be compiled in; do not link a system SQLite that might lack it), `serde`, `serde_json`. HTTP crate adds `axum`, `tokio`, and one HTTP client for the proxy. No runtime download, no npm, no CDN. A C compiler is required at build time because of bundled SQLite.

The product binary after RS-005 is `cortex-rs`. Until that crate exists, tests invoke the crate bins. Argv is identical so RS-005 is a dispatcher, not a second parser. `--db` is a global flag before the subcommand, defaulting the same way as `cli.py`: `$CORTEX_DB` if set, otherwise `$HERMES_HOME/cortex/cortex.db`, otherwise `~/.hermes/cortex/cortex.db`.

```
cortex-rs --db PATH recall QUERY --json [--limit N]
cortex-rs --db PATH status --json
cortex-rs --db PATH remember TEXT --json [--kind KIND]
cortex-rs --db PATH sleep-status --json
cortex-rs --db PATH pulse [--cols N] [--rows N] [--interval SECS] [--ascii] [--once]
cortex-rs --db PATH dream [--watch] [--follow] [--ascii] [--run-id ID] [--once]
cortex-rs serve [--bind 127.0.0.1:8090] [--db PATH] [--assets DIR]
```

## SQLite coexistence

Python's writer is `CortexStore` (`store.py`). On every open it runs `PRAGMA journal_mode=WAL`, `PRAGMA foreign_keys=ON`, `PRAGMA busy_timeout=5000`, then `_create_schema()`, then:

```sql
INSERT INTO meta(key, value) VALUES('schema_version', ?)
ON CONFLICT(key) DO UPDATE SET value=excluded.value
```

with `SCHEMA_VERSION` from `cortex_schema.py` (34 in this tree). That write is how the on-disk version moves. Rust must never open the file through that constructor, never import it, and never issue that statement.

`meta.value` is TEXT. A live database can already be ahead of this tree's constant. Rust reads the value and reports it. It does not compare it, migrate toward it, or write it back.

### How to open

1. If the path is not an existing file, stop. Do not create the parent directory and do not create an empty database. `CortexStore` does both, and an empty file then looks like a healthy store.
2. Open `file:{absolute}?mode=ro` with a read-only flag and a 5000 ms busy timeout. Immediately run `PRAGMA query_only=ON` and `PRAGMA busy_timeout=5000`. Do not run `PRAGMA journal_mode`, `wal_checkpoint`, or `foreign_keys`.
3. If open fails, return the SQLite error text and stop. Do not retry with a read-write connection. A WAL database needs the `-wal` file readable and the `-shm` file writable by the reader; a unit that cannot write `-shm` fails the open. The fix is the unit's `ReadWritePaths` (see Deployment), not a writable connection.
4. One statement, then the connection is idle. Do not hold a transaction across a poll interval, a proxy call, or a sleep. A long-lived read snapshot pins the WAL and blocks checkpoints. A leaked write transaction in the dashboard has already frozen every other writer for hours; the read path uses `query_only` and never `BEGIN IMMEDIATE`.
5. Missing **required** tables (`meta`, `memories`, `usage_records`, `access_log`, `sleep_runs`) mean this is not a Cortex database: exit 3 / HTTP 503. Missing **optional** tables are a null field plus a `note`, not a crash and not a zero. Optional: `memory_fts`, `recall_runs`, `recall_score_breakdown`, `procedure_outcomes`, `sleep_proposals`, `sleep_state_changes`, `sleep_edge_changes`, `sleep_episode_state`. The last two score tables are created on first use by `recall_explain.py` and `procedure_confidence.py`. They are not in `SCHEMA_SQL`.

### Four states, one zero

A measurement that did not run is `null` plus a `note`. It is never `0`.

`memories.active`, `cold`, `archived`, and `other` come from `GROUP BY state` whenever the `memories` table exists. A state with no rows is the integer `0`. If the table exists and has no rows at all, every count is `0` and the note is `the store is empty`. If the table is missing, every count is null.

`recalled_7d`, `graded`, and `procedures.total` / `procedures.graded` follow `plain_summary.py`: a successful query whose count is `0`, or a missing optional table, is `null` and the note `not measurable yet`. An empty recall **result list** is different: `count: 0`, exit 1, note `no match`. That zero means the query ran.

JSON field `schema_version` on every agent payload is the **contract version**, integer `1`. It is not the database version. The database version travels as the string `store_schema_version` inside `health` on `status` only. Never write contract version `1` into `meta`.

### Columns the hot path reads

`memories`: `id`, `kind`, `content`, `state`, `created_at`, `used_count`, `confidence`, `source_type`. States that exist in the prose layer: `active`, `cold`, `archived`, `quarantined`, `tombstoned`. Counts for the contract are `active`, `cold`, `archived`. Any other state is summed into `other` so it is not dropped on the floor.

`usage_records`: `usage_id`, `task_id`, `memory_id`, `created_at`, `used`, `attribution`, `outcome`, `score`. One grid cell is one `task_id`. `used_fraction = SUM(used=1) / COUNT(*)`.

`access_log`: `memory_id`, `event`, `created_at`. Recall events, matching `plain_summary.py`: `retrieved`, `selected`, `injected`. Graded usage outcomes: `helpful`, `harmful`, `validated`, `corrected`.

`recall_runs`: `recall_id`, `created_at` (sparkline and last-recall time). `recall_score_breakdown`: `task_id`, `rank`, `memory_id`, `score`, `components_json`, `created_at`. Component keys are translated with the map in `plain_summary.COMPONENT_WORDS` (`semantic` and `lexical` both become `wording match`, `semantic_fusion` becomes `found by meaning`, `usage` becomes `past usefulness`). Drop `shadow_score` and `shadow_score_delta`. Sum keys that collapse to the same words. Keep the four largest absolute values, rounded to 2 decimal places. Keys in `why` are those words, not the raw component names.

`memory_fts`: FTS5, `memory_id UNINDEXED`, `content`, `tokenize='porter unicode61'`.

`procedure_outcomes`: `procedure_id`, `outcome`, `at`, `note`. Created lazily. `procedures.total` and `procedures.graded` are both `COUNT(DISTINCT procedure_id)` when the table exists and the count is non-zero; otherwise null.

`sleep_runs`: `run_id`, `mode`, `status`, `cutoff_at`, `episodes_scanned`, `episodes_replayed`, `association_proposals`, `interference_proposals`, `edge_decay_proposals`, `lifecycle_candidates`, `consolidation_candidates`, `dependency_candidates`, `applied_changes`, `error`, `started_at`, `completed_at`, `report_json`. `mode` is `shadow` or `apply`. `status` written by `sleep.py` is `running`, `completed`, `failed`, `reverted`, `revert_partial`.

`sleep_episode_state`: `episode_id`, `last_run_id`, `replay_count`. `sleep_state_changes`: `run_id`, `memory_id`, `prior_state`, `next_state`, `created_at`, `reversed_at`. `sleep_edge_changes`: `run_id`, `reversed_at`. `sleep_proposals`: `run_id`, `kind`, `status`, `score`. Kinds the dream view names: `association`, `association_reinforcement`, `edge_downscale` (the counter column is `edge_decay_proposals`; the row kind is `edge_downscale`), `interference_review`, `lifecycle`, `consolidation`, `dependency_repair`, `context_review`.

The run-row counters stay at their insert defaults until `_finish_run` updates them. While `status='running'`, read progress from the child tables. A bar built from `episodes_replayed` during a run would show a fake `0`.

There is no next-run column. `dashboard.py` reads `cortex-sleep.timer` via `systemctl` and the label `Nightly · 03:00–05:00 local window` (`DEFAULT_SLEEP_SCHEDULE`). The headless command does not call `systemctl` and does not invent a clock time.

### Queries

Memory counts:

```sql
SELECT state, COUNT(*) AS n FROM memories GROUP BY state
```

Recalled in 7 days. Bind `?` as a UTC timestamp computed in the process, 7 days before now:

```sql
SELECT COUNT(DISTINCT memory_id) AS n FROM access_log
WHERE event IN ('retrieved','selected','injected')
  AND datetime(created_at) >= datetime(?)
```

Graded answers:

```sql
SELECT COUNT(DISTINCT task_id) AS n FROM usage_records
WHERE outcome IN ('helpful','harmful','validated','corrected')
```

Latest dream:

```sql
SELECT run_id, mode, status, applied_changes, error, started_at, completed_at,
       episodes_scanned, episodes_replayed, association_proposals, edge_decay_proposals
FROM sleep_runs ORDER BY started_at DESC LIMIT 1
```

In-progress child counts, bound to that `run_id`:

```sql
SELECT COUNT(*) AS n FROM sleep_episode_state WHERE last_run_id = ?
SELECT prior_state, next_state, COUNT(*) AS n
  FROM sleep_state_changes
  WHERE run_id = ? AND reversed_at IS NULL
  GROUP BY prior_state, next_state
SELECT kind, COUNT(*) AS n FROM sleep_proposals WHERE run_id = ? GROUP BY kind
SELECT COUNT(*) AS n FROM sleep_edge_changes WHERE run_id = ? AND reversed_at IS NULL
```

Grid, newest last. Bind the limit (`cols * rows`, default 240):

```sql
SELECT task_id, MIN(created_at) AS started_at,
       COUNT(*) AS rows,
       SUM(CASE WHEN used = 1 THEN 1 ELSE 0 END) AS used_rows
FROM usage_records
GROUP BY task_id
ORDER BY started_at DESC
LIMIT ?
```

Render oldest-of-that-window at the top-left. Pad missing cells on the left with level 0. Level rules and glyphs: `design/dot-glyphs.json`.

Sparkline: 60 buckets of `recall_runs.created_at` covering the last 60 minutes (one minute each). If `recall_runs` is absent or every bucket is empty, every cell is the empty glyph and the numeric rate is null with note `not measurable yet`. Do not print `0.0/s`.

Lexical recall. Build the MATCH string from the query by splitting on characters that are not letters or digits, wrapping each token as an FTS phrase (`"token"`, embedded quotes doubled), and joining with ` AND `. No tokens is exit 2. Do not pass the raw query to `MATCH`.

```sql
SELECT m.id, m.kind, m.content, -bm25(memory_fts) AS score
FROM memory_fts
JOIN memories m ON m.id = memory_fts.memory_id
WHERE memory_fts MATCH ? AND m.state IN ('active','cold')
ORDER BY bm25(memory_fts)
LIMIT ?
```

Default limit is 6, matching `cortex-memory search`. `score` is the lexical `-bm25` value (higher is better). It is not the Python fusion score. `ranker` is `"lexical"`.

`why` for each hit: if `recall_score_breakdown` has a row for that `memory_id`, translate `components_json` through `COMPONENT_WORDS` and set `note` to `why is the last recorded breakdown, not the lexical score`. If the table or the row is missing, `why` is `{"wording match": <score rounded to 2 decimals>}` and `note` is `lexical read model; semantic fusion stays in the Python retriever`.

Store schema and journal mode, both read-only:

```sql
SELECT value FROM meta WHERE key = 'schema_version'
PRAGMA journal_mode
```

### The one write, and it is not Rust

`remember` execs the existing Python CLI and does not insert:

```
cortex-memory --db <same path> remember <text> [--kind <kind>]
```

Python prints `{"memory_id", "created", "quarantined"}`. Map that to the contract below. `proposal` is `false`: this CLI writes a memory row (or returns the existing id). It does not stage a review proposal. `kind` is the requested kind. On a non-zero Python exit, or when `cortex-memory` is not on `PATH`, exit 3 and put the reason in `note`. Rust does not open a write connection to do this itself. The subprocess is the writer, including its existing migrate-on-open behavior. Do not pass any other Python subcommand.

## Headless JSON contract

`--json` is required for `recall`, `status`, `remember`, and `sleep-status`. Without it, exit 2. Stdout is one JSON object, UTF-8, no ANSI, no trailing prose. Memory `content` is passed through, including non-ASCII. Decorative glyphs belong in the terminal view, not in this object. `jq -e .schema_version` prints `1`.

Exit codes: `0` success, `1` recall found nothing (not a failure), `2` bad usage, `3` store unreadable or the Python writer could not be run. `status` and `sleep-status` of an empty but readable store exit 0.

`recall`:

```json
{"schema_version":1,"query":"...","ranker":"lexical","memories":[{"id":"...","kind":"...","content":"...","score":1.2,"why":{"wording match":0.62}}],"count":1,"note":null}
```

`status`:

State buckets are integers from `GROUP BY`. The nulls are instruments with no non-zero result yet.

```json
{"schema_version":1,"memories":{"active":12,"cold":0,"archived":3,"other":0},"recalled_7d":null,"graded":null,"procedures":{"total":null,"graded":null},"health":{"readable":true,"journal_mode":"wal","store_schema_version":"34","note":null},"note":null}
```

`health.note` carries `not measurable yet` for a missing instrument. Top-level `note` carries `the store is empty` when the `memories` table has no rows. State counts of `0` stay integers; `recalled_7d`, `graded`, and `procedures` stay null until a non-zero measurement exists.

`remember`:

```json
{"schema_version":1,"memory_id":"...","kind":"semantic","proposal":false,"note":"written by cortex-memory"}
```

`sleep-status`:

```json
{"schema_version":1,"last_run":{"id":"...","mode":"apply","at":"...","applied_changes":23,"status":"completed"},"next_scheduled":null,"note":"next run is not stored in cortex.db; the timer is cortex-sleep.timer when installed"}
```

`last_run` is null and the note is `no dream has run yet` when `sleep_runs` is empty. `at` is `completed_at`, or `started_at` while the run is still `running`. `applied_changes` is null while `status` is `running`, because the column has not been finished; do not report the insert default `0`.

Dream freshness, used by `status` consumers and by the terminal: `running` if status is `running`; `no_data` if there is no row; `failed` if status is `failed` and `completed_at` is within 36 hours; `fresh` if `completed` / `reverted` / `revert_partial` and `completed_at` is within 36 hours; `stale` otherwise. The constant is 36 hours. Times are taken from the stored ISO text. Do not convert time zones. Display `HH:MM` as the hour and minute that follow `T` in that stored text.

Today's Python `status` and `why` stay as they are: prose, `mode=ro`, exit 0 even when the file is missing. Do not add `--json` to `cli.py`.

## Terminal views

Glyphs, thresholds, and grid order are `design/dot-glyphs.json`. One status line, then the grid, nothing else. Colour only when stdout is a TTY and `NO_COLOR` is unset or empty. `--ascii` forces the ASCII column of the glyph file. `Ctrl-C` restores the terminal and exits 0.

`pulse` redraws in place (cursor home, no clear-and-scroll, no extra newline after the first paint). Default interval 1 second. A locked or missing store prints one line, `store unavailable: <reason>`, and the loop keeps going (exit 0). `--once` renders a single frame and exits 3 when the store is unreadable.

Status line, single spaces, words not checkmarks:

```
cortex  {n} mem  {sparkline} {rate}  last dream {state} {when}  {health}
```

`{n}` is the memory total, or `n/a`. `{rate}` is `{x}/s` with one decimal place, or `n/a`. `{state}` is `not run yet`, `running`, `stale`, `fresh`, or `failed`. `{when}` is the `HH:MM` substring, omitted for `not run yet` and `running`. `{health}` is `ok` or `unavailable`. An idle store is a flat grid of empty glyphs.

`dream` with no flag prints the latest run once and exits. `--watch` and `--follow` are the same flag: redraw until the run leaves `running` or the user hits Ctrl-C. `--run-id` selects a run; unknown id prints `no dream has run yet` and exits 0. No rows at all prints `no dream has run yet` and exits 0.

Bars use the same five grid levels, scaled by `count / max(count, denominator)`. While the run is `running`, denominators and counts come from the child queries, and unfinished run-row counters are not displayed. After `completed` or `failed`, the run-row counters are the denominators.

| line | count | denominator while running | denominator when finished |
|---|---|---|---|
| episodes | `sleep_episode_state` rows for the run | omitted (`n/a`), the scanned total is not written yet | `episodes_scanned` |
| demoted | state changes `active` → `cold`, `reversed_at` null | the count itself | the count itself |
| linked | proposals `association` plus `association_reinforcement` | the count itself | `association_proposals` when it is `>=` the child count, else the child count |
| edges | proposals `edge_downscale` | the count itself | `edge_decay_proposals` on the same rule |

Verdict, one line, only what is true:

| run | line |
|---|---|
| no row | `no dream has run yet` |
| `running` | `running {run_id}` |
| `shadow` and finished | `shadow only · no changes applied · run {run_id}` |
| `apply`, `completed`, `applied_changes` > 0 | `{applied_changes} changes applied · undo: cortex-memory sleep-undo {run_id}` |
| `failed` | `dream failed · run {run_id}` |
| `reverted` or `revert_partial` | `undone · run {run_id}` |

`undo_sleep` rejects a run whose `mode` is not `apply`. A shadow run must not print that undo command.

## HTTP, static files, auth

`cortex-rs serve` binds `127.0.0.1` only. Default `127.0.0.1:8090`. A bind address outside the loopback range is refused unless `CORTEX_SERVE_ALLOW_PUBLIC_BIND=1`. Python's CLI default is **8765** (`cli.py`); the installed user unit `scripts/cortex-dashboard.service.in` uses **8100**. Leave the Python unit where it is. Rust sits beside it. `CORTEX_PYTHON_UPSTREAM` defaults to `http://127.0.0.1:8100`.

Assets directory `CORTEX_ASSET_DIR`, default the current directory. Read from disk on each request (or cache by mtime+size). `SIGHUP` drops that cache and re-reads the auth file. Do not read `dashboard.html` once at startup; the Python server does, and a deploy then needs a restart.

| path | who answers | auth |
|---|---|---|
| `GET /` | `dashboard.html` | public |
| `GET /pulse` | `pulse.html` | public |
| `GET /phone` | `phone.html` | public |
| `GET /dashboard.css`, `/dashboard.js` | those files, if present | public |
| `GET /vendor/*` | files UI-005 commits under `vendor/`, `.js` only, no `..` | public |
| `GET /favicon.svg`, `/favicon.ico`, `/apple-touch-icon.png` | those files | public |
| `GET /api/pulse` | Rust, SQLite | gated |
| `GET /api/auth/status` | Rust | public |
| `POST /api/auth/login`, `/api/auth/logout`, `/api/auth/change-password` | Rust | login public; the other two need a session |
| `GET /api/snapshot`, `GET /api/sleep/status`, every other path | reverse proxy to `CORTEX_PYTHON_UPSTREAM` | Python applies its own gate; forward `Cookie`, `Authorization`, `Host`, `X-Forwarded-For`, `X-Forwarded-Proto`, `X-Cortex-Request` |

`/api/snapshot` is `CortexStore.dashboard_snapshot()` plus in-memory fields (`review_writes_enabled`, `review_copilot`, `sleep_runtime`, `sleep_schedule`, `auto_judge`). Reimplementing it would fork thousands of lines and invent zeros. Phase 1 proxies it. When the upstream is down or the five-second timeout fires, respond **503** with `{"error":"snapshot unavailable","note":"python dashboard upstream is not running","schema_version":1}`. Never substitute an empty snapshot.

A missing static file is 404 text `not found`, not a generated page. `/api/pulse` is 503 with `{"error":"store unavailable","note":"...","schema_version":1}` when the database cannot be opened.

`/api/pulse` body:

```json
{"schema_version":1,"buckets":[0,1,4],"status":{"memories":{"active":1,"cold":0,"archived":0},"recalled_7d":null,"last_dream":{"state":"not run yet","at":null,"mode":null,"run_id":null}},"note":null}
```

`buckets` is one integer 0..4 per grid cell, left padding included, length `cols * rows`. Query `?cols=&rows=` overrides, each clamped to 1..80. `last_dream.state` uses the freshness rules above. `recalled_7d` follows the null rule. This route is polled about every 2 seconds. It is the cheap queries above, not `dashboard_snapshot`.

Headers on every Rust response, copied from `dashboard.py`: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Permissions-Policy: camera=(), microphone=(), geolocation=()`, and the current Content-Security-Policy, including `script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net`. `dashboard.html` still loads graphology, sigma, force-graph, cytoscape, and 3d-force-graph from that CDN. Tightening the policy before UI-005 vendors those files breaks the deep-dive page. HTML and `/api/*` send `Cache-Control: no-store`. `dashboard.css`, `dashboard.js`, and `/vendor/*` send `Cache-Control: public, max-age=86400` plus an `ETag` of `mtime-size`.

### Auth parity

Auth is not a SQLite table. It is the JSON file `dashboard_auth.py` reads. Default path: `<parent of the database>/dashboard-auth.json`. Override: `CORTEX_DASHBOARD_AUTH_FILE`. Mode `0600`. Never commit it, never log it, never put it in the unit's `Environment=` lines.

File fields Rust must honour: `username`, `salt`, `password_hash`, `session_secret`, `session_version`, `iterations` (default 310000), `must_change_password`. Binary fields are URL-safe base64 without padding. Password check is PBKDF2-HMAC-SHA256 over the password bytes. Compare username and digest in constant time.

Session cookie name: `cortex_session`. Token: `b64(json).b64(hmac_sha256(session_secret, encoded_json_ascii))`. The JSON uses sorted keys and compact separators, payload `exp`, `iat`, `u`, `v`. Lifetime 12 hours (43200 seconds). Reject a bad signature, a past `exp`, a username mismatch, or a `v` that is not the file's `session_version`. Re-read the file when its `(mtime_ns, size)` changes, including the case where it appears after the process started. If the file is absent, `auth.configured` is false and gated routes are open, matching the Python server. The moment the file exists, enforce it.

Cookie on login: `cortex_session=<token>; Path=/; HttpOnly; SameSite=Strict; Max-Age=43200`, plus `Secure` when `X-Forwarded-Proto` is `https`. Logout sets `Max-Age=0`.

`GET /api/auth/status` returns `auth_enabled`, `authenticated`, `must_change_password`, `username` (empty string when anonymous).

Every POST requires header `X-Cortex-Request: 1`, else 403 `{"error":"request verification failed"}`. JSON body, object, at most 16384 bytes. Login: username length ≤ 128, password length ≤ 256, else the same 401 as a wrong password: `{"error":"Username or password is incorrect."}`. Success 200 with the cookie and `authenticated`, `must_change_password`, `username`.

Throttle: 8 attempts per 300 seconds per client, slot reserved before PBKDF2, cleared on success. Client key is the TCP peer. Use the first hop of `CF-Connecting-IP`, else `X-Forwarded-For`, only when that peer is listed in `CORTEX_DASHBOARD_TRUSTED_PROXIES`. Over the limit: 429 `{"error":"Too many attempts. Try again shortly."}`.

A session with `must_change_password` still set may call `change-password` and `logout` and `auth/status`. Any other gated route returns 403 `{"error":"password change required","must_change_password":true}`. Missing session on a gated route: 401 `{"error":"authentication required"}`. Gated `/api/pulse` also accepts HTTP Basic (`Authorization: Basic ...`) verified with the same password function, and that attempt counts toward the throttle. Change-password rules match `validate_new_password`: at least 12 characters, at most 256, and the password must not contain the username (case-insensitive). Writing the file uses a temp file in the same directory, mode `0600`, then rename. `change_password` calls `_write_new`, which stores a new `session_secret` and an incremented `session_version`. Both change, so older cookies fail closed. Match `_write_new` field for field, so a token minted by either process verifies in the other. The parity test mints a hash and a token with `python3 -c "import dashboard_auth"` from the repo root and checks that Rust accepts the token and rejects it after a password change.

Host allowlist: `CORTEX_DASHBOARD_ALLOWED_HOSTS`, comma-separated. Empty means do not check. Otherwise strip the port (keep IPv6 brackets), casefold, and require membership. Failure: 403 `{"error":"host not allowed"}`. Apply it on Rust-owned routes and forward the original `Host` upstream so Python's own check still sees the public name.

Do not log `Cookie`, `Authorization`, or the auth file.

## Deployment

Side by side with `cortex-dashboard.service`. Do not edit, stop, or retarget that unit.

`scripts/cortex-serve.service.in` is a user unit (`WantedBy=default.target`), `Type=simple`, `Restart=on-failure`, `RestartSec=5`, `NoNewPrivileges=true`, `PrivateTmp=true`, `ProtectSystem=strict`, `ProtectHome=read-only`, `ReadWritePaths=@@HERMES_HOME@@/cortex`. The read-write path is what lets a WAL reader update `-shm` and lets a password change replace the auth file. `ExecStart` is `@@BIN@@ serve --bind 127.0.0.1:@@PORT@@`. `ExecReload=/bin/kill -HUP $MAINPID`. `EnvironmentFile=@@HERMES_HOME@@/cortex/cortex-serve.env`.

`scripts/install_cortex_serve.sh` follows `scripts/install_dashboard_service.sh`: default `HERMES_HOME` is `$HOME/.hermes`, default `PORT` is `8090`, install under `${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user`. Create `cortex-serve.env` mode `0600` if missing, with `CORTEX_DB`, `CORTEX_ASSET_DIR` (the plugin directory the script lives in), `CORTEX_PYTHON_UPSTREAM=http://127.0.0.1:8100`, and `CORTEX_DASHBOARD_AUTH_FILE`. Do not copy a password into the env file. Do not create the auth file; `cortex-memory dashboard-password` remains the tool that does. `systemctl --user daemon-reload` and enable. Print the bind address and the upstream URL. If `8090` is taken, exit non-zero without killing the occupant.

`SIGTERM` finishes the requests already accepted, closes SQLite without a checkpoint, and exits 0.

The public hostname stays on the reverse proxy the operator already runs. Point that proxy at `127.0.0.1:8090` only after the smoke command passes. Until then Python on `8100` is still the public backend, and Rust can be smoked on `8090` directly.

## Phases

**Phase 1** is RS-001 through RS-005 plus the amended HTML tasks. Rust reads, serves, proxies, and renders. Python dreams, embeds, ranks, and writes.

**Later, not tasked.** In-process `remember`, review POSTs, and any port of `sleep.py`. Those fork the writer. They wait until phase 1 has run beside Python and the smoke command is the regression net. Do not open that work from these briefs.

## Task graph

`depends_on` is the rule. Wave numbers are the earliest wave in which a task can run.

| wave | tasks | run together because |
|---|---|---|
| 1 | RS-001, UI-003, UI-005 | no shared files |
| 2 | RS-002, RS-003, RS-004, UI-006 | each crate is distinct; UI-006 waits on UI-003 and UI-005, the Rust tasks wait on RS-001 |
| 3 | RS-005 | needs the three libraries |

UI-001, UI-002, and UI-004 are `superseded`. They are not dispatched. Their text stays so the intent (JSON contract, pulse pane, dream pane) is still written down. RS-002 and RS-004 implement that intent.

| task | owns | does not own |
|---|---|---|
| RS-001 | workspace, `cortex-store`, bin stubs | HTTP, glyphs, Python |
| RS-002 | `rust/crates/cortex-cli/`, `rust/Cargo.lock` | `rust/Cargo.toml`, `cli.py` |
| RS-003 | `rust/crates/cortex-http/`, `rust/Cargo.lock` | `dashboard.py`, HTML |
| RS-004 | `rust/crates/cortex-view/`, `rust/Cargo.lock` | HTML, `sleep.py` |
| RS-005 | `rust/crates/cortex/`, the two unit files, `rust/Cargo.lock`, one line in `rust/Cargo.toml` | crate internals beyond the public functions they already expose |
| UI-003 | `pulse.html` and its test | `dashboard.py` |
| UI-005 | the deep-dive split, including the Python static routes those files need | `/api/pulse` |
| UI-006 | truthful states in the HTML/JS surfaces | Rust |

## Gotchas that will waste a build

1. `schema_version` is a TEXT row in `meta`, rewritten on every Python open. Contract `"schema_version": 1` is a different number and must not be stored.
2. `mode=ro` will not create `-shm`. A read-only unit that cannot write `$HERMES_HOME/cortex` cannot open a WAL database. Do not "fix" that with a writable connection.
3. `sleep_runs` counters are filled in by `_finish_run` at the end. Watch child rows.
4. `procedure_outcomes` and `recall_score_breakdown` may not exist. Absence is `null` and a note.
5. The next dream time is a systemd timer, not a row. `next_scheduled` is null.
6. `/api/snapshot` and `/api/sleep/status` include process memory (`sleep_runtime`). Proxy them.
7. Auth is `dashboard-auth.json` plus cookie `cortex_session`. No session table. No file means the server is open.
8. CLI `--port` defaults to 8765. The dashboard unit uses 8100. Rust defaults to 8090 and does not take either socket.
9. The CSP allows `https://cdn.jsdelivr.net` because `dashboard.html` still loads five graph scripts from it (graphology 0.26.0, sigma 3.0.3, force-graph 1.51.4, cytoscape 3.34.0, 3d-force-graph 1.80.0).
10. `check_repository.py` rejects a tracked database, a tracked auth file, a home path under `/Users/`, and a root-owned Hermes home path. Write `$HERMES_HOME` in docs and unit templates.
