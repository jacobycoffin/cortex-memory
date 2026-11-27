---
task_id: RS-003
assignee: cursor-cloud
status: unassigned
layer: service
wave: 2
superseded_by: null
owner_files:
  - rust/Cargo.lock
  - rust/crates/cortex-http/
depends_on:
  - RS-001
verify:
  - cargo test --manifest-path rust/Cargo.toml -p cortex-http
  - python3 scripts/check_repository.py
---

# HTTP daemon, `/api/pulse`, static files, auth parity

> **Read first.** `design/RUST-SERVICE.md` section **HTTP, static files, auth**. Do not edit `dashboard.py`. UI-003 owns `pulse.html`. UI-005 owns the deep-dive split. You only serve bytes from disk.

## Why

The glanceable page and the deep-dive page need a fast local server that speaks the same cookie as the Python dashboard and does not hold a SQLite transaction open between polls. The big snapshot stays in Python so the numbers have one implementation.

## What to build

The `cortex-http` binary's `serve` subcommand, library-first so RS-005 can call it:

- Bind `127.0.0.1:8090` by default. Refuse a non-loopback bind unless `CORTEX_SERVE_ALLOW_PUBLIC_BIND=1`.
- Serve the static table in the spec from `CORTEX_ASSET_DIR`. Re-read on `SIGHUP` and when mtime or size changes.
- `GET /api/pulse` uses `cortex-store` and the null / four-state rules. Auth-gated exactly like `/api/snapshot` is today.
- Auth file, cookie, PBKDF2, HMAC session, throttle, host allowlist, Basic auth, and password-change write: match `dashboard_auth.py` and `dashboard.py`. The parity test mints a token with `python3 -c "import dashboard_auth"` from the repo root and checks Rust accepts it, then rejects it after `change_password` bumps `session_version`.
- Proxy every other path, including `GET /api/snapshot` and `GET /api/sleep/status`, to `CORTEX_PYTHON_UPSTREAM` (default `http://127.0.0.1:8100`) with a 5 second timeout. Upstream down: 503 and the note in the spec, not an empty snapshot.
- Copy the current CSP, including the cdn.jsdelivr.net script source. UI-005 removes the CDN loads; do not tighten the policy in this task.
- Do not log cookies, the Authorization header, or the auth file.

Replace the RS-001 stub. You may edit this crate's `Cargo.toml`. You may not edit `rust/Cargo.toml` or `dashboard.py`.

## Acceptance

- `/api/pulse` without a session is 401 when an auth file exists, and 200 when the auth file is absent.
- `/api/pulse` keys match the spec, and a fixture with no `access_log` hits returns `recalled_7d: null`.
- A token minted by `dashboard_auth.py` authorizes `/api/pulse`. After a password change through the Rust route, that token is rejected and a new one works.
- With the upstream listener closed, `GET /api/snapshot` is 503 and the body contains `schema_version` and a `note`.
- `GET /pulse` is 404 text when `pulse.html` is not in the asset dir, and 200 with that file's bytes when it is.
- A test asserts the process never writes `meta` and never issues `journal_mode=`.
- Host header not in a non-empty allowlist is 403.

## Out of scope

Writing `pulse.html` or `dashboard.html`, implementing `dashboard_snapshot` in Rust, editing the Python server, the systemd unit (RS-005).
