//! Loopback HTTP daemon for the read-only Cortex service.
//!
//! `serve` binds `127.0.0.1:8090` by default. It answers `/api/pulse` and the
//! static files from disk, checks `dashboard-auth.json` the same way the
//! Python dashboard does, and proxies every other path — including
//! `/api/snapshot` and `/api/sleep/status` — to the Python process.
//!
//! SQLite access goes through [`cortex_store::Store`]. That open is
//! `mode=ro` and `PRAGMA query_only=ON`. This crate does not write
//! `meta.schema_version` and does not issue `PRAGMA journal_mode=`.

#![forbid(unsafe_code)]

mod assets;
mod auth;
mod config;
mod proxy;
mod pulse;
mod reply;
mod serve;
mod state;
mod util;

pub use config::{parse_args, ProcessEnv, ServeConfig, ServeError};
pub use serve::{start, ServerHandle};

/// Run the `serve` subcommand.
///
/// `args` is the process argument list, including argv0. `--db` may appear
/// before or after `serve`. Returns `0` after SIGTERM or SIGINT, `2` for a
/// usage error (including a non-loopback bind without
/// `CORTEX_SERVE_ALLOW_PUBLIC_BIND=1`), and `1` when the socket cannot be bound.
pub fn run<I, S>(args: I) -> i32
where
    I: IntoIterator<Item = S>,
    S: Into<std::ffi::OsString>,
{
    serve::run(args)
}

#[cfg(test)]
mod tests;
