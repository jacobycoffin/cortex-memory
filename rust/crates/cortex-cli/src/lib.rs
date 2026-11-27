//! Headless JSON for an existing `cortex.db`.
//!
//! `recall`, `status`, and `sleep-status` read through [`cortex_store::Store`].
//! They do not create the file and do not write `meta.schema_version`.
//! `remember` execs `cortex-memory` and does not insert a row itself.
//!
//! `--json` is required. Stdout is one JSON object whose `schema_version` is
//! [`cortex_store::CONTRACT_SCHEMA_VERSION`] (the contract, not the database).
//! Exit [`EXIT_OK`] is success, [`EXIT_NO_MATCH`] is a recall with no hit,
//! [`EXIT_USAGE`] is bad usage, and [`EXIT_UNAVAILABLE`] is an unreadable store
//! or a `cortex-memory` process that could not be run.
//!
//! [`run`] takes the argv after the program name, the same sequence this binary
//! receives. Later dispatchers can call it instead of parsing a second time.

#![forbid(unsafe_code)]

mod args;
mod commands;
mod output;

/// The command finished and printed its payload.
pub const EXIT_OK: i32 = 0;
/// `recall` ran and the result list is empty.
pub const EXIT_NO_MATCH: i32 = 1;
/// Missing `--json`, a missing query or text, an unknown command, or a query with no FTS tokens.
pub const EXIT_USAGE: i32 = 2;
/// The database could not be read, or `cortex-memory` could not be run.
pub const EXIT_UNAVAILABLE: i32 = 3;

/// Run one headless command. `args` does not include the program name.
pub fn run<I, S>(args: I) -> i32
where
    I: IntoIterator<Item = S>,
    S: Into<std::ffi::OsString>,
{
    match args::parse(args) {
        Ok(invocation) => commands::dispatch(invocation),
        Err(err) => {
            eprintln!("{err}");
            EXIT_USAGE
        }
    }
}
