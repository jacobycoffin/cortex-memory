//! Read-only access to an existing `cortex.db`.
//!
//! Python's `CortexStore` is the writer: it turns on WAL, migrates, and
//! rewrites `meta.schema_version` on every open. This crate never does that.
//! It opens an existing file with `mode=ro` and `PRAGMA query_only=ON`, reads
//! the on-disk schema version as text, and runs the hot-path queries. A
//! missing file is an error. Nothing here creates a database, a parent
//! directory, or a table.
//!
//! `CONTRACT_SCHEMA_VERSION` is the agent JSON contract. It is not the
//! database version and must never be written into `meta`.

#![forbid(unsafe_code)]

mod error;
mod lexical;
mod open;
mod queries;
mod timeutil;

pub use error::{Measurement, OptionalRow, Presence, StoreError, OPTIONAL_TABLES, REQUIRED_TABLES};
pub use lexical::{
    component_words, fts_match_query, round_2, translate_components, LexicalError, LexicalHit,
    ScoreBreakdown, LEXICAL_RANKER, NOTE_LEXICAL_BREAKDOWN, NOTE_LEXICAL_ONLY,
};
pub use open::{Store, BUSY_TIMEOUT_MS};
pub use queries::{
    grid_levels, usage_level, KindCount, MemoryCounts, Procedures, SleepChildren, SleepRun,
    Sparkline, StateChangeCount, UsageBucket, DEFAULT_RECALL_LIMIT, DEFAULT_USAGE_CELL_LIMIT,
    SPARKLINE_BUCKETS,
};
pub use timeutil::{stored_hhmm, UtcTime};

/// Hours after `completed_at` during which a finished dream is fresh or failed.
pub const DREAM_FRESH_HOURS: i64 = 36;

/// Agent payload `schema_version`. Not `meta.schema_version`.
pub const CONTRACT_SCHEMA_VERSION: i64 = 1;

pub const NOTE_NOT_MEASURABLE: &str = "not measurable yet";
/// Procedures exist but nothing has graded one: the library is populated, the grading loop is not running.
pub const NOTE_PROCEDURES_UNGRADED: &str = "procedures are stored but none has been graded yet";
pub const NOTE_EMPTY_STORE: &str = "the store is empty";
pub const NOTE_NO_MATCH: &str = "no match";
pub const NOTE_NO_DREAM: &str = "no dream has run yet";
pub const NOTE_NEXT_SCHEDULED: &str =
    "next run is not stored in cortex.db; the timer is cortex-sleep.timer when installed";

/// How recent the latest dream row is, from the stored timestamps.
///
/// `NoData` is the absence of a row. Callers render that as "not run yet"
/// or `no dream has run yet`. A clock time is never invented here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DreamFreshness {
    Running,
    NoData,
    Failed,
    Fresh,
    Stale,
}

pub fn dream_freshness(run: Option<&SleepRun>, now: UtcTime) -> DreamFreshness {
    let Some(run) = run else {
        return DreamFreshness::NoData;
    };
    if run.status == "running" {
        return DreamFreshness::Running;
    }
    let within = run
        .completed_at
        .as_deref()
        .and_then(UtcTime::parse_stored)
        .is_some_and(|at| {
            let age = now.unix_secs().saturating_sub(at.unix_secs());
            // A stored stamp slightly ahead of the process clock is still current.
            at.unix_secs() > now.unix_secs() || age <= DREAM_FRESH_HOURS * 3600
        });
    match run.status.as_str() {
        "failed" if within => DreamFreshness::Failed,
        "completed" | "reverted" | "revert_partial" if within => DreamFreshness::Fresh,
        _ => DreamFreshness::Stale,
    }
}

#[cfg(test)]
mod tests;
