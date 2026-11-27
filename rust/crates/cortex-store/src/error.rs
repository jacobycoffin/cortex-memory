use std::fmt;

/// A count that must not be shown as a healthy zero.
///
/// `NotPresent` means the optional table is absent. `NotMeasurable` means the
/// query ran and the count was zero. Required tables never produce
/// `NotPresent`; a missing required table is [`StoreError::NotACortexDatabase`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Measurement {
    Count(i64),
    NotMeasurable,
    NotPresent,
}

impl Measurement {
    pub(crate) fn from_required_count(count: i64) -> Self {
        if count > 0 {
            Self::Count(count)
        } else {
            Self::NotMeasurable
        }
    }

    pub(crate) fn from_optional_count(present: bool, count: i64) -> Self {
        if !present {
            Self::NotPresent
        } else if count > 0 {
            Self::Count(count)
        } else {
            Self::NotMeasurable
        }
    }
}

/// An optional table's rows, distinct from an empty result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Presence<T> {
    Present(T),
    NotPresent,
}

/// One optional row, distinct from a missing table and from no matching row.
#[derive(Debug, Clone, PartialEq)]
pub enum OptionalRow<T> {
    NotPresent,
    Absent,
    Found(T),
}

/// Failure to read. This crate has no write success path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// The path is not an existing file, or SQLite refused the read-only open.
    /// The file was not created.
    Unreadable { message: String },
    /// A lock outlasted the busy timeout. Callers keep polling.
    Unavailable { message: String },
    /// `meta`, `memories`, `usage_records`, `access_log`, or `sleep_runs` is missing.
    NotACortexDatabase { missing: Vec<String> },
    /// A read failed for a reason other than a missing optional table.
    Query { message: String },
}

impl StoreError {
    pub fn is_unreadable(&self) -> bool {
        matches!(self, Self::Unreadable { .. })
    }

    pub fn is_unavailable(&self) -> bool {
        matches!(self, Self::Unavailable { .. })
    }
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreadable { message } => write!(f, "unreadable: {message}"),
            Self::Unavailable { message } => write!(f, "unavailable: {message}"),
            Self::NotACortexDatabase { missing } => {
                write!(f, "not a cortex database: missing {}", missing.join(", "))
            }
            Self::Query { message } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for StoreError {}

pub(crate) fn map_open_error(err: rusqlite::Error) -> StoreError {
    if is_busy(&err) {
        StoreError::Unavailable {
            message: err.to_string(),
        }
    } else {
        StoreError::Unreadable {
            message: err.to_string(),
        }
    }
}

pub(crate) fn map_query_error(err: rusqlite::Error) -> StoreError {
    if is_busy(&err) {
        StoreError::Unavailable {
            message: err.to_string(),
        }
    } else if is_unreadable_sqlite(&err) {
        // `PRAGMA query_only` does not touch the file, so a non-database is
        // only reported by the first real read.
        StoreError::Unreadable {
            message: err.to_string(),
        }
    } else if let Some(name) = missing_table(&err) {
        if REQUIRED_TABLES.iter().any(|table| *table == name) {
            StoreError::NotACortexDatabase {
                missing: vec![name],
            }
        } else {
            StoreError::Query {
                message: err.to_string(),
            }
        }
    } else {
        StoreError::Query {
            message: err.to_string(),
        }
    }
}

pub const REQUIRED_TABLES: &[&str] = &[
    "meta",
    "memories",
    "usage_records",
    "access_log",
    "sleep_runs",
];

pub const OPTIONAL_TABLES: &[&str] = &[
    "memory_fts",
    "recall_runs",
    "recall_score_breakdown",
    "procedure_outcomes",
    "sleep_proposals",
    "sleep_state_changes",
    "sleep_edge_changes",
    "sleep_episode_state",
];

fn is_unreadable_sqlite(err: &rusqlite::Error) -> bool {
    match err {
        rusqlite::Error::SqliteFailure(sqlite, _) => matches!(
            sqlite.code,
            rusqlite::ErrorCode::NotADatabase
                | rusqlite::ErrorCode::CannotOpen
                | rusqlite::ErrorCode::PermissionDenied
                | rusqlite::ErrorCode::SystemIoFailure
        ),
        rusqlite::Error::InvalidPath(_) => true,
        _ => false,
    }
}

fn is_busy(err: &rusqlite::Error) -> bool {
    match err {
        rusqlite::Error::SqliteFailure(sqlite, _) => matches!(
            sqlite.code,
            rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
        ),
        _ => false,
    }
}

fn missing_table(err: &rusqlite::Error) -> Option<String> {
    let rusqlite::Error::SqliteFailure(_, Some(message)) = err else {
        return None;
    };
    let (_, name) = message.split_once("no such table: ")?;
    let name = name
        .split(|c: char| c.is_whitespace() || c == ')' || c == ':')
        .next()?
        .trim_matches('"')
        .to_string();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}
