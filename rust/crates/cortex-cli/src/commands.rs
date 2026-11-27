//! The four headless commands. SQLite is opened only through `cortex_store`.
//! `remember` never opens it: the Python CLI is the writer.

use std::io::ErrorKind;
use std::path::Path;
use std::process::Command;

use cortex_store::{
    LexicalError, Measurement, Presence, Store, UtcTime, CONTRACT_SCHEMA_VERSION, LEXICAL_RANKER,
    NOTE_NEXT_SCHEDULED, NOTE_NOT_MEASURABLE, NOTE_NO_DREAM, NOTE_NO_MATCH,
};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};

use crate::args::{Command as Args, Invocation};
use crate::output;
use crate::{EXIT_NO_MATCH, EXIT_OK, EXIT_UNAVAILABLE, EXIT_USAGE};

pub fn dispatch(invocation: Invocation) -> i32 {
    match invocation.command {
        Args::Recall { query, limit } => recall(&invocation.db, &query, limit),
        Args::Status => status(&invocation.db),
        Args::Remember { text, kind } => remember(&invocation.db, &text, &kind),
        Args::SleepStatus => sleep_status(&invocation.db),
    }
}

fn recall(db: &Path, query: &str, limit: i64) -> i32 {
    let store = match Store::open(db) {
        Ok(store) => store,
        Err(err) => {
            return recall_stopped(query, None, None, Some(err.to_string()), EXIT_UNAVAILABLE)
        }
    };
    match store.lexical_recall(query, limit) {
        Err(LexicalError::NoTokens) => recall_stopped(
            query,
            None,
            None,
            Some("no fts tokens".to_string()),
            EXIT_USAGE,
        ),
        Err(LexicalError::Store(err)) => {
            recall_stopped(query, None, None, Some(err.to_string()), EXIT_UNAVAILABLE)
        }
        Ok(Presence::NotPresent) => recall_stopped(
            query,
            None,
            None,
            Some("memory_fts is missing".to_string()),
            EXIT_UNAVAILABLE,
        ),
        Ok(Presence::Present(hits)) if hits.is_empty() => recall_stopped(
            query,
            Some(Vec::new()),
            Some(0),
            Some(NOTE_NO_MATCH.to_string()),
            EXIT_NO_MATCH,
        ),
        Ok(Presence::Present(hits)) => {
            let count = hits.len() as i64;
            let memories = hits.into_iter().map(RecallHit::from).collect();
            recall_stopped(query, Some(memories), Some(count), None, EXIT_OK)
        }
    }
}

fn recall_stopped(
    query: &str,
    memories: Option<Vec<RecallHit>>,
    count: Option<i64>,
    note: Option<String>,
    code: i32,
) -> i32 {
    finish(
        code,
        &RecallOut {
            schema_version: CONTRACT_SCHEMA_VERSION,
            query,
            ranker: LEXICAL_RANKER,
            memories,
            count,
            note,
        },
    )
}

fn status(db: &Path) -> i32 {
    let store = match Store::open(db) {
        Ok(store) => store,
        Err(err) => return finish(EXIT_UNAVAILABLE, &unreadable_status(err.to_string())),
    };
    match load_status(&store) {
        Ok(payload) => finish(EXIT_OK, &payload),
        Err(err) => finish(EXIT_UNAVAILABLE, &unreadable_status(err.to_string())),
    }
}

fn load_status(store: &Store) -> Result<StatusOut, cortex_store::StoreError> {
    let counts = store.memory_counts()?;
    let recalled = measurement(store.recalled_7d(UtcTime::now())?);
    let graded = measurement(store.graded_answers()?);
    let procedures = store.procedures()?;
    let total = measurement(procedures.total);
    let graded_procedures = measurement(procedures.graded);
    let health_note =
        if recalled.is_none() || graded.is_none() || total.is_none() || graded_procedures.is_none()
        {
            Some(NOTE_NOT_MEASURABLE.to_string())
        } else {
            None
        };
    Ok(StatusOut {
        schema_version: CONTRACT_SCHEMA_VERSION,
        memories: MemoryCountsJson {
            active: Some(counts.active),
            cold: Some(counts.cold),
            archived: Some(counts.archived),
            other: Some(counts.other),
        },
        recalled_7d: recalled,
        graded,
        procedures: ProceduresJson {
            total,
            graded: graded_procedures,
        },
        health: HealthJson {
            readable: true,
            journal_mode: Some(store.journal_mode()?),
            store_schema_version: store.schema_version()?,
            note: health_note,
        },
        note: counts.note().map(str::to_string),
    })
}

fn unreadable_status(note: String) -> StatusOut {
    StatusOut {
        schema_version: CONTRACT_SCHEMA_VERSION,
        memories: MemoryCountsJson {
            active: None,
            cold: None,
            archived: None,
            other: None,
        },
        recalled_7d: None,
        graded: None,
        procedures: ProceduresJson {
            total: None,
            graded: None,
        },
        health: HealthJson {
            readable: false,
            journal_mode: None,
            store_schema_version: None,
            note: Some(note.clone()),
        },
        note: Some(note),
    }
}

fn measurement(value: Measurement) -> Option<i64> {
    match value {
        Measurement::Count(count) if count > 0 => Some(count),
        Measurement::Count(_) | Measurement::NotMeasurable | Measurement::NotPresent => None,
    }
}

fn sleep_status(db: &Path) -> i32 {
    let store = match Store::open(db) {
        Ok(store) => store,
        Err(err) => {
            return finish(
                EXIT_UNAVAILABLE,
                &SleepOut {
                    schema_version: CONTRACT_SCHEMA_VERSION,
                    last_run: None,
                    next_scheduled: (),
                    note: err.to_string(),
                },
            )
        }
    };
    match store.latest_sleep_run() {
        Ok(None) => finish(
            EXIT_OK,
            &SleepOut {
                schema_version: CONTRACT_SCHEMA_VERSION,
                last_run: None,
                next_scheduled: (),
                note: NOTE_NO_DREAM.to_string(),
            },
        ),
        Ok(Some(run)) => {
            let at = run.at().to_string();
            // Finished runs keep the stored count, including zero.
            // A running row is still the insert default, so it is null.
            let applied_changes = run.finished_applied_changes();
            finish(
                EXIT_OK,
                &SleepOut {
                    schema_version: CONTRACT_SCHEMA_VERSION,
                    last_run: Some(SleepLast {
                        id: run.run_id,
                        mode: run.mode,
                        at,
                        applied_changes,
                        status: run.status,
                    }),
                    next_scheduled: (),
                    note: NOTE_NEXT_SCHEDULED.to_string(),
                },
            )
        }
        Err(err) => finish(
            EXIT_UNAVAILABLE,
            &SleepOut {
                schema_version: CONTRACT_SCHEMA_VERSION,
                last_run: None,
                next_scheduled: (),
                note: err.to_string(),
            },
        ),
    }
}

fn remember(db: &Path, text: &str, kind: &str) -> i32 {
    // Python is the writer, including its migrate-on-open. This process does
    // not open the database and does not insert into `memories`.
    let output = match Command::new("cortex-memory")
        .arg("--db")
        .arg(db)
        .arg("remember")
        .arg("--")
        .arg(text)
        .arg("--kind")
        .arg(kind)
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            let note = if err.kind() == ErrorKind::NotFound {
                "cortex-memory cannot be run".to_string()
            } else {
                format!("cortex-memory cannot be run: {err}")
            };
            return remember_out(None, kind, note, EXIT_UNAVAILABLE);
        }
    };
    if !output.status.success() {
        return remember_out(None, kind, writer_failure(&output), EXIT_UNAVAILABLE);
    }
    match serde_json::from_slice::<PythonRemember>(&output.stdout) {
        Ok(parsed) => remember_out(
            Some(parsed.memory_id),
            kind,
            "written by cortex-memory".to_string(),
            EXIT_OK,
        ),
        Err(_) => remember_out(
            None,
            kind,
            "cortex-memory did not print a memory id".to_string(),
            EXIT_UNAVAILABLE,
        ),
    }
}

fn writer_failure(output: &std::process::Output) -> String {
    let code = output
        .status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "signal".to_string());
    let stderr = String::from_utf8_lossy(&output.stderr);
    match stderr.lines().find(|line| !line.trim().is_empty()) {
        Some(line) => {
            let line: String = line.trim().chars().take(400).collect();
            format!("cortex-memory exited {code}: {line}")
        }
        None => format!("cortex-memory exited {code}"),
    }
}

fn remember_out(memory_id: Option<String>, kind: &str, note: String, code: i32) -> i32 {
    finish(
        code,
        &RememberOut {
            schema_version: CONTRACT_SCHEMA_VERSION,
            memory_id,
            kind,
            proposal: false,
            note,
        },
    )
}

fn finish(code: i32, value: &impl Serialize) -> i32 {
    if output::emit(value) {
        code
    } else {
        eprintln!("failed to write json");
        EXIT_UNAVAILABLE
    }
}

#[derive(Serialize)]
struct RecallOut<'a> {
    schema_version: i64,
    query: &'a str,
    ranker: &'static str,
    memories: Option<Vec<RecallHit>>,
    count: Option<i64>,
    note: Option<String>,
}

#[derive(Serialize)]
struct RecallHit {
    id: String,
    kind: String,
    content: String,
    #[serde(serialize_with = "serialize_finite_f64")]
    score: f64,
    why: std::collections::BTreeMap<String, f64>,
    note: &'static str,
}

impl From<cortex_store::LexicalHit> for RecallHit {
    fn from(hit: cortex_store::LexicalHit) -> Self {
        Self {
            id: hit.id,
            kind: hit.kind,
            content: hit.content,
            score: hit.score,
            why: hit.why,
            note: hit.note,
        }
    }
}

fn serialize_finite_f64<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if value.is_finite() {
        serializer.serialize_f64(*value)
    } else {
        serializer.serialize_none()
    }
}

#[derive(Serialize)]
struct StatusOut {
    schema_version: i64,
    memories: MemoryCountsJson,
    recalled_7d: Option<i64>,
    graded: Option<i64>,
    procedures: ProceduresJson,
    health: HealthJson,
    note: Option<String>,
}

#[derive(Serialize)]
struct MemoryCountsJson {
    active: Option<i64>,
    cold: Option<i64>,
    archived: Option<i64>,
    other: Option<i64>,
}

#[derive(Serialize)]
struct ProceduresJson {
    total: Option<i64>,
    graded: Option<i64>,
}

#[derive(Serialize)]
struct HealthJson {
    readable: bool,
    journal_mode: Option<String>,
    store_schema_version: Option<String>,
    note: Option<String>,
}

#[derive(Serialize)]
struct SleepOut {
    schema_version: i64,
    last_run: Option<SleepLast>,
    /// There is no next-run column. This field is always JSON null.
    next_scheduled: (),
    note: String,
}

#[derive(Serialize)]
struct SleepLast {
    id: String,
    mode: String,
    at: String,
    applied_changes: Option<i64>,
    status: String,
}

#[derive(Serialize)]
struct RememberOut<'a> {
    schema_version: i64,
    memory_id: Option<String>,
    kind: &'a str,
    proposal: bool,
    note: String,
}

#[derive(Deserialize)]
struct PythonRemember {
    memory_id: String,
}
