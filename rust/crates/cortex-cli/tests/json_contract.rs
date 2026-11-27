//! Subprocess coverage for the headless JSON contract.
//!
//! Each test runs the `cortex-cli` binary. The database fixtures are created
//! here; the binary only reads them, except `remember`, which execs a fake
//! `cortex-memory` and must not create a database itself.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use cortex_cli::{EXIT_NO_MATCH, EXIT_OK, EXIT_UNAVAILABLE, EXIT_USAGE};
use cortex_store::{
    UtcTime, NOTE_EMPTY_STORE, NOTE_LEXICAL_BREAKDOWN, NOTE_LEXICAL_ONLY, NOTE_NEXT_SCHEDULED,
    NOTE_NOT_MEASURABLE, NOTE_NO_DREAM, NOTE_NO_MATCH,
};
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde_json::Value;

const REQUIRED_SQL: &str = "
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'semantic',
    content TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    used_count INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.6,
    source_type TEXT NOT NULL DEFAULT 'conversation'
);
CREATE TABLE usage_records (
    usage_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    attribution REAL NOT NULL DEFAULT 0.0,
    outcome TEXT NOT NULL DEFAULT 'pending',
    score REAL
);
CREATE TABLE access_log (
    id INTEGER PRIMARY KEY,
    memory_id TEXT NOT NULL,
    event TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE sleep_runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    cutoff_at TEXT NOT NULL,
    episodes_scanned INTEGER NOT NULL DEFAULT 0,
    episodes_replayed INTEGER NOT NULL DEFAULT 0,
    association_proposals INTEGER NOT NULL DEFAULT 0,
    interference_proposals INTEGER NOT NULL DEFAULT 0,
    edge_decay_proposals INTEGER NOT NULL DEFAULT 0,
    lifecycle_candidates INTEGER NOT NULL DEFAULT 0,
    consolidation_candidates INTEGER NOT NULL DEFAULT 0,
    dependency_candidates INTEGER NOT NULL DEFAULT 0,
    applied_changes INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    report_json TEXT NOT NULL DEFAULT '{}'
);
";

const FTS_SQL: &str = "
CREATE VIRTUAL TABLE memory_fts USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize='porter unicode61'
);
";

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("cortex-cli-{nanos}-{n}"));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Run {
    code: i32,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn cortex_cli_bin() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_cortex_cli") {
        return PathBuf::from(path);
    }
    // This Cargo does not pass CARGO_BIN_EXE_* into the integration-test
    // crate. The workspace target dir is `rust/target`.
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(profile)
        .join("cortex-cli")
}

fn invoke(args: &[&str], env: &[(&str, Option<&str>)]) -> Run {
    let bin = cortex_cli_bin();
    assert!(
        bin.is_file(),
        "cortex-cli binary is missing at {}",
        bin.display()
    );
    let mut cmd = Command::new(bin);
    cmd.args(args);
    for (key, value) in env {
        match value {
            Some(value) => {
                cmd.env(key, value);
            }
            None => {
                cmd.env_remove(key);
            }
        }
    }
    let output = cmd.output().expect("spawn cortex-cli");
    Run {
        code: output.status.code().expect("cortex-cli exited by signal"),
        stdout: output.stdout,
        stderr: output.stderr,
    }
}

fn db_args(path: &Path, rest: &[&str]) -> Vec<String> {
    let mut args = vec!["--db".to_string(), path.display().to_string()];
    args.extend(rest.iter().map(|arg| (*arg).to_string()));
    args
}

fn invoke_db(path: &Path, rest: &[&str]) -> Run {
    let args = db_args(path, rest);
    let borrowed: Vec<&str> = args.iter().map(String::as_str).collect();
    invoke(&borrowed, &[])
}

fn parse_json(run: &Run) -> Value {
    serde_json::from_slice(&run.stdout).unwrap_or_else(|err| {
        panic!(
            "stdout is not one JSON object ({err}): {}\nstderr: {}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        )
    })
}

fn assert_no_ansi(bytes: &[u8]) {
    assert!(
        !bytes.contains(&0x1b),
        "ANSI byte in output: {}",
        String::from_utf8_lossy(bytes)
    );
}

/// Bytes outside JSON strings are ASCII. Non-ASCII memory text stays inside strings.
fn assert_structural_ascii(bytes: &[u8]) {
    let mut in_string = false;
    let mut escaped = false;
    for &byte in bytes {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if byte == b'\\' {
                escaped = true;
                continue;
            }
            if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        assert!(
            byte.is_ascii(),
            "non-ASCII byte {byte:#x} outside a JSON string"
        );
        if byte == b'"' {
            in_string = true;
        }
    }
    assert!(!in_string, "unterminated JSON string");
}

fn assert_contract_stdout(run: &Run) {
    assert_no_ansi(&run.stdout);
    assert_structural_ascii(&run.stdout);
    assert!(run.stdout.ends_with(b"\n"), "payload ends with one newline");
    assert_eq!(run.stdout.iter().filter(|byte| **byte == b'\n').count(), 1);
}

fn schema_version(path: &Path) -> Option<String> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    conn.query_row(
        "SELECT value FROM meta WHERE key = 'schema_version'",
        [],
        |row| row.get(0),
    )
    .optional()
    .unwrap()
}

fn journal_mode(path: &Path) -> String {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .unwrap()
}

fn create_required(path: &Path, schema_version: Option<&str>) -> Connection {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(REQUIRED_SQL).unwrap();
    if let Some(version) = schema_version {
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?1)",
            [version],
        )
        .unwrap();
    }
    conn
}

fn insert_memory(conn: &Connection, id: &str, kind: &str, content: &str, state: &str) {
    conn.execute(
        "INSERT INTO memories(id, kind, content, state, created_at)
         VALUES(?1, ?2, ?3, ?4, '2026-11-27T12:00:00+00:00')",
        (id, kind, content, state),
    )
    .unwrap();
}

fn memories_by_id(body: &Value) -> BTreeMap<String, Value> {
    body["memories"]
        .as_array()
        .expect("memories array")
        .iter()
        .map(|hit| (hit["id"].as_str().unwrap().to_string(), hit.clone()))
        .collect()
}

#[test]
fn recall_returns_lexical_hits_and_keeps_non_ascii_inside_strings() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    let content = "deploy café — naïve 東京 health";
    {
        let conn = create_required(&path, Some("34"));
        insert_memory(&conn, "m-active", "semantic", content, "active");
        insert_memory(&conn, "m-cold", "procedure", "other deploy", "cold");
        insert_memory(&conn, "m-arch", "semantic", "deploy secret", "archived");
        insert_memory(
            &conn,
            "m-quar",
            "semantic",
            "deploy quarantine",
            "quarantined",
        );
        conn.execute_batch(FTS_SQL).unwrap();
        for (id, text) in [
            ("m-active", content),
            ("m-cold", "other deploy"),
            ("m-arch", "deploy secret"),
            ("m-quar", "deploy quarantine"),
        ] {
            conn.execute(
                "INSERT INTO memory_fts(memory_id, content) VALUES(?1, ?2)",
                (id, text),
            )
            .unwrap();
        }
        conn.execute_batch(
            "CREATE TABLE recall_score_breakdown (
                task_id TEXT NOT NULL, rank INTEGER NOT NULL, memory_id TEXT NOT NULL,
                score REAL NOT NULL, components_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO recall_score_breakdown(task_id, rank, memory_id, score, components_json, created_at)
             VALUES('task-1', 1, 'm-active', 0.41, ?1, '2026-11-27T12:00:00+00:00')",
            [r#"{"semantic":0.18,"lexical":0.10,"semantic_fusion":0.22,"usage":0.09,"shadow_score":5,"shadow_score_delta":1,"recency":0.06,"trust":0.01}"#],
        )
        .unwrap();
    }

    let run = invoke_db(&path, &["recall", "deploy health", "--json"]);
    assert_eq!(
        run.code,
        EXIT_OK,
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        run.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_contract_stdout(&run);
    assert!(
        run.stdout
            .windows(content.len())
            .any(|window| window == content.as_bytes()),
        "non-ASCII content must survive as UTF-8 inside the JSON string, got {}",
        String::from_utf8_lossy(&run.stdout)
    );

    let body = parse_json(&run);
    assert_eq!(body["schema_version"], 1);
    assert!(body["schema_version"].is_i64());
    assert_eq!(body["query"], "deploy health");
    assert_eq!(body["ranker"], "lexical");
    assert!(body["note"].is_null());
    let hits = memories_by_id(&body);
    assert_eq!(
        hits.len(),
        1,
        "archived and quarantined rows stay out: {hits:?}"
    );
    assert_eq!(body["count"], 1);
    let hit = &hits["m-active"];
    assert_eq!(hit["kind"], "semantic");
    assert_eq!(hit["content"], content);
    assert!(hit["score"].as_f64().unwrap().is_finite());
    assert_eq!(hit["note"], NOTE_LEXICAL_BREAKDOWN);
    assert_eq!(hit["why"]["wording match"], 0.28);
    assert_eq!(hit["why"]["found by meaning"], 0.22);
    assert_eq!(hit["why"]["past usefulness"], 0.09);
    assert_eq!(hit["why"]["recency"], 0.06);
    assert!(hit["why"].get("trust").is_none(), "only four why keys");
    for key in hit["why"].as_object().unwrap().keys() {
        assert!(!key.contains('_'), "{key}");
        assert!(!key.contains("shadow"), "{key}");
        assert_ne!(key, "semantic");
        assert_ne!(key, "lexical");
    }
    assert_eq!(schema_version(&path).as_deref(), Some("34"));
}

#[test]
fn recall_without_a_breakdown_uses_the_lexical_note() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    {
        let conn = create_required(&path, Some("34"));
        insert_memory(&conn, "m-cold", "procedure", "other deploy", "cold");
        conn.execute_batch(FTS_SQL).unwrap();
        conn.execute(
            "INSERT INTO memory_fts(memory_id, content) VALUES('m-cold', 'other deploy')",
            [],
        )
        .unwrap();
    }
    let run = invoke_db(&path, &["recall", "other", "--json", "--limit", "6"]);
    assert_eq!(run.code, EXIT_OK);
    assert_contract_stdout(&run);
    let body = parse_json(&run);
    let hit = &body["memories"][0];
    assert_eq!(hit["id"], "m-cold");
    assert_eq!(hit["note"], NOTE_LEXICAL_ONLY);
    let score = hit["score"].as_f64().unwrap();
    let why = hit["why"]["wording match"].as_f64().unwrap();
    let rounded = (score * 100.0).round() / 100.0;
    assert!((why - rounded).abs() < 1e-9, "why {why} score {score}");
    assert_eq!(hit["why"].as_object().unwrap().len(), 1);
    assert!(body["note"].is_null());
}

#[test]
fn recall_with_no_hits_exits_one_and_says_no_match() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    {
        let conn = create_required(&path, Some("34"));
        insert_memory(&conn, "m1", "semantic", "unrelated note", "active");
        conn.execute_batch(FTS_SQL).unwrap();
        conn.execute(
            "INSERT INTO memory_fts(memory_id, content) VALUES('m1', 'unrelated note')",
            [],
        )
        .unwrap();
    }
    let run = invoke_db(&path, &["recall", "deploy", "--json"]);
    assert_eq!(run.code, EXIT_NO_MATCH);
    assert_contract_stdout(&run);
    let body = parse_json(&run);
    assert_eq!(body["schema_version"], 1);
    assert_eq!(body["query"], "deploy");
    assert_eq!(body["ranker"], "lexical");
    assert_eq!(body["memories"], serde_json::json!([]));
    assert_eq!(body["count"], 0);
    assert_eq!(body["note"], NOTE_NO_MATCH);
}

#[test]
fn recall_without_fts_tokens_exits_two() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    {
        let conn = create_required(&path, Some("34"));
        conn.execute_batch(FTS_SQL).unwrap();
    }
    let run = invoke_db(&path, &["recall", "...", "--json"]);
    assert_eq!(run.code, EXIT_USAGE);
    assert_contract_stdout(&run);
    let body = parse_json(&run);
    assert_eq!(body["schema_version"], 1);
    assert_eq!(body["query"], "...");
    assert_eq!(body["ranker"], "lexical");
    assert!(body["memories"].is_null());
    assert!(body["count"].is_null());
    assert_eq!(body["note"], "no fts tokens");
    assert_eq!(schema_version(&path).as_deref(), Some("34"));
}

#[test]
fn recall_limit_defaults_to_six_and_rejects_a_non_positive_limit() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    {
        let conn = create_required(&path, Some("34"));
        conn.execute_batch(FTS_SQL).unwrap();
        for n in 0..7 {
            let id = format!("m{n}");
            let content = format!("tokenword number {n}");
            insert_memory(&conn, &id, "semantic", &content, "active");
            conn.execute(
                "INSERT INTO memory_fts(memory_id, content) VALUES(?1, ?2)",
                (id, content),
            )
            .unwrap();
        }
    }
    let defaulted = invoke_db(&path, &["recall", "tokenword", "--json"]);
    assert_eq!(defaulted.code, EXIT_OK);
    let body = parse_json(&defaulted);
    assert_eq!(body["count"], 6);
    assert_eq!(body["memories"].as_array().unwrap().len(), 6);

    let limited = invoke_db(&path, &["recall", "--limit", "2", "tokenword", "--json"]);
    assert_eq!(limited.code, EXIT_OK);
    let body = parse_json(&limited);
    assert_eq!(body["count"], 2);
    assert_eq!(body["memories"].as_array().unwrap().len(), 2);

    for args in [
        ["recall", "tokenword", "--json", "--limit", "0"],
        ["recall", "tokenword", "--json", "--limit", "-3"],
        ["recall", "tokenword", "--json", "--limit", "no"],
    ] {
        let run = invoke_db(&path, &args);
        assert_eq!(run.code, EXIT_USAGE, "{args:?}");
        assert!(run.stdout.is_empty(), "{args:?}");
        assert_no_ansi(&run.stderr);
    }
}

#[test]
fn recall_of_a_missing_fts_table_exits_three_without_a_zero_count() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_required(&path, Some("34"));
    let run = invoke_db(&path, &["recall", "deploy", "--json"]);
    assert_eq!(run.code, EXIT_UNAVAILABLE);
    assert_contract_stdout(&run);
    let body = parse_json(&run);
    assert_eq!(body["schema_version"], 1);
    assert!(body["memories"].is_null());
    assert!(body["count"].is_null());
    assert_eq!(body["note"], "memory_fts is missing");
    assert_eq!(schema_version(&path).as_deref(), Some("34"));
}

#[test]
fn missing_database_exits_three_and_creates_nothing() {
    let scratch = Scratch::new();
    let missing = scratch.path().join("absent").join("cortex.db");
    for command in [
        vec!["recall", "deploy", "--json"],
        vec!["status", "--json"],
        vec!["sleep-status", "--json"],
    ] {
        let mut args = vec!["--db".to_string(), missing.display().to_string()];
        args.extend(command.iter().map(|arg| (*arg).to_string()));
        let borrowed: Vec<&str> = args.iter().map(String::as_str).collect();
        let run = invoke(&borrowed, &[]);
        assert_eq!(run.code, EXIT_UNAVAILABLE, "{command:?}");
        assert_contract_stdout(&run);
        let body = parse_json(&run);
        assert_eq!(body["schema_version"], 1);
        assert!(!missing.exists());
        assert!(!missing.parent().unwrap().exists());
    }

    let garbage = scratch.path().join("notes.txt");
    std::fs::write(&garbage, b"not a database").unwrap();
    let run = invoke_db(&garbage, &["status", "--json"]);
    assert_eq!(run.code, EXIT_UNAVAILABLE);
    let body = parse_json(&run);
    assert_eq!(body["health"]["readable"], false);
    assert!(body["memories"]["active"].is_null());
    assert!(body["memories"]["cold"].is_null());
    assert!(body["recalled_7d"].is_null());
    assert!(body["graded"].is_null());
    assert!(body["procedures"]["total"].is_null());
    assert!(body["health"]["store_schema_version"].is_null());
    assert!(body["health"]["journal_mode"].is_null());
    assert_eq!(std::fs::read(&garbage).unwrap(), b"not a database");
}

#[test]
fn status_of_a_measured_store_reports_schema_34_and_leaves_it_there() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    let now = UtcTime::now().to_iso();
    {
        let conn = create_required(&path, Some("34"));
        insert_memory(&conn, "m-active", "semantic", "active fact", "active");
        insert_memory(&conn, "m-cold", "semantic", "cold fact", "cold");
        insert_memory(&conn, "m-arch", "semantic", "archived fact", "archived");
        insert_memory(&conn, "m-quar", "semantic", "quarantine", "quarantined");
        conn.execute(
            "INSERT INTO access_log(memory_id, event, created_at) VALUES('m-active', 'retrieved', ?1)",
            [&now],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO access_log(memory_id, event, created_at) VALUES('m-active', 'selected', ?1)",
            [&now],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO usage_records(usage_id, task_id, memory_id, created_at, used, outcome)
             VALUES('u1', 'task-1', 'm-active', ?1, 1, 'helpful')",
            [&now],
        )
        .unwrap();
        conn.execute_batch(
            "CREATE TABLE procedure_outcomes (
                procedure_id TEXT NOT NULL, outcome TEXT NOT NULL, at TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '');",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO procedure_outcomes(procedure_id, outcome, at) VALUES('p1', 'success', ?1)",
            [&now],
        )
        .unwrap();
        conn.pragma_update(None, "journal_mode", "wal").unwrap();
    }
    assert_eq!(schema_version(&path).as_deref(), Some("34"));

    let run = invoke_db(&path, &["status", "--json"]);
    assert_eq!(
        run.code,
        EXIT_OK,
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(run.stderr.is_empty());
    assert_contract_stdout(&run);
    let body = parse_json(&run);
    assert_eq!(body["schema_version"], 1);
    assert!(body["schema_version"].is_i64());
    assert_eq!(body["memories"]["active"], 1);
    assert_eq!(body["memories"]["cold"], 1);
    assert_eq!(body["memories"]["archived"], 1);
    assert_eq!(body["memories"]["other"], 1);
    assert_eq!(body["recalled_7d"], 1);
    assert_eq!(body["graded"], 1);
    assert_eq!(body["procedures"]["total"], 1);
    assert_eq!(body["procedures"]["graded"], 1);
    assert_eq!(body["health"]["readable"], true);
    assert_eq!(body["health"]["journal_mode"], "wal");
    assert_eq!(body["health"]["store_schema_version"], "34");
    assert!(body["health"]["store_schema_version"].is_string());
    assert!(body["health"]["note"].is_null());
    assert!(body["note"].is_null());

    assert_eq!(schema_version(&path).as_deref(), Some("34"));
    assert_eq!(journal_mode(&path), "wal");
}

#[test]
fn status_of_an_empty_store_uses_zeros_only_for_memory_counts() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_required(&path, Some("34"));
    let run = invoke_db(&path, &["status", "--json"]);
    assert_eq!(run.code, EXIT_OK);
    assert_contract_stdout(&run);
    let body = parse_json(&run);
    assert_eq!(body["memories"]["active"], 0);
    assert_eq!(body["memories"]["cold"], 0);
    assert_eq!(body["memories"]["archived"], 0);
    assert_eq!(body["memories"]["other"], 0);
    assert_eq!(body["note"], NOTE_EMPTY_STORE);
    assert!(body["recalled_7d"].is_null());
    assert!(body["graded"].is_null());
    assert!(body["procedures"]["total"].is_null());
    assert!(body["procedures"]["graded"].is_null());
    assert_eq!(body["health"]["note"], NOTE_NOT_MEASURABLE);
    assert_eq!(body["health"]["readable"], true);
    assert_eq!(body["health"]["store_schema_version"], "34");
    assert_eq!(schema_version(&path).as_deref(), Some("34"));
}

#[test]
fn status_state_zero_is_a_count_and_an_unmeasured_instrument_is_null() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    {
        let conn = create_required(&path, Some("34"));
        insert_memory(&conn, "m1", "semantic", "only active", "active");
    }
    let run = invoke_db(&path, &["status", "--json"]);
    assert_eq!(run.code, EXIT_OK);
    let body = parse_json(&run);
    assert_eq!(body["memories"]["active"], 1);
    assert_eq!(body["memories"]["cold"], 0);
    assert_eq!(body["memories"]["archived"], 0);
    assert_eq!(body["memories"]["other"], 0);
    assert!(body["note"].is_null());
    assert!(body["recalled_7d"].is_null());
    assert!(body["graded"].is_null());
    assert!(body["procedures"]["total"].is_null());
    assert!(body["procedures"]["graded"].is_null());
    assert_eq!(body["health"]["note"], NOTE_NOT_MEASURABLE);
}

#[test]
fn status_does_not_insert_a_missing_schema_row() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_required(&path, None);
    let run = invoke_db(&path, &["status", "--json"]);
    assert_eq!(run.code, EXIT_OK);
    let body = parse_json(&run);
    assert!(body["health"]["store_schema_version"].is_null());
    assert_eq!(body["health"]["readable"], true);
    assert_eq!(schema_version(&path), None);
}

#[test]
fn sleep_status_without_a_dream_returns_null_instead_of_zeros() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_required(&path, Some("34"));
    let run = invoke_db(&path, &["sleep-status", "--json"]);
    assert_eq!(run.code, EXIT_OK);
    assert!(run.stderr.is_empty());
    assert_contract_stdout(&run);
    let text = String::from_utf8(run.stdout.clone()).unwrap();
    assert!(!text.contains("applied_changes"), "{text}");
    assert!(!text.contains(":0") && !text.contains(": 0"), "{text}");
    let body = parse_json(&run);
    assert_eq!(body["schema_version"], 1);
    assert!(body["last_run"].is_null());
    assert!(body["next_scheduled"].is_null());
    assert_eq!(body["note"], NOTE_NO_DREAM);
}

#[test]
fn a_running_dream_reports_applied_changes_as_null() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    let started = "2026-11-27T03:14:00+00:00";
    {
        let conn = create_required(&path, Some("34"));
        conn.execute(
            "INSERT INTO sleep_runs(run_id, mode, status, cutoff_at, started_at, applied_changes, episodes_replayed)
             VALUES('run-live', 'shadow', 'running', '2026-11-27T00:00:00+00:00', ?1, 0, 0)",
            [started],
        )
        .unwrap();
    }
    let stored: i64 = {
        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        conn.query_row(
            "SELECT applied_changes FROM sleep_runs WHERE run_id = 'run-live'",
            [],
            |row| row.get(0),
        )
        .unwrap()
    };
    assert_eq!(
        stored, 0,
        "the insert default is the value the JSON must not copy"
    );

    let run = invoke_db(&path, &["sleep-status", "--json"]);
    assert_eq!(run.code, EXIT_OK);
    assert_contract_stdout(&run);
    let body = parse_json(&run);
    assert_eq!(body["schema_version"], 1);
    assert!(body["next_scheduled"].is_null());
    assert_eq!(body["note"], NOTE_NEXT_SCHEDULED);
    assert_eq!(body["last_run"]["id"], "run-live");
    assert_eq!(body["last_run"]["mode"], "shadow");
    assert_eq!(body["last_run"]["status"], "running");
    assert_eq!(body["last_run"]["at"], started);
    assert!(body["last_run"]["applied_changes"].is_null());
    assert_eq!(schema_version(&path).as_deref(), Some("34"));
}

#[test]
fn a_finished_dream_reports_applied_changes_and_completed_at() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    let started = "2026-11-27T03:00:00+00:00";
    let completed = "2026-11-27T04:05:00+00:00";
    {
        let conn = create_required(&path, Some("34"));
        conn.execute(
            "INSERT INTO sleep_runs(run_id, mode, status, cutoff_at, started_at, completed_at, applied_changes)
             VALUES('run-done', 'apply', 'completed', '2026-11-27T00:00:00+00:00', ?1, ?2, 23)",
            (started, completed),
        )
        .unwrap();
    }
    let run = invoke_db(&path, &["sleep-status", "--json"]);
    assert_eq!(run.code, EXIT_OK);
    assert_contract_stdout(&run);
    let body = parse_json(&run);
    assert!(body["next_scheduled"].is_null());
    assert_eq!(body["note"], NOTE_NEXT_SCHEDULED);
    assert_eq!(body["last_run"]["id"], "run-done");
    assert_eq!(body["last_run"]["mode"], "apply");
    assert_eq!(body["last_run"]["status"], "completed");
    assert_eq!(body["last_run"]["at"], completed);
    assert_eq!(body["last_run"]["applied_changes"], 23);

    let zero = scratch.path().join("zero.db");
    {
        let conn = create_required(&zero, Some("34"));
        conn.execute(
            "INSERT INTO sleep_runs(run_id, mode, status, cutoff_at, started_at, completed_at, applied_changes)
             VALUES('run-zero', 'shadow', 'completed', '2026-11-27T00:00:00+00:00', ?1, ?2, 0)",
            (started, completed),
        )
        .unwrap();
    }
    let run = invoke_db(&zero, &["sleep-status", "--json"]);
    assert_eq!(run.code, EXIT_OK);
    let body = parse_json(&run);
    assert_eq!(body["last_run"]["applied_changes"], 0);
    assert_eq!(body["last_run"]["status"], "completed");
}

#[test]
fn bad_usage_exits_two_with_plain_stderr_and_empty_stdout() {
    let cases: &[&[&str]] = &[
        &[],
        &["status"],
        &["recall", "--json"],
        &["recall", "deploy"],
        &["remember", "--json"],
        &["sleep-status"],
        &["stats", "--json"],
        &["--json", "status"],
        &["status", "--json", "extra"],
        &["--db"],
    ];
    for args in cases {
        let run = invoke(args, &[]);
        assert_eq!(
            run.code,
            EXIT_USAGE,
            "{args:?} stderr={}",
            String::from_utf8_lossy(&run.stderr)
        );
        assert!(run.stdout.is_empty(), "{args:?}");
        assert_no_ansi(&run.stderr);
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(stderr.contains("exit 2"), "{args:?}: {stderr}");
        assert!(!stderr.contains("not implemented"), "{stderr}");
    }
}

#[test]
fn db_flag_beats_cortex_db_which_beats_hermes_home() {
    let scratch = Scratch::new();
    let via_flag = scratch.path().join("flag.db");
    let via_env = scratch.path().join("env.db");
    let via_home = scratch.path().join("home").join("cortex").join("cortex.db");
    std::fs::create_dir_all(via_home.parent().unwrap()).unwrap();
    create_required(&via_flag, Some("34"));
    create_required(&via_env, Some("35"));
    create_required(&via_home, Some("36"));

    let flag = via_flag.display().to_string();
    let env = via_env.display().to_string();
    let home = scratch.path().join("home").display().to_string();
    let run = invoke(
        &["--db", &flag, "status", "--json"],
        &[
            ("CORTEX_DB", Some(env.as_str())),
            ("HERMES_HOME", Some(home.as_str())),
        ],
    );
    assert_eq!(run.code, EXIT_OK);
    assert_eq!(parse_json(&run)["health"]["store_schema_version"], "34");

    let run = invoke(
        &["status", "--json"],
        &[
            ("CORTEX_DB", Some(env.as_str())),
            ("HERMES_HOME", Some(home.as_str())),
        ],
    );
    assert_eq!(run.code, EXIT_OK);
    assert_eq!(parse_json(&run)["health"]["store_schema_version"], "35");

    let run = invoke(
        &["status", "--json"],
        &[("CORTEX_DB", None), ("HERMES_HOME", Some(home.as_str()))],
    );
    assert_eq!(
        run.code,
        EXIT_OK,
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(parse_json(&run)["health"]["store_schema_version"], "36");
    assert_eq!(schema_version(&via_home).as_deref(), Some("36"));
}

#[test]
fn hermes_home_tilde_expands_and_an_empty_cortex_db_does_not_win() {
    let scratch = Scratch::new();
    let home = scratch.path().join("home");
    let db = home.join("alt").join("cortex").join("cortex.db");
    std::fs::create_dir_all(db.parent().unwrap()).unwrap();
    create_required(&db, Some("34"));
    let home_text = home.display().to_string();
    let run = invoke(
        &["status", "--json"],
        &[
            ("HOME", Some(home_text.as_str())),
            ("HERMES_HOME", Some("~/alt")),
            ("CORTEX_DB", Some("")),
        ],
    );
    assert_eq!(
        run.code,
        EXIT_OK,
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(parse_json(&run)["health"]["store_schema_version"], "34");
    assert_eq!(schema_version(&db).as_deref(), Some("34"));
}

const FAKE_WRITER: &str = r#"#!/bin/sh
printf '%s\n' "$@" > "$CORTEX_FAKE_ARGV"
case "$CORTEX_FAKE_MODE" in
fail)
  echo remember-failed >&2
  exit 1
  ;;
badjson)
  echo not-json
  exit 0
  ;;
quarantine)
  printf '%s\n' '{"memory_id":"mem-q","created":false,"quarantined":true}'
  exit 0
  ;;
*)
  printf '%s\n' '{"memory_id":"mem-1","created":true,"quarantined":false}'
  exit 0
  ;;
esac
"#;

fn install_fake(dir: &Path) {
    let path = dir.join("cortex-memory");
    std::fs::write(&path, FAKE_WRITER).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
    }
}

fn remember(scratch: &Scratch, mode: &str, args: &[&str]) -> (Run, PathBuf) {
    let bin = scratch.path().join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    install_fake(&bin);
    let argv_file = scratch.path().join("argv.txt");
    let db = scratch.path().join("does-not-exist.db");
    let mut full = vec!["--db".to_string(), db.display().to_string()];
    full.extend(args.iter().map(|arg| (*arg).to_string()));
    let borrowed: Vec<&str> = full.iter().map(String::as_str).collect();
    let argv_text = argv_file.display().to_string();
    let path_text = bin.display().to_string();
    let run = invoke(
        &borrowed,
        &[
            ("PATH", Some(path_text.as_str())),
            ("CORTEX_FAKE_ARGV", Some(argv_text.as_str())),
            ("CORTEX_FAKE_MODE", Some(mode)),
        ],
    );
    (run, db)
}

#[test]
fn remember_maps_the_python_payload_and_does_not_open_sqlite() {
    let scratch = Scratch::new();
    let text = "say \"hello\" café";
    let (run, db) = remember(&scratch, "ok", &["remember", text, "--json"]);
    assert_eq!(
        run.code,
        EXIT_OK,
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        run.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_contract_stdout(&run);
    let body = parse_json(&run);
    assert_eq!(body["schema_version"], 1);
    assert_eq!(body["memory_id"], "mem-1");
    assert_eq!(body["kind"], "semantic");
    assert_eq!(body["proposal"], false);
    assert_eq!(body["note"], "written by cortex-memory");
    assert!(!db.exists(), "remember must not create the database");

    let argv = std::fs::read_to_string(scratch.path().join("argv.txt")).unwrap();
    assert_eq!(
        argv.lines().collect::<Vec<_>>(),
        vec![
            "--db",
            db.to_str().unwrap(),
            "remember",
            "--",
            text,
            "--kind",
            "semantic",
        ]
    );

    let (run, db) = remember(
        &scratch,
        "quarantine",
        &["remember", "--kind", "procedure", "--json", "keep this"],
    );
    assert_eq!(run.code, EXIT_OK);
    let body = parse_json(&run);
    assert_eq!(body["memory_id"], "mem-q");
    assert_eq!(body["kind"], "procedure");
    assert_eq!(body["proposal"], false);
    assert_eq!(body["note"], "written by cortex-memory");
    assert!(!db.exists());
    let argv = std::fs::read_to_string(scratch.path().join("argv.txt")).unwrap();
    assert_eq!(
        argv.lines().collect::<Vec<_>>(),
        vec![
            "--db",
            db.to_str().unwrap(),
            "remember",
            "--",
            "keep this",
            "--kind",
            "procedure"
        ]
    );
}

#[test]
fn remember_exits_three_when_the_writer_fails_or_is_missing() {
    let scratch = Scratch::new();
    let (run, db) = remember(&scratch, "fail", &["remember", "text", "--json"]);
    assert_eq!(run.code, EXIT_UNAVAILABLE);
    assert_contract_stdout(&run);
    let body = parse_json(&run);
    assert_eq!(body["schema_version"], 1);
    assert!(body["memory_id"].is_null());
    assert_eq!(body["kind"], "semantic");
    assert_eq!(body["proposal"], false);
    assert_eq!(body["note"], "cortex-memory exited 1: remember-failed");
    assert!(!db.exists());

    let (run, _) = remember(&scratch, "badjson", &["remember", "text", "--json"]);
    assert_eq!(run.code, EXIT_UNAVAILABLE);
    let body = parse_json(&run);
    assert!(body["memory_id"].is_null());
    assert_eq!(body["proposal"], false);
    assert_eq!(body["note"], "cortex-memory did not print a memory id");

    let empty = scratch.path().join("empty-path");
    std::fs::create_dir_all(&empty).unwrap();
    let missing_db = scratch.path().join("missing.db");
    let path_text = empty.display().to_string();
    let db_text = missing_db.display().to_string();
    let run = invoke(
        &["--db", &db_text, "remember", "text", "--json"],
        &[("PATH", Some(path_text.as_str()))],
    );
    assert_eq!(run.code, EXIT_UNAVAILABLE);
    assert_contract_stdout(&run);
    let body = parse_json(&run);
    assert!(body["memory_id"].is_null());
    assert_eq!(body["proposal"], false);
    assert_eq!(body["note"], "cortex-memory cannot be run");
    assert!(!missing_db.exists());

    let before = scratch.path().join("argv.txt");
    let _ = std::fs::remove_file(&before);
    let (run, _) = remember(&scratch, "ok", &["remember", "text"]);
    assert_eq!(run.code, EXIT_USAGE);
    assert!(run.stdout.is_empty());
    assert!(
        !scratch.path().join("argv.txt").exists(),
        "missing --json must not spawn cortex-memory"
    );
}
