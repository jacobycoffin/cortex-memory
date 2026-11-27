//! The `cortex-view` binary: argv, degradation, and a read that does not write.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use cortex_store::{Store, UtcTime};
use cortex_view::{load_dream, load_pulse, render_dream, render_pulse, DreamFrame};
use rusqlite::Connection;

const REQUIRED_SQL: &str = "
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
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

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("cortex-view-{nanos}-{n}"));
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

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_cortex-view"))
}

fn fingerprint(path: &Path) -> (u64, usize) {
    let bytes = std::fs::read(path).unwrap();
    let mut hash = 0xcbf29ce484222325u64;
    for byte in &bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash, bytes.len())
}

fn catalog(path: &Path) -> Vec<(String, String)> {
    let conn =
        Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let mut statement = conn
        .prepare("SELECT type, name FROM sqlite_master ORDER BY type, name")
        .unwrap();
    statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .map(|row| row.unwrap())
        .collect()
}

fn schema_row(path: &Path) -> Option<String> {
    let conn =
        Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    conn.query_row(
        "SELECT value FROM meta WHERE key = 'schema_version'",
        [],
        |row| row.get(0),
    )
    .ok()
}

fn create_required(path: &Path) {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(REQUIRED_SQL).unwrap();
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', '34')",
        [],
    )
    .unwrap();
}

#[test]
fn piped_pulse_has_no_ansi_and_ascii_is_ascii() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_required(&path);

    let output = bin()
        .args(["--db", path.to_str().unwrap(), "pulse", "--once"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!output.stdout.contains(&0x1b), "ANSI on a pipe: {output:?}");
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("cortex  0 mem  "));
    assert!(text.contains("n/a"));
    assert!(text.contains("last dream not run yet  ok"));
    assert!(text.contains('·'));
    assert!(!text.contains("0.0/s"), "{text}");

    let output = bin()
        .args(["--db", path.to_str().unwrap(), "pulse", "--ascii", "--once"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.is_ascii(), "{text}");
    assert!(!text.contains('·'));
}

#[test]
fn missing_store_once_exits_3_and_creates_nothing() {
    let scratch = Scratch::new();
    let missing = scratch.path().join("absent").join("cortex.db");
    let output = bin()
        .args(["--db", missing.to_str().unwrap(), "pulse", "--once"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(3), "{output:?}");
    let text = String::from_utf8(output.stdout).unwrap();
    let err = String::from_utf8(output.stderr).unwrap();
    assert!(text.starts_with("store unavailable:"), "{text}");
    assert!(!text.contains('\u{1b}'), "{text}");
    assert!(!missing.exists());
    assert!(!missing.parent().unwrap().exists());
    assert!(err.is_empty(), "{err}");
}

#[test]
fn bad_usage_exits_2() {
    let output = bin().args(["pulse", "--cols", "0"]).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let err = String::from_utf8(output.stderr).unwrap();
    assert!(err.starts_with("cortex-view:"), "{err}");
}

#[test]
fn dream_without_a_row_says_none_has_run() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_required(&path);
    let output = bin()
        .args(["--db", path.to_str().unwrap(), "dream", "--once"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "no dream has run yet\n"
    );

    let output = bin()
        .args([
            "--db",
            path.to_str().unwrap(),
            "dream",
            "--run-id",
            "missing",
            "--once",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "no dream has run yet\n"
    );
}

#[test]
fn reads_do_not_change_schema_version_or_tables() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_required(&path);
    let now = UtcTime::parse_stored("2026-11-27T05:01:00+00:00").unwrap();
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE recall_runs (recall_id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
             CREATE TABLE sleep_episode_state (
                episode_id INTEGER PRIMARY KEY, last_run_id TEXT NOT NULL,
                replay_count INTEGER NOT NULL DEFAULT 1);
             CREATE TABLE sleep_state_changes (
                change_id INTEGER PRIMARY KEY, run_id TEXT NOT NULL,
                prior_state TEXT NOT NULL, next_state TEXT NOT NULL,
                created_at TEXT NOT NULL, reversed_at TEXT);
             CREATE TABLE sleep_proposals (
                proposal_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL,
                status TEXT NOT NULL, score REAL NOT NULL DEFAULT 0);
             CREATE TABLE sleep_edge_changes (
                change_id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, reversed_at TEXT);",
        )
        .unwrap();
        for (id, state) in [("m1", "active"), ("m2", "cold")] {
            conn.execute(
                "INSERT INTO memories(id, kind, content, state, created_at)
                 VALUES(?1, 'semantic', 'note', ?2, '2026-11-27T04:00:00+00:00')",
                (id, state),
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO usage_records(usage_id, task_id, memory_id, created_at, used)
             VALUES('u1', 'task-spike', 'm1', '2026-11-27T04:30:00+00:00', 1)",
            [],
        )
        .unwrap();
        for id in 0..6 {
            conn.execute(
                "INSERT INTO recall_runs(recall_id, created_at) VALUES(?1, '2026-11-27T05:01:10+00:00')",
                [format!("r{id}")],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO sleep_runs(run_id, mode, status, cutoff_at, started_at, completed_at,
                 episodes_scanned, episodes_replayed, association_proposals, edge_decay_proposals, applied_changes)
             VALUES('run-live', 'apply', 'running', '2026-11-27T00:00:00+00:00',
                 '2026-11-27T04:15:02+00:00', NULL, 41, 0, 50, 60, 0)",
            [],
        )
        .unwrap();
        for episode in 1..=3 {
            conn.execute(
                "INSERT INTO sleep_episode_state(episode_id, last_run_id) VALUES(?1, 'run-live')",
                [episode],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO sleep_episode_state(episode_id, last_run_id) VALUES(9, 'run-other')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sleep_edge_changes(change_id, run_id, reversed_at) VALUES(1, 'run-live', NULL)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sleep_runs(run_id, mode, status, cutoff_at, started_at, completed_at,
                 episodes_scanned, episodes_replayed, applied_changes)
             VALUES('run-shadow', 'shadow', 'completed', '2026-11-27T00:00:00+00:00',
                 '2026-11-27T03:00:00+00:00', '2026-11-27T03:10:00+00:00', 10, 9, 4)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sleep_runs(run_id, mode, status, cutoff_at, started_at, completed_at, applied_changes)
             VALUES('run-4', 'apply', 'completed', '2026-11-27T00:00:00+00:00',
                 '2026-11-27T05:00:00+00:00', '2026-11-27T05:00:00+00:00', 4)",
            [],
        )
        .unwrap();
    }

    let before = fingerprint(&path);
    let before_catalog = catalog(&path);
    let store = Store::open(&path).unwrap();
    assert!(store.is_query_only().unwrap());
    let pulse = load_pulse(&store, 4, 2, now).unwrap();
    let pulse_text = render_pulse(&pulse);
    assert!(pulse_text.contains("cortex  2 mem  "), "{pulse_text}");
    assert!(pulse_text.contains(" 0.1/s  "), "{pulse_text}");
    assert!(
        pulse_text.contains("last dream fresh 05:00  ok"),
        "{pulse_text}"
    );
    assert!(!pulse_text.contains("0.0/s"), "{pulse_text}");
    let grid: Vec<&str> = pulse_text.lines().skip(1).collect();
    assert_eq!(grid.len(), 2);
    assert!(grid[1].ends_with('█'), "{grid:?}");

    let live = load_dream(&store, Some("run-live")).unwrap();
    let live_text = render_dream(&live);
    let episodes = live_text
        .lines()
        .find(|line| line.starts_with("episodes"))
        .unwrap();
    assert!(episodes.contains("3/n/a"), "{episodes}");
    assert!(!episodes.contains('0'), "{episodes}");
    assert!(!live_text.contains("41"), "{live_text}");
    assert!(!live_text.contains("50"), "{live_text}");
    assert!(!live_text.contains("60"), "{live_text}");
    assert!(!live_text.contains("sleep-undo"), "{live_text}");

    let shadow = load_dream(&store, Some("run-shadow")).unwrap();
    let shadow_text = render_dream(&shadow);
    assert!(shadow_text.contains("shadow only"), "{shadow_text}");
    assert!(!shadow_text.contains("sleep-undo"), "{shadow_text}");
    assert!(!shadow_text.contains('9'), "{shadow_text}");

    let apply = load_dream(&store, Some("run-4")).unwrap();
    let apply_text = render_dream(&apply);
    assert!(
        apply_text.contains("4 changes applied · undo: cortex-memory sleep-undo run-4"),
        "{apply_text}"
    );
    assert!(matches!(
        load_dream(&store, Some("nope")).unwrap(),
        DreamFrame::NoDream
    ));
    drop(store);

    assert_eq!(fingerprint(&path), before);
    assert_eq!(catalog(&path), before_catalog);
    assert_eq!(schema_row(&path).as_deref(), Some("34"));

    let output = bin()
        .args([
            "--db",
            path.to_str().unwrap(),
            "dream",
            "--ascii",
            "--once",
            "--run-id",
            "run-4",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.is_ascii(), "{text}");
    assert!(
        text.contains("undo: cortex-memory sleep-undo run-4"),
        "{text}"
    );
    assert_eq!(schema_row(&path).as_deref(), Some("34"));
    assert_eq!(fingerprint(&path), before);
}

#[test]
fn follow_on_a_finished_dream_exits_and_ctrl_c_exits_0() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_required(&path);
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute(
            "INSERT INTO sleep_runs(run_id, mode, status, cutoff_at, started_at, completed_at, applied_changes)
             VALUES('run-4', 'apply', 'completed', '2026-11-27T00:00:00+00:00',
                 '2026-11-27T05:00:00+00:00', '2026-11-27T05:00:00+00:00', 4)",
            [],
        )
        .unwrap();
    }
    let output = bin()
        .args([
            "--db",
            path.to_str().unwrap(),
            "dream",
            "--follow",
            "--run-id",
            "run-4",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("sleep-undo run-4"), "{text}");

    let mut child = bin()
        .args(["--db", path.to_str().unwrap(), "pulse", "--interval", "30"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 128];
        loop {
            match stdout.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => {
                    buf.extend_from_slice(&tmp[..n]);
                    if buf.windows(6).any(|window| window == b"cortex") {
                        let _ = tx.send(());
                    }
                }
                Err(_) => break,
            }
        }
    });
    let pid = child.id();
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done_flag = std::sync::Arc::clone(&done);
    std::thread::spawn(move || {
        for _ in 0..150 {
            if done_flag.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    });
    assert!(
        rx.recv_timeout(Duration::from_secs(10)).is_ok(),
        "pulse did not paint before the signal"
    );
    let kill = Command::new("kill")
        .args(["-INT", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(kill.success(), "kill -INT {pid}");
    let status = child.wait().unwrap();
    done.store(true, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(status.code(), Some(0), "Ctrl-C should exit 0, got {status}");
}
