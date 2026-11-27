use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rusqlite::{Connection, OpenFlags};

use super::*;

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

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("cortex-store-{nanos}-{n}"));
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

fn fingerprint(path: &Path) -> (u64, usize) {
    let bytes = std::fs::read(path).unwrap();
    let mut hash = 0xcbf29ce484222325u64;
    for byte in &bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash, bytes.len())
}

fn catalog(path: &Path) -> Vec<(String, String, Option<String>)> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let mut statement = conn
        .prepare("SELECT type, name, sql FROM sqlite_master ORDER BY type, name")
        .unwrap();
    statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .unwrap()
        .map(|row| row.unwrap())
        .collect()
}

fn schema_row(path: &Path) -> Option<String> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    conn.query_row(
        "SELECT value FROM meta WHERE key = 'schema_version'",
        [],
        |row| row.get(0),
    )
    .ok()
}

fn create_required(path: &Path, schema_version: Option<&str>) {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(REQUIRED_SQL).unwrap();
    if let Some(version) = schema_version {
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?1)",
            [version],
        )
        .unwrap();
    }
}

#[test]
fn missing_file_is_unreadable_and_creates_nothing() {
    let scratch = Scratch::new();
    let missing = scratch.path().join("absent").join("cortex.db");
    let err = Store::open(&missing).unwrap_err();
    assert!(err.is_unreadable(), "{err}");
    assert!(err.to_string().contains("unreadable"), "{err}");
    assert!(!missing.exists());
    assert!(!missing.parent().unwrap().exists());

    let garbage = scratch.path().join("notes.txt");
    std::fs::write(&garbage, b"not a database").unwrap();
    let err = Store::open(&garbage).unwrap_err();
    assert!(err.is_unreadable(), "{err}");
    assert_eq!(std::fs::read(&garbage).unwrap(), b"not a database");

    let err = Store::open(scratch.path()).unwrap_err();
    assert!(err.is_unreadable(), "{err}");
}

#[test]
fn readonly_uri_is_mode_ro_and_does_not_offer_read_write() {
    let uri = crate::open::readonly_uri(Path::new("/tmp/cortex memory.db"));
    assert_eq!(uri, "file:/tmp/cortex%20memory.db?mode=ro");
    assert!(!uri.contains("mode=rw"));
}

#[test]
fn missing_required_table_is_an_error_and_creates_nothing() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
         INSERT INTO meta(key, value) VALUES('schema_version', '34');",
    )
    .unwrap();
    drop(conn);
    let before = fingerprint(&path);
    let before_catalog = catalog(&path);
    let err = Store::open(&path).unwrap_err();
    match err {
        StoreError::NotACortexDatabase { missing } => {
            assert_eq!(
                missing,
                vec![
                    "memories".to_string(),
                    "usage_records".to_string(),
                    "access_log".to_string(),
                    "sleep_runs".to_string(),
                ]
            );
        }
        other => panic!("expected not-a-cortex-database, got {other:?}"),
    }
    assert_eq!(fingerprint(&path), before);
    assert_eq!(catalog(&path), before_catalog);
    assert_eq!(schema_row(&path).as_deref(), Some("34"));
}

#[test]
fn reads_leave_schema_version_and_tables_unchanged() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_required(&path, Some("34"));
    let now = UtcTime::from_unix_secs(1_700_000_000);
    let inside = now.saturating_sub_secs(24 * 60 * 60).to_iso();
    let outside = now.saturating_sub_secs(8 * 24 * 60 * 60).to_iso();
    let cutoff = now.saturating_sub_secs(7 * 24 * 60 * 60).to_iso();
    let just_before = now.saturating_sub_secs(7 * 24 * 60 * 60 + 1).to_iso();
    {
        let conn = Connection::open(&path).unwrap();
        for (id, state, content) in [
            ("m-active", "active", "deploy health check"),
            ("m-cold", "cold", "cold note"),
            ("m-arch", "archived", "archived note"),
            ("m-quar", "quarantined", "quarantine"),
            ("m-tomb", "tombstoned", "tombstone"),
        ] {
            conn.execute(
                "INSERT INTO memories(id, kind, content, state, created_at) VALUES(?1,'semantic',?2,?3,?4)",
                (id, content, state, &inside),
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO access_log(memory_id, event, created_at) VALUES('m-active','retrieved',?1)",
            [&inside],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO access_log(memory_id, event, created_at) VALUES('m-active','selected',?1)",
            [&inside],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO access_log(memory_id, event, created_at) VALUES('m-arch','retrieved',?1)",
            [&cutoff],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO access_log(memory_id, event, created_at) VALUES('m-cold','injected',?1)",
            [&outside],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO access_log(memory_id, event, created_at) VALUES('m-cold','retrieved',?1)",
            [&just_before],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO access_log(memory_id, event, created_at) VALUES('m-active','used',?1)",
            [&inside],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO usage_records(usage_id, task_id, memory_id, created_at, used, outcome)
             VALUES('u1','task-new','m-active','2023-11-14T22:00:00+00:00',1,'helpful')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO usage_records(usage_id, task_id, memory_id, created_at, used, outcome)
             VALUES('u2','task-new','m-cold','2023-11-14T22:05:00+00:00',0,'pending')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO usage_records(usage_id, task_id, memory_id, created_at, used, outcome)
             VALUES('u3','task-mid','m-active','2023-11-10T00:00:00+00:00',1,'pending')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO usage_records(usage_id, task_id, memory_id, created_at, used, outcome)
             VALUES('u4','task-old','m-arch','2023-11-01T00:00:00+00:00',1,'corrected')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sleep_runs(run_id, mode, status, cutoff_at, started_at, completed_at, applied_changes, episodes_replayed)
             VALUES('run-old','shadow','completed','2023-11-01T00:00:00+00:00','2023-11-13T03:00:00+00:00','2023-11-13T03:10:00+00:00',3,4)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sleep_runs(run_id, mode, status, cutoff_at, started_at, applied_changes, episodes_replayed)
             VALUES('run-new','apply','running','2023-11-14T00:00:00+00:00','2023-11-14T03:04:00+00:00',0,0)",
            [],
        )
        .unwrap();
    }

    let before = fingerprint(&path);
    let before_catalog = catalog(&path);
    let store = Store::open(&path).unwrap();
    assert!(store.is_query_only().unwrap());
    assert_eq!(
        store.busy_timeout_ms().unwrap(),
        i64::try_from(BUSY_TIMEOUT_MS).unwrap()
    );
    assert_eq!(store.schema_version().unwrap().as_deref(), Some("34"));
    assert_eq!(store.journal_mode().unwrap(), "delete");

    let counts = store.memory_counts().unwrap();
    assert_eq!(counts.active, 1);
    assert_eq!(counts.cold, 1);
    assert_eq!(counts.archived, 1);
    assert_eq!(counts.other, 2);
    assert_eq!(counts.total(), 5);
    assert_eq!(counts.note(), None);

    assert_eq!(store.recalled_7d(now).unwrap(), Measurement::Count(2));
    assert_eq!(store.graded_answers().unwrap(), Measurement::Count(2));
    assert_eq!(
        store.procedures().unwrap(),
        Procedures {
            total: Measurement::NotPresent,
            graded: Measurement::NotPresent,
        }
    );

    let latest = store.latest_sleep_run().unwrap().unwrap();
    assert_eq!(latest.run_id, "run-new");
    assert_eq!(latest.status, "running");
    assert_eq!(latest.finished_applied_changes(), None);
    assert_eq!(latest.at(), "2023-11-14T03:04:00+00:00");
    assert_eq!(stored_hhmm(latest.at()), Some("03:04"));
    assert_eq!(dream_freshness(Some(&latest), now), DreamFreshness::Running);
    let finished = store.sleep_run("run-old").unwrap().unwrap();
    assert_eq!(finished.finished_applied_changes(), Some(3));
    assert_eq!(finished.episodes_replayed, 4);
    assert!(store.sleep_run("missing").unwrap().is_none());

    let children = store.sleep_children("run-new").unwrap();
    assert_eq!(children.episodes, Presence::NotPresent);
    assert_eq!(children.state_changes, Presence::NotPresent);
    assert_eq!(children.proposals, Presence::NotPresent);
    assert_eq!(children.edges, Presence::NotPresent);

    let buckets = store.usage_buckets(2).unwrap();
    assert_eq!(
        buckets
            .iter()
            .map(|bucket| bucket.task_id.as_str())
            .collect::<Vec<_>>(),
        vec!["task-mid", "task-new"]
    );
    assert_eq!(buckets[0].level(), 4);
    assert_eq!(buckets[1].rows, 2);
    assert_eq!(buckets[1].used_rows, 1);
    assert_eq!(buckets[1].level(), 2);
    assert_eq!(grid_levels(&buckets, 4), vec![0, 0, 4, 2]);
    assert!(store.usage_buckets(0).unwrap().is_empty());

    assert_eq!(store.recall_sparkline(now).unwrap(), Presence::NotPresent);
    assert_eq!(
        store.score_breakdown("m-active").unwrap(),
        OptionalRow::NotPresent
    );
    assert!(matches!(
        store
            .lexical_recall("...", DEFAULT_RECALL_LIMIT)
            .unwrap_err(),
        LexicalError::NoTokens
    ));
    assert_eq!(
        store
            .lexical_recall("deploy health", DEFAULT_RECALL_LIMIT)
            .unwrap(),
        Presence::NotPresent
    );

    let write = store.conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', '99')",
        [],
    );
    assert!(write.is_err(), "query_only connection accepted a write");
    let mode = store.conn.execute_batch("PRAGMA journal_mode=WAL");
    assert!(mode.is_err(), "query_only connection changed journal_mode");
    assert_eq!(store.schema_version().unwrap().as_deref(), Some("34"));
    assert_eq!(store.journal_mode().unwrap(), "delete");
    drop(store);

    assert_eq!(
        fingerprint(&path),
        before,
        "a read opened or rewrote the file"
    );
    assert_eq!(catalog(&path), before_catalog, "a read created a table");
    assert_eq!(schema_row(&path).as_deref(), Some("34"));
}

#[test]
fn empty_store_counts_are_zero_and_unmeasured_counts_are_not() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_required(&path, Some("34"));
    let store = Store::open(&path).unwrap();
    let counts = store.memory_counts().unwrap();
    assert!(counts.is_empty());
    assert_eq!(counts.note(), Some(NOTE_EMPTY_STORE));
    assert_eq!(counts.active, 0);
    assert_eq!(
        store
            .recalled_7d(UtcTime::from_unix_secs(1_700_000_000))
            .unwrap(),
        Measurement::NotMeasurable
    );
    assert_eq!(store.graded_answers().unwrap(), Measurement::NotMeasurable);
    assert!(store.latest_sleep_run().unwrap().is_none());
    assert_eq!(
        dream_freshness(None, UtcTime::now()),
        DreamFreshness::NoData
    );
    assert_eq!(NOTE_NO_DREAM, "no dream has run yet");
}

#[test]
fn present_optional_tables_with_no_rows_are_not_a_missing_table() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_required(&path, Some("34"));
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE procedure_outcomes (
                procedure_id TEXT NOT NULL, outcome TEXT NOT NULL, at TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '');
             CREATE TABLE recall_runs (recall_id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
             CREATE VIRTUAL TABLE memory_fts USING fts5(
                memory_id UNINDEXED, content, tokenize='porter unicode61');",
        )
        .unwrap();
    }
    let store = Store::open(&path).unwrap();
    let procedures = store.procedures().unwrap();
    assert_eq!(procedures.total, Measurement::NotMeasurable);
    assert_eq!(procedures.graded, Measurement::NotMeasurable);
    match store
        .recall_sparkline(UtcTime::from_unix_secs(1_700_000_000))
        .unwrap()
    {
        Presence::Present(sparkline) => assert!(sparkline.all_empty()),
        Presence::NotPresent => panic!("an empty recall_runs table is present"),
    }
    assert_eq!(
        store
            .lexical_recall("deploy", DEFAULT_RECALL_LIMIT)
            .unwrap(),
        Presence::Present(Vec::new())
    );
}

#[test]
fn absent_schema_row_is_reported_and_not_written() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_required(&path, None);
    let before = fingerprint(&path);
    let store = Store::open(&path).unwrap();
    assert_eq!(store.schema_version().unwrap(), None);
    drop(store);
    assert_eq!(fingerprint(&path), before);
    assert_eq!(schema_row(&path), None);
}

#[test]
fn optional_tables_return_measured_rows_without_writing() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_required(&path, Some("34"));
    let now = UtcTime::parse_stored("2026-11-27T18:00:30+00:00").unwrap();
    let in_bucket = "2026-11-27T18:00:10+00:00";
    let oldest = "2026-11-27T17:01:00+00:00";
    let too_old = "2026-11-27T17:00:59+00:00";
    let too_new = "2026-11-27T18:01:00+00:00";
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute(
            "INSERT INTO memories(id, kind, content, state, created_at)
             VALUES('m1','semantic',?1,'active','2026-11-27T18:00:00+00:00')",
            ["deploy café — naïve health"],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memories(id, kind, content, state, created_at)
             VALUES('m2','semantic','deploy secret','archived','2026-11-27T18:00:00+00:00')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memories(id, kind, content, state, created_at)
             VALUES('m3','procedure','other deploy','cold','2026-11-27T18:00:00+00:00')",
            [],
        )
        .unwrap();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE memory_fts USING fts5(
                memory_id UNINDEXED, content, tokenize='porter unicode61');
             CREATE TABLE procedure_outcomes (
                procedure_id TEXT NOT NULL, outcome TEXT NOT NULL, at TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '');
             CREATE TABLE recall_runs (recall_id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
             CREATE TABLE recall_score_breakdown (
                task_id TEXT NOT NULL, rank INTEGER NOT NULL, memory_id TEXT NOT NULL,
                score REAL NOT NULL, components_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL);
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
        conn.execute(
            "INSERT INTO memory_fts(memory_id, content) VALUES('m1', ?1)",
            ["deploy café — naïve health"],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memory_fts(memory_id, content) VALUES('m2', 'deploy secret')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memory_fts(memory_id, content) VALUES('m3', 'other deploy')",
            [],
        )
        .unwrap();
        for (id, at) in [("p1", "t1"), ("p1", "t2"), ("p2", "t3")] {
            conn.execute(
                "INSERT INTO procedure_outcomes(procedure_id, outcome, at) VALUES(?1,'success',?2)",
                (id, at),
            )
            .unwrap();
        }
        for (id, at) in [
            ("r-hit", in_bucket),
            ("r-old-minute", oldest),
            ("r-too-old", too_old),
            ("r-too-new", too_new),
            ("r-hit-2", in_bucket),
        ] {
            conn.execute(
                "INSERT INTO recall_runs(recall_id, created_at) VALUES(?1, ?2)",
                (id, at),
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO recall_score_breakdown(task_id, rank, memory_id, score, components_json, created_at)
             VALUES('task-1', 2, 'm1', 0.41, ?1, '2026-11-27T17:00:00+00:00')",
            [r#"{"semantic":0.18,"lexical":0.10,"usage":0.09,"shadow_score":5,"shadow_score_delta":1,"recency":0.06,"trust":0.01,"ignored":"nope"}"#],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sleep_runs(run_id, mode, status, cutoff_at, started_at, episodes_replayed)
             VALUES('run-new','apply','running','2026-11-27T00:00:00+00:00','2026-11-27T03:00:00+00:00',0)",
            [],
        )
        .unwrap();
        for episode in 1..=3 {
            conn.execute(
                "INSERT INTO sleep_episode_state(episode_id, last_run_id) VALUES(?1, 'run-new')",
                [episode],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO sleep_episode_state(episode_id, last_run_id) VALUES(9, 'run-other')",
            [],
        )
        .unwrap();
        for (id, prior, next, reversed) in [
            (1_i64, "active", "cold", None),
            (2, "active", "cold", None),
            (3, "cold", "archived", None),
            (4, "active", "cold", Some("2026-11-27T04:00:00+00:00")),
        ] {
            conn.execute(
                "INSERT INTO sleep_state_changes(change_id, run_id, prior_state, next_state, created_at, reversed_at)
                 VALUES(?1, 'run-new', ?2, ?3, '2026-11-27T03:05:00+00:00', ?4)",
                (id, prior, next, reversed),
            )
            .unwrap();
        }
        for (id, kind) in [
            ("a", "association"),
            ("b", "association_reinforcement"),
            ("c", "edge_downscale"),
        ] {
            conn.execute(
                "INSERT INTO sleep_proposals(proposal_id, run_id, kind, status) VALUES(?1, 'run-new', ?2, 'proposed')",
                (id, kind),
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO sleep_edge_changes(change_id, run_id, reversed_at) VALUES(1, 'run-new', NULL)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sleep_edge_changes(change_id, run_id, reversed_at) VALUES(2, 'run-new', '2026-11-27T04:00:00+00:00')",
            [],
        )
        .unwrap();
    }

    let before = fingerprint(&path);
    let before_catalog = catalog(&path);
    let store = Store::open(&path).unwrap();
    let procedures = store.procedures().unwrap();
    assert_eq!(procedures.total, Measurement::Count(2));
    assert_eq!(procedures.graded, Measurement::Count(2));

    let sparkline = match store.recall_sparkline(now).unwrap() {
        Presence::Present(sparkline) => sparkline,
        Presence::NotPresent => panic!("recall_runs should be present"),
    };
    assert_eq!(sparkline.buckets[0], 1, "oldest minute");
    assert_eq!(sparkline.buckets[59], 2, "current minute");
    assert_eq!(sparkline.total(), 3);
    assert!(!sparkline.all_empty());

    let latest = store.latest_sleep_run().unwrap().unwrap();
    assert_eq!(latest.episodes_replayed, 0);
    assert_eq!(latest.finished_applied_changes(), None);
    let children = store.sleep_children("run-new").unwrap();
    assert_eq!(children.episodes, Presence::Present(3));
    match children.state_changes {
        Presence::Present(changes) => {
            let active_cold = changes
                .iter()
                .find(|row| row.prior_state == "active" && row.next_state == "cold")
                .unwrap();
            assert_eq!(active_cold.n, 2);
            let cold_archived = changes
                .iter()
                .find(|row| row.prior_state == "cold")
                .unwrap();
            assert_eq!(cold_archived.n, 1);
            assert_eq!(changes.len(), 2);
        }
        Presence::NotPresent => panic!("state changes should be present"),
    }
    match children.proposals {
        Presence::Present(kinds) => {
            assert_eq!(kinds.len(), 3);
            assert!(kinds
                .iter()
                .any(|kind| kind.kind == "edge_downscale" && kind.n == 1));
        }
        Presence::NotPresent => panic!("proposals should be present"),
    }
    assert_eq!(children.edges, Presence::Present(1));

    let hits = match store
        .lexical_recall("deploy health", DEFAULT_RECALL_LIMIT)
        .unwrap()
    {
        Presence::Present(hits) => hits,
        Presence::NotPresent => panic!("memory_fts should be present"),
    };
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "m1");
    assert_eq!(hits[0].content, "deploy café — naïve health");
    assert!(hits[0].score.is_finite());
    assert_eq!(hits[0].note, NOTE_LEXICAL_BREAKDOWN);
    assert_eq!(hits[0].why.get("wording match"), Some(&0.28));
    assert_eq!(hits[0].why.get("past usefulness"), Some(&0.09));
    assert_eq!(hits[0].why.get("recency"), Some(&0.06));
    assert_eq!(hits[0].why.get("trust"), Some(&0.01));
    assert!(!hits[0].why.keys().any(|key| key.contains("shadow")));
    assert_eq!(hits[0].why.len(), 4);

    let cold = match store.lexical_recall("other", 6).unwrap() {
        Presence::Present(hits) => hits,
        Presence::NotPresent => panic!("memory_fts should be present"),
    };
    assert_eq!(cold.len(), 1);
    assert_eq!(cold[0].id, "m3");
    assert_eq!(cold[0].note, NOTE_LEXICAL_ONLY);
    assert_eq!(
        cold[0].why.get("wording match").copied(),
        Some(round_2(cold[0].score))
    );

    match store.score_breakdown("m3").unwrap() {
        OptionalRow::Absent => {}
        other => panic!("expected no breakdown row, got {other:?}"),
    }
    assert_eq!(store.schema_version().unwrap().as_deref(), Some("34"));
    drop(store);
    assert_eq!(fingerprint(&path), before);
    assert_eq!(catalog(&path), before_catalog);
    assert_eq!(schema_row(&path).as_deref(), Some("34"));
}

#[test]
fn bundled_sqlite_can_create_memory_fts() {
    let scratch = Scratch::new();
    let path = scratch.path().join("fts.db");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE memory_fts USING fts5(
            memory_id UNINDEXED,
            content,
            tokenize='porter unicode61'
         );
         INSERT INTO memory_fts(memory_id, content) VALUES('m1', 'deploy health check');",
    )
    .unwrap();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memory_fts WHERE memory_fts MATCH '\"deploy\" AND \"health\"'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
    let module: String = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name = 'memory_fts'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(module.contains("fts5"), "{module}");
}

#[test]
fn reader_waits_or_is_unavailable_while_writer_holds_the_lock() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_required(&path, Some("34"));
    let store = Store::open(&path).unwrap();
    let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
    let reader = std::thread::spawn(move || {
        go_rx.recv().unwrap();
        let started = Instant::now();
        let result = store.schema_version();
        (started.elapsed(), result)
    });

    let writer = Connection::open(&path).unwrap();
    writer.busy_timeout(Duration::from_secs(2)).unwrap();
    writer
        .execute_batch("BEGIN EXCLUSIVE; CREATE TABLE writer_probe(n INTEGER); INSERT INTO writer_probe(n) VALUES(1);")
        .unwrap();
    go_tx.send(()).unwrap();
    std::thread::sleep(Duration::from_millis(500));
    writer.execute_batch("COMMIT").unwrap();
    drop(writer);

    let (elapsed, result) = reader.join().unwrap();
    match result {
        Ok(Some(version)) => {
            assert_eq!(version, "34");
            assert!(
                elapsed >= Duration::from_millis(250),
                "reader returned without waiting for the writer: {elapsed:?}"
            );
        }
        Err(err) if err.is_unavailable() => {}
        other => panic!("unexpected reader result: {other:?}"),
    }

    let committed = Connection::open(&path).unwrap();
    let probe: i64 = committed
        .query_row("SELECT n FROM writer_probe", [], |row| row.get(0))
        .unwrap();
    assert_eq!(probe, 1);
    assert_eq!(schema_row(&path).as_deref(), Some("34"));
}

#[test]
fn open_encodes_a_path_that_contains_a_space() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex memory.db");
    create_required(&path, Some("34"));
    let store = Store::open(&path).unwrap();
    assert_eq!(store.schema_version().unwrap().as_deref(), Some("34"));
}

#[test]
fn fts_match_query_wraps_tokens_and_rejects_empty() {
    assert_eq!(
        fts_match_query("deploy health-check").as_deref(),
        Some(r#""deploy" AND "health" AND "check""#)
    );
    assert_eq!(
        fts_match_query("say \"hello\"").as_deref(),
        Some(r#""say" AND "hello""#)
    );
    assert_eq!(fts_match_query("café"), Some("\"café\"".to_string()));
    assert_eq!(fts_match_query(""), None);
    assert_eq!(fts_match_query("... ---"), None);
}

#[test]
fn component_words_sum_collapse_and_drop_shadow() {
    let why = translate_components(
        r#"{"semantic":0.18,"lexical":0.10,"semantic_fusion":0.4,"usage_weight":0.09,"shadow_score":9,"shadow_score_delta":-3,"recency":0.06,"trust":0.01,"foo_bar":0.5,"noise":"skip"}"#,
    );
    assert_eq!(why.get("wording match"), Some(&0.28));
    assert_eq!(why.get("found by meaning"), Some(&0.4));
    assert_eq!(why.get("past usefulness"), Some(&0.09));
    assert_eq!(why.get("foo bar"), Some(&0.5));
    assert_eq!(why.len(), 4);
    assert!(why.get("trust").is_none());
    assert!(why.get("recency").is_none());
    assert!(!why.keys().any(|key| key.contains("shadow")));
    assert!(translate_components("[1,2]").is_empty());
    assert!(translate_components("not json").is_empty());
}

#[test]
fn usage_levels_follow_fraction_thresholds() {
    assert_eq!(usage_level(0, 4), 0);
    assert_eq!(usage_level(1, 4), 1);
    assert_eq!(usage_level(2, 4), 2);
    assert_eq!(usage_level(3, 4), 3);
    assert_eq!(usage_level(4, 4), 4);
    assert_eq!(usage_level(1, 0), 0);
}

#[test]
fn dream_freshness_uses_stored_timestamps_without_shifting_the_clock() {
    let now = UtcTime::parse_stored("2026-11-27T18:49:00+00:00").unwrap();
    assert_eq!(stored_hhmm("2026-11-27T13:49:00-05:00"), Some("13:49"));
    assert_eq!(
        UtcTime::parse_stored("2026-11-27T13:49:00-05:00").unwrap(),
        now
    );
    assert_eq!(
        UtcTime::parse_stored("2026-11-27T18:49:00.123+00:00").unwrap(),
        now
    );
    assert_eq!(dream_freshness(None, now), DreamFreshness::NoData);

    let running = sample_run("running", Some("2020-01-01T00:00:00+00:00"));
    assert_eq!(
        dream_freshness(Some(&running), now),
        DreamFreshness::Running
    );
    assert_eq!(running.at(), running.started_at.as_str());
    assert_eq!(running.finished_applied_changes(), None);

    let fresh = sample_run("completed", Some("2026-11-27T03:00:00+00:00"));
    assert_eq!(dream_freshness(Some(&fresh), now), DreamFreshness::Fresh);
    assert_eq!(fresh.finished_applied_changes(), Some(0));
    assert_eq!(fresh.at(), "2026-11-27T03:00:00+00:00");

    let failed = sample_run("failed", Some("2026-11-27T03:00:00+00:00"));
    assert_eq!(dream_freshness(Some(&failed), now), DreamFreshness::Failed);

    let stale = sample_run("completed", Some("2026-11-25T00:00:00+00:00"));
    assert_eq!(dream_freshness(Some(&stale), now), DreamFreshness::Stale);
    let old_failure = sample_run("failed", Some("2026-11-25T00:00:00+00:00"));
    assert_eq!(
        dream_freshness(Some(&old_failure), now),
        DreamFreshness::Stale
    );
    let reverted = sample_run("revert_partial", Some("2026-11-27T10:00:00+00:00"));
    assert_eq!(dream_freshness(Some(&reverted), now), DreamFreshness::Fresh);
}

#[test]
fn utc_time_round_trips_known_instants() {
    for (unix, iso) in [
        (0, "1970-01-01T00:00:00+00:00"),
        (86_400, "1970-01-02T00:00:00+00:00"),
        (951_782_400, "2000-02-29T00:00:00+00:00"),
        (1_700_000_000, "2023-11-14T22:13:20+00:00"),
        (1_709_208_000, "2024-02-29T12:00:00+00:00"),
    ] {
        let parsed = UtcTime::parse_stored(iso).unwrap();
        assert_eq!(parsed.unix_secs(), unix, "{iso}");
        assert_eq!(UtcTime::from_unix_secs(unix).to_iso(), iso);
    }
    assert!(UtcTime::parse_stored("not-a-time").is_none());
    assert!(UtcTime::parse_stored("2026-02-31T00:00:00+00:00").is_none());
}

fn sample_run(status: &str, completed_at: Option<&str>) -> SleepRun {
    SleepRun {
        run_id: "run".to_string(),
        mode: "apply".to_string(),
        status: status.to_string(),
        applied_changes: 0,
        error: None,
        started_at: "2026-11-27T03:00:00+00:00".to_string(),
        completed_at: completed_at.map(str::to_string),
        episodes_scanned: 0,
        episodes_replayed: 0,
        association_proposals: 0,
        edge_decay_proposals: 0,
    }
}
