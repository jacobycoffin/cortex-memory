//! HTTP acceptance tests. Python's `dashboard_auth` mints one token so the
//! cookie format is checked against the real writer, not a Rust copy of it.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use bytes::Bytes;
use http::header::HOST;
use http::{HeaderMap, Request};
use http_body_util::{BodyExt, Full};
use hyper_util::rt::TokioIo;
use rusqlite::Connection;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::auth::install_password;
use crate::config::ServeConfig;
use crate::start;
use crate::ServerHandle;

const SCHEMA: &str = "
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

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("cortex-http-{}-{n}", std::process::id()));
        std::fs::create_dir_all(path.join("assets").join("vendor")).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn config(dir: &Path) -> ServeConfig {
    ServeConfig {
        bind: SocketAddr::from(([127, 0, 0, 1], 0)),
        db_path: dir.join("cortex.db"),
        asset_dir: dir.join("assets"),
        auth_path: dir.join("dashboard-auth.json"),
        upstream: "http://127.0.0.1:9".to_string(),
        upstream_timeout: Duration::from_secs(5),
        allowed_hosts: std::collections::BTreeSet::new(),
        trusted_proxies: std::collections::BTreeSet::new(),
    }
}

fn init_db(path: &Path) {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(SCHEMA).unwrap();
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', '34')",
        [],
    )
    .unwrap();
    conn.execute_batch(
        "CREATE TRIGGER meta_block_update BEFORE UPDATE ON meta BEGIN SELECT RAISE(ABORT, 'meta write'); END;
         CREATE TRIGGER meta_block_insert BEFORE INSERT ON meta BEGIN SELECT RAISE(ABORT, 'meta write'); END;
         CREATE TRIGGER meta_block_delete BEFORE DELETE ON meta BEGIN SELECT RAISE(ABORT, 'meta write'); END;",
    )
    .unwrap();
    conn.pragma_update(None, "journal_mode", "DELETE").unwrap();
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

async fn boot(config: ServeConfig) -> ServerHandle {
    start(config).await.unwrap()
}

async fn exchange(
    addr: SocketAddr,
    method: &str,
    path: &str,
    host: &str,
    extra: &[(&str, &str)],
    body: &[u8],
) -> (u16, HeaderMap, Bytes) {
    let stream = TcpStream::connect(addr).await.unwrap();
    let io = TokioIo::new(stream);
    let (mut sender, connection) = hyper::client::conn::http1::handshake(io).await.unwrap();
    tokio::spawn(async move {
        let _ = connection.await;
    });
    let mut builder = Request::builder()
        .method(method)
        .uri(path)
        .header(HOST, host)
        .header("connection", "close");
    for (name, value) in extra {
        builder = builder.header(*name, *value);
    }
    let request = builder
        .body(Full::new(Bytes::copy_from_slice(body)))
        .unwrap();
    let response = tokio::time::timeout(Duration::from_secs(30), sender.send_request(request))
        .await
        .expect("request timed out")
        .unwrap();
    let status = response.status().as_u16();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    drop(sender);
    (status, headers, bytes)
}

fn json_of(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).unwrap()
}

fn assert_security(headers: &HeaderMap) {
    assert_eq!(headers["x-content-type-options"], "nosniff");
    assert_eq!(headers["x-frame-options"], "DENY");
    assert_eq!(headers["referrer-policy"], "no-referrer");
    assert_eq!(
        headers["permissions-policy"],
        "camera=(), microphone=(), geolocation=()"
    );
    let csp = headers["content-security-policy"].to_str().unwrap();
    assert!(
        csp.contains("script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net"),
        "{csp}"
    );
    assert!(csp.contains("frame-ancestors 'none'"));
}

fn cookie_token(headers: &HeaderMap) -> String {
    let value = headers["set-cookie"].to_str().unwrap();
    let pair = value.split(';').next().unwrap();
    pair.strip_prefix("cortex_session=").unwrap().to_string()
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap()
}

fn python(script: &str, vars: &[(&str, &str)]) -> std::process::Output {
    let root = repo_root();
    let mut cmd = std::process::Command::new("python3");
    cmd.current_dir(&root)
        .arg("-c")
        .arg(script)
        .env("PYTHONPATH", &root);
    for (key, value) in vars {
        cmd.env(key, value);
    }
    cmd.output().unwrap()
}

#[test]
fn usage_exit_code_is_2() {
    assert_eq!(crate::run(std::iter::once("cortex-http")), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pulse_is_open_without_an_auth_file_and_closed_once_it_appears() {
    let dir = TempDir::new();
    init_db(&dir.path().join("cortex.db"));
    let server = boot(config(dir.path())).await;
    let addr = server.local_addr();

    let (status, headers, body) = exchange(addr, "GET", "/api/pulse", "127.0.0.1", &[], b"").await;
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    assert_security(&headers);
    assert_eq!(headers["cache-control"], "no-store");
    let payload = json_of(&body);
    assert_eq!(payload["schema_version"], 1);
    assert!(payload["schema_version"].is_number());
    assert!(payload.get("store_schema_version").is_none());
    assert!(payload["status"]["recalled_7d"].is_null());
    assert_eq!(payload["note"], "the store is empty");
    assert_eq!(payload["status"]["last_dream"]["state"], "not run yet");
    assert!(payload["status"]["last_dream"]["at"].is_null());
    assert!(payload["status"]["last_dream"]["mode"].is_null());
    assert!(payload["status"]["last_dream"]["run_id"].is_null());
    assert_eq!(payload["buckets"].as_array().unwrap().len(), 240);
    assert!(payload["buckets"]
        .as_array()
        .unwrap()
        .iter()
        .all(|level| level == &json!(0)));
    for key in ["schema_version", "buckets", "status", "note"] {
        assert!(payload.get(key).is_some(), "missing {key}");
    }
    for key in ["memories", "recalled_7d", "last_dream"] {
        assert!(payload["status"].get(key).is_some(), "missing status.{key}");
    }
    for key in ["active", "cold", "archived", "other"] {
        assert!(payload["status"]["memories"].get(key).is_some());
    }

    let (status, _, _) = exchange(addr, "GET", "/api/auth/status", "127.0.0.1", &[], b"").await;
    assert_eq!(status, 200);
    let status_body = json_of(
        &exchange(addr, "GET", "/api/auth/status", "127.0.0.1", &[], b"")
            .await
            .2,
    );
    assert_eq!(status_body["auth_enabled"], false);
    assert_eq!(status_body["authenticated"], true);
    assert_eq!(status_body["username"], "");

    install_password(
        &dir.path().join("dashboard-auth.json"),
        "ada",
        "correct-horse-battery",
        1,
        false,
        1,
    );
    let (status, _, body) = exchange(addr, "GET", "/api/pulse", "127.0.0.1", &[], b"").await;
    assert_eq!(status, 401, "{}", String::from_utf8_lossy(&body));
    assert_eq!(json_of(&body)["error"], "authentication required");

    std::fs::remove_file(dir.path().join("dashboard-auth.json")).unwrap();
    let (status, _, _) = exchange(addr, "GET", "/api/pulse", "127.0.0.1", &[], b"").await;
    assert_eq!(status, 200);
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pulse_reports_counts_levels_dreams_and_null_recall() {
    let dir = TempDir::new();
    let db = dir.path().join("cortex.db");
    init_db(&db);
    let server = boot(config(dir.path())).await;
    let addr = server.local_addr();

    {
        let conn = Connection::open(&db).unwrap();
        for (id, state) in [
            ("a", "active"),
            ("c", "cold"),
            ("r", "archived"),
            ("q", "quarantined"),
        ] {
            conn.execute(
                "INSERT INTO memories(id, content, state, created_at) VALUES(?1, 'text', ?2, '2020-01-01T00:00:00+00:00')",
                [id, state],
            )
            .unwrap();
        }
    }
    let body = json_of(
        &exchange(addr, "GET", "/api/pulse", "127.0.0.1", &[], b"")
            .await
            .2,
    );
    assert_eq!(body["status"]["memories"]["active"], 1);
    assert_eq!(body["status"]["memories"]["cold"], 1);
    assert_eq!(body["status"]["memories"]["archived"], 1);
    assert_eq!(body["status"]["memories"]["other"], 1);
    assert!(body["status"]["recalled_7d"].is_null());
    assert!(body["note"].is_null());

    let clamped = json_of(
        &exchange(
            addr,
            "GET",
            "/api/pulse?cols=0&rows=100",
            "127.0.0.1",
            &[],
            b"",
        )
        .await
        .2,
    );
    assert_eq!(clamped["buckets"].as_array().unwrap().len(), 80);
    let tiny = json_of(
        &exchange(
            addr,
            "GET",
            "/api/pulse?cols=2&rows=2",
            "127.0.0.1",
            &[],
            b"",
        )
        .await
        .2,
    );
    assert_eq!(tiny["buckets"].as_array().unwrap().len(), 4);

    {
        let conn = Connection::open(&db).unwrap();
        let rows = [
            ("t1", "2020-01-01T00:00:00+00:00", 0),
            ("t2", "2020-01-02T00:00:00+00:00", 1),
            ("t2", "2020-01-02T00:00:01+00:00", 0),
            ("t2", "2020-01-02T00:00:02+00:00", 0),
            ("t2", "2020-01-02T00:00:03+00:00", 0),
            ("t3", "2020-01-03T00:00:00+00:00", 1),
            ("t3", "2020-01-03T00:00:01+00:00", 0),
            ("t4", "2020-01-04T00:00:00+00:00", 1),
            ("t4", "2020-01-04T00:00:01+00:00", 1),
            ("t4", "2020-01-04T00:00:02+00:00", 1),
            ("t4", "2020-01-04T00:00:03+00:00", 0),
            ("t5", "2020-01-05T00:00:00+00:00", 1),
        ];
        for (index, (task, at, used)) in rows.iter().enumerate() {
            conn.execute(
                "INSERT INTO usage_records(usage_id, task_id, memory_id, created_at, used) VALUES(?1, ?2, 'a', ?3, ?4)",
                rusqlite::params![format!("u{index}"), task, at, used],
            )
            .unwrap();
        }
    }
    let levels = json_of(
        &exchange(
            addr,
            "GET",
            "/api/pulse?cols=5&rows=1",
            "127.0.0.1",
            &[],
            b"",
        )
        .await
        .2,
    );
    assert_eq!(levels["buckets"], json!([0, 1, 2, 3, 4]));

    {
        let conn = Connection::open(&db).unwrap();
        conn.execute(
            "INSERT INTO access_log(memory_id, event, created_at) VALUES('a', 'noted', '2099-01-01T00:00:00+00:00')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO access_log(memory_id, event, created_at) VALUES('a', 'retrieved', '2001-01-01T00:00:00+00:00')",
            [],
        )
        .unwrap();
    }
    let still_null = json_of(
        &exchange(addr, "GET", "/api/pulse", "127.0.0.1", &[], b"")
            .await
            .2,
    );
    assert!(still_null["status"]["recalled_7d"].is_null());

    let now = cortex_store::UtcTime::now().to_iso();
    {
        let conn = Connection::open(&db).unwrap();
        conn.execute(
            "INSERT INTO access_log(memory_id, event, created_at) VALUES('a', 'retrieved', ?1)",
            [&now],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO access_log(memory_id, event, created_at) VALUES('a', 'selected', ?1)",
            [&now],
        )
        .unwrap();
    }
    let one = json_of(
        &exchange(addr, "GET", "/api/pulse", "127.0.0.1", &[], b"")
            .await
            .2,
    );
    assert_eq!(one["status"]["recalled_7d"], 1);

    let started = "2026-11-27T15:04:00+00:00";
    {
        let conn = Connection::open(&db).unwrap();
        conn.execute(
            "INSERT INTO sleep_runs(run_id, mode, status, cutoff_at, started_at, report_json)
             VALUES('run-1', 'shadow', 'running', ?1, ?1, '{}')",
            [started],
        )
        .unwrap();
    }
    let running = json_of(
        &exchange(addr, "GET", "/api/pulse", "127.0.0.1", &[], b"")
            .await
            .2,
    );
    assert_eq!(running["status"]["last_dream"]["state"], "running");
    assert_eq!(running["status"]["last_dream"]["at"], started);
    assert_eq!(running["status"]["last_dream"]["mode"], "shadow");
    assert_eq!(running["status"]["last_dream"]["run_id"], "run-1");

    let recent = cortex_store::UtcTime::now()
        .saturating_sub_secs(3600)
        .to_iso();
    {
        let conn = Connection::open(&db).unwrap();
        conn.execute(
            "UPDATE sleep_runs SET status='completed', completed_at=?1 WHERE run_id='run-1'",
            [&recent],
        )
        .unwrap();
    }
    let fresh = json_of(
        &exchange(addr, "GET", "/api/pulse", "127.0.0.1", &[], b"")
            .await
            .2,
    );
    assert_eq!(fresh["status"]["last_dream"]["state"], "fresh");
    assert_eq!(fresh["status"]["last_dream"]["at"], recent);

    let old = cortex_store::UtcTime::now()
        .saturating_sub_secs(40 * 3600)
        .to_iso();
    {
        let conn = Connection::open(&db).unwrap();
        conn.execute(
            "UPDATE sleep_runs SET completed_at=?1 WHERE run_id='run-1'",
            [&old],
        )
        .unwrap();
    }
    let stale = json_of(
        &exchange(addr, "GET", "/api/pulse", "127.0.0.1", &[], b"")
            .await
            .2,
    );
    assert_eq!(stale["status"]["last_dream"]["state"], "stale");

    {
        let conn = Connection::open(&db).unwrap();
        conn.execute(
            "UPDATE sleep_runs SET status='failed', completed_at=?1 WHERE run_id='run-1'",
            [&recent],
        )
        .unwrap();
    }
    let failed = json_of(
        &exchange(addr, "GET", "/api/pulse", "127.0.0.1", &[], b"")
            .await
            .2,
    );
    assert_eq!(failed["status"]["last_dream"]["state"], "failed");

    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn empty_store_note_is_not_a_healthy_zero_recall() {
    let dir = TempDir::new();
    init_db(&dir.path().join("cortex.db"));
    let server = boot(config(dir.path())).await;
    let body = json_of(
        &exchange(
            server.local_addr(),
            "GET",
            "/api/pulse",
            "127.0.0.1",
            &[],
            b"",
        )
        .await
        .2,
    );
    assert_eq!(body["status"]["memories"]["active"], 0);
    assert_eq!(body["status"]["memories"]["other"], 0);
    assert!(body["status"]["recalled_7d"].is_null());
    assert_eq!(body["note"], "the store is empty");
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pulse_never_writes_meta_or_assigns_journal_mode() {
    let dir = TempDir::new();
    let db = dir.path().join("cortex.db");
    init_db(&db);
    {
        let conn = Connection::open(&db).unwrap();
        conn.execute(
            "INSERT INTO memories(id, content, state, created_at) VALUES('a', 'text', 'active', '2020-01-01T00:00:00+00:00')",
            [],
        )
        .unwrap();
    }
    let mode_before = {
        let conn =
            Connection::open_with_flags(&db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let mode: String = conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        mode
    };
    assert_eq!(mode_before, "delete");
    let before = fingerprint(&db);
    let mut readonly = std::fs::metadata(&db).unwrap().permissions();
    readonly.set_readonly(true);
    std::fs::set_permissions(&db, readonly).unwrap();

    let server = boot(config(dir.path())).await;
    let (status, _, body) = exchange(
        server.local_addr(),
        "GET",
        "/api/pulse",
        "127.0.0.1",
        &[],
        b"",
    )
    .await;
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    assert_eq!(json_of(&body)["status"]["memories"]["active"], 1);
    server.shutdown().await;

    let mut writable = std::fs::metadata(&db).unwrap().permissions();
    writable.set_readonly(false);
    std::fs::set_permissions(&db, writable).unwrap();
    assert_eq!(fingerprint(&db), before);
    assert!(!PathBuf::from(format!("{}-wal", db.display())).exists());
    assert!(!PathBuf::from(format!("{}-shm", db.display())).exists());
    assert!(!PathBuf::from(format!("{}-journal", db.display())).exists());
    let conn = Connection::open(&db).unwrap();
    let mode: String = conn
        .query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .unwrap();
    let version: String = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(mode, "delete");
    assert_eq!(version, "34");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_database_is_503_and_is_not_created() {
    let dir = TempDir::new();
    let missing = dir.path().join("cortex.db");
    assert!(!missing.exists());
    let server = boot(config(dir.path())).await;
    let (status, headers, body) = exchange(
        server.local_addr(),
        "GET",
        "/api/pulse",
        "127.0.0.1",
        &[],
        b"",
    )
    .await;
    assert_eq!(status, 503);
    assert_security(&headers);
    let payload = json_of(&body);
    assert_eq!(payload["error"], "store unavailable");
    assert_eq!(payload["schema_version"], 1);
    assert!(payload["note"].as_str().unwrap().contains("does not exist"));
    assert!(!missing.exists());
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn static_files_404_until_they_exist_and_reload_on_sighup() {
    let dir = TempDir::new();
    init_db(&dir.path().join("cortex.db"));
    let server = boot(config(dir.path())).await;
    let addr = server.local_addr();
    let (status, headers, body) = exchange(addr, "GET", "/pulse", "127.0.0.1", &[], b"").await;
    assert_eq!(status, 404);
    assert_eq!(&body[..], b"not found");
    assert_eq!(headers["content-type"], "text/plain; charset=utf-8");
    assert_security(&headers);

    let pulse = dir.path().join("assets").join("pulse.html");
    std::fs::write(&pulse, b"version-one").unwrap();
    let (status, _, body) = exchange(addr, "GET", "/pulse", "127.0.0.1", &[], b"").await;
    assert_eq!(status, 200);
    assert_eq!(&body[..], b"version-one");
    assert_eq!(
        exchange(addr, "GET", "/pulse", "127.0.0.1", &[], b"")
            .await
            .1["cache-control"],
        "no-store"
    );

    let css = dir.path().join("assets").join("dashboard.css");
    std::fs::write(&css, b"body{}\n").unwrap();
    let (status, headers, body) =
        exchange(addr, "GET", "/dashboard.css", "127.0.0.1", &[], b"").await;
    assert_eq!(status, 200);
    assert_eq!(&body[..], b"body{}\n");
    assert_eq!(headers["cache-control"], "public, max-age=86400");
    let meta = std::fs::metadata(&css).unwrap();
    let mtime = meta
        .modified()
        .unwrap()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    assert_eq!(
        headers["etag"].to_str().unwrap(),
        format!("\"{mtime}-{}\"", meta.len())
    );

    let vendor = dir.path().join("assets").join("vendor").join("app.js");
    std::fs::write(&vendor, b"var app=1;\n").unwrap();
    let (status, headers, body) =
        exchange(addr, "GET", "/vendor/app.js", "127.0.0.1", &[], b"").await;
    assert_eq!(status, 200);
    assert_eq!(&body[..], b"var app=1;\n");
    assert_eq!(headers["cache-control"], "public, max-age=86400");

    let (status, _, body) = exchange(addr, "GET", "/vendor/nope.txt", "127.0.0.1", &[], b"").await;
    assert_eq!(status, 404);
    assert_eq!(&body[..], b"not found");

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /vendor/%2e%2e/secret.js HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.unwrap();
    let text = String::from_utf8_lossy(&raw);
    assert!(text.starts_with("HTTP/1.1 404"), "{text}");
    assert!(text.contains("not found"));

    let modified = std::fs::metadata(&pulse).unwrap().modified().unwrap();
    std::fs::write(&pulse, b"version-two").unwrap();
    let file = std::fs::File::options().write(true).open(&pulse).unwrap();
    file.set_modified(modified).unwrap();
    drop(file);
    let (_, _, cached) = exchange(addr, "GET", "/pulse", "127.0.0.1", &[], b"").await;
    assert_eq!(&cached[..], b"version-one");
    server.reload();
    let (_, _, fresh) = exchange(addr, "GET", "/pulse", "127.0.0.1", &[], b"").await;
    assert_eq!(&fresh[..], b"version-two");
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auth_matches_python_sessions_and_password_rules() {
    let dir = TempDir::new();
    init_db(&dir.path().join("cortex.db"));
    let auth_path = dir.path().join("dashboard-auth.json");
    install_password(&auth_path, "ada", "correct-horse-battery", 1, false, 1);
    let server = boot(config(dir.path())).await;
    let addr = server.local_addr();

    let login_body = br#"{"username":"ada","password":"correct-horse-battery"}"#;
    let (status, _, body) = exchange(
        addr,
        "POST",
        "/api/auth/login",
        "127.0.0.1",
        &[],
        login_body,
    )
    .await;
    assert_eq!(status, 403, "{}", String::from_utf8_lossy(&body));
    assert_eq!(json_of(&body)["error"], "request verification failed");

    let (status, headers, body) = exchange(
        addr,
        "POST",
        "/api/auth/login",
        "127.0.0.1",
        &[("x-cortex-request", "1")],
        login_body,
    )
    .await;
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    let logged_in = json_of(&body);
    assert_eq!(logged_in["authenticated"], true);
    assert_eq!(logged_in["username"], "ada");
    assert_eq!(logged_in["must_change_password"], false);
    let cookie = headers["set-cookie"].to_str().unwrap().to_string();
    assert!(cookie.contains("Path=/"));
    assert!(cookie.contains("HttpOnly"));
    assert!(cookie.contains("SameSite=Strict"));
    assert!(cookie.contains("Max-Age=43200"));
    assert!(!cookie.contains("Secure"));
    let token = cookie_token(&headers);
    let cookie_header = format!("cortex_session={token}");

    let verified = python(
        "import dashboard_auth, os\nauth = dashboard_auth.DashboardAuth(os.environ['AUTH_PATH'])\n\
         token = os.environ['TOKEN']\n\
         assert auth.verify_session(token) is not None\n\
         assert auth.verify_password('ada', os.environ['PASSWORD'])\n\
         print('ok')",
        &[
            ("AUTH_PATH", auth_path.to_str().unwrap()),
            ("TOKEN", &token),
            ("PASSWORD", "correct-horse-battery"),
        ],
    );
    assert!(
        verified.status.success(),
        "python rejected a Rust token: {}",
        String::from_utf8_lossy(&verified.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&verified.stdout).trim(), "ok");

    let (status, _, _) = exchange(
        addr,
        "GET",
        "/api/pulse",
        "127.0.0.1",
        &[("cookie", &cookie_header)],
        b"",
    )
    .await;
    assert_eq!(status, 200);

    let (status, headers, _) = exchange(
        addr,
        "POST",
        "/api/auth/login",
        "127.0.0.1",
        &[("x-cortex-request", "1"), ("x-forwarded-proto", "https")],
        login_body,
    )
    .await;
    assert_eq!(status, 200);
    assert!(headers["set-cookie"].to_str().unwrap().contains("Secure"));

    let bad = br#"{"username":"ada","password":"nope-nope-nope"}"#;
    let (status, _, body) = exchange(
        addr,
        "POST",
        "/api/auth/login",
        "127.0.0.1",
        &[("x-cortex-request", "1")],
        bad,
    )
    .await;
    assert_eq!(status, 401);
    assert_eq!(
        json_of(&body)["error"],
        "Username or password is incorrect."
    );

    let long_user = format!(
        "{{\"username\":\"{}\",\"password\":\"correct-horse-battery\"}}",
        "a".repeat(129)
    );
    let (status, _, body) = exchange(
        addr,
        "POST",
        "/api/auth/login",
        "127.0.0.1",
        &[("x-cortex-request", "1")],
        long_user.as_bytes(),
    )
    .await;
    assert_eq!(status, 401);
    assert_eq!(
        json_of(&body)["error"],
        "Username or password is incorrect."
    );

    let (status, _, body) = exchange(
        addr,
        "POST",
        "/api/auth/login",
        "127.0.0.1",
        &[("x-cortex-request", "1")],
        b"[]",
    )
    .await;
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"], "invalid JSON request");

    let short = br#"{"current_password":"correct-horse-battery","new_password":"short"}"#;
    let (status, _, body) = exchange(
        addr,
        "POST",
        "/api/auth/change-password",
        "127.0.0.1",
        &[("x-cortex-request", "1"), ("cookie", &cookie_header)],
        short,
    )
    .await;
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"], "Use at least 12 characters.");

    let contains =
        br#"{"current_password":"correct-horse-battery","new_password":"something-about-Ada-x"}"#;
    let (status, _, body) = exchange(
        addr,
        "POST",
        "/api/auth/change-password",
        "127.0.0.1",
        &[("x-cortex-request", "1"), ("cookie", &cookie_header)],
        contains,
    )
    .await;
    assert_eq!(status, 400);
    assert_eq!(
        json_of(&body)["error"],
        "Password cannot contain the username."
    );

    let wrong =
        br#"{"current_password":"wrong-password-here","new_password":"replacement-secret-value"}"#;
    let (status, _, body) = exchange(
        addr,
        "POST",
        "/api/auth/change-password",
        "127.0.0.1",
        &[("x-cortex-request", "1"), ("cookie", &cookie_header)],
        wrong,
    )
    .await;
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"], "Current password is incorrect.");

    let (status, headers, body) = exchange(
        addr,
        "POST",
        "/api/auth/logout",
        "127.0.0.1",
        &[("x-cortex-request", "1"), ("cookie", &cookie_header)],
        b"",
    )
    .await;
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    assert!(headers["set-cookie"]
        .to_str()
        .unwrap()
        .contains("Max-Age=0"));
    let (status, _, _) = exchange(addr, "GET", "/api/pulse", "127.0.0.1", &[], b"").await;
    assert_eq!(status, 401);

    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn python_minted_token_survives_until_rust_changes_the_password() {
    let dir = TempDir::new();
    init_db(&dir.path().join("cortex.db"));
    let auth_path = dir.path().join("dashboard-auth.json");
    let minted = python(
        "import dashboard_auth, os\n\
         auth = dashboard_auth.DashboardAuth(os.environ['AUTH_PATH'])\n\
         auth.ensure(os.environ['AUTH_USER'], os.environ['AUTH_PASSWORD'])\n\
         print(auth.issue_session())",
        &[
            ("AUTH_PATH", auth_path.to_str().unwrap()),
            ("AUTH_USER", "ada"),
            ("AUTH_PASSWORD", "correct-horse-battery"),
        ],
    );
    assert!(
        minted.status.success(),
        "dashboard_auth failed: {}",
        String::from_utf8_lossy(&minted.stderr)
    );
    let old_token = String::from_utf8(minted.stdout).unwrap();
    let old_token = old_token.trim();
    let server = boot(config(dir.path())).await;
    let addr = server.local_addr();
    let old_cookie = format!("cortex_session={old_token}");
    let (status, _, body) = exchange(
        addr,
        "GET",
        "/api/pulse",
        "127.0.0.1",
        &[("cookie", &old_cookie)],
        b"",
    )
    .await;
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));

    let change = br#"{"current_password":"correct-horse-battery","new_password":"replacement-secret-value"}"#;
    let (status, headers, body) = exchange(
        addr,
        "POST",
        "/api/auth/change-password",
        "127.0.0.1",
        &[("x-cortex-request", "1"), ("cookie", &old_cookie)],
        change,
    )
    .await;
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    assert_eq!(json_of(&body)["must_change_password"], false);
    let new_token = cookie_token(&headers);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&auth_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }

    let (status, _, _) = exchange(
        addr,
        "GET",
        "/api/pulse",
        "127.0.0.1",
        &[("cookie", &old_cookie)],
        b"",
    )
    .await;
    assert_eq!(status, 401);
    let new_cookie = format!("cortex_session={new_token}");
    let (status, _, _) = exchange(
        addr,
        "GET",
        "/api/pulse",
        "127.0.0.1",
        &[("cookie", &new_cookie)],
        b"",
    )
    .await;
    assert_eq!(status, 200);

    let checked = python(
        "import dashboard_auth, json, os\n\
         auth = dashboard_auth.DashboardAuth(os.environ['AUTH_PATH'])\n\
         old = os.environ['OLD_TOKEN']\n\
         new = os.environ['NEW_TOKEN']\n\
         if auth.verify_session(old) is not None:\n\
         \traise SystemExit('old token still valid')\n\
         if auth.verify_session(new) is None:\n\
         \traise SystemExit('new token rejected')\n\
         if not auth.verify_password(os.environ['AUTH_USER'], os.environ['NEW_PASSWORD']):\n\
         \traise SystemExit('new password rejected')\n\
         state = json.load(open(os.environ['AUTH_PATH']))\n\
         expected = {'iterations','must_change_password','password_hash','salt','session_secret','session_version','updated_at','username','version'}\n\
         if set(state) != expected:\n\
         \traise SystemExit('fields ' + ' '.join(sorted(set(state))))\n\
         if state['iterations'] != 310000 or state['version'] != 1 or state['must_change_password'] is not False:\n\
         \traise SystemExit('field values')\n\
         if not str(state['updated_at']).endswith('+00:00'):\n\
         \traise SystemExit('updated_at')\n\
         if state['session_version'] != 2:\n\
         \traise SystemExit('version')\n\
         print('ok')",
        &[
            ("AUTH_PATH", auth_path.to_str().unwrap()),
            ("AUTH_USER", "ada"),
            ("OLD_TOKEN", old_token),
            ("NEW_TOKEN", &new_token),
            ("NEW_PASSWORD", "replacement-secret-value"),
        ],
    );
    assert!(
        checked.status.success(),
        "python parity failed: {}",
        String::from_utf8_lossy(&checked.stderr)
    );
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn must_change_blocks_pulse_but_not_logout_or_change_password() {
    let dir = TempDir::new();
    init_db(&dir.path().join("cortex.db"));
    install_password(
        &dir.path().join("dashboard-auth.json"),
        "ada",
        "correct-horse-battery",
        1,
        true,
        3,
    );
    let server = boot(config(dir.path())).await;
    let addr = server.local_addr();
    let (status, headers, _) = exchange(
        addr,
        "POST",
        "/api/auth/login",
        "127.0.0.1",
        &[("x-cortex-request", "1")],
        br#"{"username":"ada","password":"correct-horse-battery"}"#,
    )
    .await;
    assert_eq!(status, 200);
    let cookie = format!("cortex_session={}", cookie_token(&headers));
    let (status, _, body) = exchange(
        addr,
        "GET",
        "/api/pulse",
        "127.0.0.1",
        &[("cookie", &cookie)],
        b"",
    )
    .await;
    assert_eq!(status, 403);
    let payload = json_of(&body);
    assert_eq!(payload["error"], "password change required");
    assert_eq!(payload["must_change_password"], true);
    let status_body = json_of(
        &exchange(
            addr,
            "GET",
            "/api/auth/status",
            "127.0.0.1",
            &[("cookie", &cookie)],
            b"",
        )
        .await
        .2,
    );
    assert_eq!(status_body["authenticated"], true);
    assert_eq!(status_body["must_change_password"], true);
    assert_eq!(status_body["username"], "ada");
    let (status, _, body) = exchange(
        addr,
        "POST",
        "/api/auth/change-password",
        "127.0.0.1",
        &[("x-cortex-request", "1"), ("cookie", &cookie)],
        br#"{"current_password":"correct-horse-battery","new_password":"short"}"#,
    )
    .await;
    assert_eq!(status, 400, "{}", String::from_utf8_lossy(&body));
    assert_eq!(json_of(&body)["error"], "Use at least 12 characters.");
    let (status, _, _) = exchange(
        addr,
        "POST",
        "/api/auth/logout",
        "127.0.0.1",
        &[("x-cortex-request", "1"), ("cookie", &cookie)],
        b"",
    )
    .await;
    assert_eq!(status, 200);
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn basic_auth_and_login_share_a_throttle() {
    let dir = TempDir::new();
    init_db(&dir.path().join("cortex.db"));
    install_password(
        &dir.path().join("dashboard-auth.json"),
        "ada",
        "correct-horse-battery",
        1,
        false,
        1,
    );
    let server = boot(config(dir.path())).await;
    let addr = server.local_addr();
    let bad = STANDARD.encode(b"ada:wrong-password");
    let basic = format!("Basic {bad}");
    for _ in 0..8 {
        let (status, _, _) = exchange(
            addr,
            "GET",
            "/api/pulse",
            "127.0.0.1",
            &[("authorization", &basic)],
            b"",
        )
        .await;
        assert_eq!(status, 401);
    }
    let (status, _, body) = exchange(
        addr,
        "POST",
        "/api/auth/login",
        "127.0.0.1",
        &[("x-cortex-request", "1")],
        br#"{"username":"ada","password":"correct-horse-battery"}"#,
    )
    .await;
    assert_eq!(status, 429, "{}", String::from_utf8_lossy(&body));
    assert_eq!(
        json_of(&body)["error"],
        "Too many attempts. Try again shortly."
    );

    let good = STANDARD.encode(b"ada:correct-horse-battery");
    let good = format!("Basic {good}");
    // The ninth basic attempt is still 401, matching the Python gate, and
    // does not clear the counter. A fresh server checks success clears it.
    server.shutdown().await;

    let server = boot(config(dir.path())).await;
    let addr = server.local_addr();
    for _ in 0..7 {
        let (status, _, _) = exchange(
            addr,
            "POST",
            "/api/auth/login",
            "127.0.0.1",
            &[("x-cortex-request", "1")],
            br#"{"username":"ada","password":"nope-nope-nope"}"#,
        )
        .await;
        assert_eq!(status, 401);
    }
    let (status, _, _) = exchange(
        addr,
        "GET",
        "/api/pulse",
        "127.0.0.1",
        &[("authorization", &good)],
        b"",
    )
    .await;
    assert_eq!(status, 200);
    for _ in 0..8 {
        let (status, _, _) = exchange(
            addr,
            "POST",
            "/api/auth/login",
            "127.0.0.1",
            &[("x-cortex-request", "1")],
            br#"{"username":"ada","password":"nope-nope-nope"}"#,
        )
        .await;
        assert_eq!(status, 401);
    }
    let (status, _, _) = exchange(
        addr,
        "POST",
        "/api/auth/login",
        "127.0.0.1",
        &[("x-cortex-request", "1")],
        br#"{"username":"ada","password":"nope-nope-nope"}"#,
    )
    .await;
    assert_eq!(status, 429);
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn throttle_trusts_forwarded_ips_only_from_a_listed_proxy() {
    let dir = TempDir::new();
    init_db(&dir.path().join("cortex.db"));
    install_password(
        &dir.path().join("dashboard-auth.json"),
        "ada",
        "correct-horse-battery",
        1,
        false,
        1,
    );
    let mut trusted = config(dir.path());
    trusted.trusted_proxies.insert("127.0.0.1".to_string());
    let server = boot(trusted).await;
    let addr = server.local_addr();
    let bad = br#"{"username":"ada","password":"nope-nope-nope"}"#;
    for _ in 0..8 {
        let (status, _, _) = exchange(
            addr,
            "POST",
            "/api/auth/login",
            "127.0.0.1",
            &[
                ("x-cortex-request", "1"),
                ("cf-connecting-ip", "10.9.9.9"),
                ("x-forwarded-for", "10.8.8.8"),
            ],
            bad,
        )
        .await;
        assert_eq!(status, 401);
    }
    let (status, _, _) = exchange(
        addr,
        "POST",
        "/api/auth/login",
        "127.0.0.1",
        &[("x-cortex-request", "1"), ("cf-connecting-ip", "10.9.9.9")],
        bad,
    )
    .await;
    assert_eq!(status, 429);
    let (status, _, _) = exchange(
        addr,
        "POST",
        "/api/auth/login",
        "127.0.0.1",
        &[("x-cortex-request", "1"), ("x-forwarded-for", "10.8.8.8")],
        bad,
    )
    .await;
    assert_eq!(status, 401);
    server.shutdown().await;

    let server = boot(config(dir.path())).await;
    let addr = server.local_addr();
    for index in 0..8 {
        let forwarded = format!("10.0.0.{index}");
        let (status, _, _) = exchange(
            addr,
            "POST",
            "/api/auth/login",
            "127.0.0.1",
            &[("x-cortex-request", "1"), ("x-forwarded-for", &forwarded)],
            bad,
        )
        .await;
        assert_eq!(status, 401);
    }
    let (status, _, _) = exchange(
        addr,
        "POST",
        "/api/auth/login",
        "127.0.0.1",
        &[("x-cortex-request", "1"), ("x-forwarded-for", "10.9.9.9")],
        bad,
    )
    .await;
    assert_eq!(status, 429);
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_allowlist_gates_rust_routes_and_the_original_host_is_proxied() {
    let dir = TempDir::new();
    init_db(&dir.path().join("cortex.db"));
    std::fs::write(dir.path().join("assets").join("pulse.html"), b"tile").unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let Ok(n) = socket.read(&mut buf).await else {
                    return;
                };
                let body = &buf[..n];
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = socket.write_all(header.as_bytes()).await;
                let _ = socket.write_all(body).await;
            });
        }
    });
    let mut cfg = config(dir.path());
    cfg.allowed_hosts.insert("localhost".to_string());
    cfg.upstream = format!("http://{upstream}");
    cfg.upstream_timeout = Duration::from_secs(2);
    let server = boot(cfg).await;
    let addr = server.local_addr();

    let (status, _, body) = exchange(addr, "GET", "/pulse", "evil.example", &[], b"").await;
    assert_eq!(status, 403, "{}", String::from_utf8_lossy(&body));
    assert_eq!(json_of(&body)["error"], "host not allowed");
    assert_security(
        &exchange(addr, "GET", "/pulse", "evil.example", &[], b"")
            .await
            .1,
    );

    let (status, _, body) = exchange(addr, "GET", "/pulse", "LOCALHOST:9999", &[], b"").await;
    assert_eq!(status, 200);
    assert_eq!(&body[..], b"tile");

    let (status, _, body) = exchange(
        addr,
        "GET",
        "/api/snapshot?x=1",
        "public.example",
        &[
            ("cookie", "cortex_session=abc"),
            ("authorization", "Basic eA=="),
            ("x-forwarded-for", "203.0.113.5"),
            ("x-forwarded-proto", "https"),
            ("x-cortex-request", "1"),
        ],
        b"",
    )
    .await;
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    let echoed = String::from_utf8_lossy(&body);
    assert!(echoed.contains("GET /api/snapshot?x=1"), "{echoed}");
    assert!(
        echoed.to_ascii_lowercase().contains("host: public.example"),
        "{echoed}"
    );
    assert!(echoed.contains("cortex_session=abc"), "{echoed}");
    assert!(
        echoed.contains("Basic eA==") || echoed.contains("Basic eA"),
        "{echoed}"
    );
    assert!(
        echoed
            .to_ascii_lowercase()
            .contains("x-forwarded-for: 203.0.113.5"),
        "{echoed}"
    );
    assert!(
        echoed
            .to_ascii_lowercase()
            .contains("x-forwarded-proto: https"),
        "{echoed}"
    );
    assert!(
        echoed.to_ascii_lowercase().contains("x-cortex-request: 1"),
        "{echoed}"
    );
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_is_503_when_the_upstream_is_down_or_slow() {
    let dir = TempDir::new();
    init_db(&dir.path().join("cortex.db"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let closed = listener.local_addr().unwrap();
    drop(listener);
    let mut cfg = config(dir.path());
    cfg.upstream = format!("http://{closed}");
    let server = boot(cfg).await;
    let (status, headers, body) = exchange(
        server.local_addr(),
        "GET",
        "/api/snapshot",
        "127.0.0.1",
        &[],
        b"",
    )
    .await;
    assert_eq!(status, 503);
    assert_security(&headers);
    let payload = json_of(&body);
    assert_eq!(payload["error"], "snapshot unavailable");
    assert_eq!(payload["note"], "python dashboard upstream is not running");
    assert_eq!(payload["schema_version"], 1);
    assert_eq!(headers["cache-control"], "no-store");
    server.shutdown().await;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let slow = listener.local_addr().unwrap();
    tokio::spawn(async move {
        if let Ok((socket, _)) = listener.accept().await {
            tokio::time::sleep(Duration::from_secs(5)).await;
            drop(socket);
        }
    });
    let mut cfg = config(dir.path());
    cfg.upstream = format!("http://{slow}");
    cfg.upstream_timeout = Duration::from_millis(200);
    let server = boot(cfg).await;
    let started = std::time::Instant::now();
    let (status, _, body) = exchange(
        server.local_addr(),
        "GET",
        "/api/sleep/status",
        "127.0.0.1",
        &[],
        b"",
    )
    .await;
    assert_eq!(status, 503);
    assert_eq!(
        json_of(&body)["note"],
        "python dashboard upstream is not running"
    );
    assert!(started.elapsed() < Duration::from_secs(3));
    server.shutdown().await;
}
