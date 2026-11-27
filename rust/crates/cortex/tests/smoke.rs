//! Coexistence smoke: Rust reads, Python's schema version stays put, and a
//! down upstream is reported instead of an empty snapshot.

use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{fs, thread};

use rusqlite::{Connection, OpenFlags};

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
CREATE TABLE kept_marker (id INTEGER PRIMARY KEY);
";

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("cortex-rs-{nanos}-{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_cortex-rs"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap()
}

fn create_store(path: &Path) {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(REQUIRED_SQL).unwrap();
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', '34')",
        [],
    )
    .unwrap();
}

struct Snapshot {
    bytes: Vec<u8>,
    tables: BTreeSet<String>,
    schema_version: Option<String>,
}

fn snapshot(path: &Path) -> Snapshot {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let mut statement = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        .unwrap();
    let tables = statement
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(|name| name.unwrap())
        .collect();
    let schema_version = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        )
        .ok();
    drop(statement);
    drop(conn);
    Snapshot {
        bytes: fs::read(path).unwrap(),
        tables,
        schema_version,
    }
}

fn assert_unchanged(path: &Path, before: &Snapshot) {
    let after = snapshot(path);
    assert_eq!(after.schema_version.as_deref(), Some("34"));
    assert_eq!(after.tables, before.tables);
    assert_eq!(after.bytes, before.bytes, "database bytes changed");
    assert!(!PathBuf::from(format!("{}-wal", path.display())).exists());
    assert!(!PathBuf::from(format!("{}-shm", path.display())).exists());
}

#[test]
fn status_json_leaves_schema_version_and_tables_unchanged() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_store(&path);
    let before = snapshot(&path);
    assert_eq!(before.schema_version.as_deref(), Some("34"));
    assert!(before.tables.contains("kept_marker"));
    assert!(before.tables.contains("memories"));

    let output = bin()
        .args(["--db", path.to_str().unwrap(), "status", "--json"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["schema_version"], 1);
    assert_eq!(payload["health"]["readable"], true);
    assert_eq!(payload["health"]["store_schema_version"], "34");
    assert_unchanged(&path, &before);
}

#[test]
fn status_does_not_create_a_missing_database() {
    let scratch = Scratch::new();
    let missing = scratch.path().join("absent").join("cortex.db");
    let output = bin()
        .args(["--db", missing.to_str().unwrap(), "status", "--json"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!missing.exists());
    assert!(!missing.parent().unwrap().exists());
}

#[test]
fn missing_command_exits_2() {
    let output = bin().output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let err = String::from_utf8(output.stderr).unwrap();
    assert!(err.contains("missing command"), "{err}");
    assert!(err.contains("cortex-rs serve"), "{err}");
}

struct Serve {
    child: Option<std::process::Child>,
    addr: SocketAddr,
    log: Arc<Mutex<String>>,
}

impl Serve {
    fn shutdown(mut self) -> ExitStatus {
        let mut child = self.child.take().unwrap();
        let _ = Command::new("kill")
            .args(["-TERM", &child.id().to_string()])
            .status();
        let started = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => return status,
                Ok(None) if started.elapsed() < Duration::from_secs(5) => {
                    thread::sleep(Duration::from_millis(30));
                }
                _ => {
                    let _ = child.kill();
                    return child.wait().unwrap();
                }
            }
        }
    }
}

impl Drop for Serve {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = Command::new("kill")
                .args(["-KILL", &child.id().to_string()])
                .status();
            let _ = child.wait();
        }
    }
}

fn closed_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

fn spawn_serve(db: &Path, auth: &Path, upstream: &str) -> Serve {
    let mut child = bin()
        .args([
            "serve",
            "--bind",
            "127.0.0.1:0",
            "--db",
            db.to_str().unwrap(),
            "--assets",
            db.parent().unwrap().to_str().unwrap(),
        ])
        .env("CORTEX_DASHBOARD_AUTH_FILE", auth)
        .env("CORTEX_PYTHON_UPSTREAM", upstream)
        .env_remove("CORTEX_SERVE_ALLOW_PUBLIC_BIND")
        .env_remove("CORTEX_DASHBOARD_ALLOWED_HOSTS")
        .env_remove("CORTEX_DASHBOARD_TRUSTED_PROXIES")
        .env_remove("CORTEX_ASSET_DIR")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stderr = child.stderr.take().unwrap();
    let log = Arc::new(Mutex::new(String::new()));
    let recorded = Arc::clone(&log);
    let (tx, rx) = std::sync::mpsc::channel();
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            {
                let mut guard = recorded.lock().unwrap();
                guard.push_str(&line);
                guard.push('\n');
            }
            if let Some(addr) = line.split("listening on ").nth(1) {
                if let Ok(addr) = addr.trim().parse() {
                    let _ = tx.send(addr);
                }
            }
        }
    });
    let addr = rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap_or_else(|err| {
            panic!("serve did not listen ({err}): {}", log.lock().unwrap());
        });
    Serve {
        child: Some(child),
        addr,
        log,
    }
}

fn http_get(addr: SocketAddr, path: &str) -> (u16, String) {
    let mut last = String::new();
    for _ in 0..20 {
        match TcpStream::connect(addr) {
            Ok(mut stream) => {
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let request = format!("GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n");
                stream.write_all(request.as_bytes()).unwrap();
                let mut raw = Vec::new();
                stream.read_to_end(&mut raw).unwrap();
                let text = String::from_utf8_lossy(&raw).into_owned();
                let status = text
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
                let body = text
                    .split_once("\r\n\r\n")
                    .map(|(_, body)| body.to_string())
                    .unwrap_or_default();
                return (status, body);
            }
            Err(err) => {
                last = err.to_string();
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
    panic!("GET {path} on {addr} failed: {last}");
}

fn json_body(body: &str) -> serde_json::Value {
    let start = body
        .find('{')
        .unwrap_or_else(|| panic!("no json object in {body}"));
    let end = body
        .rfind('}')
        .unwrap_or_else(|| panic!("no json object in {body}"));
    serde_json::from_str(&body[start..=end]).unwrap_or_else(|err| panic!("{err} in {body}"))
}

fn write_auth(path: &Path) {
    let output = Command::new("python3")
        .current_dir(repo_root())
        .arg("-c")
        .arg(
            "import dashboard_auth, os\n\
             auth = dashboard_auth.DashboardAuth(os.environ['AUTH_PATH'])\n\
             auth.ensure('cortex', 'correct-horse-battery')\n\
             print('ok')",
        )
        .env("PYTHONPATH", repo_root())
        .env("AUTH_PATH", path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "dashboard_auth failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn serve_requires_auth_for_pulse_and_reports_auth_status() {
    let scratch = Scratch::new();
    let db = scratch.path().join("cortex.db");
    create_store(&db);
    let before = snapshot(&db);
    let auth = scratch.path().join("dashboard-auth.json");
    write_auth(&auth);
    let upstream = format!("http://127.0.0.1:{}", closed_port());
    let server = spawn_serve(&db, &auth, &upstream);

    let (status, body) = http_get(server.addr, "/api/pulse");
    assert_eq!(status, 401, "{body}\n{}", server.log.lock().unwrap());
    assert!(
        json_body(&body)["error"]
            .as_str()
            .unwrap_or("")
            .contains("authentication required"),
        "{body}"
    );

    let (status, body) = http_get(server.addr, "/api/auth/status");
    assert_eq!(status, 200, "{body}");
    let payload = json_body(&body);
    assert_eq!(payload["auth_enabled"], true);
    assert_eq!(payload["authenticated"], false);

    let exit = server.shutdown();
    assert_eq!(exit.code(), Some(0), "SIGTERM should exit 0");
    assert_unchanged(&db, &before);
}

#[test]
fn snapshot_is_503_with_a_note_when_python_is_down() {
    let scratch = Scratch::new();
    let db = scratch.path().join("cortex.db");
    create_store(&db);
    let before = snapshot(&db);
    let auth = scratch.path().join("dashboard-auth.json");
    write_auth(&auth);
    let upstream = format!("http://127.0.0.1:{}", closed_port());
    let server = spawn_serve(&db, &auth, &upstream);

    let (status, body) = http_get(server.addr, "/api/snapshot");
    assert_eq!(status, 503, "{body}\n{}", server.log.lock().unwrap());
    let payload = json_body(&body);
    let note = payload["note"].as_str().unwrap_or("");
    assert!(
        note.contains("python dashboard upstream is not running"),
        "{body}"
    );
    assert_eq!(payload["schema_version"], 1);
    assert!(payload.get("memories").is_none(), "{body}");

    let exit = server.shutdown();
    assert_eq!(exit.code(), Some(0));
    assert_unchanged(&db, &before);
}

fn glyph_rows(text: &str, glyph: char) -> Vec<&str> {
    text.lines()
        .filter(|line| !line.is_empty() && line.chars().all(|ch| ch == glyph))
        .collect()
}

#[test]
fn pulse_once_prints_an_empty_glyph_grid() {
    let scratch = Scratch::new();
    let path = scratch.path().join("cortex.db");
    create_store(&path);
    let before = snapshot(&path);

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
    assert!(!output.stdout.contains(&0x1b), "ANSI on a pipe");
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(!text.contains("0.0/s"), "{text}");
    let rows = glyph_rows(&text, '·');
    assert_eq!(rows.len(), 6, "{text}");
    assert!(rows.iter().all(|row| row.chars().count() == 40), "{text}");

    let output = bin()
        .args(["--db", path.to_str().unwrap(), "pulse", "--ascii", "--once"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.is_ascii(), "{text}");
    assert!(!text.contains('·'));
    let rows = glyph_rows(&text, '.');
    assert_eq!(rows.len(), 6, "{text}");
    assert!(rows.iter().all(|row| row.chars().count() == 40), "{text}");
    assert_unchanged(&path, &before);
}

struct Install {
    scratch: Scratch,
    status: ExitStatus,
    stdout: String,
    stderr: String,
    unit: PathBuf,
    env_file: PathBuf,
    systemctl_log: PathBuf,
}

fn install(extra: &[(&str, &str)]) -> Install {
    let scratch = Scratch::new();
    let hermes = scratch.path().join("hermes");
    let unit_dir = scratch.path().join("units");
    let fake_bin = scratch.path().join("fake-bin");
    fs::create_dir_all(&hermes).unwrap();
    fs::create_dir_all(&unit_dir).unwrap();
    fs::create_dir_all(&fake_bin).unwrap();
    let systemctl_log = scratch.path().join("systemctl.log");
    let systemctl = fake_bin.join("systemctl");
    fs::write(
        &systemctl,
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\nexit 0\n",
    )
    .unwrap();
    let mut perms = fs::metadata(&systemctl).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&systemctl, perms).unwrap();
    fs::write(
        unit_dir.join("cortex-dashboard.service"),
        b"SENTINEL do not touch\n",
    )
    .unwrap();

    let mut cmd = Command::new("bash");
    cmd.arg(repo_root().join("scripts/install_cortex_serve.sh"))
        .env("HOME", scratch.path())
        .env("HERMES_HOME", &hermes)
        .env("UNIT_DIR", &unit_dir)
        .env("XDG_CONFIG_HOME", scratch.path().join("xdg"))
        .env("BIN", env!("CARGO_BIN_EXE_cortex-rs"))
        .env(
            "PATH",
            format!(
                "{}:{}",
                fake_bin.display(),
                std::env::var("PATH").unwrap_or_default()
            ),
        )
        .env("SYSTEMCTL_LOG", &systemctl_log)
        .env_remove("PORT")
        .env_remove("CORTEX_SERVE_INSTALL_NO_SYSTEMCTL");
    for (key, value) in extra {
        if value.is_empty() {
            cmd.env_remove(key);
        } else {
            cmd.env(key, value);
        }
    }
    let output = cmd.output().unwrap();
    Install {
        unit: unit_dir.join("cortex-serve.service"),
        env_file: hermes.join("cortex").join("cortex-serve.env"),
        systemctl_log,
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        status: output.status,
        scratch,
    }
}

#[test]
fn install_writes_unit_and_env_without_systemctl() {
    let installed = install(&[("CORTEX_SERVE_INSTALL_NO_SYSTEMCTL", "1")]);
    assert!(
        installed.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        installed.stdout,
        installed.stderr
    );
    let unit = fs::read_to_string(&installed.unit).unwrap();
    assert!(unit.contains("127.0.0.1:8090"), "{unit}");
    assert!(!unit.contains("cortex-dashboard.service"), "{unit}");
    let read_write = unit
        .lines()
        .find(|line| line.starts_with("ReadWritePaths="))
        .unwrap();
    assert!(read_write.ends_with("/cortex"), "{read_write}");
    assert!(
        unit.contains("ExecReload=/bin/kill -HUP $MAINPID"),
        "{unit}"
    );
    assert!(unit.contains("ProtectHome=read-only"), "{unit}");
    assert_eq!(
        fs::read(
            installed
                .scratch
                .path()
                .join("units/cortex-dashboard.service")
        )
        .unwrap(),
        b"SENTINEL do not touch\n"
    );
    assert!(
        !installed.systemctl_log.exists(),
        "systemctl was called:\n{}",
        fs::read_to_string(&installed.systemctl_log).unwrap_or_default()
    );

    let env_text = fs::read_to_string(&installed.env_file).unwrap();
    assert!(
        !env_text.to_ascii_lowercase().contains("password"),
        "{env_text}"
    );
    assert!(
        env_text.contains("CORTEX_PYTHON_UPSTREAM=http://127.0.0.1:8100"),
        "{env_text}"
    );
    assert!(env_text.contains("CORTEX_DB="), "{env_text}");
    assert!(
        env_text.contains("CORTEX_DASHBOARD_AUTH_FILE="),
        "{env_text}"
    );
    let asset = env_text
        .lines()
        .find_map(|line| line.strip_prefix("CORTEX_ASSET_DIR="))
        .unwrap();
    assert_eq!(PathBuf::from(asset).canonicalize().unwrap(), repo_root());
    let mode = fs::metadata(&installed.env_file)
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600, "env mode is {mode:o}");
    assert!(!installed
        .env_file
        .parent()
        .unwrap()
        .join("dashboard-auth.json")
        .exists());
    assert!(
        installed.stdout.contains("127.0.0.1:8090"),
        "{}",
        installed.stdout
    );
    assert!(
        installed.stdout.contains("http://127.0.0.1:8100"),
        "{}",
        installed.stdout
    );
}

#[test]
fn install_keeps_an_existing_env_file() {
    let scratch = Scratch::new();
    let hermes = scratch.path().join("hermes");
    let env_file = hermes.join("cortex").join("cortex-serve.env");
    fs::create_dir_all(env_file.parent().unwrap()).unwrap();
    fs::write(&env_file, "CORTEX_DB=/keep/this/cortex.db\n").unwrap();
    let unit_dir = scratch.path().join("units");
    fs::create_dir_all(&unit_dir).unwrap();
    let output = Command::new("bash")
        .arg(repo_root().join("scripts/install_cortex_serve.sh"))
        .env("HOME", scratch.path())
        .env("HERMES_HOME", &hermes)
        .env("UNIT_DIR", &unit_dir)
        .env("BIN", env!("CARGO_BIN_EXE_cortex-rs"))
        .env("CORTEX_SERVE_INSTALL_NO_SYSTEMCTL", "1")
        .env_remove("PORT")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let env_text = fs::read_to_string(&env_file).unwrap();
    assert_eq!(env_text, "CORTEX_DB=/keep/this/cortex.db\n");
    assert!(!env_text.to_ascii_lowercase().contains("password"));
}

#[test]
fn install_calls_systemctl_only_for_its_own_unit() {
    let installed = install(&[("CORTEX_SERVE_INSTALL_NO_SYSTEMCTL", "")]);
    assert!(
        installed.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        installed.stdout,
        installed.stderr
    );
    let log = fs::read_to_string(&installed.systemctl_log).unwrap();
    assert!(log.contains("--user daemon-reload"), "{log}");
    assert!(log.contains("--user enable cortex-serve.service"), "{log}");
    assert!(!log.contains("cortex-dashboard"), "{log}");
    assert!(!log.contains("stop"), "{log}");
    assert!(!log.contains("disable"), "{log}");
    assert_eq!(
        fs::read(
            installed
                .scratch
                .path()
                .join("units/cortex-dashboard.service")
        )
        .unwrap(),
        b"SENTINEL do not touch\n"
    );
}

#[test]
fn install_refuses_a_taken_port_without_killing_the_occupant() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let port_text = port.to_string();
    let installed = install(&[
        ("CORTEX_SERVE_INSTALL_NO_SYSTEMCTL", "1"),
        ("PORT", &port_text),
    ]);
    assert!(
        !installed.status.success(),
        "install succeeded while {} was taken\n{}",
        port,
        installed.stdout
    );
    assert!(
        installed.stderr.contains("already in use"),
        "{}",
        installed.stderr
    );
    assert!(
        installed.stderr.contains("leaving the occupant running"),
        "{}",
        installed.stderr
    );
    TcpStream::connect(listener.local_addr().unwrap()).expect("occupant was killed");
    assert!(!installed.unit.exists());
    assert!(!installed.systemctl_log.exists());
}
