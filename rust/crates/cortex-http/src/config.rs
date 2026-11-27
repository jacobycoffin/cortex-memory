//! `serve` flags and the environment the daemon reads once at startup.

use std::collections::BTreeSet;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Environment values `parse_args` consults. Tests pass a synthetic value so
/// a developer machine's `CORTEX_DB` cannot leak into the suite.
#[derive(Clone, Debug, Default)]
pub struct ProcessEnv {
    pub cortex_db: Option<OsString>,
    pub hermes_home: Option<OsString>,
    pub home: Option<OsString>,
    pub asset_dir: Option<OsString>,
    pub auth_file: Option<OsString>,
    pub upstream: Option<OsString>,
    pub allow_public_bind: bool,
    pub allowed_hosts: OsString,
    pub trusted_proxies: OsString,
}

impl ProcessEnv {
    /// Read the process environment. Missing keys stay empty.
    pub fn from_process() -> Self {
        Self {
            cortex_db: std::env::var_os("CORTEX_DB"),
            hermes_home: std::env::var_os("HERMES_HOME"),
            home: std::env::var_os("HOME"),
            asset_dir: std::env::var_os("CORTEX_ASSET_DIR"),
            auth_file: std::env::var_os("CORTEX_DASHBOARD_AUTH_FILE"),
            upstream: std::env::var_os("CORTEX_PYTHON_UPSTREAM"),
            allow_public_bind: std::env::var("CORTEX_SERVE_ALLOW_PUBLIC_BIND")
                .ok()
                .as_deref()
                == Some("1"),
            allowed_hosts: std::env::var_os("CORTEX_DASHBOARD_ALLOWED_HOSTS").unwrap_or_default(),
            trusted_proxies: std::env::var_os("CORTEX_DASHBOARD_TRUSTED_PROXIES")
                .unwrap_or_default(),
        }
    }
}

/// Everything `start` needs. Built by [`parse_args`] or by a caller that
/// already knows the paths (the RS-005 binary, and tests).
#[derive(Clone, Debug)]
pub struct ServeConfig {
    pub bind: SocketAddr,
    pub db_path: PathBuf,
    pub asset_dir: PathBuf,
    pub auth_path: PathBuf,
    pub upstream: String,
    /// Whole upstream attempt, including the response body. Default 5 seconds.
    pub upstream_timeout: Duration,
    /// Empty means do not check `Host`. Entries are already normalized.
    pub allowed_hosts: BTreeSet<String>,
    /// TCP peers whose `CF-Connecting-IP` / `X-Forwarded-For` may name the client.
    pub trusted_proxies: BTreeSet<String>,
}

/// Failure before the socket is listening, or while it is coming up.
#[derive(Debug)]
pub enum ServeError {
    Usage(String),
    Message(String),
}

impl ServeError {
    pub fn is_usage(&self) -> bool {
        matches!(self, Self::Usage(_))
    }
}

impl fmt::Display for ServeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(message) | Self::Message(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ServeError {}

const USAGE: &str = "usage: cortex-http serve [--bind 127.0.0.1:8090] [--db PATH] [--assets DIR]";

pub fn parse_args<I, S>(args: I, env: &ProcessEnv) -> Result<ServeConfig, ServeError>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let tokens = command_tokens(args);
    let mut bind_text: Option<String> = None;
    let mut db_text: Option<String> = None;
    let mut asset_text: Option<String> = None;
    let mut saw_serve = false;
    let mut iter = tokens.into_iter();
    while let Some(arg) = iter.next() {
        if arg == "serve" {
            if saw_serve {
                return Err(ServeError::Usage(format!("duplicate subcommand\n{USAGE}")));
            }
            saw_serve = true;
            continue;
        }
        if arg == "--help" || arg == "-h" {
            return Err(ServeError::Usage(USAGE.to_string()));
        }
        let (flag, inline) = split_flag(&arg);
        if !flag.starts_with("--") {
            return Err(ServeError::Usage(format!(
                "unknown argument {arg}\n{USAGE}"
            )));
        }
        let value = match inline {
            Some(value) => value.to_string(),
            None => iter
                .next()
                .ok_or_else(|| ServeError::Usage(format!("{flag} needs a value\n{USAGE}")))?,
        };
        match flag {
            "--bind" => bind_text = Some(value),
            "--db" => db_text = Some(value),
            "--assets" => asset_text = Some(value),
            other => {
                return Err(ServeError::Usage(format!(
                    "unknown argument {other}\n{USAGE}"
                )));
            }
        }
    }
    if !saw_serve {
        return Err(ServeError::Usage(USAGE.to_string()));
    }

    let bind = match bind_text {
        Some(text) => parse_bind(&text)?,
        None => SocketAddr::from(([127, 0, 0, 1], 8090)),
    };
    if !bind.ip().is_loopback() && !env.allow_public_bind {
        return Err(ServeError::Usage(format!(
            "refusing non-loopback bind {bind}; set CORTEX_SERVE_ALLOW_PUBLIC_BIND=1 to allow it"
        )));
    }

    let db_path = match db_text {
        Some(text) => expand_home(PathBuf::from(text), env.home.as_deref()),
        None => default_db(env)?,
    };
    let asset_dir = match asset_text {
        Some(text) => expand_home(PathBuf::from(text), env.home.as_deref()),
        None => match &env.asset_dir {
            Some(path) => expand_home(PathBuf::from(path), env.home.as_deref()),
            None => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        },
    };
    let auth_path = match &env.auth_file {
        Some(path) => expand_home(PathBuf::from(path), env.home.as_deref()),
        None => default_auth_path(&db_path),
    };
    let upstream = match &env.upstream {
        Some(value) => {
            let text = os_text(value, "CORTEX_PYTHON_UPSTREAM")?;
            normalize_upstream(&text)?
        }
        None => "http://127.0.0.1:8100".to_string(),
    };
    let allowed_hosts = parse_allowed_hosts(&os_text(
        &env.allowed_hosts,
        "CORTEX_DASHBOARD_ALLOWED_HOSTS",
    )?);
    let trusted_proxies = parse_trusted_proxies(&os_text(
        &env.trusted_proxies,
        "CORTEX_DASHBOARD_TRUSTED_PROXIES",
    )?);

    Ok(ServeConfig {
        bind,
        db_path,
        asset_dir,
        auth_path,
        upstream,
        upstream_timeout: Duration::from_secs(5),
        allowed_hosts,
        trusted_proxies,
    })
}

fn command_tokens<I, S>(args: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut tokens: Vec<String> = args
        .into_iter()
        .map(|arg| Into::<OsString>::into(arg).to_string_lossy().into_owned())
        .collect();
    if let Some(first) = tokens.first() {
        if first != "serve" && !first.starts_with('-') {
            tokens.remove(0);
        }
    }
    tokens
}

fn split_flag(arg: &str) -> (&str, Option<&str>) {
    if let Some((flag, value)) = arg.split_once('=') {
        if flag.starts_with("--") && flag != "--" {
            return (flag, Some(value));
        }
    }
    (arg, None)
}

fn parse_bind(text: &str) -> Result<SocketAddr, ServeError> {
    text.parse::<SocketAddr>().map_err(|_| {
        ServeError::Usage(format!(
            "invalid --bind address {text} (expected IP:port, for example 127.0.0.1:8090)"
        ))
    })
}

fn default_db(env: &ProcessEnv) -> Result<PathBuf, ServeError> {
    if let Some(path) = &env.cortex_db {
        return Ok(expand_home(PathBuf::from(path), env.home.as_deref()));
    }
    if let Some(home) = &env.hermes_home {
        return Ok(expand_home(PathBuf::from(home), env.home.as_deref())
            .join("cortex")
            .join("cortex.db"));
    }
    let home = env.home.as_ref().ok_or_else(|| {
        ServeError::Usage(
            "HOME is not set; pass --db or set CORTEX_DB (or HERMES_HOME)".to_string(),
        )
    })?;
    Ok(PathBuf::from(home)
        .join(".hermes")
        .join("cortex")
        .join("cortex.db"))
}

fn default_auth_path(db: &Path) -> PathBuf {
    match db.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join("dashboard-auth.json"),
        _ => PathBuf::from("dashboard-auth.json"),
    }
}

fn expand_home(path: PathBuf, home: Option<&OsStr>) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix("~/") {
        if let Some(home) = home {
            return PathBuf::from(home).join(rest);
        }
    }
    path
}

fn os_text(value: &OsString, name: &str) -> Result<String, ServeError> {
    value
        .to_str()
        .map(str::to_string)
        .ok_or_else(|| ServeError::Usage(format!("{name} is not valid UTF-8")))
}

fn normalize_upstream(raw: &str) -> Result<String, ServeError> {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(ServeError::Usage(
            "CORTEX_PYTHON_UPSTREAM is empty".to_string(),
        ));
    }
    let uri: http::Uri = trimmed
        .parse()
        .map_err(|_| ServeError::Usage(format!("invalid CORTEX_PYTHON_UPSTREAM")))?;
    match uri.scheme_str() {
        Some("http") | Some("https") => {}
        _ => {
            return Err(ServeError::Usage(
                "CORTEX_PYTHON_UPSTREAM must be an http or https URL".to_string(),
            ));
        }
    }
    if uri.host().is_none() {
        return Err(ServeError::Usage(
            "CORTEX_PYTHON_UPSTREAM is missing a host".to_string(),
        ));
    }
    Ok(trimmed.to_string())
}

/// Host allowlist entries, matching `dashboard.py`'s `_normalize_allowed_hosts`.
pub(crate) fn parse_allowed_hosts(raw: &str) -> BTreeSet<String> {
    let mut hosts = BTreeSet::new();
    for entry in raw.split(',') {
        if let Some(host) = normalize_host(entry) {
            hosts.insert(host);
        }
    }
    hosts
}

pub(crate) fn parse_trusted_proxies(raw: &str) -> BTreeSet<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(str::to_string)
        .collect()
}

/// Strip a port, keep IPv6 brackets, and casefold.
///
/// `localhost:8090` → `localhost`. `[::1]:8090` → `[::1]`. Empty input is
/// `None`. This follows the Python dashboard: strip, then split on `]` or
/// `:`, then casefold.
pub(crate) fn normalize_host(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let host = if let Some(stripped) = value.strip_prefix('[') {
        match stripped.find(']') {
            Some(index) => &value[..=index + 1],
            // Python does `split("]", 1)[0] + "]"`, which appends a bracket
            // when the closer is missing.
            None => {
                let mut owned = value.to_string();
                owned.push(']');
                return casefold_host(&owned);
            }
        }
    } else {
        value.split(':').next().unwrap_or(value)
    };
    let host = host.trim();
    if host.is_empty() {
        None
    } else {
        casefold_host(host)
    }
}

fn casefold_host(host: &str) -> Option<String> {
    let folded = host.to_lowercase();
    if folded.is_empty() {
        None
    } else {
        Some(folded)
    }
}

pub(crate) fn host_allowed(allowed: &BTreeSet<String>, header: Option<&str>) -> bool {
    if allowed.is_empty() {
        return true;
    }
    match header.and_then(normalize_host) {
        Some(host) => allowed.contains(&host),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_with_home() -> ProcessEnv {
        let mut env = ProcessEnv::default();
        env.home = Some("home".into());
        env
    }

    #[test]
    fn defaults_follow_the_python_paths() {
        let mut env = env_with_home();
        env.hermes_home = Some("hermes-home".into());
        let config = parse_args(["cortex-http", "serve"], &env).unwrap();
        assert_eq!(config.bind, "127.0.0.1:8090".parse().unwrap());
        assert_eq!(
            config.db_path,
            PathBuf::from("hermes-home/cortex/cortex.db")
        );
        assert_eq!(
            config.auth_path,
            PathBuf::from("hermes-home/cortex/dashboard-auth.json")
        );
        assert_eq!(config.upstream, "http://127.0.0.1:8100");
        assert!(config.allowed_hosts.is_empty());
        assert_eq!(config.upstream_timeout, Duration::from_secs(5));
    }

    #[test]
    fn db_flag_before_the_subcommand_wins_over_the_environment() {
        let mut env = env_with_home();
        env.cortex_db = Some("from-env.db".into());
        let config = parse_args(
            [
                "cortex-http",
                "--db",
                "earlier.db",
                "serve",
                "--db",
                "later.db",
                "--assets",
                "assets",
            ],
            &env,
        )
        .unwrap();
        assert_eq!(config.db_path, PathBuf::from("later.db"));
        assert_eq!(config.auth_path, PathBuf::from("dashboard-auth.json"));
        assert_eq!(config.asset_dir, PathBuf::from("assets"));
    }

    #[test]
    fn loopback_bind_is_accepted_and_a_public_bind_is_refused() {
        let env = env_with_home();
        let loopback = parse_args(
            [
                "serve",
                "--bind",
                "127.0.0.2:9",
                "--db",
                "db",
                "--bind=[::1]:8090",
            ],
            &env,
        )
        .unwrap();
        assert_eq!(loopback.bind, "[::1]:8090".parse().unwrap());

        let refused =
            parse_args(["serve", "--bind", "0.0.0.0:8090", "--db", "db"], &env).unwrap_err();
        assert!(refused.is_usage());
        assert!(refused
            .to_string()
            .contains("CORTEX_SERVE_ALLOW_PUBLIC_BIND"));

        let mut allowed = env.clone();
        allowed.allow_public_bind = true;
        let config =
            parse_args(["serve", "--bind", "0.0.0.0:8090", "--db", "db"], &allowed).unwrap();
        assert_eq!(config.bind.ip().to_string(), "0.0.0.0");
    }

    #[test]
    fn missing_serve_is_usage() {
        let err = parse_args(["cortex-http", "--db", "db"], &env_with_home()).unwrap_err();
        assert!(err.is_usage());
        assert!(err.to_string().contains("usage:"));
    }

    #[test]
    fn host_normalization_strips_ports_and_case() {
        assert_eq!(
            normalize_host(" LocalHost:8090 ").as_deref(),
            Some("localhost")
        );
        assert_eq!(normalize_host("[::1]:8090").as_deref(), Some("[::1]"));
        assert_eq!(normalize_host("127.0.0.1").as_deref(), Some("127.0.0.1"));
        assert_eq!(normalize_host("").as_deref(), None);
        assert_eq!(normalize_host("[::1").as_deref(), Some("[::1]"));

        let hosts = parse_allowed_hosts(" LocalHost:80, ,127.0.0.1:8090 ,[::1]:1 ");
        assert!(hosts.contains("localhost"));
        assert!(hosts.contains("127.0.0.1"));
        assert!(hosts.contains("[::1]"));
        assert!(host_allowed(&hosts, Some("LOCALHOST:9999")));
        assert!(!host_allowed(&hosts, Some("evil.example")));
        assert!(host_allowed(&BTreeSet::new(), Some("evil.example")));
        assert!(!host_allowed(&hosts, None));
    }

    #[test]
    fn auth_override_and_upstream_trim() {
        let mut env = env_with_home();
        env.auth_file = Some("secrets/dashboard-auth.json".into());
        env.upstream = Some("http://127.0.0.1:8100/".into());
        let config = parse_args(["serve", "--db", "dir/cortex.db"], &env).unwrap();
        assert_eq!(
            config.auth_path,
            PathBuf::from("secrets/dashboard-auth.json")
        );
        assert_eq!(config.upstream, "http://127.0.0.1:8100");
        assert_eq!(config.db_path, PathBuf::from("dir/cortex.db"));
    }
}
