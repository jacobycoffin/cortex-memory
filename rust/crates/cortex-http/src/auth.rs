//! `dashboard-auth.json` and `cortex_session`, matching `dashboard_auth.py`.
//!
//! The file is not a SQLite table. Binary fields are URL-safe base64 without
//! padding. A missing file means the gated routes are open. The moment the
//! file exists, they are enforced. Nothing here is written to a log.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, UNIX_EPOCH};

use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use base64::Engine;
use hmac::{Hmac, Mac};
use http::HeaderMap;
use pbkdf2::pbkdf2_hmac;
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::Sha256;
use subtle::ConstantTimeEq;

pub(crate) const SESSION_COOKIE: &str = "cortex_session";
pub(crate) const SESSION_SECONDS: i64 = 12 * 60 * 60;
pub(crate) const PBKDF2_ITERATIONS: u32 = 310_000;
const MAX_ATTEMPTS: usize = 8;
const ATTEMPT_WINDOW: Duration = Duration::from_secs(300);
const MAX_AUTH_BYTES: usize = 65_536;
const USERNAME_LIMIT: usize = 128;
const PASSWORD_LIMIT: usize = 256;

type HmacSha256 = Hmac<Sha256>;

#[derive(Clone)]
pub(crate) struct AuthState {
    pub username: String,
    pub salt: String,
    pub password_hash: String,
    pub session_secret: String,
    pub session_version: i64,
    pub iterations: u32,
    pub must_change_password: bool,
}

impl std::fmt::Debug for AuthState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthState")
            .field("username", &self.username)
            .field("session_version", &self.session_version)
            .field("iterations", &self.iterations)
            .field("must_change_password", &self.must_change_password)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Deserialize)]
struct AuthFile {
    username: String,
    salt: String,
    password_hash: String,
    session_secret: String,
    session_version: i64,
    #[serde(default = "default_iterations")]
    iterations: u32,
    #[serde(default)]
    must_change_password: bool,
}

fn default_iterations() -> u32 {
    PBKDF2_ITERATIONS
}

impl From<AuthFile> for AuthState {
    fn from(file: AuthFile) -> Self {
        Self {
            username: file.username,
            salt: file.salt,
            password_hash: file.password_hash,
            session_secret: file.session_secret,
            session_version: file.session_version,
            iterations: file.iterations,
            must_change_password: file.must_change_password,
        }
    }
}

#[derive(Clone)]
struct CachedAuth {
    generation: u64,
    mtime_ns: u128,
    size: u64,
    state: AuthState,
}

pub(crate) enum AuthLoad {
    /// No auth file. Gated routes are open.
    Open,
    Ready(AuthState),
    /// The file exists but cannot be trusted. Fail closed.
    Failed(&'static str),
}

pub(crate) struct AuthService {
    path: PathBuf,
    generation: Arc<AtomicU64>,
    cache: Mutex<Option<CachedAuth>>,
}

impl AuthService {
    pub(crate) fn new(path: PathBuf, generation: Arc<AtomicU64>) -> Self {
        Self {
            path,
            generation,
            cache: Mutex::new(None),
        }
    }

    pub(crate) fn load(&self) -> AuthLoad {
        let mut cache = self.cache.lock().unwrap_or_else(|err| err.into_inner());
        for _ in 0..3 {
            let before = match file_identity(&self.path) {
                Ok(identity) => identity,
                Err(()) => return AuthLoad::Failed("dashboard auth file is unreadable"),
            };
            let generation = self.generation.load(Ordering::Acquire);
            if let (Some((mtime_ns, size)), Some(cached)) = (before, cache.as_ref()) {
                if cached.generation == generation
                    && cached.mtime_ns == mtime_ns
                    && cached.size == size
                {
                    return AuthLoad::Ready(cached.state.clone());
                }
            }
            let Some((mtime_ns, size)) = before else {
                *cache = None;
                return AuthLoad::Open;
            };
            let bytes = match std::fs::read(&self.path) {
                Ok(bytes) => bytes,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    *cache = None;
                    return AuthLoad::Open;
                }
                Err(_) => return AuthLoad::Failed("dashboard auth file is unreadable"),
            };
            if bytes.len() > MAX_AUTH_BYTES {
                return AuthLoad::Failed("dashboard auth file is incomplete");
            }
            let after = match file_identity(&self.path) {
                Ok(identity) => identity,
                Err(()) => return AuthLoad::Failed("dashboard auth file is unreadable"),
            };
            if after != Some((mtime_ns, size)) {
                continue;
            }
            let text = match std::str::from_utf8(&bytes) {
                Ok(text) => text,
                Err(_) => return AuthLoad::Failed("dashboard auth file is incomplete"),
            };
            let state = match parse_auth(text) {
                Ok(state) => state,
                Err(()) => return AuthLoad::Failed("dashboard auth file is incomplete"),
            };
            *cache = Some(CachedAuth {
                generation,
                mtime_ns,
                size,
                state: state.clone(),
            });
            return AuthLoad::Ready(state);
        }
        AuthLoad::Failed("dashboard auth file changed repeatedly while reading")
    }

    /// Write a new secret and an incremented `session_version`, then return a
    /// token for the new file. Matches `DashboardAuth.change_password` /
    /// `_write_new`: both the secret and the version move, so older cookies
    /// fail closed.
    pub(crate) fn change_password(
        &self,
        current_password: &str,
        new_password: &str,
        now: i64,
    ) -> Result<ChangedPassword, AuthFailure> {
        let state = match self.load() {
            AuthLoad::Ready(state) => state,
            AuthLoad::Open => return Err(AuthFailure::Disabled),
            AuthLoad::Failed(_) => return Err(AuthFailure::Unreadable),
        };
        if !verify_password(&state, &state.username, current_password) {
            return Err(AuthFailure::CurrentPassword);
        }
        if let Some(problem) = validate_new_password(new_password, &state.username) {
            return Err(AuthFailure::InvalidPassword(problem));
        }
        let salt = random_bytes(16).map_err(|_| AuthFailure::Unreadable)?;
        let mut digest = [0u8; 32];
        pbkdf2_hmac::<Sha256>(
            new_password.as_bytes(),
            &salt,
            PBKDF2_ITERATIONS,
            &mut digest,
        );
        let secret = random_bytes(32).map_err(|_| AuthFailure::Unreadable)?;
        let new_state = AuthState {
            username: state.username.clone(),
            salt: encode_token(&salt),
            password_hash: encode_token(&digest),
            session_secret: encode_token(&secret),
            session_version: state.session_version.saturating_add(1),
            iterations: PBKDF2_ITERATIONS,
            must_change_password: false,
        };
        self.persist(&new_state)?;
        let token = issue_session(&new_state, now).map_err(|_| AuthFailure::Unreadable)?;
        Ok(ChangedPassword {
            token,
            username: new_state.username,
        })
    }

    fn persist(&self, state: &AuthState) -> Result<(), AuthFailure> {
        let updated_at = cortex_store::UtcTime::now().to_iso();
        let body = encode_file(state, &updated_at);
        let parent = self.path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent).map_err(|_| AuthFailure::Unreadable)?;
        let temporary = temp_path(&self.path);
        let _ = std::fs::remove_file(&temporary);
        {
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options
                .open(&temporary)
                .map_err(|_| AuthFailure::Unreadable)?;
            file.write_all(body.as_bytes())
                .map_err(|_| AuthFailure::Unreadable)?;
            file.sync_all().map_err(|_| AuthFailure::Unreadable)?;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if std::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o600)).is_err()
            {
                let _ = std::fs::remove_file(&temporary);
                return Err(AuthFailure::Unreadable);
            }
        }
        if std::fs::rename(&temporary, &self.path).is_err() {
            let _ = std::fs::remove_file(&temporary);
            return Err(AuthFailure::Unreadable);
        }
        let mut cache = self.cache.lock().unwrap_or_else(|err| err.into_inner());
        *cache = None;
        Ok(())
    }
}

pub(crate) struct ChangedPassword {
    pub token: String,
    pub username: String,
}

pub(crate) enum AuthFailure {
    Disabled,
    CurrentPassword,
    InvalidPassword(&'static str),
    Unreadable,
}

/// Test and fixture writer. Production password changes go through
/// [`AuthService::change_password`], which always stores
/// [`PBKDF2_ITERATIONS`].
#[cfg(test)]
pub(crate) fn install_password(
    path: &Path,
    username: &str,
    password: &str,
    iterations: u32,
    must_change: bool,
    session_version: i64,
) -> AuthState {
    let salt = random_bytes(16).expect("os rng");
    let mut digest = [0u8; 32];
    pbkdf2_hmac::<Sha256>(password.as_bytes(), &salt, iterations, &mut digest);
    let secret = random_bytes(32).expect("os rng");
    let state = AuthState {
        username: username.to_string(),
        salt: encode_token(&salt),
        password_hash: encode_token(&digest),
        session_secret: encode_token(&secret),
        session_version,
        iterations,
        must_change_password: must_change,
    };
    let body = encode_file(&state, &cortex_store::UtcTime::now().to_iso());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let temporary = temp_path(path);
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary).unwrap();
    file.write_all(body.as_bytes()).unwrap();
    file.sync_all().unwrap();
    drop(file);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    std::fs::rename(&temporary, path).unwrap();
    state
}

pub(crate) fn issue_session(state: &AuthState, now: i64) -> Result<String, ()> {
    let mut payload = Map::new();
    payload.insert("exp".to_string(), Value::from(now + SESSION_SECONDS));
    payload.insert("iat".to_string(), Value::from(now));
    payload.insert("u".to_string(), Value::String(state.username.clone()));
    payload.insert("v".to_string(), Value::from(state.session_version));
    let raw = serde_json::to_vec(&Value::Object(payload)).map_err(|_| ())?;
    let encoded = encode_token(&raw);
    let secret = decode_token(&state.session_secret)?;
    let signature = hmac_sha256(&secret, encoded.as_bytes())?;
    Ok(format!("{encoded}.{}", encode_token(&signature)))
}

pub(crate) fn verify_session(state: &AuthState, token: &str, now: i64) -> bool {
    let Some((encoded, supplied)) = token.split_once('.') else {
        return false;
    };
    let Ok(secret) = decode_token(&state.session_secret) else {
        return false;
    };
    let Ok(expected) = hmac_sha256(&secret, encoded.as_bytes()) else {
        return false;
    };
    let Ok(supplied_bytes) = decode_token(supplied) else {
        return false;
    };
    if !constant_eq(&supplied_bytes, &expected) {
        return false;
    }
    let Ok(payload) = decode_json_b64(encoded) else {
        return false;
    };
    let Some(exp) = payload.get("exp").and_then(Value::as_i64) else {
        return false;
    };
    if exp <= now {
        return false;
    }
    let Some(user) = payload.get("u").and_then(Value::as_str) else {
        return false;
    };
    if !constant_eq(user.as_bytes(), state.username.as_bytes()) {
        return false;
    }
    let version = payload.get("v").and_then(Value::as_i64).unwrap_or(-1);
    version == state.session_version
}

pub(crate) fn verify_password(state: &AuthState, username: &str, password: &str) -> bool {
    if state.iterations == 0 {
        return false;
    }
    let Ok(salt) = decode_token(&state.salt) else {
        return false;
    };
    let Ok(expected) = decode_token(&state.password_hash) else {
        return false;
    };
    let mut actual = [0u8; 32];
    pbkdf2_hmac::<Sha256>(password.as_bytes(), &salt, state.iterations, &mut actual);
    let user_ok = constant_eq(username.as_bytes(), state.username.as_bytes());
    let pass_ok = constant_eq(&actual, &expected);
    user_ok && pass_ok
}

/// `dashboard_auth.validate_new_password`.
pub(crate) fn validate_new_password(password: &str, username: &str) -> Option<&'static str> {
    let chars = password.chars().count();
    if chars < 12 {
        return Some("Use at least 12 characters.");
    }
    if chars > 256 {
        return Some("Password is too long.");
    }
    // Python `casefold()` for the ASCII usernames this file stores. An empty
    // username is contained in every password (`"" in text` is true).
    let password_folded = password.to_lowercase();
    let username_folded = username.to_lowercase();
    if password_folded.contains(&username_folded) {
        Some("Password cannot contain the username.")
    } else {
        None
    }
}

pub(crate) fn session_cookie(token: &str, secure: bool) -> String {
    let mut cookie = format!(
        "{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={SESSION_SECONDS}"
    );
    if secure {
        cookie.push_str("; Secure");
    }
    cookie
}

pub(crate) fn expired_cookie() -> String {
    format!("{SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0")
}

pub(crate) fn forwarded_https(headers: &HeaderMap) -> bool {
    headers
        .get(http::header::HeaderName::from_static("x-forwarded-proto"))
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case("https"))
}

pub(crate) fn cookie_header(headers: &HeaderMap) -> String {
    headers
        .get_all(http::header::COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .collect::<Vec<_>>()
        .join("; ")
}

/// Last `cortex_session` value, matching `SimpleCookie`'s last-assignment rule.
pub(crate) fn session_token<'a>(cookie_header: &'a str) -> Option<&'a str> {
    let mut found = None;
    for part in cookie_header.split(';') {
        let Some((name, value)) = part.split_once('=') else {
            continue;
        };
        if name.trim() == SESSION_COOKIE {
            let value = value.trim().trim_matches('"');
            if !value.is_empty() {
                found = Some(value);
            }
        }
    }
    found
}

pub(crate) fn session_from_headers(state: &AuthState, headers: &HeaderMap, now: i64) -> bool {
    let header = cookie_header(headers);
    match session_token(&header) {
        Some(token) => verify_session(state, token, now),
        None => false,
    }
}

/// `Authorization: Basic ...`, or `None` when the header is absent or malformed.
/// A malformed header does not count as an attempt.
pub(crate) fn basic_credentials(headers: &HeaderMap) -> Option<(String, String)> {
    let value = headers.get(http::header::AUTHORIZATION)?.to_str().ok()?;
    let encoded = value.strip_prefix("Basic ")?;
    if encoded.is_empty() {
        return None;
    }
    let bytes = STANDARD.decode(encoded.trim()).ok()?;
    let text = std::str::from_utf8(&bytes).ok()?;
    let (username, password) = text.split_once(':')?;
    Some((username.to_string(), password.to_string()))
}

pub(crate) fn json_str<'a>(map: &'a Map<String, Value>, key: &str) -> &'a str {
    map.get(key).and_then(Value::as_str).unwrap_or("")
}

pub(crate) fn credential_too_long(username: &str, password: &str) -> bool {
    username.chars().count() > USERNAME_LIMIT || password.chars().count() > PASSWORD_LIMIT
}

pub(crate) struct Throttle {
    attempts: Mutex<HashMap<String, Vec<Instant>>>,
}

impl Throttle {
    pub(crate) fn new() -> Self {
        Self {
            attempts: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn admit(&self, client: &str) -> bool {
        self.admit_at(client, Instant::now())
    }

    pub(crate) fn admit_at(&self, client: &str, now: Instant) -> bool {
        let mut attempts = self.attempts.lock().unwrap_or_else(|err| err.into_inner());
        if attempts.len() > 1024 {
            attempts.retain(|_, stamps| {
                stamps.retain(|stamp| now.saturating_duration_since(*stamp) < ATTEMPT_WINDOW);
                !stamps.is_empty()
            });
        }
        let recent = attempts.entry(client.to_string()).or_default();
        recent.retain(|stamp| now.saturating_duration_since(*stamp) < ATTEMPT_WINDOW);
        if recent.len() >= MAX_ATTEMPTS {
            return false;
        }
        recent.push(now);
        true
    }

    pub(crate) fn clear(&self, client: &str) {
        self.attempts
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .remove(client);
    }
}

pub(crate) fn client_key(
    peer: &str,
    headers: &HeaderMap,
    trusted: &std::collections::BTreeSet<String>,
) -> String {
    if trusted.contains(peer) {
        let forwarded = header_str(headers, "cf-connecting-ip")
            .or_else(|| header_str(headers, "x-forwarded-for"))
            .unwrap_or("");
        let client = forwarded.split(',').next().unwrap_or("").trim();
        if !client.is_empty() {
            return client.to_string();
        }
    }
    peer.to_string()
}

fn header_str<'a>(headers: &'a HeaderMap, name: &'static str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

fn parse_auth(text: &str) -> Result<AuthState, ()> {
    let file: AuthFile = serde_json::from_str(text).map_err(|_| ())?;
    if file.username.is_empty() && file.salt.is_empty() {
        return Err(());
    }
    Ok(file.into())
}

fn encode_file(state: &AuthState, updated_at: &str) -> String {
    let value = serde_json::json!({
        "iterations": state.iterations,
        "must_change_password": state.must_change_password,
        "password_hash": state.password_hash,
        "salt": state.salt,
        "session_secret": state.session_secret,
        "session_version": state.session_version,
        "updated_at": updated_at,
        "username": state.username,
        "version": 1,
    });
    let mut pretty = serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string());
    pretty.push('\n');
    pretty
}

fn encode_token(bytes: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

fn decode_token(value: &str) -> Result<Vec<u8>, ()> {
    URL_SAFE_NO_PAD.decode(value).map_err(|_| ())
}

fn decode_json_b64(encoded: &str) -> Result<Value, ()> {
    let bytes = decode_token(encoded)?;
    serde_json::from_slice(&bytes).map_err(|_| ())
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> Result<[u8; 32], ()> {
    let mut mac = HmacSha256::new_from_slice(key).map_err(|_| ())?;
    mac.update(message);
    let bytes = mac.finalize().into_bytes();
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn constant_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.ct_eq(right).into()
}

fn random_bytes(len: usize) -> Result<Vec<u8>, ()> {
    let mut bytes = vec![0u8; len];
    getrandom::getrandom(&mut bytes).map_err(|_| ())?;
    Ok(bytes)
}

fn temp_path(path: &Path) -> PathBuf {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    path.with_file_name(format!(".{name}.{}.tmp", std::process::id()))
}

fn file_identity(path: &Path) -> Result<Option<(u128, u64)>, ()> {
    match std::fs::metadata(path) {
        Ok(meta) if meta.is_file() => Ok(Some((mtime_ns(&meta), meta.len()))),
        Ok(_) => Err(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(_) => Err(()),
    }
}

fn mtime_ns(meta: &std::fs::Metadata) -> u128 {
    meta.modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn password_rules_match_python() {
        assert_eq!(
            validate_new_password("short", "ada"),
            Some("Use at least 12 characters.")
        );
        assert_eq!(
            validate_new_password(&"x".repeat(257), "ada"),
            Some("Password is too long.")
        );
        assert_eq!(
            validate_new_password("something-about-Ada-here", "ada"),
            Some("Password cannot contain the username.")
        );
        assert_eq!(
            validate_new_password("replacement-secret-value", "ada"),
            None
        );
        assert_eq!(
            validate_new_password("replacement-secret-value", ""),
            Some("Password cannot contain the username.")
        );
    }

    #[test]
    fn issued_token_roundtrips_and_rejects_expiry_and_version() {
        let state = AuthState {
            username: "ada".to_string(),
            salt: encode_token(&[7u8; 16]),
            password_hash: encode_token(&[9u8; 32]),
            session_secret: encode_token(&[3u8; 32]),
            session_version: 4,
            iterations: 1,
            must_change_password: false,
        };
        let now = 1_700_000_000;
        let token = issue_session(&state, now).unwrap();
        assert!(verify_session(&state, &token, now));
        assert!(verify_session(&state, &token, now + SESSION_SECONDS - 1));
        assert!(!verify_session(&state, &token, now + SESSION_SECONDS));
        let mut bumped = state.clone();
        bumped.session_version = 5;
        assert!(!verify_session(&bumped, &token, now));
        let mut renamed = state.clone();
        renamed.username = "ada2".to_string();
        assert!(!verify_session(&renamed, &token, now));
        let (encoded, _) = token.split_once('.').unwrap();
        let payload: Value = decode_json_b64(encoded).unwrap();
        assert_eq!(
            payload,
            serde_json::json!({"exp": now + SESSION_SECONDS, "iat": now, "u": "ada", "v": 4})
        );
    }

    #[test]
    fn throttle_reserves_eight_slots_and_expires_them() {
        let throttle = Throttle::new();
        let start = Instant::now();
        for _ in 0..8 {
            assert!(throttle.admit_at("client", start));
        }
        assert!(!throttle.admit_at("client", start + Duration::from_secs(299)));
        assert!(throttle.admit_at("other", start));
        assert!(throttle.admit_at("client", start + Duration::from_secs(300)));
        throttle.clear("other");
        assert!(throttle.admit_at("other", start));
    }

    #[test]
    fn basic_header_splits_on_the_first_colon() {
        let mut headers = HeaderMap::new();
        let token = STANDARD.encode(b"ada:secret:tail");
        headers.insert(
            http::header::AUTHORIZATION,
            format!("Basic {token}").parse().unwrap(),
        );
        let (user, password) = basic_credentials(&headers).unwrap();
        assert_eq!(user, "ada");
        assert_eq!(password, "secret:tail");
        headers.insert(http::header::AUTHORIZATION, "Basic !!!".parse().unwrap());
        assert!(basic_credentials(&headers).is_none());
    }

    #[test]
    fn cookie_parser_uses_the_last_session() {
        let header = "a=b; cortex_session=one; cortex_session=two";
        assert_eq!(session_token(header), Some("two"));
    }
}
