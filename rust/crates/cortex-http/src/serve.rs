//! Axum server. Rust owns static files, `/api/pulse`, and the auth routes.
//! Everything else is proxied. Host allowlisting applies only to the routes
//! Rust answers.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::Router;
use http::header::{HeaderName, HOST};
use http::{HeaderMap, HeaderValue, Method, StatusCode};
use http_body_util::{BodyExt, Limited};
use serde_json::{json, Map, Value};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use crate::assets::{static_route, vendor_path, CachePolicy, JS_CONTENT_TYPE};
use crate::auth::{
    basic_credentials, client_key, credential_too_long, expired_cookie, forwarded_https,
    issue_session, json_str, session_cookie, session_from_headers, verify_password, AuthFailure,
    AuthLoad, AuthService, Throttle,
};
use crate::config::{host_allowed, parse_args, ProcessEnv, ServeConfig, ServeError};
use crate::proxy::{self, build_client};
use crate::pulse;
use crate::reply::{bytes_response, invalid_json, json_response, not_found};
use crate::state::{AppState, Inner};
use crate::util::unix_now;

const CSP: &str = "default-src 'self'; style-src 'self' 'unsafe-inline'; \
script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; \
img-src 'self' data:; worker-src 'self' blob:; form-action 'self'; \
base-uri 'none'; frame-ancestors 'none'";

const JSON_LIMIT: usize = 16_384;

/// A running daemon. Dropping the handle asks it to stop; [`Self::shutdown`]
/// waits until accepted requests have finished.
pub struct ServerHandle {
    local_addr: SocketAddr,
    shutdown: Mutex<Option<oneshot::Sender<()>>>,
    generation: Arc<AtomicU64>,
    task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl ServerHandle {
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Drop the asset cache and re-read the auth file on the next request.
    /// SIGHUP calls this.
    pub fn reload(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    pub async fn shutdown(self) {
        let sender = {
            let mut guard = self.shutdown.lock().unwrap_or_else(|err| err.into_inner());
            guard.take()
        };
        let task = {
            let mut guard = self.task.lock().unwrap_or_else(|err| err.into_inner());
            guard.take()
        };
        if let Some(sender) = sender {
            let _ = sender.send(());
        }
        if let Some(task) = task {
            let _ = task.await;
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        if let Some(sender) = self
            .shutdown
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .take()
        {
            let _ = sender.send(());
        }
    }
}

pub async fn start(config: ServeConfig) -> Result<ServerHandle, ServeError> {
    let listener = TcpListener::bind(config.bind)
        .await
        .map_err(|err| ServeError::Message(format!("bind {}: {err}", config.bind)))?;
    let local_addr = listener
        .local_addr()
        .map_err(|err| ServeError::Message(format!("bind {}: {err}", config.bind)))?;
    let generation = Arc::new(AtomicU64::new(1));
    let inner = Arc::new(Inner {
        auth: AuthService::new(config.auth_path.clone(), Arc::clone(&generation)),
        assets: crate::assets::AssetStore::new(),
        throttle: Throttle::new(),
        proxy: build_client(),
        generation: Arc::clone(&generation),
        config,
    });
    let app = Router::new()
        .fallback(handle)
        .layer(middleware::from_fn(security_headers))
        .with_state(AppState { inner });
    let (sender, receiver) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        let server = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(async move {
            let _ = receiver.await;
        });
        if let Err(err) = server.await {
            eprintln!("cortex-http accept loop: {err}");
        }
    });
    Ok(ServerHandle {
        local_addr,
        shutdown: Mutex::new(Some(sender)),
        generation,
        task: Mutex::new(Some(task)),
    })
}

pub(crate) fn run<I, S>(args: I) -> i32
where
    I: IntoIterator<Item = S>,
    S: Into<std::ffi::OsString>,
{
    let config = match parse_args(args, &ProcessEnv::from_process()) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("{err}");
            return 2;
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("tokio runtime: {err}");
            return 1;
        }
    };
    match runtime.block_on(serve_until_signal(config)) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("{err}");
            if err.is_usage() {
                2
            } else {
                1
            }
        }
    }
}

#[cfg(unix)]
async fn serve_until_signal(config: ServeConfig) -> Result<(), ServeError> {
    use tokio::signal::unix::{signal, SignalKind};
    let server = start(config).await?;
    eprintln!("cortex-http listening on {}", server.local_addr());
    let mut terminate = signal(SignalKind::terminate())
        .map_err(|err| ServeError::Message(format!("signal handler: {err}")))?;
    let mut interrupt = signal(SignalKind::interrupt())
        .map_err(|err| ServeError::Message(format!("signal handler: {err}")))?;
    let mut hangup = signal(SignalKind::hangup())
        .map_err(|err| ServeError::Message(format!("signal handler: {err}")))?;
    loop {
        tokio::select! {
            _ = terminate.recv() => break,
            _ = interrupt.recv() => break,
            _ = hangup.recv() => server.reload(),
        }
    }
    server.shutdown().await;
    Ok(())
}

#[cfg(not(unix))]
async fn serve_until_signal(_config: ServeConfig) -> Result<(), ServeError> {
    Err(ServeError::Message(
        "serve requires a unix signal runtime".to_string(),
    ))
}

async fn security_headers(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    if !headers.contains_key(http::header::CACHE_CONTROL) {
        headers.insert(
            http::header::CACHE_CONTROL,
            HeaderValue::from_static("no-store"),
        );
    }
    insert_static(headers, http::header::X_CONTENT_TYPE_OPTIONS, "nosniff");
    insert_static(headers, http::header::X_FRAME_OPTIONS, "DENY");
    insert_static(headers, http::header::REFERRER_POLICY, "no-referrer");
    insert_static(
        headers,
        HeaderName::from_static("permissions-policy"),
        "camera=(), microphone=(), geolocation=()",
    );
    insert_static(
        headers,
        HeaderName::from_static("content-security-policy"),
        CSP,
    );
    response
}

fn insert_static(headers: &mut HeaderMap, name: HeaderName, value: &'static str) {
    if let Ok(value) = HeaderValue::from_str(value) {
        headers.insert(name, value);
    }
}

fn owned(method: &Method, path: &str) -> bool {
    match *method {
        Method::GET => {
            static_route(path).is_some()
                || path == "/api/pulse"
                || path == "/api/auth/status"
                || path.starts_with("/vendor/")
        }
        Method::POST => matches!(
            path,
            "/api/auth/login" | "/api/auth/logout" | "/api/auth/change-password"
        ),
        _ => false,
    }
}

async fn handle(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    req: Request,
) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    if !owned(&method, &path) {
        return proxy::forward(&state, req).await;
    }
    let query = req.uri().query().map(str::to_owned);
    let headers = req.headers().clone();
    let host = headers.get(HOST).and_then(|value| value.to_str().ok());
    if !host_allowed(&state.inner.config.allowed_hosts, host) {
        return json_response(
            StatusCode::FORBIDDEN,
            json!({"error": "host not allowed"}),
            None,
        );
    }
    match (method, path.as_str()) {
        (Method::GET, "/api/pulse") => pulse(&state, &headers, &peer, query.as_deref()).await,
        (Method::GET, "/api/auth/status") => status(&state, &headers, &peer).await,
        (Method::POST, "/api/auth/login") => login(&state, &headers, &peer, req.into_body()).await,
        (Method::POST, "/api/auth/logout") => logout(&state, &headers, &peer).await,
        (Method::POST, "/api/auth/change-password") => {
            change_password(&state, &headers, &peer, req.into_body()).await
        }
        (Method::GET, _) => static_file(&state, &path).await,
        _ => not_found(),
    }
}

async fn pulse(
    state: &AppState,
    headers: &HeaderMap,
    peer: &SocketAddr,
    query: Option<&str>,
) -> Response {
    if let Err(response) = gate(state, headers, peer, true).await {
        return response;
    }
    let db = state.inner.config.db_path.clone();
    let (cols, rows) = pulse::dimensions(query);
    let joined = tokio::task::spawn_blocking(move || -> Result<Value, cortex_store::StoreError> {
        let store = cortex_store::Store::open(&db)?;
        pulse::body(&store, cols, rows, cortex_store::UtcTime::now())
    })
    .await;
    match joined {
        Ok(Ok(value)) => json_response(StatusCode::OK, value, None),
        Ok(Err(err)) => json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({
                "error": "store unavailable",
                "note": err.to_string(),
                "schema_version": 1
            }),
            None,
        ),
        Err(_) => json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({
                "error": "store unavailable",
                "note": "pulse read failed",
                "schema_version": 1
            }),
            None,
        ),
    }
}

async fn status(state: &AppState, headers: &HeaderMap, peer: &SocketAddr) -> Response {
    match state.inner.auth.load() {
        AuthLoad::Open => json_response(
            StatusCode::OK,
            json!({
                "auth_enabled": false,
                "authenticated": true,
                "must_change_password": false,
                "username": ""
            }),
            None,
        ),
        AuthLoad::Failed(message) => auth_failed(message),
        AuthLoad::Ready(auth_state) => {
            let authenticated = if session_from_headers(&auth_state, headers, unix_now()) {
                true
            } else {
                basic_authenticates(state, headers, peer, &auth_state).await
            };
            let username = if authenticated {
                auth_state.username.clone()
            } else {
                String::new()
            };
            let must_change = authenticated && auth_state.must_change_password;
            json_response(
                StatusCode::OK,
                json!({
                    "auth_enabled": true,
                    "authenticated": authenticated,
                    "must_change_password": must_change,
                    "username": username
                }),
                None,
            )
        }
    }
}

async fn login(state: &AppState, headers: &HeaderMap, peer: &SocketAddr, body: Body) -> Response {
    if let Some(response) = require_request_header(headers) {
        return response;
    }
    match state.inner.auth.load() {
        AuthLoad::Open => {
            return json_response(
                StatusCode::BAD_REQUEST,
                json!({"error": "authentication is disabled"}),
                None,
            );
        }
        AuthLoad::Failed(message) => return auth_failed(message),
        AuthLoad::Ready(_) => {}
    }
    let client = client_key(
        &peer.ip().to_string(),
        headers,
        &state.inner.config.trusted_proxies,
    );
    if !state.inner.throttle.admit(&client) {
        return json_response(
            StatusCode::TOO_MANY_REQUESTS,
            json!({"error": "Too many attempts. Try again shortly."}),
            None,
        );
    }
    let map = match read_json(headers, body).await {
        Ok(map) => map,
        Err(response) => return response,
    };
    let username = json_str(&map, "username").to_string();
    let password = json_str(&map, "password").to_string();
    let secure = forwarded_https(headers);
    let inner = Arc::clone(&state.inner);
    let client_for_clear = client.clone();
    let outcome = tokio::task::spawn_blocking(move || -> Result<LoginOk, LoginError> {
        let auth_state = match inner.auth.load() {
            AuthLoad::Ready(state) => state,
            AuthLoad::Open => return Err(LoginError::Disabled),
            AuthLoad::Failed(_) => return Err(LoginError::Broken),
        };
        if credential_too_long(&username, &password)
            || !verify_password(&auth_state, &username, &password)
        {
            return Err(LoginError::Rejected);
        }
        inner.throttle.clear(&client_for_clear);
        let token = issue_session(&auth_state, unix_now()).map_err(|_| LoginError::Broken)?;
        Ok(LoginOk {
            token,
            must_change: auth_state.must_change_password,
            username: auth_state.username,
        })
    })
    .await;
    match outcome {
        Ok(Ok(ok)) => json_response(
            StatusCode::OK,
            json!({
                "authenticated": true,
                "must_change_password": ok.must_change,
                "username": ok.username
            }),
            Some(session_cookie(&ok.token, secure)),
        ),
        Ok(Err(LoginError::Disabled)) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"error": "authentication is disabled"}),
            None,
        ),
        Ok(Err(LoginError::Rejected)) => json_response(
            StatusCode::UNAUTHORIZED,
            json!({"error": "Username or password is incorrect."}),
            None,
        ),
        Ok(Err(LoginError::Broken)) | Err(_) => auth_failed("dashboard auth file is unreadable"),
    }
}

struct LoginOk {
    token: String,
    must_change: bool,
    username: String,
}

enum LoginError {
    Disabled,
    Rejected,
    Broken,
}

async fn logout(state: &AppState, headers: &HeaderMap, peer: &SocketAddr) -> Response {
    if let Some(response) = require_request_header(headers) {
        return response;
    }
    match state.inner.auth.load() {
        AuthLoad::Open => json_response(
            StatusCode::OK,
            json!({"success": true}),
            Some(expired_cookie()),
        ),
        AuthLoad::Failed(message) => auth_failed(message),
        AuthLoad::Ready(auth_state) => {
            if !session_ok(state, headers, peer, &auth_state).await {
                return unauthorized();
            }
            json_response(
                StatusCode::OK,
                json!({"success": true}),
                Some(expired_cookie()),
            )
        }
    }
}

async fn change_password(
    state: &AppState,
    headers: &HeaderMap,
    peer: &SocketAddr,
    body: Body,
) -> Response {
    if let Some(response) = require_request_header(headers) {
        return response;
    }
    let auth_state = match state.inner.auth.load() {
        AuthLoad::Open => {
            return json_response(
                StatusCode::BAD_REQUEST,
                json!({"error": "authentication is disabled"}),
                None,
            );
        }
        AuthLoad::Failed(message) => return auth_failed(message),
        AuthLoad::Ready(auth_state) => auth_state,
    };
    if !session_ok(state, headers, peer, &auth_state).await {
        return unauthorized();
    }
    let map = match read_json(headers, body).await {
        Ok(map) => map,
        Err(response) => return response,
    };
    let current = json_str(&map, "current_password").to_string();
    let new_password = json_str(&map, "new_password").to_string();
    let secure = forwarded_https(headers);
    let inner = Arc::clone(&state.inner);
    let outcome = tokio::task::spawn_blocking(move || {
        inner
            .auth
            .change_password(&current, &new_password, unix_now())
    })
    .await;
    match outcome {
        Ok(Ok(changed)) => json_response(
            StatusCode::OK,
            json!({
                "success": true,
                "must_change_password": false,
                "username": changed.username
            }),
            Some(session_cookie(&changed.token, secure)),
        ),
        Ok(Err(AuthFailure::Disabled)) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"error": "authentication is disabled"}),
            None,
        ),
        Ok(Err(AuthFailure::CurrentPassword)) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"error": "Current password is incorrect."}),
            None,
        ),
        Ok(Err(AuthFailure::InvalidPassword(message))) => {
            json_response(StatusCode::BAD_REQUEST, json!({"error": message}), None)
        }
        Ok(Err(AuthFailure::Unreadable)) | Err(_) => auth_failed("could not update the password"),
    }
}

/// `complete` matches `dashboard.py`'s `_require_auth(complete=...)`.
/// A session that still has `must_change_password` is rejected only when
/// `complete` is true (`/api/pulse`). Logout and change-password pass false.
async fn gate(
    state: &AppState,
    headers: &HeaderMap,
    peer: &SocketAddr,
    complete: bool,
) -> Result<(), Response> {
    match state.inner.auth.load() {
        AuthLoad::Open => Ok(()),
        AuthLoad::Failed(message) => Err(auth_failed(message)),
        AuthLoad::Ready(auth_state) => {
            if !session_ok(state, headers, peer, &auth_state).await {
                return Err(unauthorized());
            }
            if complete && auth_state.must_change_password {
                return Err(json_response(
                    StatusCode::FORBIDDEN,
                    json!({"error": "password change required", "must_change_password": true}),
                    None,
                ));
            }
            Ok(())
        }
    }
}

async fn session_ok(
    state: &AppState,
    headers: &HeaderMap,
    peer: &SocketAddr,
    auth_state: &crate::auth::AuthState,
) -> bool {
    if session_from_headers(auth_state, headers, unix_now()) {
        return true;
    }
    basic_authenticates(state, headers, peer, auth_state).await
}

async fn basic_authenticates(
    state: &AppState,
    headers: &HeaderMap,
    peer: &SocketAddr,
    auth_state: &crate::auth::AuthState,
) -> bool {
    let Some((username, password)) = basic_credentials(headers) else {
        return false;
    };
    let client = client_key(
        &peer.ip().to_string(),
        headers,
        &state.inner.config.trusted_proxies,
    );
    if !state.inner.throttle.admit(&client) {
        return false;
    }
    let iterations_state = auth_state.clone();
    let ok = tokio::task::spawn_blocking(move || {
        verify_password(&iterations_state, &username, &password)
    })
    .await
    .unwrap_or(false);
    if ok {
        state.inner.throttle.clear(&client);
    }
    ok
}

fn require_request_header(headers: &HeaderMap) -> Option<Response> {
    let ok = headers
        .get("x-cortex-request")
        .and_then(|value| value.to_str().ok())
        == Some("1");
    if ok {
        None
    } else {
        Some(json_response(
            StatusCode::FORBIDDEN,
            json!({"error": "request verification failed"}),
            None,
        ))
    }
}

fn unauthorized() -> Response {
    json_response(
        StatusCode::UNAUTHORIZED,
        json!({"error": "authentication required"}),
        None,
    )
}

fn auth_failed(message: &'static str) -> Response {
    json_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        json!({"error": message}),
        None,
    )
}

async fn read_json(headers: &HeaderMap, body: Body) -> Result<Map<String, Value>, Response> {
    let declared = headers
        .get(http::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|text| text.parse::<usize>().ok());
    let bytes = match Limited::new(body, JSON_LIMIT).collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => return Err(invalid_json()),
    };
    let Some(declared) = declared else {
        return Err(invalid_json());
    };
    if !(1..=JSON_LIMIT).contains(&declared) || bytes.len() != declared {
        return Err(invalid_json());
    }
    match serde_json::from_slice::<Value>(&bytes) {
        Ok(Value::Object(map)) => Ok(map),
        _ => Err(invalid_json()),
    }
}

async fn static_file(state: &AppState, path: &str) -> Response {
    let generation = state.inner.generation.load(Ordering::Acquire);
    if let Some(route) = static_route(path) {
        let file = state.inner.config.asset_dir.join(route.file);
        return match state.inner.assets.load(generation, route.file, &file) {
            Ok(cached) => bytes_response(
                StatusCode::OK,
                route.content_type,
                cached.bytes,
                route.cache,
                Some(&cached.etag),
            ),
            Err(_) => not_found(),
        };
    }
    if path.starts_with("/vendor/") {
        let Some(file) = vendor_path(&state.inner.config.asset_dir, path) else {
            return not_found();
        };
        return match state.inner.assets.load(generation, path, &file) {
            Ok(cached) => bytes_response(
                StatusCode::OK,
                JS_CONTENT_TYPE,
                cached.bytes,
                CachePolicy::Asset,
                Some(&cached.etag),
            ),
            Err(_) => not_found(),
        };
    }
    not_found()
}
