//! Reverse proxy for every path Rust does not own.
//!
//! `/api/snapshot` and `/api/sleep/status` stay in Python so those numbers
//! have one implementation. When the upstream is down or the timeout fires,
//! the response is 503 with the spec note — never an empty snapshot.

use bytes::Bytes;
use http::header::{
    CACHE_CONTROL, CONTENT_TYPE, ETAG, HOST, LAST_MODIFIED, LOCATION, SET_COOKIE, VARY,
    WWW_AUTHENTICATE,
};
use http::{HeaderMap, Request, Response, StatusCode};
use http_body_util::{BodyExt, Full, Limited};
use hyper::body::Incoming;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use serde_json::json;

use crate::reply::json_response;
use crate::state::AppState;

pub(crate) type ProxyClient = Client<HttpConnector, Full<Bytes>>;

const PROXY_BODY_LIMIT: usize = 2 * 1024 * 1024;

pub(crate) fn build_client() -> ProxyClient {
    Client::builder(TokioExecutor::new()).build(HttpConnector::new())
}

pub(crate) async fn forward(
    state: &AppState,
    req: Request<axum::body::Body>,
) -> Response<axum::body::Body> {
    let method = req.method().clone();
    let headers = req.headers().clone();
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|value| value.as_str().to_string())
        .unwrap_or_else(|| "/".to_string());
    let body = match Limited::new(req.into_body(), PROXY_BODY_LIMIT)
        .collect()
        .await
    {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            return json_response(
                StatusCode::PAYLOAD_TOO_LARGE,
                json!({"error": "request too large"}),
                None,
            );
        }
    };
    let upstream = format!(
        "{}{}",
        state.inner.config.upstream.trim_end_matches('/'),
        if path_and_query.starts_with('/') {
            path_and_query
        } else {
            format!("/{path_and_query}")
        }
    );
    let Ok(uri) = upstream.parse::<http::Uri>() else {
        return snapshot_unavailable();
    };
    let mut outbound = Request::builder()
        .method(method)
        .uri(uri)
        .body(Full::new(body))
        .unwrap_or_else(|_| {
            Request::builder()
                .body(Full::new(Bytes::new()))
                .expect("empty request")
        });
    copy_request_headers(&headers, outbound.headers_mut());
    // Keep the caller's Host so Python's allowlist sees the public name.
    // hyper only fills Host when it is absent.
    if let Some(host) = headers.get(HOST) {
        outbound.headers_mut().insert(HOST, host.clone());
    }

    let client = state.inner.proxy.clone();
    let timeout = state.inner.config.upstream_timeout;
    match tokio::time::timeout(timeout, client.request(outbound)).await {
        Ok(Ok(response)) => match read_response(response).await {
            Ok(response) => response,
            Err(()) => snapshot_unavailable(),
        },
        Ok(Err(_)) | Err(_) => snapshot_unavailable(),
    }
}

fn copy_request_headers(from: &HeaderMap, to: &mut HeaderMap) {
    for name in [
        "cookie",
        "authorization",
        "x-forwarded-for",
        "x-forwarded-proto",
        "x-cortex-request",
        "content-type",
    ] {
        if let Some(value) = from.get(name) {
            to.insert(name, value.clone());
        }
    }
}

async fn read_response(response: Response<Incoming>) -> Result<Response<axum::body::Body>, ()> {
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = Limited::new(response.into_body(), 8 * 1024 * 1024)
        .collect()
        .await
        .map_err(|_| ())?
        .to_bytes();
    let mut builder = Response::builder().status(status);
    if let Some(map) = builder.headers_mut() {
        copy_response_headers(&headers, map);
    }
    builder.body(axum::body::Body::from(bytes)).map_err(|_| ())
}

fn copy_response_headers(from: &HeaderMap, to: &mut HeaderMap) {
    for name in [
        &CONTENT_TYPE,
        &CACHE_CONTROL,
        &ETAG,
        &LOCATION,
        &WWW_AUTHENTICATE,
        &VARY,
        &LAST_MODIFIED,
    ] {
        if let Some(value) = from.get(name) {
            to.insert(name.clone(), value.clone());
        }
    }
    for value in from.get_all(SET_COOKIE) {
        to.append(SET_COOKIE, value.clone());
    }
}

pub(crate) fn snapshot_unavailable() -> Response<axum::body::Body> {
    json_response(
        StatusCode::SERVICE_UNAVAILABLE,
        json!({
            "error": "snapshot unavailable",
            "note": "python dashboard upstream is not running",
            "schema_version": 1
        }),
        None,
    )
}
