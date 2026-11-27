//! Response builders. Security headers are added by middleware so every
//! route, including the proxy, gets the same set.

use axum::body::Body;
use axum::response::Response;
use bytes::Bytes;
use http::header::{CACHE_CONTROL, CONTENT_TYPE, ETAG, SET_COOKIE};
use http::{HeaderValue, StatusCode};
use serde_json::Value;

use crate::assets::CachePolicy;

pub(crate) fn json_response(
    status: StatusCode,
    value: Value,
    set_cookie: Option<String>,
) -> Response {
    let body =
        serde_json::to_vec(&value).unwrap_or_else(|_| br#"{"error":"internal error"}"#.to_vec());
    let mut builder = Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json; charset=utf-8")
        .header(CACHE_CONTROL, "no-store");
    if let Some(cookie) = set_cookie {
        if let Ok(value) = HeaderValue::from_str(&cookie) {
            builder = builder.header(SET_COOKIE, value);
        }
    }
    builder
        .body(Body::from(body))
        .unwrap_or_else(|_| Response::new(Body::from(r#"{"error":"internal error"}"#)))
}

pub(crate) fn bytes_response(
    status: StatusCode,
    content_type: &'static str,
    body: Bytes,
    cache: CachePolicy,
    etag: Option<&str>,
) -> Response {
    let mut builder = Response::builder()
        .status(status)
        .header(CONTENT_TYPE, content_type);
    match cache {
        CachePolicy::NoStore => {
            builder = builder.header(CACHE_CONTROL, "no-store");
        }
        CachePolicy::Asset => {
            builder = builder.header(CACHE_CONTROL, "public, max-age=86400");
            if let Some(etag) = etag {
                let quoted = format!("\"{etag}\"");
                if let Ok(value) = HeaderValue::from_str(&quoted) {
                    builder = builder.header(ETAG, value);
                }
            }
        }
    }
    builder
        .body(Body::from(body))
        .unwrap_or_else(|_| Response::new(Body::from("not found")))
}

pub(crate) fn not_found() -> Response {
    bytes_response(
        StatusCode::NOT_FOUND,
        "text/plain; charset=utf-8",
        Bytes::from_static(b"not found"),
        CachePolicy::NoStore,
        None,
    )
}

pub(crate) fn invalid_json() -> Response {
    json_response(
        StatusCode::BAD_REQUEST,
        serde_json::json!({"error": "invalid JSON request"}),
        None,
    )
}
