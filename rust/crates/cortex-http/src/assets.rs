//! Static files from `CORTEX_ASSET_DIR`. Re-read when mtime or size changes,
//! and when SIGHUP bumps the generation counter. A missing file is a 404,
//! not a generated page.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

use bytes::Bytes;

use crate::util::percent_decode;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum CachePolicy {
    /// HTML and `/api/*`.
    NoStore,
    /// `dashboard.css`, `dashboard.js`, and `/vendor/*`.
    Asset,
}

pub(crate) struct StaticRoute {
    pub file: &'static str,
    pub content_type: &'static str,
    pub cache: CachePolicy,
}

const HTML: &str = "text/html; charset=utf-8";
const CSS: &str = "text/css; charset=utf-8";
const JS: &str = "text/javascript; charset=utf-8";

pub(crate) fn static_route(path: &str) -> Option<StaticRoute> {
    Some(match path {
        "/" => StaticRoute {
            file: "dashboard.html",
            content_type: HTML,
            cache: CachePolicy::NoStore,
        },
        "/pulse" => StaticRoute {
            file: "pulse.html",
            content_type: HTML,
            cache: CachePolicy::NoStore,
        },
        "/phone" => StaticRoute {
            file: "phone.html",
            content_type: HTML,
            cache: CachePolicy::NoStore,
        },
        "/dashboard.css" => StaticRoute {
            file: "dashboard.css",
            content_type: CSS,
            cache: CachePolicy::Asset,
        },
        "/dashboard.js" => StaticRoute {
            file: "dashboard.js",
            content_type: JS,
            cache: CachePolicy::Asset,
        },
        "/favicon.svg" => StaticRoute {
            file: "favicon.svg",
            content_type: "image/svg+xml",
            cache: CachePolicy::NoStore,
        },
        "/favicon.ico" => StaticRoute {
            file: "favicon.ico",
            content_type: "image/x-icon",
            cache: CachePolicy::NoStore,
        },
        "/apple-touch-icon.png" => StaticRoute {
            file: "apple-touch-icon.png",
            content_type: "image/png",
            cache: CachePolicy::NoStore,
        },
        _ => return None,
    })
}

#[derive(Clone)]
pub(crate) struct CachedFile {
    generation: u64,
    mtime_ns: u128,
    size: u64,
    pub etag: String,
    pub bytes: Bytes,
}

pub(crate) struct AssetStore {
    cache: Mutex<HashMap<String, CachedFile>>,
}

impl AssetStore {
    pub(crate) fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Return the file bytes, using the cache only when generation, mtime, and
    /// size all still match. A missing file is `NotFound` and is not cached.
    pub(crate) fn load(
        &self,
        generation: u64,
        key: &str,
        path: &Path,
    ) -> std::io::Result<CachedFile> {
        let meta = std::fs::metadata(path)?;
        if !meta.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "not a file",
            ));
        }
        let mtime_ns = mtime_ns(&meta);
        let size = meta.len();
        {
            let cache = self.cache.lock().unwrap_or_else(|err| err.into_inner());
            if let Some(hit) = cache.get(key) {
                if hit.generation == generation && hit.mtime_ns == mtime_ns && hit.size == size {
                    return Ok(hit.clone());
                }
            }
        }
        let bytes = Bytes::from(std::fs::read(path)?);
        let cached = CachedFile {
            generation,
            mtime_ns,
            size,
            etag: format!("{mtime_ns}-{size}"),
            bytes,
        };
        self.cache
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .insert(key.to_string(), cached.clone());
        Ok(cached)
    }
}

/// Resolve `/vendor/...` to a `.js` file that stays inside `asset_dir/vendor`.
///
/// `..`, absolute paths, and symlinks that leave the directory are rejected.
/// A missing file is `None` (the route is still ours, so the caller sends 404
/// rather than proxying).
pub(crate) fn vendor_path(asset_dir: &Path, request_path: &str) -> Option<PathBuf> {
    let rest = request_path.strip_prefix("/vendor/")?;
    if rest.is_empty() || rest.contains('\0') || rest.contains('\\') {
        return None;
    }
    let decoded = percent_decode(rest, false);
    if decoded.is_empty() || decoded.contains('\0') || decoded.contains('\\') {
        return None;
    }
    if !decoded.to_ascii_lowercase().ends_with(".js") {
        return None;
    }
    let relative = Path::new(&decoded);
    if relative.is_absolute() {
        return None;
    }
    if relative
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return None;
    }
    let vendor_root = asset_dir.join("vendor");
    let root = vendor_root.canonicalize().ok()?;
    let file = vendor_root.join(relative).canonicalize().ok()?;
    if file.is_file() && file.starts_with(&root) {
        Some(file)
    } else {
        None
    }
}

pub(crate) const JS_CONTENT_TYPE: &str = JS;

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

    fn scratch() -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("cortex-http-assets-{nanos}-{}", std::process::id()));
        std::fs::create_dir_all(path.join("vendor")).unwrap();
        path
    }

    #[test]
    fn vendor_rejects_escape_and_non_js() {
        let dir = scratch();
        let js = dir.join("vendor").join("app.js");
        std::fs::write(&js, b"var x=1;\n").unwrap();
        assert!(vendor_path(&dir, "/vendor/app.js").is_some());
        assert!(vendor_path(&dir, "/vendor/app.txt").is_none());
        assert!(vendor_path(&dir, "/vendor/%2e%2e/app.js").is_none());
        assert!(vendor_path(&dir, "/vendor/../secret.js").is_none());
        assert!(vendor_path(&dir, "/vendor/foo/../../app.js").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn vendor_rejects_a_symlink_that_leaves_the_directory() {
        let dir = scratch();
        let link = dir.join("vendor").join("host.js");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/etc/hostname", &link).unwrap();
            assert!(vendor_path(&dir, "/vendor/host.js").is_none());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_reloads_when_mtime_or_generation_changes() {
        let dir = scratch();
        let file = dir.join("pulse.html");
        std::fs::write(&file, b"aaaaaaaa").unwrap();
        let store = AssetStore::new();
        let first = store.load(1, "pulse.html", &file).unwrap();
        assert_eq!(&first.bytes[..], b"aaaaaaaa");
        let cached = store.load(1, "pulse.html", &file).unwrap();
        assert_eq!(cached.bytes, first.bytes);

        let modified = std::fs::metadata(&file).unwrap().modified().unwrap();
        std::fs::write(&file, b"bbbbbbbb").unwrap();
        let handle = std::fs::File::options().write(true).open(&file).unwrap();
        handle.set_modified(modified).unwrap();
        drop(handle);
        let still = store.load(1, "pulse.html", &file).unwrap();
        assert_eq!(&still.bytes[..], b"aaaaaaaa");
        let reloaded = store.load(2, "pulse.html", &file).unwrap();
        assert_eq!(&reloaded.bytes[..], b"bbbbbbbb");

        std::fs::write(&file, b"cccc").unwrap();
        let changed = store.load(2, "pulse.html", &file).unwrap();
        assert_eq!(&changed.bytes[..], b"cccc");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
