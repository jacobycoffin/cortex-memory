//! Open an existing database without creating it.

use std::path::Path;
use std::time::Duration;

use rusqlite::{Connection, OpenFlags};

use crate::error::{map_open_error, map_query_error, StoreError, REQUIRED_TABLES};

/// Milliseconds SQLite retries a locked read before [`StoreError::Unavailable`].
pub const BUSY_TIMEOUT_MS: u64 = 5000;

/// A read-only connection to an existing Cortex database.
///
/// The connection is `Send` and not `Sync`. Each method runs one statement and
/// leaves the connection idle. Do not hold it across a poll interval.
pub struct Store {
    pub(crate) conn: Connection,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Store")
    }
}

impl Store {
    /// Open `path` if it is an existing file.
    ///
    /// The URI is `file:{absolute}?mode=ro` with the read-only flag. The next
    /// statements are `PRAGMA query_only=ON` and `PRAGMA busy_timeout=5000`.
    /// A missing file, a directory, or a failed open returns
    /// [`StoreError::Unreadable`] and does not create a file or a directory.
    /// Open is never retried read-write. `PRAGMA journal_mode`,
    /// `wal_checkpoint`, and `foreign_keys` are not issued.
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        if !path.is_file() {
            return Err(StoreError::Unreadable {
                message: format!("database file does not exist: {}", path.display()),
            });
        }
        let absolute = path.canonicalize().map_err(|err| StoreError::Unreadable {
            message: format!("database file does not exist: {err}"),
        })?;
        let uri = readonly_uri(&absolute);
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI;
        let conn = Connection::open_with_flags(&uri, flags).map_err(map_open_error)?;
        conn.busy_timeout(Duration::from_millis(BUSY_TIMEOUT_MS))
            .map_err(map_open_error)?;
        // query_only is a connection flag, not a schema write. busy_timeout is
        // set again as the pragma the coexistence contract names.
        conn.pragma_update(None, "query_only", "ON")
            .map_err(map_open_error)?;
        conn.pragma_update(None, "busy_timeout", BUSY_TIMEOUT_MS as i64)
            .map_err(map_open_error)?;
        let store = Self { conn };
        store.ensure_required_tables()?;
        Ok(store)
    }

    /// `PRAGMA query_only` is on. Reads stay reads.
    pub fn is_query_only(&self) -> Result<bool, StoreError> {
        let value: i64 = self
            .conn
            .query_row("PRAGMA query_only", [], |row| row.get(0))
            .map_err(map_query_error)?;
        Ok(value != 0)
    }

    /// The busy timeout in milliseconds. The open path sets 5000.
    pub fn busy_timeout_ms(&self) -> Result<i64, StoreError> {
        self.conn
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .map_err(map_query_error)
    }

    pub(crate) fn has_table(&self, name: &str) -> Result<bool, StoreError> {
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
                [name],
                |row| row.get(0),
            )
            .map_err(map_query_error)?;
        Ok(count > 0)
    }

    fn ensure_required_tables(&self) -> Result<(), StoreError> {
        let mut missing = Vec::new();
        for name in REQUIRED_TABLES {
            if !self.has_table(name)? {
                missing.push((*name).to_string());
            }
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(StoreError::NotACortexDatabase { missing })
        }
    }
}

pub(crate) fn readonly_uri(absolute: &Path) -> String {
    let mut encoded = String::from("file:");
    for byte in path_bytes(absolute) {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded.push_str("?mode=ro");
    encoded
}

fn path_bytes(path: &Path) -> Vec<u8> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes().to_vec()
    }
    #[cfg(not(unix))]
    {
        path.to_string_lossy().into_owned().into_bytes()
    }
}
