//! One JSON object on stdout. Strings keep their UTF-8 code points.
//! `serde_json`'s compact formatter escapes quotes, backslashes, and ASCII
//! controls, so every byte outside a string value stays ASCII.

use std::io::Write;

use serde::Serialize;

/// Write `value` as one compact JSON object and a trailing newline.
///
/// The object is buffered first, so a serializer failure does not leave a
/// partial payload on stdout.
pub fn emit(value: &impl Serialize) -> bool {
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::new(&mut buf);
    if value.serialize(&mut serializer).is_err() {
        return false;
    }
    buf.push(b'\n');
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&buf).is_ok() && stdout.flush().is_ok()
}
