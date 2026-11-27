//! `GET /api/pulse`. Cheap reads through `cortex-store`. The connection is
//! opened for this call and dropped before the handler returns, so a poll
//! does not pin a SQLite snapshot.
//!
//! This module does not issue `PRAGMA journal_mode` and does not write `meta`.

use cortex_store::{
    dream_freshness, grid_levels, DreamFreshness, Measurement, Store, UtcTime,
    CONTRACT_SCHEMA_VERSION,
};
use serde_json::{json, Value};

use crate::util::query_value;

pub(crate) const DEFAULT_COLS: i64 = 40;
pub(crate) const DEFAULT_ROWS: i64 = 6;
const MIN_DIM: i64 = 1;
const MAX_DIM: i64 = 80;

pub(crate) fn dimensions(query: Option<&str>) -> (i64, i64) {
    (
        parse_dim(query_value(query, "cols").as_deref(), DEFAULT_COLS),
        parse_dim(query_value(query, "rows").as_deref(), DEFAULT_ROWS),
    )
}

pub(crate) fn parse_dim(raw: Option<&str>, default: i64) -> i64 {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return default;
    };
    match raw.parse::<i64>() {
        Ok(value) => value.clamp(MIN_DIM, MAX_DIM),
        Err(_) => default,
    }
}

pub(crate) fn body(
    store: &Store,
    cols: i64,
    rows: i64,
    now: UtcTime,
) -> Result<Value, cortex_store::StoreError> {
    let counts = store.memory_counts()?;
    let recalled = store.recalled_7d(now)?;
    let cells = cols.saturating_mul(rows).max(0);
    let buckets = store.usage_buckets(cells)?;
    let levels = grid_levels(&buckets, cells as usize);
    let run = store.latest_sleep_run()?;
    let freshness = dream_freshness(run.as_ref(), now);
    let last_dream = match run.as_ref() {
        Some(run) => json!({
            "state": dream_state(freshness),
            "at": run.at(),
            "mode": run.mode,
            "run_id": run.run_id,
        }),
        None => json!({
            "state": "not run yet",
            "at": Value::Null,
            "mode": Value::Null,
            "run_id": Value::Null,
        }),
    };
    let recalled_7d = match recalled {
        Measurement::Count(count) => json!(count),
        Measurement::NotMeasurable | Measurement::NotPresent => Value::Null,
    };
    let note = match counts.note() {
        Some(text) => json!(text),
        None => Value::Null,
    };
    Ok(json!({
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "buckets": levels,
        "status": {
            "memories": {
                "active": counts.active,
                "cold": counts.cold,
                "archived": counts.archived,
                "other": counts.other,
            },
            "recalled_7d": recalled_7d,
            "last_dream": last_dream,
        },
        "note": note,
    }))
}

fn dream_state(freshness: DreamFreshness) -> &'static str {
    match freshness {
        DreamFreshness::Running => "running",
        DreamFreshness::NoData => "not run yet",
        DreamFreshness::Failed => "failed",
        DreamFreshness::Fresh => "fresh",
        DreamFreshness::Stale => "stale",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_clamp_and_fall_back() {
        assert_eq!(parse_dim(None, DEFAULT_COLS), 40);
        assert_eq!(parse_dim(Some(""), 40), 40);
        assert_eq!(parse_dim(Some("abc"), 40), 40);
        assert_eq!(parse_dim(Some("0"), 40), 1);
        assert_eq!(parse_dim(Some("81"), 40), 80);
        assert_eq!(parse_dim(Some("6"), 40), 6);
        assert_eq!(dimensions(Some("cols=2&rows=3")), (2, 3));
        assert_eq!(
            dimensions(Some("cols=&rows=")),
            (DEFAULT_COLS, DEFAULT_ROWS)
        );
    }
}
