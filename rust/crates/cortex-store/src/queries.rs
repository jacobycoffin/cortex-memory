//! Typed reads. Each method is one statement, then the connection is idle.

use rusqlite::OptionalExtension;

use crate::error::{map_query_error, Measurement, Presence, StoreError};
use crate::open::Store;
use crate::timeutil::UtcTime;

use super::lexical::{translate_components, ScoreBreakdown, NOTE_LEXICAL_BREAKDOWN};

pub const DEFAULT_RECALL_LIMIT: i64 = 6;
pub const DEFAULT_USAGE_CELL_LIMIT: i64 = 240;
pub const SPARKLINE_BUCKETS: usize = 60;

const RECALL_EVENTS: &[&str] = &["retrieved", "selected", "injected"];
const GRADED_OUTCOMES: &[&str] = &["helpful", "harmful", "validated", "corrected"];
const RECALLED_WINDOW_SECS: i64 = 7 * 24 * 60 * 60;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryCounts {
    pub active: i64,
    pub cold: i64,
    pub archived: i64,
    pub other: i64,
}

impl MemoryCounts {
    pub fn total(&self) -> i64 {
        self.active + self.cold + self.archived + self.other
    }

    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    /// `the store is empty` when `memories` exists and has no rows.
    pub fn note(&self) -> Option<&'static str> {
        if self.is_empty() {
            Some(crate::NOTE_EMPTY_STORE)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Procedures {
    pub total: Measurement,
    pub graded: Measurement,
}

/// One `sleep_runs` row. Counter columns stay at their insert defaults until
/// the run finishes, so [`Self::finished_applied_changes`] is `None` while
/// `status` is `running`.
#[derive(Debug, Clone, PartialEq)]
pub struct SleepRun {
    pub run_id: String,
    pub mode: String,
    pub status: String,
    pub applied_changes: i64,
    pub error: Option<String>,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub episodes_scanned: i64,
    pub episodes_replayed: i64,
    pub association_proposals: i64,
    pub edge_decay_proposals: i64,
}

impl SleepRun {
    /// `None` while the run is `running`: the column is still the insert default.
    pub fn finished_applied_changes(&self) -> Option<i64> {
        if self.status == "running" {
            None
        } else {
            Some(self.applied_changes)
        }
    }

    /// `completed_at`, or `started_at` while the run is still `running`.
    pub fn at(&self) -> &str {
        if self.status == "running" {
            self.started_at.as_str()
        } else {
            self.completed_at
                .as_deref()
                .unwrap_or(self.started_at.as_str())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateChangeCount {
    pub prior_state: String,
    pub next_state: String,
    pub n: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KindCount {
    pub kind: String,
    pub n: i64,
}

/// Child-table counts for one dream. Each field is [`Presence::NotPresent`]
/// when that optional table does not exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SleepChildren {
    pub episodes: Presence<i64>,
    pub state_changes: Presence<Vec<StateChangeCount>>,
    pub proposals: Presence<Vec<KindCount>>,
    pub edges: Presence<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UsageBucket {
    pub task_id: String,
    pub started_at: String,
    pub rows: i64,
    pub used_rows: i64,
}

impl UsageBucket {
    pub fn used_fraction(&self) -> f64 {
        if self.rows <= 0 {
            0.0
        } else {
            self.used_rows as f64 / self.rows as f64
        }
    }

    pub fn level(&self) -> u8 {
        usage_level(self.used_rows, self.rows)
    }
}

/// Counts of `recall_runs` in the last 60 minutes, oldest minute first.
///
/// An all-zero window is a present instrument with nothing in it. Callers
/// render empty glyphs and a null rate. A missing `recall_runs` table is
/// [`Presence::NotPresent`], not this value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sparkline {
    pub buckets: [i64; SPARKLINE_BUCKETS],
}

impl Sparkline {
    pub fn all_empty(&self) -> bool {
        self.buckets.iter().all(|count| *count == 0)
    }

    pub fn total(&self) -> i64 {
        self.buckets.iter().sum()
    }
}

/// `0` when the fraction is 0; `1` at or under 0.25; `2` at or under 0.50;
/// `3` at or under 0.75; `4` above that.
pub fn usage_level(used_rows: i64, rows: i64) -> u8 {
    if rows <= 0 || used_rows <= 0 {
        return 0;
    }
    let fraction = used_rows as f64 / rows as f64;
    if fraction <= 0.25 {
        1
    } else if fraction <= 0.50 {
        2
    } else if fraction <= 0.75 {
        3
    } else {
        4
    }
}

/// Levels for a grid, oldest of the window first, empty cells padded on the left.
pub fn grid_levels(buckets_oldest_first: &[UsageBucket], cells: usize) -> Vec<u8> {
    let mut levels: Vec<u8> = buckets_oldest_first
        .iter()
        .map(UsageBucket::level)
        .collect();
    if levels.len() > cells {
        let extra = levels.len() - cells;
        levels.drain(0..extra);
    }
    let pad = cells.saturating_sub(levels.len());
    let mut out = vec![0; pad];
    out.append(&mut levels);
    out
}

impl Store {
    /// `meta.schema_version` as text. `None` when the row is absent.
    /// This never writes the row and never compares it to a constant.
    pub fn schema_version(&self) -> Result<Option<String>, StoreError> {
        match self.conn.query_row(
            "SELECT CAST(value AS TEXT) FROM meta WHERE key = 'schema_version'",
            [],
            |row| row.get::<_, Option<String>>(0),
        ) {
            Ok(value) => Ok(value),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => Err(map_query_error(err)),
        }
    }

    /// `PRAGMA journal_mode` with no assignment, so the mode is not changed.
    pub fn journal_mode(&self) -> Result<String, StoreError> {
        self.conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .map_err(map_query_error)
    }

    pub fn memory_counts(&self) -> Result<MemoryCounts, StoreError> {
        let mut counts = MemoryCounts {
            active: 0,
            cold: 0,
            archived: 0,
            other: 0,
        };
        let mut statement = self
            .conn
            .prepare("SELECT state, COUNT(*) AS n FROM memories GROUP BY state")
            .map_err(map_query_error)?;
        let rows = statement
            .query_map([], |row| {
                Ok((row.get::<_, Option<String>>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(map_query_error)?;
        for row in rows {
            let (state, n) = row.map_err(map_query_error)?;
            match state.as_deref() {
                Some("active") => counts.active += n,
                Some("cold") => counts.cold += n,
                Some("archived") => counts.archived += n,
                _ => counts.other += n,
            }
        }
        Ok(counts)
    }

    /// Distinct memories recalled in the 7 days before `now`.
    ///
    /// The cutoff is a UTC timestamp computed here. A count of zero is
    /// [`Measurement::NotMeasurable`], not a healthy zero.
    pub fn recalled_7d(&self, now: UtcTime) -> Result<Measurement, StoreError> {
        let cutoff = now.saturating_sub_secs(RECALLED_WINDOW_SECS).to_iso();
        let sql = format!(
            "SELECT COUNT(DISTINCT memory_id) AS n FROM access_log \
             WHERE event IN ({}) AND datetime(created_at) >= datetime(?1)",
            quoted_list(RECALL_EVENTS)
        );
        let count = self.count(&sql, [cutoff])?;
        Ok(Measurement::from_required_count(count))
    }

    /// Distinct graded `task_id`s. A count of zero is not measurable yet.
    pub fn graded_answers(&self) -> Result<Measurement, StoreError> {
        let sql = format!(
            "SELECT COUNT(DISTINCT task_id) AS n FROM usage_records WHERE outcome IN ({})",
            quoted_list(GRADED_OUTCOMES)
        );
        let count = self.count(&sql, ())?;
        Ok(Measurement::from_required_count(count))
    }

    /// `procedures.total` and `procedures.graded` are the same distinct count.
    ///
    /// The table is optional. Absence is [`Measurement::NotPresent`]. A present
    /// table whose count is zero is [`Measurement::NotMeasurable`].
    pub fn procedures(&self) -> Result<Procedures, StoreError> {
        if !self.has_table("procedure_outcomes")? {
            return Ok(Procedures {
                total: Measurement::NotPresent,
                graded: Measurement::NotPresent,
            });
        }
        let count = self.count(
            "SELECT COUNT(DISTINCT procedure_id) AS n FROM procedure_outcomes",
            (),
        )?;
        let measurement = Measurement::from_optional_count(true, count);
        Ok(Procedures {
            total: measurement,
            graded: measurement,
        })
    }

    pub fn latest_sleep_run(&self) -> Result<Option<SleepRun>, StoreError> {
        self.sleep_run_query(
            "SELECT run_id, mode, status, applied_changes, error, started_at, completed_at, \
             episodes_scanned, episodes_replayed, association_proposals, edge_decay_proposals \
             FROM sleep_runs ORDER BY started_at DESC LIMIT 1",
            (),
        )
    }

    pub fn sleep_run(&self, run_id: &str) -> Result<Option<SleepRun>, StoreError> {
        self.sleep_run_query(
            "SELECT run_id, mode, status, applied_changes, error, started_at, completed_at, \
             episodes_scanned, episodes_replayed, association_proposals, edge_decay_proposals \
             FROM sleep_runs WHERE run_id = ?1",
            [run_id],
        )
    }

    pub fn sleep_children(&self, run_id: &str) -> Result<SleepChildren, StoreError> {
        Ok(SleepChildren {
            episodes: self.optional_count(
                "sleep_episode_state",
                "SELECT COUNT(*) AS n FROM sleep_episode_state WHERE last_run_id = ?1",
                run_id,
            )?,
            state_changes: self.optional_state_changes(run_id)?,
            proposals: self.optional_proposals(run_id)?,
            edges: self.optional_count(
                "sleep_edge_changes",
                "SELECT COUNT(*) AS n FROM sleep_edge_changes \
                 WHERE run_id = ?1 AND reversed_at IS NULL",
                run_id,
            )?,
        })
    }

    /// Newest `limit` tasks, returned oldest-of-that-window first.
    ///
    /// `limit` is `cols * rows` at the view. A non-positive limit returns no
    /// rows and is not passed to SQL (`LIMIT -1` would be unbounded).
    pub fn usage_buckets(&self, limit: i64) -> Result<Vec<UsageBucket>, StoreError> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        let mut statement = self
            .conn
            .prepare(
                "SELECT task_id, MIN(created_at) AS started_at, \
                 COUNT(*) AS rows, \
                 SUM(CASE WHEN used = 1 THEN 1 ELSE 0 END) AS used_rows \
                 FROM usage_records \
                 GROUP BY task_id \
                 ORDER BY started_at DESC \
                 LIMIT ?1",
            )
            .map_err(map_query_error)?;
        let rows = statement
            .query_map([limit], |row| {
                Ok(UsageBucket {
                    task_id: row.get(0)?,
                    started_at: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    rows: row.get(2)?,
                    used_rows: row.get::<_, Option<i64>>(3)?.unwrap_or(0),
                })
            })
            .map_err(map_query_error)?;
        let mut buckets = Vec::new();
        for row in rows {
            buckets.push(row.map_err(map_query_error)?);
        }
        buckets.reverse();
        Ok(buckets)
    }

    /// Sixty one-minute buckets ending at the minute that contains `now`.
    pub fn recall_sparkline(&self, now: UtcTime) -> Result<Presence<Sparkline>, StoreError> {
        if !self.has_table("recall_runs")? {
            return Ok(Presence::NotPresent);
        }
        let (start, end) = sparkline_window(now);
        let mut statement = self
            .conn
            .prepare(
                "SELECT created_at FROM recall_runs \
                 WHERE julianday(created_at) >= julianday(?1) \
                   AND julianday(created_at) < julianday(?2)",
            )
            .map_err(map_query_error)?;
        let rows = statement
            .query_map([start.to_iso(), end.to_iso()], |row| {
                row.get::<_, String>(0)
            })
            .map_err(map_query_error)?;
        let mut buckets = [0_i64; SPARKLINE_BUCKETS];
        let start_minute = start.unix_secs().div_euclid(60);
        for row in rows {
            let text = row.map_err(map_query_error)?;
            let Some(at) = UtcTime::parse_stored(&text) else {
                continue;
            };
            let index = at.unix_secs().div_euclid(60) - start_minute;
            if (0..SPARKLINE_BUCKETS as i64).contains(&index) {
                buckets[index as usize] += 1;
            }
        }
        Ok(Presence::Present(Sparkline { buckets }))
    }

    /// Latest translated score breakdown for one memory.
    pub fn score_breakdown(
        &self,
        memory_id: &str,
    ) -> Result<crate::OptionalRow<ScoreBreakdown>, StoreError> {
        if !self.has_table("recall_score_breakdown")? {
            return Ok(crate::OptionalRow::NotPresent);
        }
        let mut statement = self
            .conn
            .prepare(
                "SELECT task_id, rank, memory_id, score, components_json, created_at \
                 FROM recall_score_breakdown \
                 WHERE memory_id = ?1 \
                 ORDER BY created_at DESC \
                 LIMIT 1",
            )
            .map_err(map_query_error)?;
        let found = statement
            .query_row([memory_id], map_breakdown)
            .optional()
            .map_err(map_query_error)?;
        Ok(match found {
            Some(row) => crate::OptionalRow::Found(row),
            None => crate::OptionalRow::Absent,
        })
    }

    fn count(&self, sql: &str, params: impl rusqlite::Params) -> Result<i64, StoreError> {
        self.conn
            .query_row(sql, params, |row| row.get(0))
            .map_err(map_query_error)
    }

    fn optional_count(
        &self,
        table: &str,
        sql: &str,
        run_id: &str,
    ) -> Result<Presence<i64>, StoreError> {
        if !self.has_table(table)? {
            return Ok(Presence::NotPresent);
        }
        self.count(sql, [run_id]).map(Presence::Present)
    }

    fn optional_state_changes(
        &self,
        run_id: &str,
    ) -> Result<Presence<Vec<StateChangeCount>>, StoreError> {
        if !self.has_table("sleep_state_changes")? {
            return Ok(Presence::NotPresent);
        }
        let mut statement = self
            .conn
            .prepare(
                "SELECT prior_state, next_state, COUNT(*) AS n \
                 FROM sleep_state_changes \
                 WHERE run_id = ?1 AND reversed_at IS NULL \
                 GROUP BY prior_state, next_state",
            )
            .map_err(map_query_error)?;
        let rows = statement
            .query_map([run_id], |row| {
                Ok(StateChangeCount {
                    prior_state: row.get(0)?,
                    next_state: row.get(1)?,
                    n: row.get(2)?,
                })
            })
            .map_err(map_query_error)?;
        let mut counts = Vec::new();
        for row in rows {
            counts.push(row.map_err(map_query_error)?);
        }
        Ok(Presence::Present(counts))
    }

    fn optional_proposals(&self, run_id: &str) -> Result<Presence<Vec<KindCount>>, StoreError> {
        if !self.has_table("sleep_proposals")? {
            return Ok(Presence::NotPresent);
        }
        let mut statement = self
            .conn
            .prepare(
                "SELECT kind, COUNT(*) AS n FROM sleep_proposals \
                 WHERE run_id = ?1 GROUP BY kind",
            )
            .map_err(map_query_error)?;
        let rows = statement
            .query_map([run_id], |row| {
                Ok(KindCount {
                    kind: row.get(0)?,
                    n: row.get(1)?,
                })
            })
            .map_err(map_query_error)?;
        let mut counts = Vec::new();
        for row in rows {
            counts.push(row.map_err(map_query_error)?);
        }
        Ok(Presence::Present(counts))
    }

    fn sleep_run_query(
        &self,
        sql: &str,
        params: impl rusqlite::Params,
    ) -> Result<Option<SleepRun>, StoreError> {
        self.conn
            .query_row(sql, params, map_sleep_run)
            .optional()
            .map_err(map_query_error)
    }
}

fn map_sleep_run(row: &rusqlite::Row<'_>) -> rusqlite::Result<SleepRun> {
    Ok(SleepRun {
        run_id: row.get(0)?,
        mode: row.get(1)?,
        status: row.get(2)?,
        applied_changes: row.get(3)?,
        error: row.get(4)?,
        started_at: row.get(5)?,
        completed_at: row.get(6)?,
        episodes_scanned: row.get(7)?,
        episodes_replayed: row.get(8)?,
        association_proposals: row.get(9)?,
        edge_decay_proposals: row.get(10)?,
    })
}

fn map_breakdown(row: &rusqlite::Row<'_>) -> rusqlite::Result<ScoreBreakdown> {
    let components_json: String = row.get(4)?;
    Ok(ScoreBreakdown {
        task_id: row.get(0)?,
        rank: row.get(1)?,
        memory_id: row.get(2)?,
        score: row.get(3)?,
        why: translate_components(&components_json),
        components_json,
        created_at: row.get(5)?,
        note: NOTE_LEXICAL_BREAKDOWN,
    })
}

fn quoted_list(values: &[&str]) -> String {
    values
        .iter()
        .map(|value| format!("'{value}'"))
        .collect::<Vec<_>>()
        .join(",")
}

fn sparkline_window(now: UtcTime) -> (UtcTime, UtcTime) {
    // Sixty clock minutes, the current minute last. 18:00:30 → 17:01:00 .. 18:01:00.
    let minute = now.unix_secs().div_euclid(60) * 60;
    let start = minute - (SPARKLINE_BUCKETS as i64 - 1) * 60;
    let end = minute + 60;
    (UtcTime::from_unix_secs(start), UtcTime::from_unix_secs(end))
}
