//! One status line and one usage grid.

use cortex_store::{
    dream_freshness, grid_levels, stored_hhmm, DreamFreshness, Presence, Sparkline, Store,
    StoreError, UtcTime, SPARKLINE_BUCKETS,
};

use crate::glyphs::{grid_cell, spark_cell, sparkline_level};

pub const DEFAULT_COLS: usize = 40;
pub const DEFAULT_ROWS: usize = 6;

/// Words on the status line. A clock is never invented for the two that have none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DreamWord {
    NotRunYet,
    Running,
    Stale,
    Fresh,
    Failed,
}

impl DreamWord {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotRunYet => "not run yet",
            Self::Running => "running",
            Self::Stale => "stale",
            Self::Fresh => "fresh",
            Self::Failed => "failed",
        }
    }

    /// `HH:MM` is omitted while a dream has not finished.
    pub fn shows_when(self) -> bool {
        matches!(self, Self::Stale | Self::Fresh | Self::Failed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    Ok,
    Unavailable,
}

impl Health {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Unavailable => "unavailable",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamGlance {
    pub word: DreamWord,
    pub when: Option<String>,
}

/// Everything `render_pulse` needs. `sparkline` is `None` when the instrument
/// is absent or every bucket is empty: empty glyphs and `n/a`, never `0.0/s`.
#[derive(Debug, Clone, PartialEq)]
pub struct PulseFrame {
    pub memories: Option<i64>,
    pub sparkline: Option<[i64; SPARKLINE_BUCKETS]>,
    pub dream: DreamGlance,
    pub health: Health,
    pub levels: Vec<u8>,
    pub cols: usize,
    pub rows: usize,
    pub ascii: bool,
    pub color: bool,
}

pub fn render_pulse(frame: &PulseFrame) -> String {
    let mut text = status_line(frame);
    text.push('\n');
    let fitted = fit_levels(&frame.levels, frame.cols.saturating_mul(frame.rows));
    if frame.cols == 0 || frame.rows == 0 {
        return text;
    }
    for row in 0..frame.rows {
        let start = row * frame.cols;
        for level in &fitted[start..start + frame.cols] {
            text.push_str(&grid_cell(*level, frame.ascii, frame.color));
        }
        text.push('\n');
    }
    text
}

fn status_line(frame: &PulseFrame) -> String {
    let memories = match frame.memories {
        Some(count) => count.to_string(),
        None => "n/a".to_string(),
    };
    let spark = render_sparkline(frame.sparkline.as_ref(), frame.ascii, frame.color);
    let rate = rate_text(frame.sparkline.as_ref());
    let when = if frame.dream.word.shows_when() {
        frame
            .dream
            .when
            .as_deref()
            .map(|hhmm| format!(" {hhmm}"))
            .unwrap_or_default()
    } else {
        String::new()
    };
    format!(
        "cortex  {memories} mem  {spark} {rate}  last dream {}{when}  {}",
        frame.dream.word.as_str(),
        frame.health.as_str()
    )
}

fn render_sparkline(
    buckets: Option<&[i64; SPARKLINE_BUCKETS]>,
    ascii: bool,
    color: bool,
) -> String {
    let levels = spark_levels(buckets);
    let mut text = String::new();
    for level in levels {
        text.push_str(&spark_cell(level, ascii, color));
    }
    text
}

fn spark_levels(buckets: Option<&[i64; SPARKLINE_BUCKETS]>) -> [u8; SPARKLINE_BUCKETS] {
    let Some(buckets) = buckets else {
        return [0; SPARKLINE_BUCKETS];
    };
    if buckets.iter().all(|count| *count == 0) {
        return [0; SPARKLINE_BUCKETS];
    }
    let max = buckets.iter().copied().max().unwrap_or(0);
    let mut levels = [0; SPARKLINE_BUCKETS];
    for (index, count) in buckets.iter().enumerate() {
        levels[index] = sparkline_level(*count, max);
    }
    levels
}

/// Recalls in the current minute, as events per second. `None` when the
/// sparkline is not measurable.
pub fn rate_per_second(buckets: Option<&[i64; SPARKLINE_BUCKETS]>) -> Option<f64> {
    let buckets = buckets?;
    if buckets.iter().all(|count| *count == 0) {
        return None;
    }
    Some(buckets[SPARKLINE_BUCKETS - 1] as f64 / 60.0)
}

fn rate_text(buckets: Option<&[i64; SPARKLINE_BUCKETS]>) -> String {
    match rate_per_second(buckets) {
        Some(rate) => format!("{rate:.1}/s"),
        None => "n/a".to_string(),
    }
}

fn fit_levels(levels: &[u8], cells: usize) -> Vec<u8> {
    let mut levels: Vec<u8> = levels.iter().map(|level| (*level).min(4)).collect();
    if levels.len() > cells {
        let extra = levels.len() - cells;
        levels.drain(0..extra);
    }
    let pad = cells.saturating_sub(levels.len());
    let mut out = vec![0; pad];
    out.append(&mut levels);
    out
}

pub fn dream_glance(run: Option<&cortex_store::SleepRun>, now: UtcTime) -> DreamGlance {
    let word = match dream_freshness(run, now) {
        DreamFreshness::NoData => DreamWord::NotRunYet,
        DreamFreshness::Running => DreamWord::Running,
        DreamFreshness::Failed => DreamWord::Failed,
        DreamFreshness::Fresh => DreamWord::Fresh,
        DreamFreshness::Stale => DreamWord::Stale,
    };
    let when = if word.shows_when() {
        run.and_then(|run| stored_hhmm(run.at()).map(str::to_string))
    } else {
        None
    };
    DreamGlance { word, when }
}

/// Read one pulse frame and drop the connection before the caller sleeps.
pub fn load_pulse(
    store: &Store,
    cols: usize,
    rows: usize,
    now: UtcTime,
) -> Result<PulseFrame, StoreError> {
    let memories = store.memory_counts()?.total();
    let cells = cols.saturating_mul(rows);
    let buckets = store.usage_buckets(cells as i64)?;
    let levels = grid_levels(&buckets, cells);
    let sparkline = match store.recall_sparkline(now)? {
        Presence::NotPresent => None,
        Presence::Present(spark) if spark.all_empty() => None,
        Presence::Present(Sparkline { buckets }) => Some(buckets),
    };
    let run = store.latest_sleep_run()?;
    Ok(PulseFrame {
        memories: Some(memories),
        sparkline,
        dream: dream_glance(run.as_ref(), now),
        health: Health::Ok,
        levels,
        cols,
        rows,
        ascii: false,
        color: false,
    })
}
