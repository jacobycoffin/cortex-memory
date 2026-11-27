//! Terminal dot views of an existing `cortex.db`.
//!
//! `render_pulse` and `render_dream` are pure. The binary opens the database
//! through [`cortex_store::Store`], which is `mode=ro` and `query_only`, and
//! drops that connection before sleeping until the next paint. This crate
//! never writes `meta.schema_version` and never opens SQLite read-write.
//!
//! Glyphs are the columns of `design/dot-glyphs.json`. `--ascii` selects the
//! ASCII column. Colour is added only when the caller sets it, which the
//! binary does when stdout is a terminal and `NO_COLOR` is unset or empty.

#![forbid(unsafe_code)]

mod cli;
mod dream;
mod glyphs;
mod pulse;
mod run;

pub use cli::{
    default_db, parse_args, use_color, Args, Command, DbEnv, DreamOpts, Parsed, PulseOpts,
    DEFAULT_INTERVAL, USAGE,
};
pub use dream::{
    dream_from_run, is_running, load_dream, render_dream, stored_hms, verdict, DreamFrame,
    DreamRun, Linked, Scaled, BAR_WIDTH, NO_DREAM,
};
pub use glyphs::{sparkline_level, GRID_ASCII, GRID_GLYPHS, SPARK_ASCII, SPARK_GLYPHS};
pub use pulse::{
    dream_glance, load_pulse, rate_per_second, render_pulse, DreamGlance, DreamWord, Health,
    PulseFrame, DEFAULT_COLS, DEFAULT_ROWS,
};
pub use run::{paint_frame, run_frames, store_unavailable_line, RunOpts, Tick, CURSOR_HOME};

#[cfg(test)]
mod tests;
