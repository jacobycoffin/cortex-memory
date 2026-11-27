//! One dream cycle: four bars and a verdict that only says what is true.

use cortex_store::{KindCount, Presence, SleepChildren, SleepRun, Store, StoreError};

use crate::glyphs::bar;

pub const BAR_WIDTH: usize = 20;
pub const NO_DREAM: &str = "no dream has run yet";

/// A count whose scale may be unknown. `None` is "not measurable", never a
/// stand-in zero. `denominator: None` renders `n/a`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scaled {
    pub count: Option<i64>,
    pub denominator: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Linked {
    pub association: Option<i64>,
    pub reinforcement: Option<i64>,
    pub denominator: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamRun {
    pub run_id: String,
    pub mode: String,
    pub status: String,
    pub started_hms: Option<String>,
    pub episodes: Scaled,
    pub demoted: Scaled,
    pub linked: Linked,
    pub edges: Scaled,
    /// `None` while `status` is `running`: the column is still the insert default.
    pub applied_changes: Option<i64>,
    pub ascii: bool,
    pub color: bool,
    pub bar_width: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DreamFrame {
    NoDream,
    Run(DreamRun),
}

pub fn render_dream(frame: &DreamFrame) -> String {
    let DreamFrame::Run(run) = frame else {
        return format!("{NO_DREAM}\n");
    };
    let mut text = header(run);
    text.push('\n');
    text.push_str(&bar_line(
        "episodes  ",
        run.episodes.count,
        run.episodes.denominator,
        &episodes_text(run.episodes),
        run,
    ));
    text.push('\n');
    text.push_str(&bar_line(
        "demoted   ",
        run.demoted.count,
        run.demoted.denominator,
        &demoted_text(run.demoted.count, run.ascii),
        run,
    ));
    text.push('\n');
    let linked_count = match (run.linked.association, run.linked.reinforcement) {
        (Some(association), Some(reinforcement)) => Some(association + reinforcement),
        _ => None,
    };
    text.push_str(&bar_line(
        "linked    ",
        linked_count,
        run.linked.denominator,
        &linked_text(run.linked),
        run,
    ));
    text.push('\n');
    text.push_str(&bar_line(
        "edges     ",
        run.edges.count,
        run.edges.denominator,
        &edges_text(run.edges.count),
        run,
    ));
    text.push('\n');
    text.push_str(&verdict(run));
    text.push('\n');
    text
}

fn header(run: &DreamRun) -> String {
    match &run.started_hms {
        Some(hms) => format!("dream  run {}  {}  started {hms}", run.run_id, run.mode),
        None => format!("dream  run {}  {}", run.run_id, run.mode),
    }
}

fn bar_line(
    label: &str,
    count: Option<i64>,
    denom: Option<i64>,
    text: &str,
    run: &DreamRun,
) -> String {
    let cells = bar(count, denom, run.bar_width, run.ascii, run.color);
    format!("{label}{cells} {text}")
}

fn episodes_text(scaled: Scaled) -> String {
    match (scaled.count, scaled.denominator) {
        (None, None) => "n/a".to_string(),
        (None, Some(denom)) => format!("n/a/{denom}"),
        (Some(count), None) => format!("{count}/n/a"),
        (Some(count), Some(denom)) => format!("{count}/{denom}"),
    }
}

fn demoted_text(count: Option<i64>, ascii: bool) -> String {
    let Some(count) = count else {
        return "n/a".to_string();
    };
    let arrow = if ascii { "->" } else { "→" };
    format!("{count}  active {arrow} cold")
}

fn linked_text(linked: Linked) -> String {
    match (linked.association, linked.reinforcement) {
        (Some(association), Some(reinforcement)) => {
            format!("{association} new associations, {reinforcement} reinforced")
        }
        _ => "n/a".to_string(),
    }
}

fn edges_text(count: Option<i64>) -> String {
    match count {
        Some(count) => format!("{count} decay proposals"),
        None => "n/a".to_string(),
    }
}

/// Verdict table from the service spec, first match wins.
///
/// A finished shadow run is claimed before `failed`, so every non-running
/// shadow says `shadow only` and none of them name `sleep-undo`.
pub fn verdict(run: &DreamRun) -> String {
    let sep = if run.ascii { " - " } else { " · " };
    if run.status == "running" {
        return format!("running {}", run.run_id);
    }
    if run.mode == "shadow" {
        return format!("shadow only{sep}no changes applied{sep}run {}", run.run_id);
    }
    if run.mode == "apply" && run.status == "completed" {
        let applied = run.applied_changes.unwrap_or(0);
        if applied > 0 {
            return format!(
                "{applied} changes applied{sep}undo: cortex-memory sleep-undo {}",
                run.run_id
            );
        }
        return format!("{applied} changes applied{sep}run {}", run.run_id);
    }
    if run.status == "failed" {
        return format!("dream failed{sep}run {}", run.run_id);
    }
    if run.status == "reverted" || run.status == "revert_partial" {
        return format!("undone{sep}run {}", run.run_id);
    }
    format!("{}{sep}run {}", run.status, run.run_id)
}

/// Child-table counts for a `running` row. Run-row counters stay at their
/// insert defaults until the dream finishes, so `episodes_replayed` is not read.
pub fn dream_from_run(run: &SleepRun, children: &SleepChildren) -> DreamRun {
    let running = run.status == "running";
    DreamRun {
        run_id: run.run_id.clone(),
        mode: run.mode.clone(),
        status: run.status.clone(),
        started_hms: stored_hms(&run.started_at).map(str::to_string),
        episodes: episodes(run, children, running),
        demoted: demoted(children),
        linked: linked(run, children, running),
        edges: edges(run, children, running),
        applied_changes: if running {
            None
        } else {
            Some(run.applied_changes)
        },
        ascii: false,
        color: false,
        bar_width: BAR_WIDTH,
    }
}

fn episodes(run: &SleepRun, children: &SleepChildren, running: bool) -> Scaled {
    let count = match children.episodes {
        Presence::NotPresent => None,
        Presence::Present(count) => Some(count),
    };
    Scaled {
        count,
        // While the run is unfinished the scanned total is still the insert
        // default, so the denominator is omitted. `episodes_replayed` is never
        // the count.
        denominator: if running {
            None
        } else {
            Some(run.episodes_scanned)
        },
    }
}

fn demoted(children: &SleepChildren) -> Scaled {
    match &children.state_changes {
        Presence::NotPresent => Scaled {
            count: None,
            denominator: None,
        },
        Presence::Present(rows) => {
            let count = rows
                .iter()
                .filter(|row| row.prior_state == "active" && row.next_state == "cold")
                .map(|row| row.n)
                .sum();
            Scaled {
                count: Some(count),
                denominator: Some(count),
            }
        }
    }
}

fn linked(run: &SleepRun, children: &SleepChildren, running: bool) -> Linked {
    let Presence::Present(kinds) = &children.proposals else {
        return Linked {
            association: None,
            reinforcement: None,
            denominator: None,
        };
    };
    let association = kind_count(kinds, "association");
    let reinforcement = kind_count(kinds, "association_reinforcement");
    let sum = association + reinforcement;
    let denominator = if running || run.association_proposals < sum {
        sum
    } else {
        run.association_proposals
    };
    Linked {
        association: Some(association),
        reinforcement: Some(reinforcement),
        denominator: Some(denominator),
    }
}

fn edges(run: &SleepRun, children: &SleepChildren, running: bool) -> Scaled {
    let Presence::Present(kinds) = &children.proposals else {
        return Scaled {
            count: None,
            denominator: None,
        };
    };
    // The row kind is `edge_downscale`. `edge_decay_proposals` is the finished
    // counter, and `sleep_edge_changes` is a different table.
    let count = kind_count(kinds, "edge_downscale");
    let denominator = if running || run.edge_decay_proposals < count {
        count
    } else {
        run.edge_decay_proposals
    };
    Scaled {
        count: Some(count),
        denominator: Some(denominator),
    }
}

fn kind_count(kinds: &[KindCount], kind: &str) -> i64 {
    kinds
        .iter()
        .find(|row| row.kind == kind)
        .map(|row| row.n)
        .unwrap_or(0)
}

/// `HH:MM:SS` after `T` in stored text. The zone is not applied.
pub fn stored_hms(text: &str) -> Option<&str> {
    let index = text.find('T')?;
    let hms = text.get(index + 1..)?.get(0..8)?;
    let bytes = hms.as_bytes();
    if bytes.len() == 8
        && bytes[2] == b':'
        && bytes[5] == b':'
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| index == 2 || index == 5 || byte.is_ascii_digit())
    {
        Some(hms)
    } else {
        None
    }
}

pub fn load_dream(store: &Store, run_id: Option<&str>) -> Result<DreamFrame, StoreError> {
    let run = match run_id {
        Some(run_id) => store.sleep_run(run_id)?,
        None => store.latest_sleep_run()?,
    };
    let Some(run) = run else {
        return Ok(DreamFrame::NoDream);
    };
    let children = store.sleep_children(&run.run_id)?;
    Ok(DreamFrame::Run(dream_from_run(&run, &children)))
}

pub fn is_running(frame: &DreamFrame) -> bool {
    matches!(frame, DreamFrame::Run(run) if run.status == "running")
}
