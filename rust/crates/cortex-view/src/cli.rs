//! Argv shared with the future `cortex-rs` dispatcher.
//!
//! `--db` is global and comes before the subcommand. The default path matches
//! `cli.py`: `$CORTEX_DB` if set, otherwise `$HERMES_HOME/cortex/cortex.db`,
//! otherwise `~/.hermes/cortex/cortex.db`. The first path that exists wins, so a profile
//! `HERMES_HOME` still finds the shared brain.

use std::env;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::time::Duration;

use crate::pulse::{DEFAULT_COLS, DEFAULT_ROWS};

pub const DEFAULT_INTERVAL: Duration = Duration::from_secs(1);
const MAX_GRID: usize = 400;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DbEnv {
    pub cortex_db: Option<OsString>,
    pub hermes_home: Option<OsString>,
    pub home: Option<OsString>,
}

impl DbEnv {
    pub fn from_process() -> Self {
        Self {
            cortex_db: env::var_os("CORTEX_DB"),
            hermes_home: env::var_os("HERMES_HOME"),
            home: env::var_os("HOME"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub db: PathBuf,
    pub command: Command,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Pulse(PulseOpts),
    Dream(DreamOpts),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PulseOpts {
    pub cols: usize,
    pub rows: usize,
    pub interval: Duration,
    pub ascii: bool,
    pub once: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamOpts {
    pub watch: bool,
    pub ascii: bool,
    pub run_id: Option<String>,
    pub once: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Parsed {
    Help,
    Run(Args),
}

pub const USAGE: &str = "\
cortex-view [--db PATH] pulse [--cols N] [--rows N] [--interval SECS] [--ascii] [--once]
cortex-view [--db PATH] dream [--watch | --follow] [--ascii] [--run-id ID] [--once]

--db defaults to $CORTEX_DB, else $HERMES_HOME/cortex/cortex.db,
else ~/.hermes/cortex/cortex.db.
pulse grid defaults to 40x6. --interval defaults to 1 second.
--watch and --follow redraw until the dream leaves running.
Ctrl-C exits 0. --once exits 3 when the store cannot be read.";

pub fn parse_args<I, S>(args: I, env: &DbEnv) -> Result<Parsed, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let args: Vec<OsString> = args
        .into_iter()
        .map(|arg| arg.as_ref().to_os_string())
        .collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Ok(Parsed::Help);
    }
    let mut index = 0;
    let mut db = None;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--" {
            index += 1;
            break;
        }
        if !is_flag(arg) {
            break;
        }
        if arg == "--db" || starts_with(arg, "--db=") {
            db = Some(take_path(&args, &mut index, "--db")?);
            continue;
        }
        return Err(format!("unknown flag {}", display(arg)));
    }
    let Some(command) = args.get(index) else {
        return Err("missing command".to_string());
    };
    index += 1;
    let command = match command.to_str() {
        Some("pulse") => Command::Pulse(parse_pulse(&args, &mut index)?),
        Some("dream") => Command::Dream(parse_dream(&args, &mut index)?),
        Some(other) => return Err(format!("unknown command {other}")),
        None => return Err("command is not utf-8".to_string()),
    };
    if index != args.len() {
        return Err(format!("unexpected argument {}", display(&args[index])));
    }
    Ok(Parsed::Run(Args {
        db: db.unwrap_or_else(|| default_db(env)),
        command,
    }))
}

fn parse_pulse(args: &[OsString], index: &mut usize) -> Result<PulseOpts, String> {
    let mut opts = PulseOpts {
        cols: DEFAULT_COLS,
        rows: DEFAULT_ROWS,
        interval: DEFAULT_INTERVAL,
        ascii: false,
        once: false,
    };
    while *index < args.len() {
        let arg = &args[*index];
        if arg == "--ascii" {
            opts.ascii = true;
            *index += 1;
        } else if arg == "--once" {
            opts.once = true;
            *index += 1;
        } else if arg == "--cols" || starts_with(arg, "--cols=") {
            opts.cols = take_grid(&args, index, "--cols")?;
        } else if arg == "--rows" || starts_with(arg, "--rows=") {
            opts.rows = take_grid(&args, index, "--rows")?;
        } else if arg == "--interval" || starts_with(arg, "--interval=") {
            opts.interval = take_interval(&args, index)?;
        } else if is_flag(arg) {
            return Err(format!("unknown flag {}", display(arg)));
        } else {
            return Err(format!("unexpected argument {}", display(arg)));
        }
    }
    Ok(opts)
}

fn parse_dream(args: &[OsString], index: &mut usize) -> Result<DreamOpts, String> {
    let mut opts = DreamOpts {
        watch: false,
        ascii: false,
        run_id: None,
        once: false,
    };
    while *index < args.len() {
        let arg = &args[*index];
        if arg == "--ascii" {
            opts.ascii = true;
            *index += 1;
        } else if arg == "--once" {
            opts.once = true;
            *index += 1;
        } else if arg == "--watch" || arg == "--follow" {
            opts.watch = true;
            *index += 1;
        } else if arg == "--run-id" || starts_with(arg, "--run-id=") {
            let run_id = take_string(&args, index, "--run-id")?;
            if run_id.is_empty() {
                return Err("missing value for --run-id".to_string());
            }
            opts.run_id = Some(run_id);
        } else if is_flag(arg) {
            return Err(format!("unknown flag {}", display(arg)));
        } else {
            return Err(format!("unexpected argument {}", display(arg)));
        }
    }
    Ok(opts)
}

pub fn default_db(env: &DbEnv) -> PathBuf {
    if let Some(path) = &env.cortex_db {
        return PathBuf::from(path);
    }
    let home_root = match &env.home {
        Some(home) if !home.is_empty() => PathBuf::from(home),
        _ => PathBuf::from("."),
    };
    let root = match &env.hermes_home {
        Some(path) if !path.is_empty() => expand_user(path, env.home.as_deref()),
        _ => home_root.join(".hermes"),
    };
    let primary = root.join("cortex").join("cortex.db");
    // The same rule as `cortex-cli`: a profile `HERMES_HOME` (`.../profiles/<name>`) holds no
    // `cortex/` directory, and the brain lives in the shared `~/.hermes/cortex/cortex.db`. The
    // documented fallback was missing here too, so the unified binary could not read the store while
    // the service (given `CORTEX_DB`) could.
    if !primary.exists() {
        let shared = home_root.join(".hermes").join("cortex").join("cortex.db");
        if shared.exists() {
            return shared;
        }
    }
    primary
}

fn expand_user(path: &OsStr, home: Option<&OsStr>) -> PathBuf {
    let Some(text) = path.to_str() else {
        return PathBuf::from(path);
    };
    if text == "~" {
        return home
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(path));
    }
    if let Some(rest) = text.strip_prefix("~/") {
        if let Some(home) = home {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(path)
}

fn take_path(args: &[OsString], index: &mut usize, flag: &str) -> Result<PathBuf, String> {
    Ok(PathBuf::from(take_os(args, index, flag)?))
}

fn take_string(args: &[OsString], index: &mut usize, flag: &str) -> Result<String, String> {
    let value = take_os(args, index, flag)?;
    value
        .to_str()
        .map(str::to_string)
        .ok_or_else(|| format!("{flag} is not utf-8"))
}

fn take_grid(args: &[OsString], index: &mut usize, flag: &str) -> Result<usize, String> {
    let text = take_string(args, index, flag)?;
    let value: usize = text
        .parse()
        .map_err(|_| format!("{flag} must be an integer from 1 to {MAX_GRID}"))?;
    if (1..=MAX_GRID).contains(&value) {
        Ok(value)
    } else {
        Err(format!("{flag} must be an integer from 1 to {MAX_GRID}"))
    }
}

fn take_interval(args: &[OsString], index: &mut usize) -> Result<Duration, String> {
    let text = take_string(args, index, "--interval")?;
    let seconds: f64 = text
        .parse()
        .map_err(|_| "--interval must be a positive number of seconds".to_string())?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("--interval must be a positive number of seconds".to_string());
    }
    Ok(Duration::from_secs_f64(seconds))
}

fn take_os(args: &[OsString], index: &mut usize, flag: &str) -> Result<OsString, String> {
    let arg = &args[*index];
    let prefix = format!("{flag}=");
    if let Some(rest) = strip_prefix(arg, &prefix) {
        *index += 1;
        if rest.is_empty() {
            return Err(format!("missing value for {flag}"));
        }
        return Ok(rest);
    }
    *index += 1;
    let Some(value) = args.get(*index) else {
        return Err(format!("missing value for {flag}"));
    };
    if is_flag(value) && !looks_like_number(value) {
        return Err(format!("missing value for {flag}"));
    }
    *index += 1;
    Ok(value.clone())
}

fn looks_like_number(arg: &OsStr) -> bool {
    let Some(text) = arg.to_str() else {
        return false;
    };
    let text = text.strip_prefix('-').unwrap_or(text);
    !text.is_empty() && text.chars().next().is_some_and(|c| c.is_ascii_digit())
}

fn is_flag(arg: &OsStr) -> bool {
    starts_with(arg, "-")
}

fn starts_with(arg: &OsStr, prefix: &str) -> bool {
    arg.to_str().is_some_and(|text| text.starts_with(prefix))
}

fn strip_prefix(arg: &OsStr, prefix: &str) -> Option<OsString> {
    arg.to_str()
        .and_then(|text| text.strip_prefix(prefix))
        .map(OsString::from)
}

fn display(arg: &OsStr) -> String {
    arg.to_string_lossy().into_owned()
}

/// Colour only when stdout is a terminal and `NO_COLOR` is unset or empty.
pub fn use_color(stdout_is_tty: bool, no_color: Option<&str>) -> bool {
    if !stdout_is_tty {
        return false;
    }
    !matches!(no_color, Some(value) if !value.is_empty())
}
