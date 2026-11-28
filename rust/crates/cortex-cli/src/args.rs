//! Argv for the four headless commands.
//!
//! `--db` is global and must come before the subcommand. `--json` is required
//! on every command. `--limit` belongs to `recall`. `--kind` belongs to `remember`.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::PathBuf;

use cortex_store::DEFAULT_RECALL_LIMIT;

const USAGE: &str = "\
usage:
  cortex-cli [--db PATH] recall QUERY --json [--limit N]
  cortex-cli [--db PATH] status --json
  cortex-cli [--db PATH] remember TEXT --json [--kind KIND]
  cortex-cli [--db PATH] sleep-status --json

--db defaults to $CORTEX_DB, else $HERMES_HOME/cortex/cortex.db, else ~/.hermes/cortex/cortex.db.
--json is required. stdout is one JSON object and schema_version is 1.

exit 0  the command finished
exit 1  recall found nothing
exit 2  bad usage
exit 3  the store is unreadable, or cortex-memory could not be run";

pub struct Invocation {
    pub db: PathBuf,
    pub command: Command,
}

pub enum Command {
    Recall { query: String, limit: i64 },
    Status,
    Remember { text: String, kind: String },
    SleepStatus,
}

pub struct Usage {
    message: String,
}

impl Usage {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for Usage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}\n\n{USAGE}", self.message)
    }
}

pub fn parse<I, S>(args: I) -> Result<Invocation, Usage>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let args: Vec<OsString> = args.into_iter().map(Into::into).collect();
    let mut index = 0;
    let mut db = None;
    while index < args.len() {
        let arg = &args[index];
        if arg.as_os_str() == OsStr::new("--") {
            return Err(Usage::new("missing command"));
        }
        if arg.as_os_str() == OsStr::new("--db") {
            index += 1;
            let Some(value) = args.get(index) else {
                return Err(Usage::new("--db needs a path"));
            };
            db = Some(PathBuf::from(value));
            index += 1;
            continue;
        }
        let text = utf8(arg)?;
        if text.starts_with('-') {
            return Err(Usage::new(format!("unknown flag {text}")));
        }
        break;
    }
    if index >= args.len() {
        return Err(Usage::new("missing command"));
    }
    let name = utf8(&args[index])?;
    index += 1;
    let db = db.unwrap_or_else(default_db);
    let command = match name {
        "recall" => parse_recall(&args[index..])?,
        "status" => {
            parse_empty(&args[index..], "status")?;
            Command::Status
        }
        "remember" => parse_remember(&args[index..])?,
        "sleep-status" => {
            parse_empty(&args[index..], "sleep-status")?;
            Command::SleepStatus
        }
        other => return Err(Usage::new(format!("unknown command {other}"))),
    };
    Ok(Invocation { db, command })
}

fn parse_recall(args: &[OsString]) -> Result<Command, Usage> {
    let tail = parse_tail(args, true, false)?;
    let query = one(
        tail.positionals,
        "recall needs a query",
        "recall takes one query",
    )?;
    Ok(Command::Recall {
        query,
        limit: tail.limit.unwrap_or(DEFAULT_RECALL_LIMIT),
    })
}

fn parse_remember(args: &[OsString]) -> Result<Command, Usage> {
    let tail = parse_tail(args, false, true)?;
    let text = one(
        tail.positionals,
        "remember needs text",
        "remember takes one text",
    )?;
    Ok(Command::Remember {
        text,
        kind: tail.kind.unwrap_or_else(|| "semantic".to_string()),
    })
}

fn parse_empty(args: &[OsString], command: &str) -> Result<(), Usage> {
    let tail = parse_tail(args, false, false)?;
    if !tail.positionals.is_empty() {
        return Err(Usage::new(format!("{command} takes no arguments")));
    }
    Ok(())
}

struct Tail {
    json: bool,
    positionals: Vec<String>,
    limit: Option<i64>,
    kind: Option<String>,
}

fn parse_tail(args: &[OsString], allow_limit: bool, allow_kind: bool) -> Result<Tail, Usage> {
    let mut tail = Tail {
        json: false,
        positionals: Vec::new(),
        limit: None,
        kind: None,
    };
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg.as_os_str() == OsStr::new("--") {
            index += 1;
            while index < args.len() {
                tail.positionals.push(utf8(&args[index])?.to_string());
                index += 1;
            }
            break;
        }
        let text = utf8(arg)?;
        if text == "--json" {
            tail.json = true;
            index += 1;
            continue;
        }
        if text == "--limit" {
            if !allow_limit {
                return Err(Usage::new("unknown flag --limit"));
            }
            index += 1;
            let raw = args
                .get(index)
                .ok_or_else(|| Usage::new("--limit needs a positive integer"))?;
            let raw = utf8(raw)?;
            let limit: i64 = raw
                .parse()
                .map_err(|_| Usage::new("--limit needs a positive integer"))?;
            if limit <= 0 {
                return Err(Usage::new("--limit needs a positive integer"));
            }
            tail.limit = Some(limit);
            index += 1;
            continue;
        }
        if text == "--kind" {
            if !allow_kind {
                return Err(Usage::new("unknown flag --kind"));
            }
            index += 1;
            let raw = args
                .get(index)
                .ok_or_else(|| Usage::new("--kind needs a value"))?;
            let raw = utf8(raw)?;
            if raw.is_empty() {
                return Err(Usage::new("--kind needs a value"));
            }
            tail.kind = Some(raw.to_string());
            index += 1;
            continue;
        }
        if text.starts_with('-') {
            return Err(Usage::new(format!("unknown flag {text}")));
        }
        tail.positionals.push(text.to_string());
        index += 1;
    }
    if !tail.json {
        return Err(Usage::new("--json is required"));
    }
    Ok(tail)
}

fn one(mut items: Vec<String>, missing: &str, extra: &str) -> Result<String, Usage> {
    if items.is_empty() {
        return Err(Usage::new(missing));
    }
    if items.len() != 1 {
        return Err(Usage::new(extra));
    }
    Ok(items.pop().unwrap())
}

fn utf8(arg: &OsStr) -> Result<&str, Usage> {
    arg.to_str()
        .ok_or_else(|| Usage::new("argument is not utf-8"))
}

/// `$CORTEX_DB`, else `$HERMES_HOME/cortex/cortex.db`, else `~/.hermes/cortex/cortex.db`.
///
/// An empty variable is unset, matching `cli.py`. `~` is expanded only on
/// `HERMES_HOME`, which is the path `cli.py` passes through `expanduser`. The first path that
/// exists wins, so the shared brain is found from a profile `HERMES_HOME` as well.
fn default_db() -> PathBuf {
    let cortex_db = std::env::var_os("CORTEX_DB").filter(|value| !value.is_empty());
    let hermes_home = std::env::var_os("HERMES_HOME").filter(|value| !value.is_empty());
    pick_db(cortex_db, hermes_home, home_dir())
}

/// The resolution rule as a function of the environment, so it can be tested without mutating
/// process-wide variables.
fn pick_db(cortex_db: Option<OsString>, hermes_home: Option<OsString>, home_root: PathBuf) -> PathBuf {
    if let Some(value) = cortex_db {
        return PathBuf::from(value);
    }
    let base = match hermes_home {
        Some(value) => expand_home(PathBuf::from(value)),
        None => home_root.clone().join(".hermes"),
    };
    let primary = base.join("cortex").join("cortex.db");
    // A profile `HERMES_HOME` (`.../profiles/<name>`) has no `cortex/` directory: the brain lives in
    // the shared `~/.hermes/cortex/cortex.db`. Without this fallback a hand-run CLI - exactly what a
    // headless agent does - reports "database file does not exist" while the service, which receives
    // `CORTEX_DB` from its environment file, works fine.
    if !primary.exists() {
        let shared = home_root.join(".hermes").join("cortex").join("cortex.db");
        if shared.exists() {
            return shared;
        }
    }
    primary
}

fn home_dir() -> PathBuf {
    match std::env::var_os("HOME") {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        _ => PathBuf::from("."),
    }
}

fn expand_home(path: PathBuf) -> PathBuf {
    let Some(text) = path.to_str() else {
        return path;
    };
    if text == "~" {
        return home_dir();
    }
    if let Some(rest) = text.strip_prefix("~/") {
        return home_dir().join(rest);
    }
    path
}

#[cfg(test)]
mod tests {
    use super::pick_db;
    use std::ffi::OsString;
    use std::fs;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("cortex-args-{}-{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("scratch root");
        root
    }

    fn touch_shared(home: &PathBuf) -> PathBuf {
        let shared = home.join(".hermes").join("cortex");
        fs::create_dir_all(&shared).expect("shared dir");
        let db = shared.join("cortex.db");
        fs::write(&db, b"x").expect("shared db");
        db
    }

    #[test]
    fn cortex_db_wins_over_everything() {
        let home = scratch("explicit");
        touch_shared(&home);
        let picked = pick_db(
            Some(OsString::from("/explicit/cortex.db")),
            Some(OsString::from("/some/profile")),
            home,
        );
        assert_eq!(picked, PathBuf::from("/explicit/cortex.db"));
    }

    #[test]
    fn a_profile_hermes_home_falls_back_to_the_shared_brain() {
        // The lifeboat layout: HERMES_HOME points at .../profiles/<name>, which holds no cortex/ dir.
        let home = scratch("profile");
        let shared = touch_shared(&home);
        let profile = home.join("lifeboat").join("profiles").join("teknium");
        fs::create_dir_all(&profile).expect("profile dir");
        let picked = pick_db(None, Some(OsString::from(profile.to_str().unwrap())), home);
        assert_eq!(picked, shared);
    }

    #[test]
    fn with_nothing_on_disk_the_profile_path_is_reported() {
        // The error should name the path the operator configured, not a path they never set.
        let home = scratch("empty");
        let profile = home.join("lifeboat").join("profiles").join("teknium");
        fs::create_dir_all(&profile).expect("profile dir");
        let picked = pick_db(None, Some(OsString::from(profile.to_str().unwrap())), home);
        assert_eq!(picked, profile.join("cortex").join("cortex.db"));
    }

    #[test]
    fn without_hermes_home_the_default_is_dot_hermes() {
        let home = scratch("default");
        let shared = touch_shared(&home);
        let picked = pick_db(None, None, home);
        assert_eq!(picked, shared);
    }
}
