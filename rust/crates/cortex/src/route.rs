//! Pick which crate owns this argv.
//!
//! The only decision made here is which subcommand is present. Flag values
//! and errors stay with that crate's parser. `--db` is global and comes
//! before the subcommand; `serve` also accepts `--bind` and `--assets`
//! anywhere, matching `cortex-http`.

use std::ffi::{OsStr, OsString};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Cli,
    Http,
    View,
    Help,
}

pub(crate) const USAGE: &str = "\
usage:
  cortex-rs [--db PATH] recall QUERY --json [--limit N]
  cortex-rs [--db PATH] status --json
  cortex-rs [--db PATH] remember TEXT --json [--kind KIND]
  cortex-rs [--db PATH] sleep-status --json
  cortex-rs [--db PATH] pulse [--cols N] [--rows N] [--interval SECS] [--ascii] [--once]
  cortex-rs [--db PATH] dream [--watch] [--follow] [--ascii] [--run-id ID] [--once]
  cortex-rs serve [--bind 127.0.0.1:8090] [--db PATH] [--assets DIR]

--db defaults to $CORTEX_DB, else $HERMES_HOME/cortex/cortex.db,
else ~/.hermes/cortex/cortex.db.
--json is required for recall, status, remember, and sleep-status.

exit 0  the command finished
exit 1  recall found nothing
exit 2  bad usage
exit 3  the store is unreadable, or cortex-memory could not be run";

const VALUE_FLAGS: &[&str] = &["--db", "--bind", "--assets"];

pub(crate) fn classify<I, S>(args: I) -> Result<Kind, String>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let args: Vec<OsString> = args.into_iter().map(Into::into).collect();
    let command = command_name(&args)?;
    let Some(command) = command else {
        if args.iter().any(|arg| arg == "--help" || arg == "-h") {
            return Ok(Kind::Help);
        }
        return Err("missing command".to_string());
    };
    match command.as_str() {
        "recall" | "status" | "remember" | "sleep-status" => Ok(Kind::Cli),
        "pulse" | "dream" => Ok(Kind::View),
        "serve" => Ok(Kind::Http),
        other => Err(format!("unknown command {other}")),
    }
}

/// The subcommand token, if one was present.
///
/// A value flag consumes the following token so a path is not treated as the
/// command. `--` ends flags; the next token is the command, and the target
/// parser decides whether that `--` is allowed.
fn command_name(args: &[OsString]) -> Result<Option<String>, String> {
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let text = utf8(arg)?;
        if text == "--" {
            let Some(next) = args.get(index + 1) else {
                return Ok(None);
            };
            return Ok(Some(utf8(next)?.to_string()));
        }
        if VALUE_FLAGS.contains(&text) {
            let Some(value) = args.get(index + 1) else {
                return Err(format!("{text} needs a value"));
            };
            utf8(value)?;
            index += 2;
            continue;
        }
        if is_value_flag_inline(text) {
            index += 1;
            continue;
        }
        if text.starts_with('-') {
            index += 1;
            continue;
        }
        return Ok(Some(text.to_string()));
    }
    Ok(None)
}

fn is_value_flag_inline(text: &str) -> bool {
    VALUE_FLAGS
        .iter()
        .any(|flag| text.starts_with(flag) && text.as_bytes().get(flag.len()) == Some(&b'='))
}

fn utf8(arg: &OsStr) -> Result<&str, String> {
    arg.to_str()
        .ok_or_else(|| "argument is not utf-8".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(args: &[&str]) -> Result<Kind, String> {
        classify(args.iter().copied().map(OsString::from))
    }

    #[test]
    fn spec_argv_reaches_the_crate_that_parses_it() {
        assert_eq!(
            kind(&["--db", "db", "status", "--json"]).unwrap(),
            Kind::Cli
        );
        assert_eq!(
            kind(&["--db", "db", "recall", "query", "--json", "--limit", "3"]).unwrap(),
            Kind::Cli
        );
        assert_eq!(
            kind(&["--db", "db", "remember", "text", "--json", "--kind", "semantic"]).unwrap(),
            Kind::Cli
        );
        assert_eq!(
            kind(&["--db", "db", "sleep-status", "--json"]).unwrap(),
            Kind::Cli
        );
        assert_eq!(
            kind(&["--db", "db", "pulse", "--cols", "4", "--once"]).unwrap(),
            Kind::View
        );
        assert_eq!(
            kind(&["--db", "db", "dream", "--follow", "--ascii"]).unwrap(),
            Kind::View
        );
        assert_eq!(
            kind(&["serve", "--bind", "127.0.0.1:0", "--db", "db"]).unwrap(),
            Kind::Http
        );
        assert_eq!(
            kind(&["--db", "db", "--bind", "127.0.0.1:8090", "serve"]).unwrap(),
            Kind::Http
        );
        assert_eq!(kind(&["--help"]).unwrap(), Kind::Help);
        assert_eq!(kind(&["-h"]).unwrap(), Kind::Help);
    }

    #[test]
    fn a_path_is_not_the_command() {
        assert_eq!(
            kind(&["--db", "serve", "status", "--json"]).unwrap(),
            Kind::Cli
        );
        assert!(kind(&["--db"]).unwrap_err().contains("--db needs a value"));
        assert!(kind(&[]).unwrap_err().contains("missing command"));
        assert!(kind(&["nope"])
            .unwrap_err()
            .contains("unknown command nope"));
    }
}
