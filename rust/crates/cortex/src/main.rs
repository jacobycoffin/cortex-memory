//! `cortex-rs` dispatches the phase-1 argv to the crates that already implement it.
//!
//! `recall`, `status`, `remember`, and `sleep-status` go to `cortex-cli`.
//! `pulse` and `dream` go to `cortex-view`. `serve` goes to `cortex-http`.
//! This binary does not open SQLite and does not re-parse command flags.

mod route;
mod view;

use std::ffi::OsString;
use std::io::Write;

fn main() {
    let args: Vec<OsString> = std::env::args_os().collect();
    let code = match route::classify(args.iter().skip(1).cloned()) {
        Ok(route::Kind::Cli) => cortex_cli::run(args.iter().skip(1).cloned()),
        // `cortex-http` expects the process argument list, including argv0.
        Ok(route::Kind::Http) => cortex_http::run(args.iter().cloned()),
        Ok(route::Kind::View) => view::run(args.iter().skip(1).cloned()),
        Ok(route::Kind::Help) => {
            println!("{}", route::USAGE);
            0
        }
        Err(message) => {
            eprintln!("cortex-rs: {message}");
            eprintln!("{}", route::USAGE);
            2
        }
    };
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    std::process::exit(code);
}
