#![forbid(unsafe_code)]

use std::io::Write;

fn main() {
    let code = cortex_cli::run(std::env::args_os().skip(1));
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    std::process::exit(code);
}
