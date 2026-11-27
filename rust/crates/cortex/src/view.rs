//! Pulse and dream, through `cortex-view`'s parser and frame loop.
//!
//! The database is opened by `load_pulse` and `load_dream`, which use
//! `cortex-store` (`mode=ro`, `PRAGMA query_only=ON`). That connection is
//! dropped before the next paint. This module does not issue SQL.

use std::ffi::OsStr;
use std::io::{self, IsTerminal, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use cortex_store::{Store, StoreError};
use cortex_view::{
    is_running, load_dream, load_pulse, parse_args, render_dream, render_pulse, run_frames,
    use_color, Command, DbEnv, DreamFrame, DreamOpts, Parsed, PulseOpts, RunOpts, Tick,
};

use crate::route::USAGE;

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sigint(_sig: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

pub(crate) fn run<I, S>(args: I) -> i32
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let parsed = match parse_args(args, &DbEnv::from_process()) {
        Ok(parsed) => parsed,
        Err(message) => {
            eprintln!("cortex-rs: {message}");
            eprintln!("{USAGE}");
            return 2;
        }
    };
    let args = match parsed {
        Parsed::Help => {
            println!("{USAGE}");
            return 0;
        }
        Parsed::Run(args) => args,
    };
    install_sigint();
    let stdout = io::stdout();
    let tty = stdout.is_terminal();
    let color = use_color(tty, std::env::var("NO_COLOR").ok().as_deref());
    let mut out = stdout.lock();
    match args.command {
        Command::Pulse(opts) => run_pulse(&mut out, &args.db, &opts, tty, color),
        Command::Dream(opts) => run_dream(&mut out, &args.db, &opts, tty, color),
    }
}

fn run_pulse(out: &mut impl Write, db: &Path, opts: &PulseOpts, tty: bool, color: bool) -> i32 {
    run_frames(
        out,
        || {
            let store = Store::open(db)?;
            let mut frame = load_pulse(&store, opts.cols, opts.rows, cortex_store::UtcTime::now())?;
            drop(store);
            frame.ascii = opts.ascii;
            frame.color = color;
            Ok(Tick::Again(render_pulse(&frame)))
        },
        &loop_opts(opts.once, opts.interval, tty),
        &STOP,
    )
}

fn run_dream(out: &mut impl Write, db: &Path, opts: &DreamOpts, tty: bool, color: bool) -> i32 {
    let watch = opts.watch && !opts.once;
    run_frames(
        out,
        || match load_styled_dream(db, opts, color) {
            Ok(frame) => {
                let running = is_running(&frame);
                let text = render_dream(&frame);
                if watch && running {
                    Ok(Tick::Again(text))
                } else {
                    Ok(Tick::Stop(text))
                }
            }
            Err(err) => Err(err),
        },
        &loop_opts(opts.once, cortex_view::DEFAULT_INTERVAL, tty),
        &STOP,
    )
}

fn load_styled_dream(db: &Path, opts: &DreamOpts, color: bool) -> Result<DreamFrame, StoreError> {
    let store = Store::open(db)?;
    let mut frame = load_dream(&store, opts.run_id.as_deref())?;
    drop(store);
    if let DreamFrame::Run(run) = &mut frame {
        run.ascii = opts.ascii;
        run.color = color;
    }
    Ok(frame)
}

fn loop_opts(once: bool, interval: std::time::Duration, tty: bool) -> RunOpts {
    RunOpts {
        once,
        interval,
        tty,
    }
}

fn install_sigint() {
    // SAFETY: the handler only stores a lock-free atomic. It does not
    // allocate, lock, or touch the terminal. Ctrl-C then makes the poll
    // loop return, which shows the cursor before exiting 0.
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_sigint as *const () as libc::sighandler_t,
        );
    }
}
