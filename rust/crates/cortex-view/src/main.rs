//! `cortex-view pulse` and `cortex-view dream`.
//!
//! Reads go through `cortex-store`. A missing or locked database prints one
//! line and, unless `--once` was given, the loop keeps waiting.

use std::io::{self, IsTerminal, Write};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use cortex_store::{Store, StoreError};
use cortex_view::{
    is_running, load_dream, load_pulse, parse_args, render_dream, render_pulse, run_frames,
    use_color, Command, DbEnv, DreamFrame, DreamOpts, Parsed, PulseOpts, RunOpts, Tick, USAGE,
};

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sigint(_sig: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

fn main() -> ExitCode {
    let parsed = match parse_args(std::env::args_os().skip(1), &DbEnv::from_process()) {
        Ok(parsed) => parsed,
        Err(message) => {
            eprintln!("cortex-view: {message}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let args = match parsed {
        Parsed::Help => {
            println!("{USAGE}");
            return ExitCode::from(0);
        }
        Parsed::Run(args) => args,
    };
    install_sigint();
    let stdout = io::stdout();
    let tty = stdout.is_terminal();
    let color = use_color(tty, std::env::var("NO_COLOR").ok().as_deref());
    let mut out = stdout.lock();
    let code = match args.command {
        Command::Pulse(opts) => run_pulse(&mut out, &args.db, &opts, tty, color),
        Command::Dream(opts) => run_dream(&mut out, &args.db, &opts, tty, color),
    };
    ExitCode::from(code)
}

fn run_pulse(
    out: &mut impl Write,
    db: &std::path::Path,
    opts: &PulseOpts,
    tty: bool,
    color: bool,
) -> u8 {
    let code = run_frames(
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
    );
    code as u8
}

fn run_dream(
    out: &mut impl Write,
    db: &std::path::Path,
    opts: &DreamOpts,
    tty: bool,
    color: bool,
) -> u8 {
    let watch = opts.watch && !opts.once;
    let code = run_frames(
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
    );
    code as u8
}

fn load_styled_dream(
    db: &std::path::Path,
    opts: &DreamOpts,
    color: bool,
) -> Result<DreamFrame, StoreError> {
    let store = Store::open(db)?;
    let mut frame = load_dream(&store, opts.run_id.as_deref())?;
    drop(store);
    if let DreamFrame::Run(run) = &mut frame {
        run.ascii = opts.ascii;
        run.color = color;
    }
    Ok(frame)
}

fn loop_opts(once: bool, interval: Duration, tty: bool) -> RunOpts {
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
