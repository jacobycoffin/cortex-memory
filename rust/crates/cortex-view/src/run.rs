//! Paint frames. A terminal redraws in place; a pipe prints the bytes it was given.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use cortex_store::StoreError;

pub const CURSOR_HOME: &[u8] = b"\x1b[H";
const CURSOR_HIDE: &[u8] = b"\x1b[?25l";
const CURSOR_SHOW: &[u8] = b"\x1b[?25h";
const ERASE_LINE: &[u8] = b"\x1b[2K";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tick {
    /// Paint and keep polling.
    Again(String),
    /// Paint this frame and return.
    Stop(String),
}

#[derive(Debug, Clone, Copy)]
pub struct RunOpts {
    pub once: bool,
    pub interval: Duration,
    pub tty: bool,
}

pub fn store_unavailable_line(err: &StoreError) -> String {
    format!("store unavailable: {err}\n")
}

/// Write `frame` and nothing after it. A redraw prefixes cursor-home.
pub fn paint_frame(out: &mut impl Write, frame: &str, redraw: bool) -> io::Result<()> {
    if redraw {
        out.write_all(CURSOR_HOME)?;
    }
    out.write_all(frame.as_bytes())
}

/// Poll `next` until it stops, `--once` is set, or `stop` is raised.
///
/// The second and later terminal paints are cursor-home plus the frame.
/// The frame's own trailing newline is the last byte of that write.
/// Ctrl-C is the caller's `stop` flag; this returns 0 after showing the cursor.
/// `--once` returns 3 when a tick cannot read the store.
pub fn run_frames(
    out: &mut impl Write,
    mut next: impl FnMut() -> Result<Tick, StoreError>,
    opts: &RunOpts,
    stop: &AtomicBool,
) -> i32 {
    let mut painted = false;
    let mut hid_cursor = false;
    let live = opts.tty && !opts.once;
    loop {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        match next() {
            Ok(tick) => {
                let (text, done) = match tick {
                    Tick::Again(text) => (text, false),
                    Tick::Stop(text) => (text, true),
                };
                if live && !hid_cursor {
                    if out.write_all(CURSOR_HIDE).is_err() {
                        return finish(out, false);
                    }
                    hid_cursor = true;
                }
                if paint_frame(out, &text, live && painted).is_err() || out.flush().is_err() {
                    return finish(out, hid_cursor);
                }
                painted = true;
                if done || opts.once {
                    break;
                }
            }
            Err(err) => {
                if live
                    && painted
                    && out
                        .write_all(CURSOR_HOME)
                        .and_then(|_| out.write_all(ERASE_LINE))
                        .is_err()
                {
                    return finish(out, hid_cursor);
                }
                let line = store_unavailable_line(&err);
                if out.write_all(line.as_bytes()).is_err() || out.flush().is_err() {
                    return finish(out, hid_cursor);
                }
                // A piped run cannot recover: the store is missing or unreadable, and re-reading it
                // every interval painted 20+ identical lines and never exited. Fail once, with the
                // exit code `--once` already uses, so a supervisor sees the fault. An interactive
                // terminal keeps retrying, where a transient lock clears on its own.
                if opts.once || !live {
                    finish(out, hid_cursor);
                    return 3;
                }
            }
        }
        if !sleep_until(opts.interval, stop) {
            break;
        }
    }
    finish(out, hid_cursor);
    0
}

fn finish(out: &mut impl Write, hid_cursor: bool) -> i32 {
    if hid_cursor {
        let _ = out.write_all(CURSOR_SHOW);
        let _ = out.flush();
    }
    0
}

fn sleep_until(interval: Duration, stop: &AtomicBool) -> bool {
    let step = Duration::from_millis(50);
    let started = Instant::now();
    while started.elapsed() < interval {
        if stop.load(Ordering::SeqCst) {
            return false;
        }
        let remaining = interval.saturating_sub(started.elapsed());
        thread::sleep(remaining.min(step));
    }
    !stop.load(Ordering::SeqCst)
}
