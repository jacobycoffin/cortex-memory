use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use cortex_store::{
    grid_levels, KindCount, Presence, SleepChildren, SleepRun, StateChangeCount, UsageBucket,
    UtcTime, SPARKLINE_BUCKETS,
};

use super::*;

fn glance_idle() -> DreamGlance {
    DreamGlance {
        word: DreamWord::NotRunYet,
        when: None,
    }
}

fn pulse(levels: Vec<u8>, ascii: bool, memories: Option<i64>) -> PulseFrame {
    PulseFrame {
        memories,
        sparkline: None,
        dream: glance_idle(),
        health: Health::Ok,
        levels,
        cols: 4,
        rows: 2,
        ascii,
        color: false,
    }
}

fn expect_grid(memories: &str, ascii: bool, rows: [&str; 2]) -> String {
    let spark = if ascii { "." } else { "▁" };
    format!(
        "cortex  {memories} mem  {} n/a  last dream not run yet  ok\n{}\n{}\n",
        spark.repeat(SPARKLINE_BUCKETS),
        rows[0],
        rows[1],
    )
}

fn glyph(ascii: bool, level: u8) -> &'static str {
    if ascii {
        GRID_ASCII[level as usize]
    } else {
        GRID_GLYPHS[level as usize]
    }
}

fn row(ascii: bool, levels: &[u8]) -> String {
    levels.iter().map(|level| glyph(ascii, *level)).collect()
}

#[test]
fn sparkline_bucket_count_matches_the_glyph_file() {
    assert_eq!(SPARKLINE_BUCKETS, 60);
}

#[test]
fn pulse_grids_match_exact_strings_for_both_glyph_sets() {
    for ascii in [false, true] {
        let empty_row = row(ascii, &[0, 0, 0, 0]);
        let empty = expect_grid("0", ascii, [&empty_row, &empty_row]);
        let no_buckets = grid_levels(&[], 8);
        assert_eq!(
            render_pulse(&pulse(no_buckets, ascii, Some(0))),
            empty,
            "empty grid ascii={ascii}"
        );

        let zeros: Vec<UsageBucket> = (0..8)
            .map(|index| UsageBucket {
                task_id: format!("t{index}"),
                started_at: format!("2026-11-27T00:00:{index:02}+00:00"),
                rows: 4,
                used_rows: 0,
            })
            .collect();
        let zero_levels = grid_levels(&zeros, 8);
        assert!(zero_levels.iter().all(|level| *level == 0));
        assert_eq!(
            render_pulse(&pulse(zero_levels, ascii, Some(0))),
            empty,
            "all-zero grid ascii={ascii}"
        );

        let full: Vec<UsageBucket> = (0..8)
            .map(|index| UsageBucket {
                task_id: format!("f{index}"),
                started_at: format!("2026-11-27T01:00:{index:02}+00:00"),
                rows: 1,
                used_rows: 1,
            })
            .collect();
        let saturated = grid_levels(&full, 8);
        assert!(saturated.iter().all(|level| *level == 4));
        let saturated_row = row(ascii, &[4, 4, 4, 4]);
        assert_eq!(
            render_pulse(&pulse(saturated, ascii, Some(8))),
            expect_grid("8", ascii, [&saturated_row, &saturated_row]),
            "saturated grid ascii={ascii}"
        );

        let spike_buckets = vec![UsageBucket {
            task_id: "spike".to_string(),
            started_at: "2026-11-27T02:00:00+00:00".to_string(),
            rows: 4,
            used_rows: 4,
        }];
        let spike = grid_levels(&spike_buckets, 8);
        assert_eq!(spike, vec![0, 0, 0, 0, 0, 0, 0, 4]);
        let quiet = row(ascii, &[0, 0, 0, 0]);
        let spiked = row(ascii, &[0, 0, 0, 4]);
        let rendered = render_pulse(&pulse(spike, ascii, Some(1)));
        assert_eq!(
            rendered,
            expect_grid("1", ascii, [&quiet, &spiked]),
            "one spike ascii={ascii}"
        );
        assert!(
            !rendered.contains('\u{1b}'),
            "colour leaked into a plain frame"
        );
        if ascii {
            assert!(rendered.is_ascii(), "{rendered}");
        }
        assert_eq!(rendered.chars().filter(|ch| *ch == '\n').count(), 3);
        assert!(!rendered.ends_with("\n\n"));
    }
}

#[test]
fn default_grid_is_40_by_6_empty_glyphs() {
    let frame = PulseFrame {
        cols: DEFAULT_COLS,
        rows: DEFAULT_ROWS,
        levels: grid_levels(&[], DEFAULT_COLS * DEFAULT_ROWS),
        ..pulse(Vec::new(), false, Some(0))
    };
    let rendered = render_pulse(&frame);
    let mut lines = rendered.lines();
    let _status = lines.next().unwrap();
    let grid: Vec<&str> = lines.collect();
    assert_eq!(grid.len(), 6);
    assert!(grid.iter().all(|line| line.chars().count() == 40));
    assert!(grid.iter().all(|line| line.chars().all(|ch| ch == '·')));
}

#[test]
fn missing_rate_is_na_and_a_measured_minute_is_one_decimal() {
    let mut quiet = pulse(vec![0; 8], false, Some(2));
    quiet.sparkline = Some([0; SPARKLINE_BUCKETS]);
    let rendered = render_pulse(&quiet);
    assert!(rendered.contains(" n/a  last dream"));
    assert!(!rendered.contains("0.0/s"), "{rendered}");

    let mut current = [0; SPARKLINE_BUCKETS];
    current[SPARKLINE_BUCKETS - 1] = 252;
    let mut busy = pulse(vec![0; 8], false, Some(2));
    busy.sparkline = Some(current);
    let rendered = render_pulse(&busy);
    assert!(rendered.contains(" 4.2/s  last dream"));
    assert!(rendered.contains('█'));
    assert!(!rendered.contains("0.0/s"));

    let mut earlier = [0; SPARKLINE_BUCKETS];
    earlier[0] = 60;
    let mut measured_zero = pulse(vec![0; 8], true, Some(2));
    measured_zero.sparkline = Some(earlier);
    let rendered = render_pulse(&measured_zero);
    assert!(rendered.contains(" 0.0/s  last dream"), "{rendered}");
    assert!(rendered.is_ascii());
}

#[test]
fn status_line_uses_dream_words_and_omits_the_clock_while_running() {
    let mut frame = pulse(vec![0; 8], false, None);
    frame.health = Health::Unavailable;
    let rendered = render_pulse(&frame);
    assert!(rendered.starts_with("cortex  n/a mem  "));
    assert!(rendered.contains("last dream not run yet  unavailable\n"));
    assert!(!rendered.contains("0 mem"));

    frame.memories = Some(12);
    frame.health = Health::Ok;
    frame.dream = DreamGlance {
        word: DreamWord::Running,
        when: Some("04:15".to_string()),
    };
    let rendered = render_pulse(&frame);
    assert!(rendered.contains("last dream running  ok\n"), "{rendered}");
    assert!(!rendered.contains("04:15"));

    frame.dream = DreamGlance {
        word: DreamWord::Fresh,
        when: Some("04:15".to_string()),
    };
    assert!(render_pulse(&frame).contains("last dream fresh 04:15  ok\n"));

    frame.dream = DreamGlance {
        word: DreamWord::Failed,
        when: Some("03:10".to_string()),
    };
    assert!(render_pulse(&frame).contains("last dream failed 03:10  ok\n"));

    frame.dream = DreamGlance {
        word: DreamWord::Stale,
        when: Some("01:02".to_string()),
    };
    assert!(render_pulse(&frame).contains("last dream stale 01:02  ok\n"));
}

#[test]
fn dream_glance_follows_the_36_hour_rule_without_moving_the_clock() {
    let now = UtcTime::parse_stored("2026-11-27T12:00:00+00:00").unwrap();
    assert_eq!(dream_glance(None, now).word, DreamWord::NotRunYet);
    assert_eq!(dream_glance(None, now).when, None);

    let mut run = sample_run("run", "apply", "running");
    run.started_at = "2026-11-27T04:15:02+00:00".to_string();
    let glance = dream_glance(Some(&run), now);
    assert_eq!(glance.word, DreamWord::Running);
    assert_eq!(glance.when, None);

    run.status = "completed".to_string();
    run.completed_at = Some("2026-11-27T11:00:00+00:00".to_string());
    let glance = dream_glance(Some(&run), now);
    assert_eq!(glance.word, DreamWord::Fresh);
    assert_eq!(glance.when.as_deref(), Some("11:00"));

    run.status = "failed".to_string();
    let glance = dream_glance(Some(&run), now);
    assert_eq!(glance.word, DreamWord::Failed);
    assert_eq!(glance.when.as_deref(), Some("11:00"));

    run.status = "completed".to_string();
    run.completed_at = Some("2026-11-25T11:00:00+00:00".to_string());
    let glance = dream_glance(Some(&run), now);
    assert_eq!(glance.word, DreamWord::Stale);
    assert_eq!(glance.when.as_deref(), Some("11:00"));
}

#[test]
fn colour_wraps_glyphs_only_when_asked() {
    let mut frame = pulse(vec![4; 8], false, Some(1));
    frame.color = true;
    let rendered = render_pulse(&frame);
    assert!(rendered.contains('\u{1b}'), "{rendered}");
    assert!(rendered.contains('█'));
    frame.color = false;
    assert!(!render_pulse(&frame).contains('\u{1b}'));
}

#[test]
fn redraw_second_write_has_no_extra_newline() {
    let frame = render_pulse(&pulse(vec![0, 4, 0, 0, 0, 0, 0, 0], false, Some(0)));
    assert!(frame.ends_with('\n'));
    assert!(!frame.ends_with("\n\n"));
    let mut buf = Vec::new();
    paint_frame(&mut buf, &frame, false).unwrap();
    paint_frame(&mut buf, &frame, true).unwrap();
    let second = &buf[frame.len()..];
    let mut expected = CURSOR_HOME.to_vec();
    expected.extend_from_slice(frame.as_bytes());
    assert_eq!(second, expected.as_slice());
    assert!(!second.ends_with(b"\n\n"));
    assert_eq!(
        second.iter().filter(|byte| **byte == b'\n').count(),
        frame.bytes().filter(|byte| *byte == b'\n').count()
    );
}

#[test]
fn live_loop_second_frame_is_cursor_home_plus_the_frame() {
    let frame = render_pulse(&pulse(grid_levels(&[], 8), true, Some(0)));
    let stop = AtomicBool::new(false);
    let mut paints = 0;
    let mut out = Vec::new();
    let code = run_frames(
        &mut out,
        || {
            paints += 1;
            if paints >= 2 {
                stop.store(true, Ordering::SeqCst);
            }
            Ok(Tick::Again(frame.clone()))
        },
        &RunOpts {
            once: false,
            interval: Duration::from_millis(1),
            tty: true,
        },
        &stop,
    );
    assert_eq!(code, 0);
    let hide = b"\x1b[?25l";
    let show = b"\x1b[?25h";
    let second_start = hide.len() + frame.len();
    let second_end = out.len() - show.len();
    let second = &out[second_start..second_end];
    let mut expected = CURSOR_HOME.to_vec();
    expected.extend_from_slice(frame.as_bytes());
    assert_eq!(second, expected.as_slice());
    assert!(!second.ends_with(b"\n\n"));
}

#[test]
fn store_error_once_exits_3_and_the_loop_stays_alive() {
    let stop = AtomicBool::new(false);
    let mut out = Vec::new();
    let code = run_frames(
        &mut out,
        || {
            Err(cortex_store::StoreError::Unreadable {
                message: "gone".to_string(),
            })
        },
        &RunOpts {
            once: true,
            interval: Duration::from_secs(30),
            tty: false,
        },
        &stop,
    );
    assert_eq!(code, 3);
    assert_eq!(out, b"store unavailable: unreadable: gone\n");
    assert!(!out.contains(&0x1b));

    // A live terminal keeps retrying: a transient lock clears, and the line repaints in place.
    let mut errors = 0;
    let mut out = Vec::new();
    let code = run_frames(
        &mut out,
        || {
            errors += 1;
            if errors >= 2 {
                stop.store(true, Ordering::SeqCst);
            }
            Err(cortex_store::StoreError::Unavailable {
                message: "locked".to_string(),
            })
        },
        &RunOpts {
            once: false,
            interval: Duration::from_millis(1),
            tty: true,
        },
        &stop,
    );
    assert_eq!(code, 0);
    let paints = String::from_utf8_lossy(&out).matches("store unavailable").count();
    assert_eq!(paints, 2, "an interactive terminal retries");

    // A pipe cannot recover, so it fails once with a code a supervisor can see. It used to repaint
    // the same line every interval and never exit.
    let mut out = Vec::new();
    let code = run_frames(
        &mut out,
        || {
            Err(cortex_store::StoreError::Unavailable {
                message: "locked".to_string(),
            })
        },
        &RunOpts {
            once: false,
            interval: Duration::from_millis(1),
            tty: false,
        },
        &AtomicBool::new(false),
    );
    assert_eq!(code, 3);
    assert_eq!(
        std::str::from_utf8(&out).unwrap(),
        "store unavailable: unavailable: locked\n"
    );
}

#[test]
fn running_episode_count_is_the_child_rows_not_the_replayed_default() {
    let mut run = sample_run("run-live", "apply", "running");
    run.episodes_replayed = 0;
    run.episodes_scanned = 41;
    run.association_proposals = 50;
    run.edge_decay_proposals = 60;
    run.applied_changes = 0;
    run.completed_at = None;
    let mut children = present_children();
    children.episodes = Presence::Present(3);
    let mut frame = dream_from_run(&run, &children);
    assert!(frame.applied_changes.is_none());
    frame.bar_width = 4;
    let rendered = render_dream(&DreamFrame::Run(frame));
    let expected = "\
dream  run run-live  apply  started 04:15:02
episodes  ···· 3/n/a
demoted   ···· 0  active → cold
linked    ···· 0 new associations, 0 reinforced
edges     ···· 0 decay proposals
running run-live
";
    assert_eq!(rendered, expected);
    let episodes = rendered
        .lines()
        .find(|line| line.starts_with("episodes"))
        .unwrap();
    assert!(episodes.contains("3/n/a"), "{episodes}");
    assert!(!episodes.contains('0'), "{episodes}");
    assert!(!rendered.contains("41"), "{rendered}");
    assert!(!rendered.contains("50"), "{rendered}");
    assert!(!rendered.contains("60"), "{rendered}");
    assert!(!rendered.contains("sleep-undo"), "{rendered}");
}

#[test]
fn finished_shadow_verdict_does_not_offer_sleep_undo() {
    let mut run = sample_run("run-shadow", "shadow", "completed");
    run.applied_changes = 4;
    run.episodes_replayed = 9;
    run.episodes_scanned = 10;
    let mut children = present_children();
    children.episodes = Presence::Present(3);
    let rendered = render_dream(&DreamFrame::Run(dream_from_run(&run, &children)));
    assert!(
        rendered.contains("shadow only · no changes applied · run run-shadow"),
        "{rendered}"
    );
    assert!(rendered.contains("3/10"), "{rendered}");
    assert!(!rendered.contains("sleep-undo"), "{rendered}");
    assert!(!rendered.contains('9'), "{rendered}");
    assert!(!rendered.contains("4 changes"), "{rendered}");

    run.status = "failed".to_string();
    let failed = render_dream(&DreamFrame::Run(dream_from_run(&run, &children)));
    assert!(failed.contains("shadow only"), "{failed}");
    assert!(!failed.contains("sleep-undo"), "{failed}");
}

#[test]
fn apply_verdict_names_the_run_on_the_undo_line() {
    let mut run = sample_run("run-4", "apply", "completed");
    run.applied_changes = 4;
    let rendered = render_dream(&DreamFrame::Run(dream_from_run(&run, &present_children())));
    assert!(
        rendered.contains("4 changes applied · undo: cortex-memory sleep-undo run-4"),
        "{rendered}"
    );

    run.applied_changes = 0;
    let none = render_dream(&DreamFrame::Run(dream_from_run(&run, &present_children())));
    assert!(none.contains("0 changes applied · run run-4"), "{none}");
    assert!(!none.contains("sleep-undo"), "{none}");

    run.status = "failed".to_string();
    run.applied_changes = 4;
    let failed = render_dream(&DreamFrame::Run(dream_from_run(&run, &present_children())));
    assert!(failed.contains("dream failed · run run-4"), "{failed}");
    assert!(!failed.contains("sleep-undo"), "{failed}");

    run.status = "reverted".to_string();
    let reverted = render_dream(&DreamFrame::Run(dream_from_run(&run, &present_children())));
    assert!(reverted.contains("undone · run run-4"), "{reverted}");

    run.status = "revert_partial".to_string();
    let partial = render_dream(&DreamFrame::Run(dream_from_run(&run, &present_children())));
    assert!(partial.contains("undone · run run-4"), "{partial}");
}

#[test]
fn no_dream_is_one_line() {
    assert_eq!(render_dream(&DreamFrame::NoDream), "no dream has run yet\n");
    assert!(!render_dream(&DreamFrame::NoDream).contains('0'));
}

#[test]
fn edge_bars_count_edge_downscale_proposals() {
    let mut run = sample_run("run-edges", "apply", "running");
    run.edge_decay_proposals = 9;
    run.episodes_replayed = 0;
    let mut children = present_children();
    children.edges = Presence::Present(9);
    children.proposals = Presence::Present(vec![KindCount {
        kind: "edge_downscale".to_string(),
        n: 2,
    }]);
    let mut frame = dream_from_run(&run, &children);
    frame.bar_width = 4;
    let rendered = render_dream(&DreamFrame::Run(frame));
    let edges = rendered
        .lines()
        .find(|line| line.starts_with("edges"))
        .unwrap();
    assert!(edges.contains("████ 2 decay proposals"), "{edges}");
    assert!(!rendered.contains('9'), "{rendered}");

    run.status = "completed".to_string();
    run.edge_decay_proposals = 8;
    let mut frame = dream_from_run(&run, &children);
    frame.bar_width = 4;
    let rendered = render_dream(&DreamFrame::Run(frame));
    let edges = rendered
        .lines()
        .find(|line| line.starts_with("edges"))
        .unwrap();
    assert!(edges.contains("░··· 2 decay proposals"), "{edges}");
    assert!(!edges.contains('8'), "{edges}");
}

#[test]
fn linked_bar_uses_both_association_kinds() {
    let mut run = sample_run("run-link", "apply", "completed");
    run.association_proposals = 4;
    let mut children = present_children();
    children.proposals = Presence::Present(vec![
        KindCount {
            kind: "association".to_string(),
            n: 1,
        },
        KindCount {
            kind: "association_reinforcement".to_string(),
            n: 1,
        },
    ]);
    let mut frame = dream_from_run(&run, &children);
    frame.bar_width = 4;
    let rendered = render_dream(&DreamFrame::Run(frame));
    let linked = rendered
        .lines()
        .find(|line| line.starts_with("linked"))
        .unwrap();
    assert!(
        linked.contains("▒▒·· 1 new associations, 1 reinforced"),
        "{linked}"
    );
}

#[test]
fn demoted_counts_active_to_cold_only() {
    let run = sample_run("run-demo", "apply", "completed");
    let mut children = present_children();
    children.state_changes = Presence::Present(vec![
        StateChangeCount {
            prior_state: "active".to_string(),
            next_state: "cold".to_string(),
            n: 2,
        },
        StateChangeCount {
            prior_state: "cold".to_string(),
            next_state: "archived".to_string(),
            n: 5,
        },
    ]);
    let rendered = render_dream(&DreamFrame::Run(dream_from_run(&run, &children)));
    assert!(rendered.contains("2  active → cold"), "{rendered}");
    assert!(!rendered.contains("5  active"), "{rendered}");
}

#[test]
fn missing_child_tables_are_na_not_zero() {
    let mut running = sample_run("run", "apply", "running");
    running.started_at = "2026-11-27T04:00:00+00:00".to_string();
    running.episodes_scanned = 8;
    running.episodes_replayed = 0;
    running.association_proposals = 12;
    running.edge_decay_proposals = 15;
    let missing = SleepChildren {
        episodes: Presence::NotPresent,
        state_changes: Presence::NotPresent,
        proposals: Presence::NotPresent,
        edges: Presence::NotPresent,
    };
    let rendered = render_dream(&DreamFrame::Run(dream_from_run(&running, &missing)));
    assert!(
        rendered.contains("episodes") && rendered.contains(" n/a"),
        "{rendered}"
    );
    let episodes = line(&rendered, "episodes");
    assert!(episodes.contains("n/a"), "{episodes}");
    assert!(!episodes.contains('0'), "{episodes}");
    assert!(!rendered.contains("12"), "{rendered}");
    assert!(!rendered.contains("15"), "{rendered}");
    assert!(!rendered.contains('8'), "{rendered}");

    let mut finished = sample_run("run", "apply", "completed");
    finished.episodes_scanned = 8;
    let rendered = render_dream(&DreamFrame::Run(dream_from_run(&finished, &missing)));
    let episodes = line(&rendered, "episodes");
    assert!(episodes.contains("n/a/8"), "{episodes}");
    assert!(!episodes.contains("0/"), "{episodes}");
}

#[test]
fn ascii_dream_is_ascii_only() {
    let mut run = sample_run("run-shadow", "shadow", "completed");
    run.applied_changes = 4;
    let mut frame = dream_from_run(&run, &present_children());
    frame.ascii = true;
    let rendered = render_dream(&DreamFrame::Run(frame));
    assert!(rendered.is_ascii(), "{rendered}");
    assert!(rendered.contains("shadow only - no changes applied - run run-shadow"));
    assert!(rendered.contains("active -> cold"));
    assert!(!rendered.contains("sleep-undo"));
}

#[test]
fn glyph_constants_and_rendered_cells_come_from_dot_glyphs_json() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../design/dot-glyphs.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_glyphs(&json["grid"]["glyphs"], &GRID_GLYPHS);
    assert_glyphs(&json["grid"]["ascii"], &GRID_ASCII);
    assert_glyphs(&json["sparkline"]["glyphs"], &SPARK_GLYPHS);
    assert_glyphs(&json["sparkline"]["ascii"], &SPARK_ASCII);
    assert_eq!(json["sparkline"]["buckets"].as_u64(), Some(60));
    assert_eq!(json["layout"]["cols"].as_u64(), Some(40));
    assert_eq!(json["layout"]["rows"].as_u64(), Some(6));

    let mut counts = [0; SPARKLINE_BUCKETS];
    for (index, count) in (0..8).enumerate() {
        counts[index] = count as i64;
    }
    for ascii in [false, true] {
        let mut frame = pulse(vec![0, 1, 2, 3, 4, 0, 0, 0], ascii, Some(3));
        frame.sparkline = Some(counts);
        let rendered = render_pulse(&frame);
        let status = rendered.lines().next().unwrap();
        let spark = status
            .split(" mem  ")
            .nth(1)
            .unwrap()
            .split(' ')
            .next()
            .unwrap();
        assert_eq!(spark.chars().count(), 60, "{spark}");
        assert_cells(spark, if ascii { &SPARK_ASCII } else { &SPARK_GLYPHS });
        for line in rendered.lines().skip(1) {
            assert_cells(line, if ascii { &GRID_ASCII } else { &GRID_GLYPHS });
        }

        let mut dream = dream_from_run(
            &sample_run("run", "apply", "completed"),
            &present_children(),
        );
        dream.ascii = ascii;
        dream.bar_width = 20;
        let rendered = render_dream(&DreamFrame::Run(dream));
        for label in ["episodes", "demoted", "linked", "edges"] {
            let line = line(&rendered, label);
            let cells: String = line.chars().skip(10).take(20).collect();
            assert_eq!(cells.chars().count(), 20, "{line}");
            assert_cells(&cells, if ascii { &GRID_ASCII } else { &GRID_GLYPHS });
        }
    }
}

#[test]
fn argv_matches_the_service_contract() {
    let env = DbEnv {
        cortex_db: None,
        hermes_home: Some("/opt/hermes".into()),
        home: Some("/home/someone".into()),
    };
    match parse_args(["pulse"], &env).unwrap() {
        Parsed::Run(args) => {
            assert_eq!(args.db, PathBuf::from("/opt/hermes/cortex/cortex.db"));
            match args.command {
                Command::Pulse(opts) => {
                    assert_eq!(opts.cols, 40);
                    assert_eq!(opts.rows, 6);
                    assert_eq!(opts.interval, Duration::from_secs(1));
                    assert!(!opts.ascii);
                    assert!(!opts.once);
                }
                other => panic!("pulse parsed as {other:?}"),
            }
        }
        other => panic!("{other:?}"),
    }

    match parse_args(
        [
            "--db",
            "mem.db",
            "pulse",
            "--cols",
            "10",
            "--rows=3",
            "--interval",
            "2.5",
            "--ascii",
            "--once",
        ],
        &env,
    )
    .unwrap()
    {
        Parsed::Run(args) => {
            assert_eq!(args.db, PathBuf::from("mem.db"));
            match args.command {
                Command::Pulse(opts) => {
                    assert_eq!(opts.cols, 10);
                    assert_eq!(opts.rows, 3);
                    assert_eq!(opts.interval, Duration::from_secs_f64(2.5));
                    assert!(opts.ascii && opts.once);
                }
                other => panic!("{other:?}"),
            }
        }
        other => panic!("{other:?}"),
    }

    match parse_args(["dream", "--follow", "--run-id", "abc", "--watch"], &env).unwrap() {
        Parsed::Run(args) => match args.command {
            Command::Dream(opts) => {
                assert!(opts.watch);
                assert_eq!(opts.run_id.as_deref(), Some("abc"));
                assert!(!opts.once);
            }
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }

    assert!(parse_args(std::iter::empty::<&str>(), &env).is_err());
    assert!(parse_args(["pulse", "--cols", "0"], &env).is_err());
    assert!(parse_args(["pulse", "--interval", "0"], &env).is_err());
    assert!(parse_args(["pulse", "--nope"], &env).is_err());
    assert!(parse_args(["nope"], &env).is_err());
    assert!(parse_args(["--db"], &env).is_err());
    assert!(matches!(
        parse_args(["--help"], &env).unwrap(),
        Parsed::Help
    ));
}

#[test]
fn default_db_matches_cli_py() {
    let home = DbEnv {
        cortex_db: None,
        hermes_home: None,
        home: Some("/home/someone".into()),
    };
    assert_eq!(
        default_db(&home),
        PathBuf::from("/home/someone/.hermes/cortex/cortex.db")
    );
    let hermes = DbEnv {
        cortex_db: None,
        hermes_home: Some("~/agents".into()),
        home: Some("/home/someone".into()),
    };
    assert_eq!(
        default_db(&hermes),
        PathBuf::from("/home/someone/agents/cortex/cortex.db")
    );
    let set = DbEnv {
        cortex_db: Some("/tmp/cortex.db".into()),
        hermes_home: Some("/opt/hermes".into()),
        home: Some("/home/someone".into()),
    };
    assert_eq!(default_db(&set), PathBuf::from("/tmp/cortex.db"));
}

#[test]
fn a_profile_hermes_home_finds_the_shared_brain() {
    // The lifeboat layout: HERMES_HOME is .../profiles/<name>, which has no cortex/ dir. The store
    // is the shared ~/.hermes/cortex/cortex.db, and the documented fallback used to be missing, so
    // the unified binary reported "database file does not exist" on the machine that has the brain.
    let root = std::env::temp_dir().join(format!("cortex-view-db-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join(".hermes").join("cortex")).expect("shared dir");
    let shared = root.join(".hermes").join("cortex").join("cortex.db");
    std::fs::write(&shared, b"x").expect("shared db");
    let profile = root.join("lifeboat").join("profiles").join("teknium");
    std::fs::create_dir_all(&profile).expect("profile dir");

    let env = DbEnv {
        cortex_db: None,
        hermes_home: Some(profile.to_str().unwrap().into()),
        home: Some(root.to_str().unwrap().into()),
    };
    assert_eq!(default_db(&env), shared);
}

#[test]
fn colour_follows_the_tty_and_no_color() {
    assert!(!use_color(false, None));
    assert!(!use_color(false, Some("")));
    assert!(use_color(true, None));
    assert!(use_color(true, Some("")));
    assert!(!use_color(true, Some("1")));
}

fn json_strings(value: &serde_json::Value) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_str().unwrap().to_string())
        .collect()
}

fn assert_glyphs(value: &serde_json::Value, expected: &[&str]) {
    let got = json_strings(value);
    let expected: Vec<String> = expected.iter().map(|glyph| (*glyph).to_string()).collect();
    assert_eq!(got, expected);
}

fn assert_cells(text: &str, allowed: &[&str]) {
    let allowed: std::collections::HashSet<char> =
        allowed.iter().flat_map(|glyph| glyph.chars()).collect();
    for ch in text.chars() {
        assert!(
            allowed.contains(&ch),
            "rendered glyph {ch:?} is not in design/dot-glyphs.json"
        );
    }
}

fn line<'a>(text: &'a str, label: &str) -> &'a str {
    text.lines().find(|line| line.starts_with(label)).unwrap()
}

fn sample_run(id: &str, mode: &str, status: &str) -> SleepRun {
    SleepRun {
        run_id: id.to_string(),
        mode: mode.to_string(),
        status: status.to_string(),
        applied_changes: 0,
        error: None,
        started_at: "2026-11-27T04:15:02+00:00".to_string(),
        completed_at: Some("2026-11-27T04:20:00+00:00".to_string()),
        episodes_scanned: 0,
        episodes_replayed: 0,
        association_proposals: 0,
        edge_decay_proposals: 0,
    }
}

fn present_children() -> SleepChildren {
    SleepChildren {
        episodes: Presence::Present(0),
        state_changes: Presence::Present(Vec::new()),
        proposals: Presence::Present(Vec::new()),
        edges: Presence::Present(0),
    }
}
