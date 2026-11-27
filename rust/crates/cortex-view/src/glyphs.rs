//! Glyphs from `design/dot-glyphs.json`.
//!
//! The renderer does not open that file. The constants are the file, and the
//! tests fail when a rendered cell is not one of its code points.

pub const GRID_GLYPHS: [&str; 5] = ["·", "░", "▒", "▓", "█"];
pub const GRID_ASCII: [&str; 5] = [".", ":", "+", "*", "#"];
pub const SPARK_GLYPHS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
pub const SPARK_ASCII: [&str; 8] = [".", ":", "-", "~", "=", "+", "*", "#"];

/// `min(7, floor(8 * count / max_count))` when `max_count > 0`, else the empty cell.
pub fn sparkline_level(count: i64, max_count: i64) -> u8 {
    if max_count <= 0 || count <= 0 {
        return 0;
    }
    let scaled = (8.0 * count as f64 / max_count as f64).floor();
    if scaled <= 0.0 {
        0
    } else if scaled >= 7.0 {
        7
    } else {
        scaled as u8
    }
}

pub fn grid_cell(level: u8, ascii: bool, color: bool) -> String {
    paint(level.min(4), ascii, color, false)
}

pub fn spark_cell(level: u8, ascii: bool, color: bool) -> String {
    paint(level.min(7), ascii, color, true)
}

fn paint(level: u8, ascii: bool, color: bool, spark: bool) -> String {
    let glyph = if spark {
        if ascii {
            SPARK_ASCII[level as usize]
        } else {
            SPARK_GLYPHS[level as usize]
        }
    } else if ascii {
        GRID_ASCII[level as usize]
    } else {
        GRID_GLYPHS[level as usize]
    };
    if !color {
        return glyph.to_string();
    }
    format!("\x1b[{}m{glyph}\x1b[0m", color_code(level, spark))
}

fn color_code(level: u8, spark: bool) -> &'static str {
    if spark {
        match level {
            0 => "2",
            1 | 2 => "34",
            3 | 4 => "36",
            5 | 6 => "32",
            _ => "1;97",
        }
    } else {
        match level {
            0 => "2",
            1 => "34",
            2 => "36",
            3 => "33",
            _ => "1;97",
        }
    }
}

/// Fill `width` cells. `filled` of them take `level`; the rest are empty.
/// A fraction of 1 fills the bar. An unknown scale (`denom` absent) stays empty
/// so a running episode count is not drawn as a finished bar.
pub fn bar(
    count: Option<i64>,
    denom: Option<i64>,
    width: usize,
    ascii: bool,
    color: bool,
) -> String {
    let mut out = String::new();
    if width == 0 {
        return out;
    }
    let (level, filled) = bar_fill(count, denom, width);
    for index in 0..width {
        let cell = if index < filled { level } else { 0 };
        out.push_str(&grid_cell(cell, ascii, color));
    }
    out
}

fn bar_fill(count: Option<i64>, denom: Option<i64>, width: usize) -> (u8, usize) {
    let (Some(count), Some(denom)) = (count, denom) else {
        return (0, 0);
    };
    let scale = count.max(denom);
    if scale <= 0 || count <= 0 {
        return (0, 0);
    }
    let fraction = count as f64 / scale as f64;
    let level = cortex_store::usage_level(count, scale);
    let mut filled = (fraction * width as f64).round() as usize;
    if filled == 0 {
        filled = 1;
    }
    (level, filled.min(width))
}
