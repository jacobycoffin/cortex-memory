//! UTC instants for cutoffs and sparkline windows.
//!
//! Stored text is parsed so a comparison is an absolute instant. Display of
//! `HH:MM` does not use this conversion: [`stored_hhmm`] returns the hour and
//! minute that follow `T` in the stored text.

use std::time::{SystemTime, UNIX_EPOCH};

/// A UTC instant at whole-second resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct UtcTime {
    unix_secs: i64,
}

impl UtcTime {
    pub fn now() -> Self {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs() as i64)
            .unwrap_or(0);
        Self { unix_secs: secs }
    }

    pub fn from_unix_secs(unix_secs: i64) -> Self {
        Self { unix_secs }
    }

    pub fn unix_secs(self) -> i64 {
        self.unix_secs
    }

    pub fn saturating_sub_secs(self, secs: i64) -> Self {
        Self {
            unix_secs: self.unix_secs.saturating_sub(secs),
        }
    }

    /// `YYYY-MM-DDTHH:MM:SS+00:00`, which SQLite `datetime()` accepts.
    pub fn to_iso(self) -> String {
        let (year, month, day, hour, minute, second) = self.parts();
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}+00:00")
    }

    /// Parse a timestamp written by the Python store, or a SQLite `datetime`.
    ///
    /// Offsets are applied. Fractional seconds are truncated. Text that is not
    /// a calendar timestamp returns `None`.
    pub fn parse_stored(text: &str) -> Option<Self> {
        let text = text.trim();
        if text.len() < 10 {
            return None;
        }
        let year: i32 = text.get(0..4)?.parse().ok()?;
        if text.as_bytes().get(4) != Some(&b'-') || text.as_bytes().get(7) != Some(&b'-') {
            return None;
        }
        let month: u32 = text.get(5..7)?.parse().ok()?;
        let day: u32 = text.get(8..10)?.parse().ok()?;
        let max_day = days_in_month(year, month)?;
        if day < 1 || day > max_day {
            return None;
        }
        if text.len() == 10 {
            let days = days_from_civil(year, month, day)?;
            return Some(Self::from_unix_secs(days.checked_mul(86_400)?));
        }
        let sep = *text.as_bytes().get(10)?;
        if sep != b'T' && sep != b' ' {
            return None;
        }
        if text.len() < 19 {
            return None;
        }
        if text.as_bytes().get(13) != Some(&b':') || text.as_bytes().get(16) != Some(&b':') {
            return None;
        }
        let hour: u32 = text.get(11..13)?.parse().ok()?;
        let minute: u32 = text.get(14..16)?.parse().ok()?;
        let second: u32 = text.get(17..19)?.parse().ok()?;
        if hour > 23 || minute > 59 || second > 60 {
            return None;
        }
        let rest = skip_fraction(&text[19..])?;
        let offset = parse_offset(rest)?;
        let days = days_from_civil(year, month, day)?;
        let local = days
            .checked_mul(86_400)?
            .checked_add(i64::from(hour) * 3600)?
            .checked_add(i64::from(minute) * 60)?
            .checked_add(i64::from(second))?;
        Some(Self::from_unix_secs(local.checked_sub(offset)?))
    }

    fn parts(self) -> (i32, u32, u32, u32, u32, u32) {
        let days = self.unix_secs.div_euclid(86_400);
        let tod = self.unix_secs.rem_euclid(86_400) as u32;
        let (year, month, day) = civil_from_days(days);
        let hour = tod / 3600;
        let minute = (tod % 3600) / 60;
        let second = tod % 60;
        (year, month, day, hour, minute, second)
    }
}

/// Hour and minute that follow `T` in stored text. The zone is not applied.
pub fn stored_hhmm(text: &str) -> Option<&str> {
    let index = text.find('T')?;
    let hm = text.get(index + 1..)?.get(0..5)?;
    let bytes = hm.as_bytes();
    if bytes.len() == 5
        && bytes[0].is_ascii_digit()
        && bytes[1].is_ascii_digit()
        && bytes[2] == b':'
        && bytes[3].is_ascii_digit()
        && bytes[4].is_ascii_digit()
    {
        Some(hm)
    } else {
        None
    }
}

fn skip_fraction(rest: &str) -> Option<&str> {
    if let Some(stripped) = rest.strip_prefix('.') {
        let digits = stripped.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits == 0 {
            return None;
        }
        Some(&stripped[digits..])
    } else {
        Some(rest)
    }
}

fn parse_offset(rest: &str) -> Option<i64> {
    if rest.is_empty() {
        return Some(0);
    }
    if rest == "Z" || rest == "z" {
        return Some(0);
    }
    let mut chars = rest.chars();
    let sign = match chars.next()? {
        '+' => 1,
        '-' => -1,
        _ => return None,
    };
    let body: String = chars.collect();
    let (hour_text, minute_text) = if let Some((hour, minute)) = body.split_once(':') {
        if minute.len() != 2 {
            return None;
        }
        (hour, minute)
    } else if body.len() == 4 {
        (&body[..2], &body[2..])
    } else if body.len() == 2 {
        (body.as_str(), "00")
    } else {
        return None;
    };
    if hour_text.len() != 2 {
        return None;
    }
    let hour: i64 = hour_text.parse().ok()?;
    let minute: i64 = minute_text.parse().ok()?;
    if !(0..=23).contains(&hour) || !(0..=59).contains(&minute) {
        return None;
    }
    Some(sign * (hour * 3600 + minute * 60))
}

fn is_leap(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i32, month: u32) -> Option<u32> {
    Some(match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(year) => 29,
        2 => 28,
        _ => return None,
    })
}

fn days_from_civil(year: i32, month: u32, day: u32) -> Option<i64> {
    if !(1..=12).contains(&month) || day < 1 {
        return None;
    }
    let year = year as i64 - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = (year - era * 400) as u64;
    let day_of_year =
        (153 * (month as i64 + if month > 2 { -3 } else { 9 }) + 2) / 5 + day as i64 - 1;
    if !(0..=365).contains(&day_of_year) {
        return None;
    }
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year as u64;
    Some(era * 146_097 + day_of_era as i64 - 719_468)
}

fn civil_from_days(mut days: i64) -> (i32, u32, u32) {
    days += 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let day_of_era = (days - era * 146_097) as u64;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146_096) / 365;
    let year = year_of_era as i64 + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_part = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_part + 2) / 5 + 1;
    let month = if month_part < 10 {
        month_part + 3
    } else {
        month_part - 9
    };
    let year = if month <= 2 { year + 1 } else { year };
    (year as i32, month as u32, day as u32)
}
