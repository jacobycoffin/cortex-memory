//! Lexical recall. The MATCH string is built here. The raw query is never passed to FTS.

use std::collections::BTreeMap;
use std::fmt;

use rusqlite::OptionalExtension;

use crate::error::{map_query_error, Presence, StoreError};
use crate::open::Store;

pub const LEXICAL_RANKER: &str = "lexical";
pub const NOTE_LEXICAL_BREAKDOWN: &str =
    "why is the last recorded breakdown, not the lexical score";
pub const NOTE_LEXICAL_ONLY: &str =
    "lexical read model; semantic fusion stays in the Python retriever";

const DROPPED_COMPONENTS: &[&str] = &["shadow_score", "shadow_score_delta"];

#[derive(Debug)]
pub enum LexicalError {
    /// The query produced no letter or digit tokens. Callers exit 2.
    NoTokens,
    Store(StoreError),
}

impl fmt::Display for LexicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTokens => write!(f, "no fts tokens"),
            Self::Store(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for LexicalError {}

impl From<StoreError> for LexicalError {
    fn from(err: StoreError) -> Self {
        Self::Store(err)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LexicalHit {
    pub id: String,
    pub kind: String,
    pub content: String,
    /// `-bm25(memory_fts)`. Higher is better. This is not the Python fusion score.
    pub score: f64,
    pub why: BTreeMap<String, f64>,
    pub note: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoreBreakdown {
    pub task_id: String,
    pub rank: i64,
    pub memory_id: String,
    pub score: f64,
    pub why: BTreeMap<String, f64>,
    pub components_json: String,
    pub created_at: String,
    pub note: &'static str,
}

/// Plain words for one ranking component, matching `plain_summary.COMPONENT_WORDS`.
pub fn component_words(key: &str) -> String {
    let mapped = match key {
        "semantic" | "semantic_similarity" | "lexical" => "wording match",
        "keyword" | "bm25" => "keyword match",
        "semantic_fusion" => "found by meaning",
        "recency" => "recency",
        "importance" => "importance",
        "trust" => "trust",
        "confidence" => "confidence",
        "currentness" => "up-to-dateness",
        "usage" | "usage_weight" => "past usefulness",
        "activation" => "how often it comes up",
        "graph" | "graph_expansion" | "association" => "connected memories",
        "entity_anchored" | "entity_boost" => "named-entity match",
        "project_match" => "same project",
        "scope_match" => "same scope",
        "goal_match" => "same goal",
        "system_match" => "same system",
        "version_match" => "same version",
        "precondition_match" => "preconditions match",
        _ => return key.replace('_', " "),
    };
    mapped.to_string()
}

/// Round half away from zero to two decimal places.
pub fn round_2(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    (value * 100.0).round() / 100.0
}

/// Translate `components_json` into at most four plain-word weights.
///
/// `shadow_score` and `shadow_score_delta` are dropped. Keys that collapse to
/// the same words are summed. The four largest absolute values are kept.
pub fn translate_components(components_json: &str) -> BTreeMap<String, f64> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(components_json) else {
        return BTreeMap::new();
    };
    let Some(object) = value.as_object() else {
        return BTreeMap::new();
    };
    let mut summed: BTreeMap<String, f64> = BTreeMap::new();
    for (key, component) in object {
        if DROPPED_COMPONENTS.contains(&key.as_str()) {
            continue;
        }
        let Some(number) = component.as_f64() else {
            continue;
        };
        if !number.is_finite() || number == 0.0 {
            continue;
        }
        *summed.entry(component_words(key)).or_insert(0.0) += number;
    }
    let mut ranked: Vec<(String, f64)> = summed
        .into_iter()
        .filter(|(_, value)| *value != 0.0)
        .collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .abs()
            .partial_cmp(&left.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(4);
    ranked
        .into_iter()
        .map(|(word, value)| (word, round_2(value)))
        .collect()
}

/// Split on characters that are not letters or digits, quote each token as an
/// FTS phrase, and join with ` AND `. `None` when the query has no tokens.
pub fn fts_match_query(query: &str) -> Option<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in query.chars() {
        if ch.is_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(quote_phrase(&current));
            current.clear();
        }
    }
    if !current.is_empty() {
        tokens.push(quote_phrase(&current));
    }
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" AND "))
    }
}

fn quote_phrase(token: &str) -> String {
    let mut out = String::from("\"");
    for ch in token.chars() {
        if ch == '"' {
            out.push('"');
        }
        out.push(ch);
    }
    out.push('"');
    out
}

impl Store {
    /// Lexical hits for `query`. `memory_fts` is optional.
    ///
    /// A missing table is [`Presence::NotPresent`]. No tokens is
    /// [`LexicalError::NoTokens`] and does not call `MATCH`. The default limit
    /// used by the CLI is [`crate::DEFAULT_RECALL_LIMIT`].
    pub fn lexical_recall(
        &self,
        query: &str,
        limit: i64,
    ) -> Result<Presence<Vec<LexicalHit>>, LexicalError> {
        let Some(match_query) = fts_match_query(query) else {
            return Err(LexicalError::NoTokens);
        };
        if !self.has_table("memory_fts")? {
            return Ok(Presence::NotPresent);
        }
        if limit <= 0 {
            return Ok(Presence::Present(Vec::new()));
        }
        let breakdown_present = self.has_table("recall_score_breakdown")?;
        let mut statement = self
            .conn
            .prepare(
                "SELECT m.id, m.kind, m.content, -bm25(memory_fts) AS score \
                 FROM memory_fts \
                 JOIN memories m ON m.id = memory_fts.memory_id \
                 WHERE memory_fts MATCH ?1 AND m.state IN ('active','cold') \
                 ORDER BY bm25(memory_fts) \
                 LIMIT ?2",
            )
            .map_err(map_query_error)?;
        let rows = statement
            .query_map(rusqlite::params![match_query, limit], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f64>(3)?,
                ))
            })
            .map_err(map_query_error)?;
        let mut hits = Vec::new();
        for row in rows {
            let (id, kind, content, score) = row.map_err(map_query_error)?;
            let (why, note) = self.why_for_hit(&id, score, breakdown_present)?;
            hits.push(LexicalHit {
                id,
                kind,
                content,
                score,
                why,
                note,
            });
        }
        Ok(Presence::Present(hits))
    }

    fn why_for_hit(
        &self,
        memory_id: &str,
        lexical_score: f64,
        breakdown_present: bool,
    ) -> Result<(BTreeMap<String, f64>, &'static str), StoreError> {
        if !breakdown_present {
            return Ok((lexical_why(lexical_score), NOTE_LEXICAL_ONLY));
        }
        let components: Option<String> = self
            .conn
            .query_row(
                "SELECT components_json FROM recall_score_breakdown \
                 WHERE memory_id = ?1 ORDER BY created_at DESC LIMIT 1",
                [memory_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(map_query_error)?;
        match components {
            Some(json) => Ok((translate_components(&json), NOTE_LEXICAL_BREAKDOWN)),
            None => Ok((lexical_why(lexical_score), NOTE_LEXICAL_ONLY)),
        }
    }
}

fn lexical_why(score: f64) -> BTreeMap<String, f64> {
    let mut why = BTreeMap::new();
    why.insert("wording match".to_string(), round_2(score));
    why
}
