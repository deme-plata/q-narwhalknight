//! x-algorithm scorer — HTTP sidecar for the Twitter MCP.
//!
//! Layer 1 (this scaffold): returns calibrated stub predictions based on
//! heuristic features (length, exclamation density, link presence, etc.).
//! The shape of the response matches what the real xAI Phoenix engine
//! would return, so MCP-side callers don't need to change when Layer 2
//! lands.
//!
//! Layer 2 (follow-up): wraps the xAI x-algorithm Python inference path
//! (github.com/xai-org/x-algorithm, Apache-2.0). The repo ships ~3 GB of
//! pretrained model artifacts. The sidecar runs the Phoenix ranking head
//! as a long-lived Python subprocess and exposes /score on this same
//! endpoint contract.
//!
//! API contract (stable from Layer 1 forward):
//!   POST /score   { draft_text, context? } → ScoreResponse
//!   GET  /health  → 200 OK
//!   GET  /version → { layer: 1|2, model: "stub-v1"|"phoenix-v1", ... }

use anyhow::Result;
use axum::{
    extract::Json as JsonExtract,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tracing::{info, warn};

// ════════════════════════════════════════════════════════════════════════
// Wire types — stable across Layer 1 (stub) and Layer 2 (Phoenix)
// ════════════════════════════════════════════════════════════════════════

#[derive(Deserialize, Debug)]
struct ScoreRequest {
    draft_text: String,
    #[serde(default)]
    context: Option<ScoreContext>,
}

#[derive(Deserialize, Debug, Default)]
struct ScoreContext {
    /// Optional: an array of recent tweets from the same author (helps
    /// the model calibrate to author style). Layer 1 ignores this.
    #[serde(default)]
    recent_tweets: Vec<String>,
    /// Optional: target audience hint (e.g. "developers", "general").
    /// Layer 1 ignores this.
    #[serde(default)]
    target_audience: Option<String>,
}

#[derive(Serialize, Debug)]
struct ScoreResponse {
    /// Aggregate engagement score in [0.0, 1.0]. Higher = more predicted
    /// reach + interactions.
    predicted_engagement: f64,
    /// Aggregate risk of blocks/mutes/reports in [0.0, 1.0]. Higher = worse.
    /// Spec §5.3: MCP server SHOULD refuse to queue drafts with neg_signal_risk > 0.15.
    negative_signal_risk: f64,
    /// Per-action probability decomposition.
    per_action_probabilities: PerActionProbabilities,
    /// Suggested alternative phrasings the agent can iterate to.
    variant_suggestions: Vec<VariantSuggestion>,
    /// Which layer/model produced this response.
    model_version: String,
}

#[derive(Serialize, Debug, Default)]
struct PerActionProbabilities {
    favorite: f64,
    reply: f64,
    repost: f64,
    quote: f64,
    click: f64,
    profile_visit: f64,
    video_view: f64,
    photo_expand: f64,
    share: f64,
    dwell_time: f64,
    // Negative:
    block: f64,
    mute: f64,
    report: f64,
}

#[derive(Serialize, Debug)]
struct VariantSuggestion {
    text: String,
    score_delta: f64,
    why: String,
}

#[derive(Serialize)]
struct VersionResponse {
    layer: u8,
    model: &'static str,
    description: &'static str,
}

// ════════════════════════════════════════════════════════════════════════
// Layer 1 scoring — heuristic stub
// ════════════════════════════════════════════════════════════════════════

/// Layer 1 scoring uses transparent heuristics matching what xAI's open-source
/// scorer is known to weight (engagement-bait, brevity, link density, etc.).
/// The numbers are calibrated to fall in plausible ranges relative to xAI's
/// own reported per-action weights, so downstream tooling that operates on
/// these scores doesn't need to be re-tuned when Layer 2 replaces the engine.
fn score_layer1(req: &ScoreRequest) -> ScoreResponse {
    let text = &req.draft_text;
    let len = text.chars().count() as f64;

    // ── Engagement-positive features ────────────────────────────────────
    // xAI's algorithm rewards: tweets that are 80–200 chars, contain a
    // question (likely reply), contain a link OR media (likely click),
    // include @-mentions (likely engagement from mentioned account),
    // and are NOT all caps.
    let len_score = if len >= 80.0 && len <= 200.0 {
        0.7
    } else if len < 40.0 {
        0.25 // too short, less engagement
    } else if len > 270.0 {
        0.4 // near the cap, dwell hurt
    } else {
        0.55
    };
    let has_question = text.contains('?');
    let has_link = text.contains("http://") || text.contains("https://");
    let has_mention = text.contains('@');
    let all_caps_words = text
        .split_whitespace()
        .filter(|w| w.len() > 3 && w.chars().all(|c| !c.is_alphabetic() || c.is_uppercase()))
        .count();
    let all_caps_density = all_caps_words as f64 / (text.split_whitespace().count().max(1) as f64);

    // ── Engagement-negative features ────────────────────────────────────
    // Excess punctuation (!?!?), inflammatory keywords, very long all-caps
    // sequences correlate with block/mute/report.
    let excess_punct = text
        .chars()
        .filter(|c| *c == '!' || *c == '?')
        .count() as f64
        / (len.max(1.0));
    let inflammatory_words = ["FRAUD", "SCAM", "LIE", "EVIL", "DESTROY"]
        .iter()
        .filter(|kw| text.to_uppercase().contains(*kw))
        .count();

    // ── Compute per-action probabilities ────────────────────────────────
    let mut actions = PerActionProbabilities {
        favorite: clamp(len_score * 0.8 + (has_link as u8 as f64) * 0.05),
        reply: clamp(len_score * 0.4 + (has_question as u8 as f64) * 0.3),
        repost: clamp(len_score * 0.3 + (has_link as u8 as f64) * 0.1),
        quote: clamp(len_score * 0.15 + (has_question as u8 as f64) * 0.1),
        click: clamp((has_link as u8 as f64) * 0.65 + len_score * 0.1),
        profile_visit: clamp(len_score * 0.2 + (has_mention as u8 as f64) * 0.1),
        video_view: 0.0,
        photo_expand: 0.0,
        share: clamp(len_score * 0.2),
        dwell_time: clamp(len_score * 0.5 - excess_punct * 0.4),
        block: clamp(inflammatory_words as f64 * 0.05 + all_caps_density * 0.3),
        mute: clamp(inflammatory_words as f64 * 0.03 + all_caps_density * 0.2 + excess_punct * 0.5),
        report: clamp(inflammatory_words as f64 * 0.08),
    };
    // Cap absurd predictions
    actions.dwell_time = actions.dwell_time.max(0.0);

    // ── Aggregate scores ────────────────────────────────────────────────
    let predicted_engagement = clamp(
        actions.favorite * 0.30
            + actions.reply * 0.25
            + actions.repost * 0.20
            + actions.quote * 0.10
            + actions.click * 0.10
            + actions.share * 0.05,
    );
    let negative_signal_risk =
        clamp(actions.block * 0.4 + actions.mute * 0.3 + actions.report * 0.3);

    // ── Variant suggestions (Layer 1: heuristic; Layer 2: model-generated) ──
    let mut variants = Vec::new();
    if len > 200.0 {
        variants.push(VariantSuggestion {
            text: format!("{}…", &text.chars().take(180).collect::<String>()),
            score_delta: 0.04,
            why: "shorter, hits the 80–200 char engagement sweet spot".to_string(),
        });
    }
    if !has_question && len_score < 0.6 {
        variants.push(VariantSuggestion {
            text: format!("{} — what do you think?", text.trim_end_matches('.')),
            score_delta: 0.06,
            why: "adds an explicit reply hook".to_string(),
        });
    }
    if all_caps_density > 0.2 {
        let toned_down: String = text
            .split_whitespace()
            .map(|w| {
                if w.len() > 3 && w.chars().all(|c| !c.is_alphabetic() || c.is_uppercase()) {
                    w.to_lowercase()
                } else {
                    w.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        variants.push(VariantSuggestion {
            text: toned_down,
            score_delta: 0.05,
            why: "lower-cased all-caps words; reduces mute/block risk".to_string(),
        });
    }

    ScoreResponse {
        predicted_engagement,
        negative_signal_risk,
        per_action_probabilities: actions,
        variant_suggestions: variants,
        model_version: "stub-v1".to_string(),
    }
}

fn clamp(x: f64) -> f64 {
    x.max(0.0).min(1.0)
}

// ════════════════════════════════════════════════════════════════════════
// HTTP handlers
// ════════════════════════════════════════════════════════════════════════

async fn score_handler(
    JsonExtract(req): JsonExtract<ScoreRequest>,
) -> Result<Json<ScoreResponse>, StatusCode> {
    if req.draft_text.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    if req.draft_text.len() > 32_000 {
        warn!("draft_text > 32 KB; refusing");
        return Err(StatusCode::PAYLOAD_TOO_LARGE);
    }
    Ok(Json(score_layer1(&req)))
}

async fn health_handler() -> &'static str {
    "ok"
}

async fn version_handler() -> Json<VersionResponse> {
    Json(VersionResponse {
        layer: 1,
        model: "stub-v1",
        description: "Heuristic stub; replace with xAI Phoenix engine in Layer 2",
    })
}

// ════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════

#[derive(Parser, Debug)]
#[clap(name = "x-algorithm-scorer")]
struct Args {
    #[arg(long, env = "SCORER_PORT", default_value_t = 8090)]
    port: u16,
    /// Bind only to localhost (default) or 0.0.0.0 (for containerized deployment).
    #[arg(long, env = "SCORER_BIND_ALL", default_value_t = false)]
    bind_all: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,x_algorithm_scorer=debug".into()),
        )
        .init();

    let args = Args::parse();

    let app = Router::new()
        .route("/score", post(score_handler))
        .route("/health", get(health_handler))
        .route("/version", get(version_handler));

    let host = if args.bind_all { "0.0.0.0" } else { "127.0.0.1" };
    let addr: SocketAddr = format!("{}:{}", host, args.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("x-algorithm-scorer Layer 1 (stub-v1) listening on {}", addr);
    info!("POST /score, GET /health, GET /version");

    axum::serve(listener, app).await?;
    Ok(())
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn r(text: &str) -> ScoreResponse {
        score_layer1(&ScoreRequest {
            draft_text: text.to_string(),
            context: None,
        })
    }

    #[test]
    fn empty_short_text_low_engagement() {
        let s = r("hi");
        assert!(s.predicted_engagement < 0.3);
    }

    #[test]
    fn well_sized_question_high_reply_probability() {
        let s = r("Just shipped the new Agent Fiber Lane spec — what does this mean for chains that lack agent-priority endpoints? Three reasons matter.");
        assert!(s.per_action_probabilities.reply > 0.4);
        assert!(s.predicted_engagement > 0.4);
    }

    #[test]
    fn all_caps_triggers_negative_signal_risk() {
        let s = r("THIS PROJECT IS A SCAM AND EVERYONE INVOLVED IS A FRAUD AND A LIE");
        assert!(s.negative_signal_risk > 0.15);
        assert!(s.per_action_probabilities.block > 0.05);
    }

    #[test]
    fn over_long_text_suggests_shortening() {
        let long = "x".repeat(250);
        let s = r(&long);
        assert!(s.variant_suggestions.iter().any(|v| v.why.contains("shorter")));
    }

    #[test]
    fn link_increases_click_probability() {
        let s = r("New AFL-1 protocol spec landed: https://github.com/deme-plata/q-narwhalknight/pull/91 — open standard, Apache-2.0.");
        assert!(s.per_action_probabilities.click > 0.5);
    }
}
