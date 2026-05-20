//! REST handler: `GET /api/v1/agent/panel/:addr` — Agent Activity Panel API
//!
//! Returns the top-K scored tasks for `addr`, grouped into the three panel
//! zones (NOW / QUEUED / DONE) per `docs/agent-activity-panel-spec.md` §2.
//! Built on the 6-trait pipeline in `pipeline.rs` + the concrete impls in
//! `concrete.rs`.
//!
//! Query params:
//!   - `mode=owner|embed` (default: owner)
//!     - owner mode: viewer is the wallet's owner, sees full data + approval
//!       affordances + Local trust tier
//!     - embed mode: read-only view, Local tasks hidden, trust tier shown as
//!       Observed
//!   - `zone=now|queued|done|all` (default: all) — server-side zone filter so
//!     each panel section can independently refresh
//!   - `limit=N` (default: 50, max: 200) — top-K cap
//!
//! Response shape:
//! ```json
//! {
//!   "wallet": "qnk7154929a...",
//!   "viewer_mode": "owner",
//!   "computed_at": "2026-05-20T15:30:00Z",
//!   "zones": {
//!     "now":    [{"task_id": "...", "task_type": "MempoolTx", ...}, ...],
//!     "queued": [...],
//!     "done":   [...]
//!   },
//!   "total_after_filter": 47,
//!   "total_after_select": 47
//! }
//! ```

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::AppState;
use crate::wallet_auth::AuthenticatedWallet;
use super::concrete::{
    build_default_panel_pipeline, TaskCandidate, TaskStatus,
};
use super::pipeline::{PanelContext, ViewerMode};

// ============ REQUEST/RESPONSE SHAPES ============

#[derive(Debug, Deserialize)]
pub struct PanelQuery {
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub zone: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct PanelResponse {
    pub wallet: String,
    pub viewer_mode: String,
    pub computed_at: String,
    pub zones: PanelZones,
    pub total_after_filter: usize,
    pub total_after_select: usize,
}

#[derive(Debug, Serialize)]
pub struct PanelZones {
    pub now: Vec<TaskCandidate>,
    pub queued: Vec<TaskCandidate>,
    pub done: Vec<TaskCandidate>,
}

#[derive(Debug, Serialize)]
struct PanelError {
    error: String,
    detail: String,
}

// ============ HANDLER ============

/// `GET /api/v1/agent/panel/:addr`
///
/// SECURITY (v10.10.7): `mode=owner` requires X-Wallet-Auth whose signing
/// address matches the path :addr. `mode=embed` is public + filtered
/// (TrustTier::Local tasks are hidden by EmbedVisibilityFilter so private
/// labels/memos don't leak to external viewers).
pub async fn get_agent_panel(
    State(state): State<Arc<AppState>>,
    Path(addr_str): Path<String>,
    Query(params): Query<PanelQuery>,
    auth: Option<AuthenticatedWallet>,
) -> impl IntoResponse {
    // Parse wallet address from string (accepts "qnk<hex>" or raw hex).
    let wallet = match parse_wallet_address(&addr_str) {
        Ok(w) => w,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(PanelError {
                    error: "INVALID_WALLET_ADDRESS".to_string(),
                    detail: e,
                }),
            )
                .into_response();
        }
    };

    // Resolve viewer mode + enforce auth for owner mode.
    let requested_mode = params.mode.as_deref().unwrap_or("owner");
    let viewer_mode = match requested_mode {
        "owner" => {
            // mode=owner requires X-Wallet-Auth signing this exact path with
            // the panel wallet's address. Anything else is observed/embed.
            match auth {
                Some(a) if a.address == wallet => ViewerMode::Owner,
                Some(_) => {
                    return (
                        StatusCode::FORBIDDEN,
                        Json(PanelError {
                            error: "AUTH_MISMATCH".to_string(),
                            detail: "X-Wallet-Auth address does not match path :addr; mode=owner requires the wallet to authenticate itself".to_string(),
                        }),
                    )
                        .into_response();
                }
                None => {
                    return (
                        StatusCode::UNAUTHORIZED,
                        Json(PanelError {
                            error: "AUTH_REQUIRED".to_string(),
                            detail: "mode=owner requires X-Wallet-Auth header signing this path; use mode=embed for public read".to_string(),
                        }),
                    )
                        .into_response();
                }
            }
        }
        "embed" => ViewerMode::Embed,
        other => {
            return (
                StatusCode::BAD_REQUEST,
                Json(PanelError {
                    error: "INVALID_VIEWER_MODE".to_string(),
                    detail: format!("expected 'owner' or 'embed', got '{}'", other),
                }),
            )
                .into_response();
        }
    };

    // Resolve zone filter (post-pipeline; the pipeline runs all sources
    // regardless and we partition the output).
    let zone_filter = ZoneFilter::from_str(params.zone.as_deref().unwrap_or("all"));

    // Limit cap (server-enforced max).
    let limit = params.limit.unwrap_or(50).min(200);

    // Build context (v10.10.10: use the new constructor that initialises
    // the query_data map for QueryHydrator stages).
    let ctx = PanelContext::new(wallet, viewer_mode, state.clone());

    // Build and run the default pipeline (v10.10.10: factory takes the
    // shared SeenTracker so the previously-seen filter sees prior runs).
    let pipeline = build_default_panel_pipeline(super::seen_tracker::global());
    let scored = pipeline.run(&ctx).await;

    // v10.10.10: record final candidates' scores into the ScoreHistory ring
    // buffer so /api/v1/agent/score-history/:addr can serve them back. This
    // is the "killer next move" foundation (docs/x-algorithm-deeper-dive
    // -2026-05-20.md §2.6).
    {
        let history = super::score_history::global();
        let viewer_wallet = hex::encode(wallet);
        let at_unix = ctx.now.timestamp();
        let entries: Vec<super::score_history::ScoreEntry> = scored
            .iter()
            .map(|c| super::score_history::ScoreEntry {
                at_unix,
                viewer_wallet: viewer_wallet.clone(),
                task_id: c.task_id.clone(),
                task_type: format!("{:?}", c.task_type),
                status: format!("{:?}", c.status),
                score: c.score.clone().unwrap_or_else(|| {
                    super::scorers::ScoreReport {
                        total: 0.0,
                        components: Vec::new(),
                    }
                }),
                selected: true,
            })
            .collect();
        history.record_batch(entries);

        // Also mark these task_ids as "seen" so the next pipeline run for
        // this viewer doesn't re-surface them. (The SeenRecorderSideEffect
        // in the factory does this too, but doing it here lets us scope
        // to the actual selected list — the SideEffect path doesn't yet
        // have access to the post-selection list.)
        let tracker = super::seen_tracker::global();
        for c in &scored {
            tracker.mark_seen(&viewer_wallet, &c.task_id);
        }

        // v10.10.10 persistence: fire-and-forget — spawn background tasks
        // that flush the in-memory rings to disk. Not awaited because the
        // request-response path shouldn't block on I/O. The persist files
        // overwrite atomically via rename so a crash mid-write doesn't
        // corrupt anything.
        history.clone().spawn_persist();
        tracker.clone().spawn_persist();
    }
    let total_after_select = scored.len();
    // (The pipeline's TopK selector already capped to 50; we cap further
    // if the caller asked for less than the default.)
    let truncated: Vec<TaskCandidate> = scored.into_iter().take(limit).collect();
    let total_after_filter = truncated.len();

    // Partition into the three zones.
    let mut now_zone: Vec<TaskCandidate> = Vec::new();
    let mut queued_zone: Vec<TaskCandidate> = Vec::new();
    let mut done_zone: Vec<TaskCandidate> = Vec::new();
    for t in truncated {
        match t.status {
            TaskStatus::Executing => now_zone.push(t),
            TaskStatus::PendingApproval => queued_zone.push(t),
            TaskStatus::Confirmed | TaskStatus::Failed | TaskStatus::Expired => {
                done_zone.push(t)
            }
        }
    }

    // Apply zone filter if narrower than 'all'.
    let zones = match zone_filter {
        ZoneFilter::All => PanelZones { now: now_zone, queued: queued_zone, done: done_zone },
        ZoneFilter::Now => PanelZones { now: now_zone, queued: Vec::new(), done: Vec::new() },
        ZoneFilter::Queued => PanelZones { now: Vec::new(), queued: queued_zone, done: Vec::new() },
        ZoneFilter::Done => PanelZones { now: Vec::new(), queued: Vec::new(), done: done_zone },
    };

    let response = PanelResponse {
        wallet: format!("qnk{}", hex::encode(wallet)),
        viewer_mode: match viewer_mode {
            ViewerMode::Owner => "owner".to_string(),
            ViewerMode::Embed => "embed".to_string(),
        },
        computed_at: ctx.now.to_rfc3339(),
        zones,
        total_after_filter,
        total_after_select,
    };

    (StatusCode::OK, Json(response)).into_response()
}

// ============ SCORE HISTORY ENDPOINTS (v10.10.10) ============

#[derive(Debug, Deserialize)]
pub struct ScoreHistoryQuery {
    /// How many entries to return, newest-first. Default 100, max 1000.
    #[serde(default)]
    pub limit: Option<usize>,
    /// If "summary", returns the `ScoreHistorySummary` instead of the raw
    /// entry list. Useful for the calibration audit endpoint.
    #[serde(default)]
    pub view: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ScoreHistoryResponse {
    pub wallet: String,
    pub entries: Vec<super::score_history::ScoreEntry>,
    pub total_returned: usize,
}

/// `GET /api/v1/agent/score-history/:addr`
///
/// Returns the in-memory score history for `addr`. Requires X-Wallet-Auth
/// matching the path :addr — this is owner-only because the score breakdown
/// can include signals derived from private state (mempool backlog,
/// reserve ratios) that a third-party shouldn't see.
///
/// `?view=summary` returns the `ScoreHistorySummary` instead — what the
/// calibration audit will eventually consume.
pub async fn get_score_history(
    State(_state): State<Arc<AppState>>,
    Path(addr_str): Path<String>,
    Query(params): Query<ScoreHistoryQuery>,
    auth: Option<AuthenticatedWallet>,
) -> impl IntoResponse {
    let wallet = match parse_wallet_address(&addr_str) {
        Ok(w) => w,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(PanelError {
                    error: "INVALID_WALLET_ADDRESS".to_string(),
                    detail: e,
                }),
            )
                .into_response();
        }
    };

    // Owner-only — no embed mode here. Score breakdowns include other-user
    // signals that we don't want to leak.
    let authed = match auth {
        Some(a) if a.address == wallet => a,
        Some(_) => {
            return (
                StatusCode::FORBIDDEN,
                Json(PanelError {
                    error: "AUTH_MISMATCH".to_string(),
                    detail: "X-Wallet-Auth address does not match path :addr".to_string(),
                }),
            )
                .into_response();
        }
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(PanelError {
                    error: "AUTH_REQUIRED".to_string(),
                    detail: "/api/v1/agent/score-history requires X-Wallet-Auth".to_string(),
                }),
            )
                .into_response();
        }
    };

    let wallet_hex = hex::encode(authed.address);
    let history = super::score_history::global();

    if params.view.as_deref() == Some("summary") {
        match history.summary(&wallet_hex) {
            Some(s) => return (StatusCode::OK, Json(s)).into_response(),
            None => {
                return (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "wallet": wallet_hex,
                        "entries": 0,
                        "note": "no score history recorded yet for this wallet — submit some panel queries first",
                    })),
                )
                    .into_response();
            }
        }
    }

    let limit = params.limit.unwrap_or(100).min(1000);
    let entries = history.get_recent(&wallet_hex, limit);
    let total_returned = entries.len();
    (
        StatusCode::OK,
        Json(ScoreHistoryResponse {
            wallet: format!("qnk{}", wallet_hex),
            entries,
            total_returned,
        }),
    )
        .into_response()
}

// ============ HELPERS ============

#[derive(Debug, Clone, Copy)]
enum ZoneFilter {
    All,
    Now,
    Queued,
    Done,
}

impl ZoneFilter {
    fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "now" => ZoneFilter::Now,
            "queued" => ZoneFilter::Queued,
            "done" => ZoneFilter::Done,
            _ => ZoneFilter::All,
        }
    }
}

/// Accept either "qnk<64hex>" or raw "<64hex>". Returns the 32-byte address.
fn parse_wallet_address(s: &str) -> Result<[u8; 32], String> {
    let clean = s.trim_start_matches("qnk").trim_start_matches("QNK");
    if clean.len() != 64 {
        return Err(format!(
            "expected 64-hex address (optionally prefixed with 'qnk'), got {} chars",
            clean.len()
        ));
    }
    let bytes = hex::decode(clean)
        .map_err(|e| format!("hex decode failed: {}", e))?;
    if bytes.len() != 32 {
        return Err(format!("expected 32 bytes, got {}", bytes.len()));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

// ============ TESTS ============

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_wallet_address_with_qnk_prefix() {
        let s = format!("qnk{}", "ab".repeat(32));
        let bytes = parse_wallet_address(&s).unwrap();
        assert_eq!(bytes, [0xab; 32]);
    }

    #[test]
    fn parse_wallet_address_without_prefix() {
        let s = "ab".repeat(32);
        let bytes = parse_wallet_address(&s).unwrap();
        assert_eq!(bytes, [0xab; 32]);
    }

    #[test]
    fn parse_wallet_address_too_short_errors() {
        let s = "ab".repeat(10);
        let r = parse_wallet_address(&s);
        assert!(r.is_err());
    }

    #[test]
    fn zone_filter_parses_known_values() {
        assert!(matches!(ZoneFilter::from_str("now"), ZoneFilter::Now));
        assert!(matches!(ZoneFilter::from_str("QUEUED"), ZoneFilter::Queued));
        assert!(matches!(ZoneFilter::from_str("done"), ZoneFilter::Done));
        assert!(matches!(ZoneFilter::from_str("anything-else"), ZoneFilter::All));
    }
}
