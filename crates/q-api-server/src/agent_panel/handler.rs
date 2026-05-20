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

    // Build context.
    let now = chrono::Utc::now();
    let ctx = PanelContext {
        wallet,
        viewer_mode,
        state: state.clone(),
        now,
    };

    // Build and run the default pipeline.
    let pipeline = build_default_panel_pipeline();
    let scored = pipeline.run(&ctx).await;
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
        computed_at: now.to_rfc3339(),
        zones,
        total_after_filter,
        total_after_select,
    };

    (StatusCode::OK, Json(response)).into_response()
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
