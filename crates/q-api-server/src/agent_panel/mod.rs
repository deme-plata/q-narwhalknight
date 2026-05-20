//! Agent Activity Panel — backend module.
//!
//! Per `docs/agent-activity-panel-spec.md`. The panel surface on quillon.xyz
//! shows what an agent is doing across three zones: NOW (executing) /
//! QUEUED (awaiting human or external trigger) / DONE (last 24 h completed).
//!
//! Architecture borrows directly from xAI's Home Mixer pattern in
//! `github.com/xai-org/x-algorithm` (Apache-2.0): six composable trait-types
//! that any decision-streaming workload fits.
//!
//! Module layout (Codex to fill in as work progresses):
//! - `pipeline`  — the six trait definitions (this file pulls them together)
//! - `sources`   — concrete `Source` impls per task class
//! - `hydrators` — concrete `Hydrator` impls for context enrichment
//! - `filters`   — concrete `Filter` impls for age/visibility/correctness
//! - `scorers`   — concrete `Scorer` impls for ranking
//! - `selectors` — concrete `Selector` impls (Top-K, FIFO, by-status)
//! - `handler`   — `GET /api/v1/agent/panel/{addr}` REST handler

pub mod pipeline;
pub mod scorers;

/// v10.10.5: concrete Source/Filter/Scorer/SideEffect impls + the
/// build_default_panel_pipeline factory. Replaces the per-feature
/// sub-modules placeholder.
pub mod concrete;

/// v10.10.5: GET /api/v1/agent/panel/:addr REST handler.
pub mod handler;

/// v10.10.10: per-wallet "tasks already shown" tracker so polling agents
/// don't see the same task forever. Mirrors xAI's
/// `home-mixer/filters/previously_seen_posts_filter.rs`.
pub mod seen_tracker;

/// v10.10.10: in-memory `ScoreReport` history per wallet so we can run
/// calibration audits + future hard-negative mining without re-running
/// the pipeline against past chain state. The "killer next move" from
/// docs/x-algorithm-deeper-dive-2026-05-20.md §2.6.
pub mod score_history;
