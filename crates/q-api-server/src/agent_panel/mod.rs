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

// Sub-modules below are stubs — Codex implements per the spec roadmap layers L1-L5.
// pub mod sources;
// pub mod hydrators;
// pub mod filters;
// pub mod scorers;
// pub mod selectors;
// pub mod handler;
