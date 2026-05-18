//! q-sync-optimizers — scientific/mathematical building blocks for the Q-NarwhalKnight
//! turbo-sync chunk scheduler.
//!
//! Each module is intentionally small, stateless-ish, and easy to test in
//! isolation. Integration into `crates/q-storage/src/turbo_sync.rs` happens in a
//! follow-up pass; today this crate just builds the toolkit.
//!
//! # Modules
//! 1. [`kalman_bdp`]    — Kalman bandwidth–delay product chunk-size estimator.
//! 2. [`beta_score`]    — Bayesian peer reliability (Beta distribution + Thompson sampling).
//! 3. [`cubic_aimd`]    — CUBIC-style AIMD per-peer congestion window.
//! 4. [`littles_law`]   — Little's law parallelism estimator (combined with CUBIC).
//! 5. [`chunk_floor`]   — Information-theoretic chunk-size floor (`log₂(peers) × bits/block`).
//! 6. [`markov_peer`]   — 3-state {Fast, Slow, Stalled} Markov peer model.
//! 7. [`ewmv_rtt`]      — Exponentially-weighted moving variance over RTT for adaptive timeout.
//! 8. [`peer_fail_pred`] — Hand-tuned logistic regression for "peer about to fail" probability.
//!
//! # Design constraints
//! - All algorithms <10 µs per call (HOT-PATH).
//! - All math in `f64`; cast to integers only at the boundary.
//! - No `unwrap()` on user/network input.
//! - No global mutable state — state flows through `&mut self`.

#![deny(missing_debug_implementations)]

pub mod beta_score;
pub mod chunk_floor;
pub mod cubic_aimd;
pub mod ewmv_rtt;
pub mod kalman_bdp;
pub mod littles_law;
pub mod markov_peer;
pub mod peer_fail_pred;

// Re-exports for convenience.
pub use beta_score::{BetaCounter, BetaScoreRegistry};
pub use chunk_floor::ChunkFloorEstimator;
pub use cubic_aimd::{CubicRegistry, CubicWindow};
pub use ewmv_rtt::EwmvRtt;
pub use kalman_bdp::{KalmanBdpEstimator, KalmanSnapshot};
pub use littles_law::LittlesLawEstimator;
pub use markov_peer::{MarkovPeer, MarkovRegistry, PeerState};
pub use peer_fail_pred::{PeerFailFeatures, PeerFailPredictor};
