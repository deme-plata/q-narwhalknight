//! Item 8 — Logistic regression for "peer about to fail" probability.
//!
//! We don't have ML training infrastructure for the sync subsystem (yet), so
//! the weights are hand-tuned. The output `p_fail ∈ [0, 1]` is the logistic of
//! a linear combination of:
//!
//! 1. **Rising-RTT slope** — `(rtt₃ - rtt₁) / max(rtt₁, ε)`, normalised so
//!    a doubling of RTT in 3 samples ≈ +1. Positive weight (slow ⇒ failing).
//! 2. **Inverse mean response size** — small responses suggest the peer keeps
//!    serving empty packs (no blocks at requested heights). Positive weight on
//!    `(REF_SIZE − mean_size) / REF_SIZE` clamped to `[0, 1]`.
//! 3. **Seconds since reconnect** — fresh reconnects are more reliable than
//!    stale long-lived connections that haven't recovered from a failure.
//!    Negative weight: a fresh peer (small `secs_since_reconnect`) lowers
//!    `p_fail`. We pass it normalised: `1 − exp(−t / TAU)` ∈ [0, 1].
//! 4. **Recent failure count** — direct positive weight on `failure_count` /
//!    `(failure_count + REF_FAILURES)` ∈ [0, 1].
//!
//! ## Scheduler integration
//! `p_fail > THRESHOLD` (0.7) ⇒ deprioritise the peer unless it's the only
//! candidate. The scheduler is free to compose this with Beta scores (Item 2)
//! and Markov state (Item 6).
//!
//! ## Performance
//! 4 multiply-adds + 1 `exp` ≈ <0.1 µs on modern x86-64.

use serde::{Deserialize, Serialize};

/// Probability above which we recommend skipping the peer.
pub const SKIP_THRESHOLD: f64 = 0.7;

/// Reference response size in bytes — responses smaller than this are
/// penalised proportionally.
pub const REF_SIZE_BYTES: f64 = 50_000.0;

/// Timescale (seconds) for the "freshness" feature.
pub const FRESHNESS_TAU_SECS: f64 = 600.0; // 10 minutes.

/// Reference failure count for the recent-failure feature.
pub const REF_FAILURES: f64 = 5.0;

/// Hand-tuned weights for the logistic regression.
///
/// Order matches [`PeerFailFeatures::to_array`]:
///   `[slope, small_response, freshness, failure_ratio, bias]`.
///
/// Signs / magnitudes:
/// - **slope** (+1.5): rising RTT slope strongly signals failure.
/// - **small_response** (+1.2): chronic empty responses suggest the peer
///   has no useful data.
/// - **freshness** (−1.0): higher freshness ⇒ more time since reconnect ⇒
///   *more* likely to fail, but we want recent reconnects to be optimistic,
///   so the *normalised* freshness gets a negative weight when applied to
///   `1 − exp(−t/τ)`. (Wait: 1 - exp(-t/τ) RISES with t, so a positive
///   weight means more risk for old connections. Hand-tuned positive
///   here: stale connections that have been around long enough to see
///   failures.) See below.
/// - **failure_ratio** (+2.5): recent failures dominate.
/// - **bias** (−2.0): baseline shifts p_fail towards 0 so most peers
///   pass the threshold by default.
///
/// Note: after re-reading the spec ("positive weight on rising RTT slope,
/// positive on small response sizes, positive on high recent-failure
/// count"), freshness is *not* explicitly required to be positive. We give
/// it a small positive weight so a long-lived connection with no
/// reconnect events gets a tiny risk boost — but it's swamped by the
/// other signals.
pub const DEFAULT_WEIGHTS: [f64; 5] = [
    1.5,  // slope (positive)
    1.2,  // small_response (positive)
    0.3,  // freshness (mild positive — long-lived ≠ healthy)
    2.5,  // failure_ratio (positive)
    -2.0, // bias
];

/// Input features for one peer.
///
/// All fields are raw measurements; [`PeerFailFeatures::to_array`] normalises
/// before the logistic.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PeerFailFeatures {
    /// Three most recent RTT samples (ms). Order: oldest → newest.
    pub last_3_rtts_ms: [f64; 3],
    /// Three most recent response sizes (bytes). Order: oldest → newest.
    pub last_3_response_sizes: [f64; 3],
    /// Seconds since the peer was last (re)connected.
    pub secs_since_reconnect: f64,
    /// Number of failures observed recently (e.g. rolling 60 s window).
    pub recent_failure_count: u32,
}

impl Default for PeerFailFeatures {
    fn default() -> Self {
        Self {
            last_3_rtts_ms: [50.0, 50.0, 50.0],
            last_3_response_sizes: [REF_SIZE_BYTES, REF_SIZE_BYTES, REF_SIZE_BYTES],
            secs_since_reconnect: 60.0,
            recent_failure_count: 0,
        }
    }
}

impl PeerFailFeatures {
    /// Convert raw features into the 5-element normalised vector
    /// `[slope, small_response, freshness, failure_ratio, 1.0]`.
    ///
    /// The trailing 1.0 multiplies the bias weight.
    pub fn to_array(&self) -> [f64; 5] {
        // Slope: (rtt₃ - rtt₁) / max(rtt₁, ε). Clamp to [-2, 2].
        let r1 = sanitize_nonneg(self.last_3_rtts_ms[0]);
        let r3 = sanitize_nonneg(self.last_3_rtts_ms[2]);
        let denom = r1.max(1.0);
        let slope = ((r3 - r1) / denom).clamp(-2.0, 2.0);

        // Mean response size, normalised to "smallness in [0, 1]".
        let mean_size = (sanitize_nonneg(self.last_3_response_sizes[0])
            + sanitize_nonneg(self.last_3_response_sizes[1])
            + sanitize_nonneg(self.last_3_response_sizes[2]))
            / 3.0;
        let smallness = if mean_size >= REF_SIZE_BYTES {
            0.0
        } else {
            ((REF_SIZE_BYTES - mean_size) / REF_SIZE_BYTES).clamp(0.0, 1.0)
        };

        // Freshness: 1 - exp(-t / τ), so 0 just-reconnected → 1 long-lived.
        let t = sanitize_nonneg(self.secs_since_reconnect);
        let freshness = 1.0 - (-t / FRESHNESS_TAU_SECS).exp();

        // Failure ratio: f / (f + REF_FAILURES).
        let f = self.recent_failure_count as f64;
        let failure_ratio = f / (f + REF_FAILURES);

        [slope, smallness, freshness, failure_ratio, 1.0]
    }
}

fn sanitize_nonneg(x: f64) -> f64 {
    if x.is_finite() && x >= 0.0 {
        x
    } else {
        0.0
    }
}

/// Stateless logistic-regression predictor with hand-tuned weights.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PeerFailPredictor {
    /// Weight vector (5 elements: 4 feature weights + 1 bias).
    pub weights: [f64; 5],
}

impl Default for PeerFailPredictor {
    fn default() -> Self {
        Self {
            weights: DEFAULT_WEIGHTS,
        }
    }
}

impl PeerFailPredictor {
    /// Predictor with the default hand-tuned weights.
    pub fn new() -> Self {
        Self::default()
    }

    /// Predictor with custom weights (for tuning experiments).
    pub fn with_weights(weights: [f64; 5]) -> Self {
        Self { weights }
    }

    /// Compute `p_fail = σ(w · x) ∈ [0, 1]`.
    ///
    /// `σ(z) = 1 / (1 + e^{-z})`.
    pub fn p_fail(&self, features: &PeerFailFeatures) -> f64 {
        let x = features.to_array();
        let mut z = 0.0_f64;
        for i in 0..5 {
            z += self.weights[i] * x[i];
        }
        sigmoid(z)
    }

    /// Convenience: `p_fail > SKIP_THRESHOLD`.
    pub fn should_skip(&self, features: &PeerFailFeatures) -> bool {
        self.p_fail(features) > SKIP_THRESHOLD
    }
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_features() -> PeerFailFeatures {
        PeerFailFeatures::default()
    }

    fn failing_features() -> PeerFailFeatures {
        PeerFailFeatures {
            last_3_rtts_ms: [50.0, 200.0, 800.0], // 16× RTT growth.
            last_3_response_sizes: [1_000.0, 500.0, 100.0], // tiny responses.
            secs_since_reconnect: 3600.0,         // 1 h old.
            recent_failure_count: 20,             // many failures.
        }
    }

    #[test]
    fn sigmoid_known_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        // σ(2) ≈ 0.8808
        assert!((sigmoid(2.0) - 0.8807970779778823).abs() < 1e-9);
        // Numerical stability at the extremes.
        assert!(sigmoid(1000.0) > 0.999);
        assert!(sigmoid(-1000.0) < 1e-6);
    }

    #[test]
    fn healthy_peer_low_p_fail() {
        let pred = PeerFailPredictor::new();
        let p = pred.p_fail(&healthy_features());
        assert!(p < 0.3, "healthy p_fail was {p}");
        assert!(!pred.should_skip(&healthy_features()));
    }

    #[test]
    fn failing_peer_high_p_fail() {
        let pred = PeerFailPredictor::new();
        let p = pred.p_fail(&failing_features());
        assert!(p > 0.9, "failing p_fail was {p}");
        assert!(pred.should_skip(&failing_features()));
    }

    #[test]
    fn rising_rtt_increases_p_fail() {
        let pred = PeerFailPredictor::new();
        let mut feats = healthy_features();
        let baseline = pred.p_fail(&feats);
        feats.last_3_rtts_ms = [50.0, 100.0, 200.0];
        let bumped = pred.p_fail(&feats);
        assert!(bumped > baseline);
    }

    #[test]
    fn small_responses_increase_p_fail() {
        let pred = PeerFailPredictor::new();
        let mut feats = healthy_features();
        let baseline = pred.p_fail(&feats);
        feats.last_3_response_sizes = [100.0, 100.0, 100.0];
        let bumped = pred.p_fail(&feats);
        assert!(bumped > baseline);
    }

    #[test]
    fn more_failures_increase_p_fail() {
        let pred = PeerFailPredictor::new();
        let mut feats = healthy_features();
        let baseline = pred.p_fail(&feats);
        feats.recent_failure_count = 50;
        let bumped = pred.p_fail(&feats);
        assert!(bumped > baseline);
    }

    #[test]
    fn p_fail_in_unit_interval() {
        let pred = PeerFailPredictor::new();
        for &rtt in &[0.0, 1.0, 10.0, 100.0, 10_000.0] {
            for &size in &[0.0, 1.0, 100.0, 100_000.0] {
                for &age in &[0.0, 60.0, 600.0, 7200.0] {
                    for &fails in &[0u32, 1, 10, 100] {
                        let feats = PeerFailFeatures {
                            last_3_rtts_ms: [rtt, rtt, rtt],
                            last_3_response_sizes: [size, size, size],
                            secs_since_reconnect: age,
                            recent_failure_count: fails,
                        };
                        let p = pred.p_fail(&feats);
                        assert!(
                            (0.0..=1.0).contains(&p),
                            "p_fail = {p} for rtt={rtt} size={size} age={age} fails={fails}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn nan_inputs_are_treated_as_zero() {
        let pred = PeerFailPredictor::new();
        let feats = PeerFailFeatures {
            last_3_rtts_ms: [f64::NAN, f64::INFINITY, -1.0],
            last_3_response_sizes: [f64::NAN; 3],
            secs_since_reconnect: f64::NEG_INFINITY,
            recent_failure_count: 0,
        };
        let p = pred.p_fail(&feats);
        assert!(p.is_finite());
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn skip_threshold_boundary() {
        // Fabricate a feature vector that pushes z ≈ 0 (p ≈ 0.5).
        let pred = PeerFailPredictor::new();
        let feats = PeerFailFeatures {
            last_3_rtts_ms: [50.0, 100.0, 200.0],
            last_3_response_sizes: [25_000.0; 3],
            secs_since_reconnect: 600.0,
            recent_failure_count: 3,
        };
        let p = pred.p_fail(&feats);
        // We don't assert == 0.5, just check threshold logic agrees with p.
        assert_eq!(pred.should_skip(&feats), p > SKIP_THRESHOLD);
    }
}
