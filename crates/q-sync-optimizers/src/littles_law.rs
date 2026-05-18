//! Item 4 — Little's law parallelism estimator.
//!
//! From Little's Law: `L = λ × W` where
//!   L = number of in-flight requests,
//!   λ = throughput (requests/sec, or here: blocks/sec),
//!   W = mean response time (RTT in seconds).
//!
//! For our purposes: `optimal_inflight = target_throughput * mean_rtt_secs`.
//!
//! - `target_throughput` defaults to 1000 blocks/sec (override per-peer).
//! - `mean_rtt` is an EWMA over measured chunk RTTs (α = 0.2).
//!
//! When this estimate disagrees with the CUBIC cwnd (Item 3), the scheduler
//! takes the min — never push more in-flight than either Little's law or
//! CUBIC sanctions.
//!
//! ## Prometheus
//! Expose `qnk_optimal_inflight` gauge per peer. The scheduler reads it via
//! [`LittlesLawEstimator::optimal_inflight`] each tick.

use serde::{Deserialize, Serialize};

/// Default EWMA smoothing factor for RTT.
pub const DEFAULT_ALPHA: f64 = 0.2;
/// Default target throughput in blocks/sec.
pub const DEFAULT_TARGET_BPS: f64 = 1000.0;
/// Minimum sane in-flight count.
pub const MIN_INFLIGHT: u32 = 1;
/// Maximum sane in-flight count (matches CUBIC ceiling).
pub const MAX_INFLIGHT: u32 = 64;

/// Stateful estimator. One per peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LittlesLawEstimator {
    /// EWMA of measured chunk RTT (seconds). None until first sample.
    mean_rtt_secs: Option<f64>,
    /// Target blocks-per-second (λ).
    target_bps: f64,
    /// EWMA α (smoothing factor in [0,1]).
    alpha: f64,
}

impl Default for LittlesLawEstimator {
    fn default() -> Self {
        Self {
            mean_rtt_secs: None,
            target_bps: DEFAULT_TARGET_BPS,
            alpha: DEFAULT_ALPHA,
        }
    }
}

impl LittlesLawEstimator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_target(target_bps: f64) -> Self {
        Self {
            target_bps: target_bps.max(0.0),
            ..Default::default()
        }
    }

    /// Record one chunk RTT in milliseconds.
    pub fn record_rtt_ms(&mut self, rtt_ms: f64) {
        if !rtt_ms.is_finite() || rtt_ms <= 0.0 {
            return;
        }
        let secs = rtt_ms / 1000.0;
        self.mean_rtt_secs = Some(match self.mean_rtt_secs {
            None => secs,
            Some(prev) => self.alpha * secs + (1.0 - self.alpha) * prev,
        });
    }

    /// Current EWMA RTT in milliseconds (None if no samples yet).
    pub fn mean_rtt_ms(&self) -> Option<f64> {
        self.mean_rtt_secs.map(|s| s * 1000.0)
    }

    /// `L = λ × W`. Returns `None` if no RTT samples yet.
    pub fn optimal_inflight(&self) -> Option<u32> {
        let w = self.mean_rtt_secs?;
        let l = self.target_bps * w;
        let rounded = l.round();
        let clipped = if rounded < MIN_INFLIGHT as f64 {
            MIN_INFLIGHT
        } else if rounded > MAX_INFLIGHT as f64 {
            MAX_INFLIGHT
        } else {
            rounded as u32
        };
        Some(clipped)
    }

    /// Combine with CUBIC cwnd — take the min, since whichever is smaller
    /// is the binding constraint.
    pub fn combined_with_cubic(&self, cwnd: u32) -> u32 {
        match self.optimal_inflight() {
            Some(l) => l.min(cwnd),
            None => cwnd, // No RTT data yet — defer entirely to CUBIC.
        }
    }

    /// Update the target throughput.
    pub fn set_target_bps(&mut self, bps: f64) {
        if bps.is_finite() && bps >= 0.0 {
            self.target_bps = bps;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_samples_returns_none() {
        let est = LittlesLawEstimator::new();
        assert!(est.optimal_inflight().is_none());
    }

    #[test]
    fn known_value_30ms_1000bps() {
        // L = 1000 b/s × 0.030 s = 30
        let mut est = LittlesLawEstimator::new();
        est.record_rtt_ms(30.0);
        assert_eq!(est.optimal_inflight(), Some(30));
    }

    #[test]
    fn ewma_smoothing() {
        let mut est = LittlesLawEstimator::new();
        est.record_rtt_ms(100.0); // first sample → mean = 100
        est.record_rtt_ms(0.0); // ignored (≤0)
        let m1 = est.mean_rtt_ms().unwrap();
        assert!((m1 - 100.0).abs() < 1e-9);
        est.record_rtt_ms(200.0);
        // mean = 0.2 * 200 + 0.8 * 100 = 120
        let m2 = est.mean_rtt_ms().unwrap();
        assert!((m2 - 120.0).abs() < 1e-9);
    }

    #[test]
    fn clamped_to_ceiling() {
        let mut est = LittlesLawEstimator::with_target(10_000.0);
        est.record_rtt_ms(1000.0); // 1s RTT → L = 10000, clamped to 64.
        assert_eq!(est.optimal_inflight(), Some(MAX_INFLIGHT));
    }

    #[test]
    fn clamped_to_floor() {
        let mut est = LittlesLawEstimator::with_target(1.0);
        est.record_rtt_ms(1.0); // 1 × 0.001 = 0.001, clamped to 1.
        assert_eq!(est.optimal_inflight(), Some(MIN_INFLIGHT));
    }

    #[test]
    fn combined_takes_min() {
        let mut est = LittlesLawEstimator::new();
        est.record_rtt_ms(30.0); // L = 30
        assert_eq!(est.combined_with_cubic(50), 30);
        assert_eq!(est.combined_with_cubic(20), 20);
    }

    #[test]
    fn combined_no_data_defers_to_cubic() {
        let est = LittlesLawEstimator::new();
        assert_eq!(est.combined_with_cubic(42), 42);
    }

    #[test]
    fn nan_rtt_ignored() {
        let mut est = LittlesLawEstimator::new();
        est.record_rtt_ms(f64::NAN);
        est.record_rtt_ms(-1.0);
        assert!(est.optimal_inflight().is_none());
    }
}
