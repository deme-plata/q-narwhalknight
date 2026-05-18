//! Item 1 — Kalman Bandwidth–Delay Product chunk-size estimator.
//!
//! Replaces a constant `chunk_size_kb = 512` with a value derived from the
//! Apollo Kalman bandwidth predictor already wired into `turbo_sync.rs`.
//!
//! ## Math
//! BDP (bits) = bandwidth (bits/sec) × RTT (sec)
//!            = bandwidth_mbps × 1_000_000 × rtt_ms / 1000
//!            = bandwidth_mbps × 1000 × rtt_ms     (bits)
//!
//! Convert to KiB: BDP_bits / 8 / 1024.
//!
//! Implementation matches the formula in the spec:
//! `chunk_size_kb = (bandwidth_mbps * 1000 / 8) * rtt_ms / 1024`.
//!
//! ## Clamping
//! - Minimum: 64 KB (set by [`min_chunk_size_kb`]; can be overridden by Item 5's
//!   information-theoretic floor).
//! - Maximum: 4096 KB.
//! - When Kalman confidence < `MIN_CONFIDENCE` (0.5), fall back to 512 KB.

use serde::{Deserialize, Serialize};

/// Hard floor (KiB) — used in absence of a higher floor from Item 5.
pub const HARD_FLOOR_KB: u32 = 64;
/// Hard ceiling (KiB) — block-pack responses must fit in libp2p frame budget.
pub const HARD_CEILING_KB: u32 = 4096;
/// Confidence below which we ignore the Kalman estimate.
pub const MIN_CONFIDENCE: f64 = 0.5;
/// Fallback when confidence is too low.
pub const FALLBACK_KB: u32 = 512;
/// Recommended recompute period.
pub const RECOMPUTE_PERIOD_SECS: u64 = 30;

/// Snapshot of Apollo Kalman state (from `turbo_sync.rs::apollo_kalman_*`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KalmanSnapshot {
    /// Estimated bandwidth in megabits per second.
    pub bandwidth_mbps: f64,
    /// Estimated round-trip latency in milliseconds.
    pub latency_ms: f64,
    /// Estimator confidence in `[0.0, 1.0]`.
    pub confidence: f64,
}

impl KalmanSnapshot {
    pub fn new(bandwidth_mbps: f64, latency_ms: f64, confidence: f64) -> Self {
        Self { bandwidth_mbps, latency_ms, confidence }
    }
}

/// Stateless estimator. Holds the dynamic floor that Item 5 can update.
#[derive(Debug, Clone)]
pub struct KalmanBdpEstimator {
    /// Effective floor (KiB). At least [`HARD_FLOOR_KB`]; can be raised by Item 5.
    pub floor_kb: u32,
    /// Effective ceiling (KiB). At most [`HARD_CEILING_KB`].
    pub ceiling_kb: u32,
}

impl Default for KalmanBdpEstimator {
    fn default() -> Self {
        Self {
            floor_kb: HARD_FLOOR_KB,
            ceiling_kb: HARD_CEILING_KB,
        }
    }
}

impl KalmanBdpEstimator {
    /// New estimator with the default floor/ceiling.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allow Item 5 (info-theoretic floor) to raise the minimum.
    /// Floor is `max(HARD_FLOOR_KB, requested)` and never exceeds ceiling.
    pub fn set_floor_kb(&mut self, kb: u32) {
        let raised = kb.max(HARD_FLOOR_KB);
        self.floor_kb = raised.min(self.ceiling_kb);
    }

    /// Compute chunk size in KiB given a Kalman snapshot.
    ///
    /// Returns [`FALLBACK_KB`] (clamped to floor/ceiling) when
    /// `snapshot.confidence < MIN_CONFIDENCE`.
    pub fn chunk_size_kb(&self, snapshot: KalmanSnapshot) -> u32 {
        if !snapshot.confidence.is_finite() || snapshot.confidence < MIN_CONFIDENCE {
            return self.clamp(FALLBACK_KB);
        }
        // Reject nonsense inputs (NaN, negative).
        if !snapshot.bandwidth_mbps.is_finite()
            || !snapshot.latency_ms.is_finite()
            || snapshot.bandwidth_mbps <= 0.0
            || snapshot.latency_ms <= 0.0
        {
            return self.clamp(FALLBACK_KB);
        }

        // chunk_size_kb = (bandwidth_mbps * 1000 / 8) * rtt_ms / 1024
        //               = bandwidth_mbps * rtt_ms * 1000 / 8 / 1024
        let bdp_kb = snapshot.bandwidth_mbps * 1000.0 / 8.0 * snapshot.latency_ms / 1024.0;
        // Round half-to-even via .round() then cast.
        let rounded = bdp_kb.round();
        // Saturating cast — rounded can be huge for crazy inputs.
        let raw = if rounded < 0.0 {
            0u32
        } else if rounded > u32::MAX as f64 {
            u32::MAX
        } else {
            rounded as u32
        };
        self.clamp(raw)
    }

    fn clamp(&self, kb: u32) -> u32 {
        kb.clamp(self.floor_kb, self.ceiling_kb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bdp_known_values() {
        // 100 Mbps, 100 ms → BDP = 100 * 100 * 1000 / 8 / 1024 = 1220.7 KiB → 1221.
        let est = KalmanBdpEstimator::new();
        let snap = KalmanSnapshot::new(100.0, 100.0, 1.0);
        let kb = est.chunk_size_kb(snap);
        assert!(
            (1220..=1221).contains(&kb),
            "expected ≈1221, got {kb}"
        );
    }

    #[test]
    fn low_confidence_falls_back_to_512() {
        let est = KalmanBdpEstimator::new();
        let snap = KalmanSnapshot::new(1000.0, 50.0, 0.49);
        assert_eq!(est.chunk_size_kb(snap), 512);
    }

    #[test]
    fn high_bdp_capped_at_ceiling() {
        // 10 Gbps × 100 ms → way over 4 MiB.
        let est = KalmanBdpEstimator::new();
        let snap = KalmanSnapshot::new(10_000.0, 100.0, 1.0);
        assert_eq!(est.chunk_size_kb(snap), HARD_CEILING_KB);
    }

    #[test]
    fn tiny_bdp_lifted_to_floor() {
        // 1 Mbps × 1 ms → BDP ≈ 0.12 KiB → floor 64.
        let est = KalmanBdpEstimator::new();
        let snap = KalmanSnapshot::new(1.0, 1.0, 1.0);
        assert_eq!(est.chunk_size_kb(snap), HARD_FLOOR_KB);
    }

    #[test]
    fn dynamic_floor_respected() {
        let mut est = KalmanBdpEstimator::new();
        est.set_floor_kb(256);
        let snap = KalmanSnapshot::new(1.0, 1.0, 1.0);
        assert_eq!(est.chunk_size_kb(snap), 256);
    }

    #[test]
    fn floor_cannot_drop_below_hard_floor() {
        let mut est = KalmanBdpEstimator::new();
        est.set_floor_kb(10);
        assert_eq!(est.floor_kb, HARD_FLOOR_KB);
    }

    #[test]
    fn nan_inputs_fall_back() {
        let est = KalmanBdpEstimator::new();
        let snap = KalmanSnapshot::new(f64::NAN, 50.0, 0.9);
        assert_eq!(est.chunk_size_kb(snap), 512);
    }

    #[test]
    fn nonzero_confidence_threshold_boundary() {
        // confidence == MIN_CONFIDENCE should be accepted (>=).
        let est = KalmanBdpEstimator::new();
        let snap = KalmanSnapshot::new(100.0, 100.0, MIN_CONFIDENCE);
        let kb = est.chunk_size_kb(snap);
        assert!(kb > FALLBACK_KB, "with strong BDP and threshold conf we expect > fallback, got {kb}");
    }
}
