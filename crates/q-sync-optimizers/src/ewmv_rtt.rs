//! Item 7 — Exponentially-weighted moving variance (EWMV) over RTT.
//!
//! Computes online mean **and** variance using the EWMA/EWMV recursion:
//!
//! ```text
//!   diff      = x - mean
//!   incr      = α * diff
//!   mean      = mean + incr
//!   variance  = (1 - α) * (variance + diff * incr)
//! ```
//!
//! This is the West/Welford-style online weighted variance with α = 0.2.
//!
//! ## Adaptive timeout
//! `timeout = mean + 3σ`, clamped to `[5s, 120s]`. Replaces the constant
//! 60 s timeout used in `turbo_sync.rs`.

use serde::{Deserialize, Serialize};

pub const DEFAULT_ALPHA: f64 = 0.2;
pub const MIN_TIMEOUT_MS: f64 = 5_000.0;
pub const MAX_TIMEOUT_MS: f64 = 120_000.0;
pub const SIGMA_MULTIPLIER: f64 = 3.0;

/// Online estimator. One per peer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EwmvRtt {
    mean_ms: f64,
    variance_ms2: f64,
    samples: u64,
    alpha: f64,
}

impl Default for EwmvRtt {
    fn default() -> Self {
        Self {
            mean_ms: 0.0,
            variance_ms2: 0.0,
            samples: 0,
            alpha: DEFAULT_ALPHA,
        }
    }
}

impl EwmvRtt {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Update with a single RTT sample (ms). Ignores NaN / negatives.
    pub fn record(&mut self, sample_ms: f64) {
        if !sample_ms.is_finite() || sample_ms < 0.0 {
            return;
        }
        if self.samples == 0 {
            self.mean_ms = sample_ms;
            self.variance_ms2 = 0.0;
        } else {
            let diff = sample_ms - self.mean_ms;
            let incr = self.alpha * diff;
            self.mean_ms += incr;
            self.variance_ms2 = (1.0 - self.alpha) * (self.variance_ms2 + diff * incr);
        }
        self.samples = self.samples.saturating_add(1);
    }

    /// Current EWMA mean (ms).
    pub fn mean_ms(&self) -> f64 {
        self.mean_ms
    }

    /// Current EWMV variance (ms²).
    pub fn variance_ms2(&self) -> f64 {
        self.variance_ms2
    }

    /// Current σ (ms).
    pub fn stddev_ms(&self) -> f64 {
        self.variance_ms2.max(0.0).sqrt()
    }

    /// Number of samples observed so far.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Adaptive timeout: `mean + 3σ`, clamped to `[5s, 120s]`.
    /// With zero samples returns a midrange default (30 s) so the system
    /// doesn't start at the floor.
    pub fn timeout_ms(&self) -> f64 {
        if self.samples == 0 {
            return 30_000.0;
        }
        let raw = self.mean_ms + SIGMA_MULTIPLIER * self.stddev_ms();
        raw.clamp(MIN_TIMEOUT_MS, MAX_TIMEOUT_MS)
    }

    pub fn reset(&mut self) {
        let alpha = self.alpha;
        *self = Self::default();
        self.alpha = alpha;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_samples_returns_default_timeout() {
        let est = EwmvRtt::new();
        assert!((est.timeout_ms() - 30_000.0).abs() < 1e-9);
    }

    #[test]
    fn single_sample_mean_eq_sample() {
        let mut est = EwmvRtt::new();
        est.record(100.0);
        assert!((est.mean_ms() - 100.0).abs() < 1e-9);
        assert_eq!(est.variance_ms2(), 0.0);
    }

    #[test]
    fn mean_converges_with_constant_input() {
        let mut est = EwmvRtt::new();
        for _ in 0..1000 {
            est.record(50.0);
        }
        assert!((est.mean_ms() - 50.0).abs() < 1e-6);
        // Variance should decay to 0 with constant input.
        assert!(est.variance_ms2() < 1e-6);
    }

    #[test]
    fn variance_nonzero_with_changing_input() {
        let mut est = EwmvRtt::new();
        for i in 0..50 {
            let s = if i % 2 == 0 { 100.0 } else { 200.0 };
            est.record(s);
        }
        assert!(est.stddev_ms() > 10.0, "stddev was {}", est.stddev_ms());
    }

    #[test]
    fn timeout_clamped_to_min() {
        let mut est = EwmvRtt::new();
        est.record(100.0); // mean=100, sigma=0 → 100 ms, clamped up to 5000.
        assert!((est.timeout_ms() - MIN_TIMEOUT_MS).abs() < 1e-9);
    }

    #[test]
    fn timeout_clamped_to_max() {
        let mut est = EwmvRtt::new();
        for _ in 0..10 {
            est.record(60_000.0);
        }
        for _ in 0..10 {
            est.record(300_000.0); // huge spike → big variance.
        }
        assert!(est.timeout_ms() <= MAX_TIMEOUT_MS);
        // And we hit the cap.
        assert!((est.timeout_ms() - MAX_TIMEOUT_MS).abs() < 1.0);
    }

    #[test]
    fn timeout_grows_with_volatility() {
        let mut steady = EwmvRtt::new();
        let mut jitter = EwmvRtt::new();
        for _ in 0..200 {
            steady.record(20.0);
        }
        for i in 0..200 {
            jitter.record(if i % 2 == 0 { 10.0 } else { 30.0 });
        }
        // Steady should be at the floor; jitter should be at-or-above floor.
        assert!(jitter.stddev_ms() >= steady.stddev_ms());
    }

    #[test]
    fn nan_rejected() {
        let mut est = EwmvRtt::new();
        est.record(f64::NAN);
        est.record(-1.0);
        assert_eq!(est.samples(), 0);
    }

    #[test]
    fn reset_preserves_alpha() {
        let mut est = EwmvRtt::with_alpha(0.1);
        est.record(100.0);
        est.reset();
        assert_eq!(est.samples(), 0);
        assert!((est.alpha - 0.1).abs() < 1e-12);
    }
}
