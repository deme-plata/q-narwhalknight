/// 🚀 Project APOLLO Phase 5: KALMAN - PID Rate Controller (THRUST CONTROL)
///
/// Self-tuning sync rate control using PID (Proportional-Integral-Derivative):
/// - P (Proportional): React to current throughput error
/// - I (Integral): Correct accumulated error over time
/// - D (Derivative): Dampen oscillations, anticipate changes
///
/// Aerospace analogy:
/// - THRUST CONTROL: Like rocket engine throttling to maintain trajectory
/// - Too fast: Risk of overwhelming peers/network
/// - Too slow: Wasted bandwidth capacity
/// - PID: Optimal balance with smooth convergence
///
/// Automatic tuning based on:
/// - Current throughput vs target
/// - Network latency changes
/// - Peer response times
/// - Buffer utilization
///
/// Expected improvement: Self-tuning optimal parameters, no manual config needed

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// PID controller for sync rate management
#[derive(Clone, Debug)]
pub struct PIDRateController {
    /// Proportional gain: How strongly to react to current error
    /// Higher = faster response, but more oscillation
    pub kp: f64,

    /// Integral gain: How strongly to react to accumulated error
    /// Higher = eliminates steady-state error, but can cause overshoot
    pub ki: f64,

    /// Derivative gain: How strongly to react to rate of change
    /// Higher = more damping, but can cause sluggish response
    pub kd: f64,

    /// Target throughput (blocks per second)
    pub target_throughput: f64,

    /// Current output rate (blocks per second)
    pub current_rate: f64,

    /// Accumulated integral error
    integral: f64,

    /// Previous error (for derivative calculation)
    last_error: f64,

    /// Last update time
    last_update: Instant,

    /// Minimum output rate (blocks/sec)
    pub min_rate: f64,

    /// Maximum output rate (blocks/sec)
    pub max_rate: f64,

    /// Anti-windup: Maximum integral value
    integral_max: f64,

    /// History for analysis (last N updates)
    history: VecDeque<PIDSample>,

    /// Enable auto-tuning
    auto_tune: bool,

    /// Samples for auto-tuning
    tune_samples: Vec<TuningSample>,
}

/// Sample for PID history
#[derive(Clone, Debug)]
struct PIDSample {
    timestamp: Instant,
    error: f64,
    output: f64,
    throughput: f64,
}

/// Sample for auto-tuning (Ziegler-Nichols method)
#[derive(Clone, Debug)]
struct TuningSample {
    timestamp: Instant,
    output: f64,
    response: f64,
}

impl PIDRateController {
    /// Create new PID controller with default parameters
    pub fn new(target_throughput: f64) -> Self {
        Self {
            // Default PID gains (conservative)
            kp: 0.5,
            ki: 0.1,
            kd: 0.05,
            target_throughput,
            current_rate: target_throughput * 0.5, // Start at 50% target
            integral: 0.0,
            last_error: 0.0,
            last_update: Instant::now(),
            min_rate: 10.0,        // Minimum 10 blocks/sec
            max_rate: 10_000.0,    // Maximum 10k blocks/sec
            integral_max: 1000.0,  // Anti-windup limit
            history: VecDeque::with_capacity(100),
            auto_tune: true,
            tune_samples: Vec::with_capacity(100),
        }
    }

    /// Create with custom PID gains
    pub fn with_gains(target: f64, kp: f64, ki: f64, kd: f64) -> Self {
        let mut controller = Self::new(target);
        controller.kp = kp;
        controller.ki = ki;
        controller.kd = kd;
        controller
    }

    /// Create aggressive controller (faster convergence, more oscillation)
    pub fn aggressive(target: f64) -> Self {
        Self::with_gains(target, 1.0, 0.3, 0.1)
    }

    /// Create conservative controller (slower convergence, smoother)
    pub fn conservative(target: f64) -> Self {
        Self::with_gains(target, 0.3, 0.05, 0.02)
    }

    /// Update controller with current throughput measurement
    /// Returns the new target rate to use
    pub fn update(&mut self, current_throughput: f64) -> f64 {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();

        // Skip if update too soon (< 10ms)
        if dt < 0.01 {
            return self.current_rate;
        }

        // Calculate error (positive = we're behind target)
        let error = self.target_throughput - current_throughput;

        // Proportional term
        let p_term = self.kp * error;

        // Integral term (with anti-windup)
        self.integral += error * dt;
        self.integral = self.integral.clamp(-self.integral_max, self.integral_max);
        let i_term = self.ki * self.integral;

        // Derivative term (rate of change of error)
        let derivative = if dt > 0.0 {
            (error - self.last_error) / dt
        } else {
            0.0
        };
        let d_term = self.kd * derivative;

        // Calculate new rate
        let pid_output = p_term + i_term + d_term;
        let new_rate = (self.current_rate + pid_output).clamp(self.min_rate, self.max_rate);

        // Store for history
        if self.history.len() >= 100 {
            self.history.pop_front();
        }
        self.history.push_back(PIDSample {
            timestamp: now,
            error,
            output: new_rate,
            throughput: current_throughput,
        });

        // Store for auto-tuning
        if self.auto_tune {
            self.tune_samples.push(TuningSample {
                timestamp: now,
                output: new_rate,
                response: current_throughput,
            });

            // Attempt auto-tune every 50 samples
            if self.tune_samples.len() >= 50 {
                self.attempt_auto_tune();
            }
        }

        // Update state
        self.last_error = error;
        self.last_update = now;
        self.current_rate = new_rate;

        debug!(
            "🎛️ [PID] target={:.1}, actual={:.1}, error={:.1}, P={:.2}, I={:.2}, D={:.2}, rate={:.1}",
            self.target_throughput, current_throughput, error, p_term, i_term, d_term, new_rate
        );

        new_rate
    }

    /// Set new target throughput
    pub fn set_target(&mut self, target: f64) {
        if (target - self.target_throughput).abs() > 10.0 {
            info!(
                "🎯 [PID] Target changed: {:.1} → {:.1} blocks/sec",
                self.target_throughput, target
            );
        }
        self.target_throughput = target;
    }

    /// Get current output rate
    pub fn get_rate(&self) -> f64 {
        self.current_rate
    }

    /// Reset controller state
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.last_error = 0.0;
        self.last_update = Instant::now();
        self.current_rate = self.target_throughput * 0.5;
        self.history.clear();
        self.tune_samples.clear();
    }

    /// Attempt auto-tuning using simplified Ziegler-Nichols method
    fn attempt_auto_tune(&mut self) {
        if self.tune_samples.len() < 50 {
            return;
        }

        // Calculate oscillation characteristics
        let (period, amplitude) = self.detect_oscillation();

        if let (Some(tu), Some(ku)) = (period, amplitude) {
            // Ziegler-Nichols PID tuning formulas
            let new_kp = 0.6 * ku;
            let new_ki = if tu > 0.0 { 2.0 * new_kp / tu } else { self.ki };
            let new_kd = new_kp * tu / 8.0;

            // Only apply if significantly different (>10% change)
            // Guard against division by zero with safe fallback
            let kp_diff = if self.kp > f64::EPSILON { (new_kp - self.kp).abs() / self.kp } else { 1.0 };
            let ki_diff = if self.ki > f64::EPSILON { (new_ki - self.ki).abs() / self.ki } else { 1.0 };
            let kd_diff = if self.kd > f64::EPSILON { (new_kd - self.kd).abs() / self.kd } else { 1.0 };

            if kp_diff > 0.1 || ki_diff > 0.1 || kd_diff > 0.1 {
                info!(
                    "🔧 [PID AUTO-TUNE] Adjusting gains: Kp={:.3}→{:.3}, Ki={:.3}→{:.3}, Kd={:.3}→{:.3}",
                    self.kp, new_kp, self.ki, new_ki, self.kd, new_kd
                );

                // Apply with smoothing (blend 20% new, 80% old)
                self.kp = 0.8 * self.kp + 0.2 * new_kp;
                self.ki = 0.8 * self.ki + 0.2 * new_ki;
                self.kd = 0.8 * self.kd + 0.2 * new_kd;
            }
        }

        // Clear samples after tuning attempt
        self.tune_samples.clear();
    }

    /// Detect oscillation period and amplitude
    fn detect_oscillation(&self) -> (Option<f64>, Option<f64>) {
        if self.tune_samples.len() < 20 {
            return (None, None);
        }

        // Find zero crossings to detect period
        let mean: f64 = self.tune_samples.iter().map(|s| s.response).sum::<f64>()
            / self.tune_samples.len() as f64;

        let mut crossings: Vec<Instant> = Vec::new();
        let mut was_above = self.tune_samples[0].response > mean;

        for sample in &self.tune_samples[1..] {
            let is_above = sample.response > mean;
            if is_above != was_above {
                crossings.push(sample.timestamp);
                was_above = is_above;
            }
        }

        // Need at least 2 crossings for period
        let period = if crossings.len() >= 2 {
            let total_time: f64 = crossings
                .windows(2)
                .map(|w| w[1].duration_since(w[0]).as_secs_f64())
                .sum();
            let avg_half_period = total_time / (crossings.len() - 1) as f64;
            Some(avg_half_period * 2.0) // Full period is 2x half-period
        } else {
            None
        };

        // Calculate amplitude
        let max = self
            .tune_samples
            .iter()
            .map(|s| s.response)
            .fold(f64::NEG_INFINITY, f64::max);
        let min = self
            .tune_samples
            .iter()
            .map(|s| s.response)
            .fold(f64::INFINITY, f64::min);
        let amplitude = if max > min && mean > 0.0 {
            Some((max - min) / (2.0 * mean))
        } else {
            None
        };

        (period, amplitude)
    }

    /// Get metrics for monitoring
    pub fn get_metrics(&self) -> PIDMetrics {
        let recent_errors: Vec<f64> = self.history.iter().map(|s| s.error).collect();
        let avg_error = if recent_errors.is_empty() {
            0.0
        } else {
            recent_errors.iter().sum::<f64>() / recent_errors.len() as f64
        };

        PIDMetrics {
            kp: self.kp,
            ki: self.ki,
            kd: self.kd,
            target: self.target_throughput,
            current_rate: self.current_rate,
            integral: self.integral,
            last_error: self.last_error,
            avg_error,
            samples_collected: self.history.len(),
        }
    }

    /// Enable/disable auto-tuning
    pub fn set_auto_tune(&mut self, enabled: bool) {
        self.auto_tune = enabled;
        if !enabled {
            self.tune_samples.clear();
        }
    }
}

impl Default for PIDRateController {
    fn default() -> Self {
        Self::new(1000.0) // Default target: 1000 blocks/sec
    }
}

/// PID metrics for monitoring
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PIDMetrics {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    pub target: f64,
    pub current_rate: f64,
    pub integral: f64,
    pub last_error: f64,
    pub avg_error: f64,
    pub samples_collected: usize,
}

/// Cascaded PID for multi-level control
/// (e.g., outer loop controls block rate, inner loop controls request rate)
pub struct CascadedPID {
    /// Outer loop (slower, coarser control)
    pub outer: PIDRateController,

    /// Inner loop (faster, finer control)
    pub inner: PIDRateController,
}

impl CascadedPID {
    pub fn new(outer_target: f64, inner_target: f64) -> Self {
        let mut outer = PIDRateController::conservative(outer_target);
        let inner = PIDRateController::aggressive(inner_target);

        // Outer loop should be slower
        outer.kp *= 0.5;
        outer.ki *= 0.5;
        outer.kd *= 0.5;

        Self { outer, inner }
    }

    /// Update with both measurements
    /// Returns (outer_rate, inner_rate)
    pub fn update(&mut self, outer_measurement: f64, inner_measurement: f64) -> (f64, f64) {
        // Outer loop determines target for inner loop
        let outer_output = self.outer.update(outer_measurement);
        self.inner.set_target(outer_output);

        // Inner loop tracks outer's target
        let inner_output = self.inner.update(inner_measurement);

        (outer_output, inner_output)
    }
}

/// Adaptive rate limiter using PID
pub struct AdaptiveRateLimiter {
    controller: PIDRateController,
    tokens: f64,
    last_refill: Instant,
}

impl AdaptiveRateLimiter {
    pub fn new(target_rate: f64) -> Self {
        Self {
            controller: PIDRateController::new(target_rate),
            tokens: target_rate, // Start with full bucket
            last_refill: Instant::now(),
        }
    }

    /// Try to acquire tokens, returns true if allowed
    pub fn try_acquire(&mut self, count: f64, current_throughput: f64) -> bool {
        // Update PID with current throughput
        let rate = self.controller.update(current_throughput);

        // Refill tokens based on elapsed time
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + rate * elapsed).min(rate * 2.0); // Max 2 seconds buffer
        self.last_refill = now;

        // Try to consume
        if self.tokens >= count {
            self.tokens -= count;
            true
        } else {
            false
        }
    }

    /// Get current rate
    pub fn get_rate(&self) -> f64 {
        self.controller.get_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_creation() {
        let pid = PIDRateController::new(1000.0);
        assert_eq!(pid.target_throughput, 1000.0);
        assert!(pid.current_rate > 0.0);
    }

    #[test]
    fn test_pid_update() {
        let mut pid = PIDRateController::new(1000.0);

        // Simulate being below target
        std::thread::sleep(Duration::from_millis(20));
        let rate1 = pid.update(500.0); // 50% of target

        // Rate should increase
        assert!(rate1 > pid.min_rate);

        // Simulate being above target
        std::thread::sleep(Duration::from_millis(20));
        let rate2 = pid.update(1500.0); // 150% of target

        // Rate should be lower than previous
        assert!(rate2 <= rate1);
    }

    #[test]
    fn test_pid_bounds() {
        let mut pid = PIDRateController::new(1000.0);
        pid.min_rate = 100.0;
        pid.max_rate = 5000.0;

        // Even with extreme error, should stay in bounds
        std::thread::sleep(Duration::from_millis(20));
        let rate = pid.update(0.0); // 0 throughput, huge error
        assert!(rate >= pid.min_rate);
        assert!(rate <= pid.max_rate);
    }

    #[test]
    fn test_pid_reset() {
        let mut pid = PIDRateController::new(1000.0);
        std::thread::sleep(Duration::from_millis(20));
        pid.update(500.0);

        pid.reset();

        assert_eq!(pid.integral, 0.0);
        assert_eq!(pid.last_error, 0.0);
        assert!(pid.history.is_empty());
    }

    #[test]
    fn test_cascaded_pid() {
        let mut cascaded = CascadedPID::new(1000.0, 100.0);

        std::thread::sleep(Duration::from_millis(20));
        let (outer, inner) = cascaded.update(800.0, 90.0);

        assert!(outer > 0.0);
        assert!(inner > 0.0);
    }

    #[test]
    fn test_adaptive_rate_limiter() {
        let mut limiter = AdaptiveRateLimiter::new(100.0);

        // Should be able to acquire initially
        assert!(limiter.try_acquire(10.0, 50.0));

        // Rate should be positive
        assert!(limiter.get_rate() > 0.0);
    }
}
