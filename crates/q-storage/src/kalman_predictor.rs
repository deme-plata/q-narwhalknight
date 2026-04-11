/// 🚀 Project APOLLO Phase 5: KALMAN - Network Predictor (NAVIGATION SYSTEM)
///
/// Kalman filter for optimal network state estimation:
/// - State: [bandwidth, latency, packet_loss]
/// - Predicts optimal chunk size, timeout, concurrency
/// - Filters noisy measurements for stable control
///
/// Aerospace analogy:
/// - NAVIGATION SYSTEM: Like spacecraft navigation using noisy sensor data
/// - Combines predictions with measurements for optimal state estimate
/// - Anticipates network changes before they impact sync
///
/// Key features:
/// - Optimal bandwidth-delay product calculation
/// - Adaptive timeout based on latency variance
/// - Predictive congestion avoidance
/// - Self-tuning measurement noise estimation
///
/// Expected improvement: Self-tuning optimal parameters, 20-40% better throughput

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Network state vector: [bandwidth_bps, latency_ms, loss_rate]
#[derive(Clone, Debug, Default)]
pub struct NetworkState {
    /// Bandwidth in bytes per second
    pub bandwidth_bps: f64,

    /// Round-trip latency in milliseconds
    pub latency_ms: f64,

    /// Packet loss rate (0.0 - 1.0)
    pub loss_rate: f64,

    /// Jitter (latency variance) in milliseconds
    pub jitter_ms: f64,

    /// Confidence in estimate (0.0 - 1.0)
    pub confidence: f64,
}

impl NetworkState {
    /// Calculate bandwidth-delay product (optimal window size)
    pub fn bandwidth_delay_product(&self) -> u64 {
        // BDP = bandwidth * RTT
        // In bytes: (bytes/sec) * (ms / 1000) = bytes
        let bdp = self.bandwidth_bps * (self.latency_ms / 1000.0);
        (bdp * 2.0) as u64 // 2x BDP for full pipeline utilization
    }

    /// Calculate optimal chunk size based on network conditions
    pub fn optimal_chunk_size(&self) -> usize {
        let bdp = self.bandwidth_delay_product();

        // Clamp to reasonable range
        let min_chunk = 10_000; // 10 KB minimum
        let max_chunk = 10_000_000; // 10 MB maximum

        (bdp as usize).clamp(min_chunk, max_chunk)
    }

    /// Calculate optimal timeout based on latency + jitter
    pub fn optimal_timeout(&self) -> Duration {
        // Timeout = RTT + 4*jitter (99.99% of packets within this)
        let timeout_ms = self.latency_ms + 4.0 * self.jitter_ms;

        // Clamp to reasonable range
        // v10.2.10: Reduced max from 60s to 15s to match adaptive timeout ceiling.
        // Prevents Kalman from independently converging to high values (e.g., 21.4s)
        // that caused the 2026-04-11 Epsilon sync stall.
        let min_timeout = Duration::from_millis(100);
        let max_timeout = Duration::from_secs(15);

        Duration::from_millis(timeout_ms as u64).clamp(min_timeout, max_timeout)
    }

    /// Calculate optimal concurrency (parallel requests)
    pub fn optimal_concurrency(&self) -> usize {
        // More concurrency hides latency but risks congestion
        // Optimal = BDP / typical_request_size * loss_penalty
        let bdp = self.bandwidth_delay_product() as f64;
        let typical_request = 50_000.0; // 50 KB typical response
        let loss_penalty = 1.0 - self.loss_rate; // Reduce for lossy networks

        let optimal = (bdp / typical_request * loss_penalty) as usize;

        // Clamp to reasonable range
        optimal.clamp(4, 128)
    }
}

/// Kalman filter state (simplified 1D per variable)
#[derive(Clone, Debug)]
struct KalmanState {
    /// Current estimate
    x: f64,

    /// Estimate uncertainty (covariance)
    p: f64,

    /// Process noise (how much state changes between updates)
    q: f64,

    /// Measurement noise (how noisy our measurements are)
    r: f64,
}

impl KalmanState {
    fn new(initial: f64, uncertainty: f64, process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            x: initial,
            p: uncertainty,
            q: process_noise,
            r: measurement_noise,
        }
    }

    /// Predict step (time update)
    fn predict(&mut self) {
        // State prediction: x_k|k-1 = x_k-1 (no motion model)
        // Uncertainty prediction: P_k|k-1 = P_k-1 + Q
        self.p += self.q;
    }

    /// Update step (measurement update)
    fn update(&mut self, measurement: f64) {
        // Kalman gain: K = P / (P + R)
        let k = self.p / (self.p + self.r);

        // State update: x = x + K * (z - x)
        self.x = self.x + k * (measurement - self.x);

        // Uncertainty update: P = (1 - K) * P
        self.p = (1.0 - k) * self.p;
    }

    /// Combined predict + update
    fn filter(&mut self, measurement: f64) -> f64 {
        self.predict();
        self.update(measurement);
        self.x
    }

    /// Get confidence (inverse of uncertainty, normalized)
    fn confidence(&self) -> f64 {
        1.0 / (1.0 + self.p)
    }
}

/// Kalman network predictor (NAVIGATION SYSTEM)
pub struct KalmanNetworkPredictor {
    /// Bandwidth filter
    bandwidth: KalmanState,

    /// Latency filter
    latency: KalmanState,

    /// Loss rate filter
    loss: KalmanState,

    /// Jitter filter
    jitter: KalmanState,

    /// Last update time
    last_update: Instant,

    /// Measurement history for noise estimation
    measurements: VecDeque<NetworkMeasurement>,

    /// Enable adaptive noise estimation
    adaptive_noise: bool,
}

/// Single network measurement
#[derive(Clone, Debug)]
struct NetworkMeasurement {
    timestamp: Instant,
    bandwidth: f64,
    latency: f64,
    loss: f64,
}

impl KalmanNetworkPredictor {
    /// Create new predictor with default parameters
    pub fn new() -> Self {
        Self {
            // Bandwidth: start at 10 MB/s, moderate uncertainty
            bandwidth: KalmanState::new(10_000_000.0, 1_000_000.0, 100_000.0, 500_000.0),

            // Latency: start at 50ms
            latency: KalmanState::new(50.0, 100.0, 5.0, 20.0),

            // Loss: start at 1%
            loss: KalmanState::new(0.01, 0.1, 0.001, 0.01),

            // Jitter: start at 10ms
            jitter: KalmanState::new(10.0, 50.0, 2.0, 10.0),

            last_update: Instant::now(),
            measurements: VecDeque::with_capacity(100),
            adaptive_noise: true,
        }
    }

    /// Create with custom initial state
    pub fn with_initial(bandwidth: f64, latency: f64, loss: f64) -> Self {
        let mut predictor = Self::new();
        predictor.bandwidth.x = bandwidth;
        predictor.latency.x = latency;
        predictor.loss.x = loss;
        predictor
    }

    /// Update predictor with new measurements
    pub fn update(
        &mut self,
        measured_bandwidth: f64,
        measured_latency: f64,
        measured_loss: f64,
    ) -> NetworkState {
        // Validate inputs - reject invalid measurements
        if measured_bandwidth <= 0.0 || !measured_bandwidth.is_finite() {
            warn!("🔭 [KALMAN] Rejecting invalid bandwidth: {}", measured_bandwidth);
            return self.get_state();
        }
        if measured_latency <= 0.0 || !measured_latency.is_finite() {
            warn!("🔭 [KALMAN] Rejecting invalid latency: {}", measured_latency);
            return self.get_state();
        }
        if measured_loss < 0.0 || measured_loss > 1.0 || !measured_loss.is_finite() {
            warn!("🔭 [KALMAN] Rejecting invalid loss rate: {}", measured_loss);
            return self.get_state();
        }

        let now = Instant::now();

        // Store measurement for noise estimation
        if self.measurements.len() >= 100 {
            self.measurements.pop_front();
        }
        self.measurements.push_back(NetworkMeasurement {
            timestamp: now,
            bandwidth: measured_bandwidth,
            latency: measured_latency,
            loss: measured_loss,
        });

        // Adaptive noise estimation
        if self.adaptive_noise && self.measurements.len() >= 10 {
            self.update_noise_estimates();
        }

        // Filter measurements
        let filtered_bw = self.bandwidth.filter(measured_bandwidth);
        let filtered_lat = self.latency.filter(measured_latency);
        let filtered_loss = self.loss.filter(measured_loss);

        // Calculate jitter from latency variance
        let jitter = self.calculate_jitter();
        let filtered_jitter = self.jitter.filter(jitter);

        // Calculate average confidence
        let confidence = (self.bandwidth.confidence()
            + self.latency.confidence()
            + self.loss.confidence()
            + self.jitter.confidence())
            / 4.0;

        self.last_update = now;

        let state = NetworkState {
            bandwidth_bps: filtered_bw.max(1000.0), // Minimum 1 KB/s
            latency_ms: filtered_lat.max(1.0),       // Minimum 1ms
            loss_rate: filtered_loss.clamp(0.0, 1.0),
            jitter_ms: filtered_jitter.max(0.0),
            confidence,
        };

        debug!(
            "🎯 [KALMAN] BW={:.1}MB/s, Lat={:.1}ms, Loss={:.2}%, Jitter={:.1}ms, Conf={:.2}",
            state.bandwidth_bps / 1_000_000.0,
            state.latency_ms,
            state.loss_rate * 100.0,
            state.jitter_ms,
            state.confidence
        );

        state
    }

    /// Update with request/response timing
    pub fn update_from_request(
        &mut self,
        bytes_received: u64,
        duration: Duration,
        success: bool,
    ) -> NetworkState {
        let duration_secs = duration.as_secs_f64().max(0.001);

        // Calculate bandwidth from this sample
        let bandwidth = bytes_received as f64 / duration_secs;

        // Latency approximation (half of round-trip for single request)
        let latency = duration.as_millis() as f64 / 2.0;

        // Loss from success/failure
        let loss = if success { 0.0 } else { 1.0 };

        self.update(bandwidth, latency, loss)
    }

    /// Get current state estimate (without new measurement)
    pub fn get_state(&self) -> NetworkState {
        // Just return predicted state
        NetworkState {
            bandwidth_bps: self.bandwidth.x.max(1000.0),
            latency_ms: self.latency.x.max(1.0),
            loss_rate: self.loss.x.clamp(0.0, 1.0),
            jitter_ms: self.jitter.x.max(0.0),
            confidence: (self.bandwidth.confidence()
                + self.latency.confidence()
                + self.loss.confidence()
                + self.jitter.confidence())
                / 4.0,
        }
    }

    /// Predict state at future time (for planning)
    pub fn predict_future(&self, duration: Duration) -> NetworkState {
        // Simple prediction: assume state remains constant with increased uncertainty
        let mut state = self.get_state();

        // Reduce confidence based on prediction horizon
        let decay = (-duration.as_secs_f64() / 60.0).exp(); // 1-minute half-life
        state.confidence *= decay;

        state
    }

    /// Calculate jitter from recent latency measurements
    fn calculate_jitter(&self) -> f64 {
        if self.measurements.len() < 2 {
            return 10.0; // Default jitter
        }

        let latencies: Vec<f64> = self.measurements.iter().map(|m| m.latency).collect();

        let mean: f64 = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let variance: f64 = latencies
            .iter()
            .map(|&l| (l - mean).powi(2))
            .sum::<f64>()
            / latencies.len() as f64;

        variance.sqrt()
    }

    /// Update noise estimates from measurement history
    fn update_noise_estimates(&mut self) {
        if self.measurements.len() < 10 {
            return;
        }

        // Calculate variance of recent measurements for R (measurement noise)
        let bw_values: Vec<f64> = self.measurements.iter().map(|m| m.bandwidth).collect();
        let lat_values: Vec<f64> = self.measurements.iter().map(|m| m.latency).collect();
        let loss_values: Vec<f64> = self.measurements.iter().map(|m| m.loss).collect();

        let bw_variance = variance(&bw_values);
        let lat_variance = variance(&lat_values);
        let loss_variance = variance(&loss_values);

        // Update measurement noise (R) with smoothing
        self.bandwidth.r = 0.9 * self.bandwidth.r + 0.1 * bw_variance.sqrt();
        self.latency.r = 0.9 * self.latency.r + 0.1 * lat_variance.sqrt();
        self.loss.r = 0.9 * self.loss.r + 0.1 * loss_variance.sqrt();
    }

    /// Get predictor metrics
    pub fn get_metrics(&self) -> KalmanMetrics {
        let state = self.get_state();
        KalmanMetrics {
            bandwidth_mbps: state.bandwidth_bps / 1_000_000.0,
            latency_ms: state.latency_ms,
            loss_percent: state.loss_rate * 100.0,
            jitter_ms: state.jitter_ms,
            confidence: state.confidence,
            optimal_chunk_kb: state.optimal_chunk_size() / 1024,
            optimal_timeout_ms: state.optimal_timeout().as_millis() as u64,
            optimal_concurrency: state.optimal_concurrency(),
            bdp_bytes: state.bandwidth_delay_product(),
            samples_collected: self.measurements.len(),
        }
    }

    /// Reset predictor to initial state
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for KalmanNetworkPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper: calculate variance
fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Kalman predictor metrics for monitoring
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KalmanMetrics {
    pub bandwidth_mbps: f64,
    pub latency_ms: f64,
    pub loss_percent: f64,
    pub jitter_ms: f64,
    pub confidence: f64,
    pub optimal_chunk_kb: usize,
    pub optimal_timeout_ms: u64,
    pub optimal_concurrency: usize,
    pub bdp_bytes: u64,
    pub samples_collected: usize,
}

/// Combined PID + Kalman controller for optimal sync
pub struct AdaptiveSyncController {
    /// Kalman predictor for network state
    pub kalman: KalmanNetworkPredictor,

    /// PID controller for rate control
    pub pid: super::pid_controller::PIDRateController,
}

impl AdaptiveSyncController {
    pub fn new(target_throughput: f64) -> Self {
        Self {
            kalman: KalmanNetworkPredictor::new(),
            pid: super::pid_controller::PIDRateController::new(target_throughput),
        }
    }

    /// Update with measurement, returns recommended settings
    pub fn update(
        &mut self,
        bytes_received: u64,
        duration: Duration,
        success: bool,
        current_throughput: f64,
    ) -> SyncSettings {
        // Update Kalman filter
        let network = self.kalman.update_from_request(bytes_received, duration, success);

        // Update PID controller
        let rate = self.pid.update(current_throughput);

        // Combine for optimal settings
        SyncSettings {
            target_rate: rate,
            chunk_size: network.optimal_chunk_size(),
            timeout: network.optimal_timeout(),
            concurrency: network.optimal_concurrency(),
            confidence: network.confidence,
        }
    }

    /// Get current recommended settings
    pub fn get_settings(&self) -> SyncSettings {
        let network = self.kalman.get_state();
        SyncSettings {
            target_rate: self.pid.get_rate(),
            chunk_size: network.optimal_chunk_size(),
            timeout: network.optimal_timeout(),
            concurrency: network.optimal_concurrency(),
            confidence: network.confidence,
        }
    }
}

/// Recommended sync settings from adaptive controller
#[derive(Clone, Debug)]
pub struct SyncSettings {
    /// Target blocks per second
    pub target_rate: f64,

    /// Optimal chunk size (bytes)
    pub chunk_size: usize,

    /// Optimal request timeout
    pub timeout: Duration,

    /// Optimal concurrent requests
    pub concurrency: usize,

    /// Confidence in these settings
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_state_bdp() {
        let state = NetworkState {
            bandwidth_bps: 100_000_000.0, // 100 MB/s
            latency_ms: 50.0,             // 50ms RTT
            loss_rate: 0.0,
            jitter_ms: 5.0,
            confidence: 1.0,
        };

        // BDP = 100MB/s * 50ms = 5MB
        let bdp = state.bandwidth_delay_product();
        assert!(bdp > 5_000_000 && bdp < 15_000_000);
    }

    #[test]
    fn test_network_state_optimal_timeout() {
        let state = NetworkState {
            bandwidth_bps: 10_000_000.0,
            latency_ms: 100.0,
            loss_rate: 0.0,
            jitter_ms: 20.0,
            confidence: 1.0,
        };

        // Timeout = 100 + 4*20 = 180ms
        let timeout = state.optimal_timeout();
        assert!(timeout.as_millis() >= 180);
    }

    #[test]
    fn test_kalman_predictor_creation() {
        let predictor = KalmanNetworkPredictor::new();
        let state = predictor.get_state();

        assert!(state.bandwidth_bps > 0.0);
        assert!(state.latency_ms > 0.0);
        assert!(state.confidence > 0.0);
    }

    #[test]
    fn test_kalman_update() {
        let mut predictor = KalmanNetworkPredictor::new();

        // Update with measurements
        let state = predictor.update(50_000_000.0, 30.0, 0.0);

        // Should move towards measurements
        assert!(state.bandwidth_bps > 1_000_000.0);
        assert!(state.latency_ms < 100.0);
    }

    #[test]
    fn test_kalman_from_request() {
        let mut predictor = KalmanNetworkPredictor::new();

        // Simulate a request: 1 MB in 100ms
        let state = predictor.update_from_request(1_000_000, Duration::from_millis(100), true);

        // Bandwidth should be ~10 MB/s
        assert!(state.bandwidth_bps > 5_000_000.0);
    }

    #[test]
    fn test_adaptive_controller() {
        let mut controller = AdaptiveSyncController::new(1000.0);

        let settings = controller.update(100_000, Duration::from_millis(50), true, 500.0);

        assert!(settings.target_rate > 0.0);
        assert!(settings.chunk_size > 0);
        assert!(settings.concurrency > 0);
    }
}
