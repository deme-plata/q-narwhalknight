/// Timing Obfuscation for Q-NarwhalKnight
///
/// This module implements advanced timing obfuscation techniques to protect
/// against traffic correlation attacks. It provides:
///
/// - Adaptive timing jitter based on traffic patterns
/// - Constant-rate transmission mode
/// - Burst scheduling to hide activity patterns
/// - Request batching and delayed release
/// - Quantum-enhanced randomness for unpredictability
/// - Circuit-aware timing to avoid fingerprinting
///
/// These techniques make it significantly harder for adversaries to correlate
/// traffic patterns across entry and exit nodes of the Tor network.

use anyhow::{anyhow, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, Mutex, RwLock, Semaphore};
use tokio::time::sleep;
use tracing::{debug, info, trace, warn};

/// Timing obfuscation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObfuscationMode {
    /// No obfuscation (for testing or low-security scenarios)
    Disabled,
    /// Light jitter only (low overhead)
    Light,
    /// Medium obfuscation with adaptive jitter
    Medium,
    /// Heavy obfuscation with constant-rate transmission
    Heavy,
    /// Maximum obfuscation with all techniques enabled
    Paranoid,
}

impl ObfuscationMode {
    pub fn name(&self) -> &'static str {
        match self {
            ObfuscationMode::Disabled => "Disabled",
            ObfuscationMode::Light => "Light",
            ObfuscationMode::Medium => "Medium",
            ObfuscationMode::Heavy => "Heavy",
            ObfuscationMode::Paranoid => "Paranoid",
        }
    }

    /// Get base jitter range in milliseconds
    pub fn base_jitter_ms(&self) -> (u64, u64) {
        match self {
            ObfuscationMode::Disabled => (0, 0),
            ObfuscationMode::Light => (5, 25),
            ObfuscationMode::Medium => (10, 100),
            ObfuscationMode::Heavy => (50, 300),
            ObfuscationMode::Paranoid => (100, 500),
        }
    }

    /// Get overhead multiplier (1.0 = no overhead)
    pub fn overhead_multiplier(&self) -> f64 {
        match self {
            ObfuscationMode::Disabled => 1.0,
            ObfuscationMode::Light => 1.05,
            ObfuscationMode::Medium => 1.15,
            ObfuscationMode::Heavy => 1.35,
            ObfuscationMode::Paranoid => 1.60,
        }
    }
}

/// Configuration for timing obfuscation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingObfuscationConfig {
    /// Obfuscation mode
    pub mode: ObfuscationMode,
    /// Enable constant-rate transmission
    pub constant_rate: bool,
    /// Target rate for constant-rate mode (bytes per second)
    pub target_rate_bps: u64,
    /// Enable request batching
    pub batch_requests: bool,
    /// Batch interval
    pub batch_interval: Duration,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Enable adaptive jitter based on traffic patterns
    pub adaptive_jitter: bool,
    /// Minimum jitter (milliseconds)
    pub min_jitter_ms: u64,
    /// Maximum jitter (milliseconds)
    pub max_jitter_ms: u64,
    /// Use quantum-enhanced randomness if available
    pub quantum_entropy: bool,
    /// Enable burst scheduling
    pub burst_scheduling: bool,
    /// Burst window duration
    pub burst_window: Duration,
    /// Maximum delay for any single request
    pub max_delay: Duration,
}

impl Default for TimingObfuscationConfig {
    fn default() -> Self {
        Self {
            mode: ObfuscationMode::Medium,
            constant_rate: false,
            target_rate_bps: 50_000, // 50 KB/s
            batch_requests: true,
            batch_interval: Duration::from_millis(100),
            max_batch_size: 10,
            adaptive_jitter: true,
            min_jitter_ms: 10,
            max_jitter_ms: 100,
            quantum_entropy: true,
            burst_scheduling: false,
            burst_window: Duration::from_secs(5),
            max_delay: Duration::from_secs(2),
        }
    }
}

/// Request to be scheduled with timing obfuscation
#[derive(Debug)]
pub struct ScheduledRequest {
    /// Unique request ID
    pub id: u64,
    /// Request payload
    pub payload: Vec<u8>,
    /// Creation time
    pub created_at: Instant,
    /// Scheduled release time
    pub release_at: Option<Instant>,
    /// Priority (higher = more urgent)
    pub priority: u8,
    /// Circuit ID for isolation
    pub circuit_id: Option<u64>,
}

impl ScheduledRequest {
    pub fn new(id: u64, payload: Vec<u8>) -> Self {
        Self {
            id,
            payload,
            created_at: Instant::now(),
            release_at: None,
            priority: 5,
            circuit_id: None,
        }
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_circuit(mut self, circuit_id: u64) -> Self {
        self.circuit_id = Some(circuit_id);
        self
    }
}

/// Response with timing metadata
#[derive(Debug)]
pub struct ScheduledResponse {
    /// Request ID
    pub request_id: u64,
    /// Actual delay applied
    pub delay_applied: Duration,
    /// Was this batched with other requests
    pub was_batched: bool,
    /// Jitter component of delay
    pub jitter_ms: u64,
}

/// Statistics for timing obfuscation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingStats {
    /// Total requests processed
    pub requests_processed: u64,
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Total delay applied (microseconds)
    pub total_delay_us: u64,
    /// Average delay per request (microseconds)
    pub avg_delay_us: u64,
    /// Maximum delay applied
    pub max_delay_us: u64,
    /// Requests batched together
    pub batched_requests: u64,
    /// Dummy packets sent (for constant-rate mode)
    pub dummy_packets: u64,
    /// Quantum entropy used (if available)
    pub quantum_entropy_used: bool,
}

/// Burst scheduler state
struct BurstScheduler {
    /// Window start time
    window_start: Instant,
    /// Requests in current window
    requests_in_window: u64,
    /// Target requests per window
    target_per_window: u64,
    /// Is currently in burst mode
    in_burst: bool,
}

impl BurstScheduler {
    fn new(target_per_window: u64) -> Self {
        Self {
            window_start: Instant::now(),
            requests_in_window: 0,
            target_per_window,
            in_burst: false,
        }
    }

    fn should_release(&mut self, window_duration: Duration) -> bool {
        let now = Instant::now();

        // Check if window has elapsed
        if now.duration_since(self.window_start) >= window_duration {
            self.window_start = now;
            self.requests_in_window = 0;
            self.in_burst = rand::thread_rng().gen_bool(0.3); // 30% chance of burst
        }

        self.requests_in_window += 1;

        // In burst mode, release immediately
        if self.in_burst {
            return true;
        }

        // Otherwise, distribute evenly across window
        let expected_so_far = (now.duration_since(self.window_start).as_millis() as u64
            * self.target_per_window)
            / window_duration.as_millis() as u64;

        self.requests_in_window <= expected_so_far.max(1)
    }
}

/// Traffic pattern analyzer for adaptive jitter
struct TrafficPatternAnalyzer {
    /// Recent inter-packet intervals
    intervals: VecDeque<Duration>,
    /// Recent packet sizes
    sizes: VecDeque<usize>,
    /// Last packet time
    last_packet: Option<Instant>,
    /// Maximum samples to keep
    max_samples: usize,
}

impl TrafficPatternAnalyzer {
    fn new(max_samples: usize) -> Self {
        Self {
            intervals: VecDeque::with_capacity(max_samples),
            sizes: VecDeque::with_capacity(max_samples),
            last_packet: None,
            max_samples,
        }
    }

    fn record_packet(&mut self, size: usize) {
        let now = Instant::now();

        if let Some(last) = self.last_packet {
            let interval = now.duration_since(last);
            self.intervals.push_back(interval);
            if self.intervals.len() > self.max_samples {
                self.intervals.pop_front();
            }
        }

        self.sizes.push_back(size);
        if self.sizes.len() > self.max_samples {
            self.sizes.pop_front();
        }

        self.last_packet = Some(now);
    }

    /// Calculate recommended jitter based on traffic pattern
    fn recommended_jitter(&self, base_min: u64, base_max: u64) -> Duration {
        if self.intervals.len() < 10 {
            // Not enough data, use base values
            let jitter = rand::thread_rng().gen_range(base_min..=base_max);
            return Duration::from_millis(jitter);
        }

        // Calculate mean and variance of intervals
        let intervals_ms: Vec<f64> = self.intervals.iter()
            .map(|d| d.as_millis() as f64)
            .collect();

        let mean: f64 = intervals_ms.iter().sum::<f64>() / intervals_ms.len() as f64;

        let variance: f64 = intervals_ms.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / intervals_ms.len() as f64;

        let std_dev = variance.sqrt();

        // If traffic is already very variable, add less jitter
        // If traffic is regular, add more jitter
        let regularity = 1.0 - (std_dev / mean.max(1.0)).min(1.0);

        let jitter_range = base_max - base_min;
        let adaptive_jitter = base_min + (jitter_range as f64 * regularity) as u64;

        // Add some randomness
        let mut rng = rand::thread_rng();
        let final_jitter = if adaptive_jitter > base_min {
            rng.gen_range(base_min..=adaptive_jitter)
        } else {
            base_min
        };

        Duration::from_millis(final_jitter)
    }

    /// Get pattern summary
    fn pattern_summary(&self) -> PatternSummary {
        let avg_interval = if self.intervals.is_empty() {
            Duration::ZERO
        } else {
            let total: Duration = self.intervals.iter().sum();
            total / self.intervals.len() as u32
        };

        let avg_size = if self.sizes.is_empty() {
            0
        } else {
            self.sizes.iter().sum::<usize>() / self.sizes.len()
        };

        PatternSummary {
            samples: self.intervals.len(),
            avg_interval,
            avg_size,
        }
    }
}

/// Traffic pattern summary
#[derive(Debug, Clone)]
pub struct PatternSummary {
    pub samples: usize,
    pub avg_interval: Duration,
    pub avg_size: usize,
}

/// Main timing obfuscation engine
pub struct TimingObfuscator {
    config: TimingObfuscationConfig,
    stats: Arc<RwLock<TimingStats>>,
    pattern_analyzer: Arc<RwLock<TrafficPatternAnalyzer>>,
    burst_scheduler: Arc<Mutex<BurstScheduler>>,
    request_queue: Arc<Mutex<VecDeque<ScheduledRequest>>>,
    request_counter: Arc<std::sync::atomic::AtomicU64>,
    rng: Arc<Mutex<ChaCha20Rng>>,
    quantum_entropy_pool: Arc<RwLock<Vec<u8>>>,
    release_semaphore: Arc<Semaphore>,
}

impl TimingObfuscator {
    /// Create a new timing obfuscator
    pub fn new(config: TimingObfuscationConfig) -> Self {
        info!("⏱️ Creating Timing Obfuscator");
        info!("   Mode: {}", config.mode.name());
        info!("   Jitter: {}ms - {}ms", config.min_jitter_ms, config.max_jitter_ms);
        info!("   Batch requests: {}", config.batch_requests);
        info!("   Adaptive jitter: {}", config.adaptive_jitter);

        let rng = ChaCha20Rng::from_rng(&mut rand::thread_rng());

        Self {
            config,
            stats: Arc::new(RwLock::new(TimingStats::default())),
            pattern_analyzer: Arc::new(RwLock::new(TrafficPatternAnalyzer::new(1000))),
            burst_scheduler: Arc::new(Mutex::new(BurstScheduler::new(100))),
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
            request_counter: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            rng: Arc::new(Mutex::new(rng)),
            quantum_entropy_pool: Arc::new(RwLock::new(Vec::new())),
            release_semaphore: Arc::new(Semaphore::new(100)),
        }
    }

    /// Add quantum entropy to the pool
    pub async fn add_quantum_entropy(&self, entropy: &[u8]) {
        if !self.config.quantum_entropy {
            return;
        }

        let mut pool = self.quantum_entropy_pool.write().await;
        pool.extend_from_slice(entropy);

        // Limit pool size
        if pool.len() > 4096 {
            let drain_count = pool.len() - 4096;
            pool.drain(0..drain_count);
        }

        debug!("Added {} bytes of quantum entropy (pool size: {})", entropy.len(), pool.len());
    }

    /// Get next random value, preferring quantum entropy
    async fn next_random(&self) -> u64 {
        // Try quantum entropy first
        {
            let mut pool = self.quantum_entropy_pool.write().await;
            if pool.len() >= 8 {
                let bytes: Vec<u8> = pool.drain(0..8).collect();
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes);
                return u64::from_le_bytes(arr);
            }
        }

        // Fall back to ChaCha20
        let mut rng = self.rng.lock().await;
        rng.gen()
    }

    /// Calculate jitter for a request
    async fn calculate_jitter(&self, payload_size: usize) -> Duration {
        if self.config.mode == ObfuscationMode::Disabled {
            return Duration::ZERO;
        }

        // Get base jitter range from mode
        let (base_min, base_max) = self.config.mode.base_jitter_ms();

        // Adjust based on config
        let min = self.config.min_jitter_ms.max(base_min);
        let max = self.config.max_jitter_ms.min(base_max * 2);

        // Use adaptive jitter if enabled
        if self.config.adaptive_jitter {
            let analyzer = self.pattern_analyzer.read().await;
            return analyzer.recommended_jitter(min, max);
        }

        // Simple random jitter
        let random = self.next_random().await;
        let range = max - min;
        let jitter = min + (random % (range + 1));

        Duration::from_millis(jitter)
    }

    /// Schedule a request with timing obfuscation
    pub async fn schedule(&self, payload: Vec<u8>) -> Result<u64> {
        let id = self.request_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let request = ScheduledRequest::new(id, payload);

        // Add to queue if batching
        if self.config.batch_requests {
            let mut queue = self.request_queue.lock().await;
            queue.push_back(request);
            debug!("Request {} queued (queue size: {})", id, queue.len());
        }

        Ok(id)
    }

    /// Apply timing obfuscation to a request and return after delay
    pub async fn obfuscate(&self, payload: &[u8]) -> Result<ScheduledResponse> {
        let id = self.request_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let start = Instant::now();

        // Acquire semaphore to limit concurrent operations
        let _permit = self.release_semaphore.acquire().await
            .map_err(|e| anyhow!("Semaphore closed: {}", e))?;

        // Record for pattern analysis
        {
            let mut analyzer = self.pattern_analyzer.write().await;
            analyzer.record_packet(payload.len());
        }

        // Calculate and apply jitter
        let jitter = self.calculate_jitter(payload.len()).await;

        // Check burst scheduler
        let should_delay = if self.config.burst_scheduling {
            let mut scheduler = self.burst_scheduler.lock().await;
            !scheduler.should_release(self.config.burst_window)
        } else {
            true
        };

        // Apply delay
        let actual_delay = if should_delay && jitter > Duration::ZERO {
            // Cap at max delay
            let capped_delay = jitter.min(self.config.max_delay);
            sleep(capped_delay).await;
            capped_delay
        } else {
            Duration::ZERO
        };

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.requests_processed += 1;
            stats.bytes_processed += payload.len() as u64;
            stats.total_delay_us += actual_delay.as_micros() as u64;
            stats.avg_delay_us = stats.total_delay_us / stats.requests_processed;
            if actual_delay.as_micros() as u64 > stats.max_delay_us {
                stats.max_delay_us = actual_delay.as_micros() as u64;
            }
        }

        trace!(
            "Request {} obfuscated: {}ms delay, {} bytes",
            id,
            actual_delay.as_millis(),
            payload.len()
        );

        Ok(ScheduledResponse {
            request_id: id,
            delay_applied: actual_delay,
            was_batched: false,
            jitter_ms: actual_delay.as_millis() as u64,
        })
    }

    /// Process batched requests
    pub async fn process_batch(&self) -> Vec<ScheduledRequest> {
        let mut queue = self.request_queue.lock().await;

        if queue.is_empty() {
            return Vec::new();
        }

        // Get batch
        let batch_size = queue.len().min(self.config.max_batch_size);
        let batch: Vec<_> = queue.drain(0..batch_size).collect();

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.batched_requests += batch.len() as u64;
        }

        debug!("Processing batch of {} requests", batch.len());
        batch
    }

    /// Generate dummy traffic for constant-rate mode
    pub async fn generate_dummy_packet(&self, target_size: usize) -> Vec<u8> {
        let mut stats = self.stats.write().await;
        stats.dummy_packets += 1;

        // Generate random-looking dummy data
        let random = self.next_random().await;
        let mut rng = ChaCha20Rng::seed_from_u64(random);

        (0..target_size).map(|_| rng.gen()).collect()
    }

    /// Start constant-rate transmission loop
    pub async fn start_constant_rate(&self, tx: mpsc::Sender<Vec<u8>>) {
        if !self.config.constant_rate {
            return;
        }

        info!("⏱️ Starting constant-rate transmission at {} bps", self.config.target_rate_bps);

        let bytes_per_tick = (self.config.target_rate_bps / 10) as usize; // 100ms ticks
        let tick_interval = Duration::from_millis(100);

        loop {
            let start = Instant::now();

            // Check if we have queued data
            let mut data = {
                let mut queue = self.request_queue.lock().await;
                if let Some(request) = queue.pop_front() {
                    request.payload
                } else {
                    // Generate dummy data
                    self.generate_dummy_packet(bytes_per_tick).await
                }
            };

            // Pad to target size if needed
            if data.len() < bytes_per_tick {
                let padding = self.generate_dummy_packet(bytes_per_tick - data.len()).await;
                data.extend(padding);
            }

            // Send data
            if tx.send(data).await.is_err() {
                warn!("Constant-rate transmission channel closed");
                break;
            }

            // Wait for next tick
            let elapsed = start.elapsed();
            if elapsed < tick_interval {
                sleep(tick_interval - elapsed).await;
            }
        }
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> TimingStats {
        let stats = self.stats.read().await;
        let mut result = stats.clone();

        // Check if quantum entropy is being used
        let pool = self.quantum_entropy_pool.read().await;
        result.quantum_entropy_used = !pool.is_empty();

        result
    }

    /// Get traffic pattern summary
    pub async fn get_pattern_summary(&self) -> PatternSummary {
        let analyzer = self.pattern_analyzer.read().await;
        analyzer.pattern_summary()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = TimingStats::default();
    }
}

/// Circuit-aware timing obfuscation wrapper
pub struct CircuitTimingObfuscator {
    /// Base obfuscator
    obfuscator: Arc<TimingObfuscator>,
    /// Per-circuit timing offsets
    circuit_offsets: Arc<RwLock<std::collections::HashMap<u64, Duration>>>,
    /// Circuit timing history
    circuit_history: Arc<RwLock<std::collections::HashMap<u64, VecDeque<Instant>>>>,
}

impl CircuitTimingObfuscator {
    pub fn new(config: TimingObfuscationConfig) -> Self {
        Self {
            obfuscator: Arc::new(TimingObfuscator::new(config)),
            circuit_offsets: Arc::new(RwLock::new(std::collections::HashMap::new())),
            circuit_history: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Set timing offset for a circuit
    pub async fn set_circuit_offset(&self, circuit_id: u64, offset: Duration) {
        let mut offsets = self.circuit_offsets.write().await;
        offsets.insert(circuit_id, offset);
    }

    /// Record circuit activity
    pub async fn record_circuit_activity(&self, circuit_id: u64) {
        let mut history = self.circuit_history.write().await;
        let times = history.entry(circuit_id).or_insert_with(VecDeque::new);
        times.push_back(Instant::now());

        // Keep last 100 entries
        while times.len() > 100 {
            times.pop_front();
        }
    }

    /// Obfuscate with circuit awareness
    pub async fn obfuscate_for_circuit(
        &self,
        circuit_id: u64,
        payload: &[u8],
    ) -> Result<ScheduledResponse> {
        // Get circuit offset
        let offset = {
            let offsets = self.circuit_offsets.read().await;
            offsets.get(&circuit_id).copied().unwrap_or(Duration::ZERO)
        };

        // Record activity
        self.record_circuit_activity(circuit_id).await;

        // Apply base obfuscation
        let mut response = self.obfuscator.obfuscate(payload).await?;

        // Add circuit-specific offset
        if offset > Duration::ZERO {
            sleep(offset).await;
            response.delay_applied += offset;
        }

        Ok(response)
    }

    /// Calculate recommended offset for new circuit
    pub async fn recommend_offset_for_new_circuit(&self) -> Duration {
        let history = self.circuit_history.read().await;

        if history.is_empty() {
            return Duration::ZERO;
        }

        // Find gaps in recent activity across all circuits
        let mut all_times: Vec<Instant> = history.values()
            .flat_map(|times| times.iter().copied())
            .collect();

        all_times.sort();

        // Find average gap
        if all_times.len() < 2 {
            return Duration::from_millis(50);
        }

        let gaps: Vec<Duration> = all_times.windows(2)
            .map(|w| w[1].duration_since(w[0]))
            .collect();

        let total: Duration = gaps.iter().sum();
        let avg_gap = total / gaps.len() as u32;

        // Recommend offset that doesn't align with existing patterns
        let mut rng = rand::thread_rng();
        let offset_ms = rng.gen_range(0..avg_gap.as_millis() as u64);

        Duration::from_millis(offset_ms)
    }

    /// Get underlying obfuscator
    pub fn obfuscator(&self) -> Arc<TimingObfuscator> {
        Arc::clone(&self.obfuscator)
    }
}

/// Defensive timing wrapper for sensitive operations
pub struct DefensiveTiming {
    /// Minimum operation time
    min_time: Duration,
    /// Maximum variance
    max_variance: Duration,
}

impl DefensiveTiming {
    pub fn new(min_time: Duration, max_variance: Duration) -> Self {
        Self { min_time, max_variance }
    }

    /// Execute an operation with constant-time padding
    pub async fn execute<F, T>(&self, operation: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        let start = Instant::now();

        // Execute the operation
        let result = operation.await;

        // Calculate remaining time to hit minimum
        let elapsed = start.elapsed();
        if elapsed < self.min_time {
            // Add random variance
            let mut rng = rand::thread_rng();
            let variance = Duration::from_micros(
                rng.gen_range(0..self.max_variance.as_micros() as u64)
            );

            let remaining = self.min_time - elapsed + variance;
            sleep(remaining).await;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_timing_obfuscator_creation() {
        let config = TimingObfuscationConfig::default();
        let obfuscator = TimingObfuscator::new(config);

        let stats = obfuscator.get_stats().await;
        assert_eq!(stats.requests_processed, 0);
    }

    #[tokio::test]
    async fn test_obfuscation_applies_delay() {
        let mut config = TimingObfuscationConfig::default();
        config.min_jitter_ms = 10;
        config.max_jitter_ms = 50;
        config.adaptive_jitter = false;

        let obfuscator = TimingObfuscator::new(config);
        let payload = vec![0u8; 100];

        let start = Instant::now();
        let response = obfuscator.obfuscate(&payload).await.unwrap();
        let elapsed = start.elapsed();

        assert!(response.delay_applied >= Duration::from_millis(10));
        assert!(elapsed >= Duration::from_millis(10));
    }

    #[tokio::test]
    async fn test_disabled_mode_no_delay() {
        let mut config = TimingObfuscationConfig::default();
        config.mode = ObfuscationMode::Disabled;

        let obfuscator = TimingObfuscator::new(config);
        let payload = vec![0u8; 100];

        let response = obfuscator.obfuscate(&payload).await.unwrap();
        assert_eq!(response.delay_applied, Duration::ZERO);
    }

    #[tokio::test]
    async fn test_quantum_entropy_pool() {
        let config = TimingObfuscationConfig::default();
        let obfuscator = TimingObfuscator::new(config);

        // Add entropy
        let entropy = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        obfuscator.add_quantum_entropy(&entropy).await;

        let stats = obfuscator.get_stats().await;
        assert!(stats.quantum_entropy_used);
    }

    #[tokio::test]
    async fn test_circuit_timing_obfuscator() {
        let config = TimingObfuscationConfig::default();
        let obfuscator = CircuitTimingObfuscator::new(config);

        // Set offset for circuit
        obfuscator.set_circuit_offset(1, Duration::from_millis(50)).await;

        // Record activity
        obfuscator.record_circuit_activity(1).await;

        // Get recommended offset
        let offset = obfuscator.recommend_offset_for_new_circuit().await;
        assert!(offset <= Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_defensive_timing() {
        let timing = DefensiveTiming::new(
            Duration::from_millis(100),
            Duration::from_millis(20),
        );

        let start = Instant::now();
        timing.execute(async {
            // Fast operation
            42
        }).await;
        let elapsed = start.elapsed();

        // Should take at least min_time
        assert!(elapsed >= Duration::from_millis(100));
    }

    #[test]
    fn test_obfuscation_mode_overhead() {
        assert_eq!(ObfuscationMode::Disabled.overhead_multiplier(), 1.0);
        assert!(ObfuscationMode::Light.overhead_multiplier() > 1.0);
        assert!(ObfuscationMode::Paranoid.overhead_multiplier() > ObfuscationMode::Heavy.overhead_multiplier());
    }
}
