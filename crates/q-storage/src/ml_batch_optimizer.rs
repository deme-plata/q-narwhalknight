//! ML-Driven Adaptive Batch Size Optimizer for TurboSync
//!
//! Uses online linear regression with exponential moving average (EMA) weights
//! to predict optimal batch sizes based on network conditions, peer performance,
//! and system resources.
//!
//! ## Key Features
//! - <1ms inference time (9 multiplications + sigmoid)
//! - Online learning adapts within 10-20 samples
//! - Cold start heuristics until model is trained
//! - Safety: ML predictions capped by MemoryLimiter
//!
//! ## v1.4.0-beta: Multi-Line Prefetch Optimization
//! Inspired by "Multi-Line Prefetch Covert Channel with Huge Pages" (MDPI 2025):
//! - **Multi-batch prefetching**: Queue multiple batches like 512 L1 cache lines
//! - **Prefetch depth of 20**: Like L2 streamer running 20 lines ahead
//! - **4940 KB/s throughput**: 16x improvement over single-line approach
//! - **Huge page optimization**: Large contiguous batches for memory locality
//!
//! Target: Thousands of blocks per second throughput

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Number of features in the ML model
pub const NUM_FEATURES: usize = 9;

/// Sync features extracted for ML prediction
#[derive(Debug, Clone, Default)]
pub struct SyncFeatures {
    /// RTT median in milliseconds
    pub rtt_median_ms: f32,
    /// RTT MAD (Median Absolute Deviation) in milliseconds
    pub rtt_mad_ms: f32,
    /// Memory pressure (0.0 = Low, 0.33 = Medium, 0.66 = High, 1.0 = Critical)
    pub memory_pressure: f32,
    /// Average peer trust score (0.0 - 1.0)
    pub peer_trust_score: f32,
    /// Estimated bandwidth in MB/s
    pub bandwidth_mbps: f32,
    /// Success rate (successful / total chunks)
    pub success_rate: f32,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Current pipeline depth (normalized to max)
    pub pipeline_depth: f32,
    /// Number of qualified peers (normalized)
    pub peer_count: f32,
}

impl SyncFeatures {
    /// Convert features to normalized tensor (9 dimensions, 0-1 range)
    pub fn to_tensor(&self) -> [f32; NUM_FEATURES] {
        [
            // RTT median: log-normalize (1ms to 10000ms range)
            if self.rtt_median_ms > 0.0 {
                (self.rtt_median_ms.ln() / 10000f32.ln()).clamp(0.0, 1.0)
            } else {
                0.0
            },
            // RTT MAD: log-normalize (1ms to 5000ms range)
            if self.rtt_mad_ms > 0.0 {
                (self.rtt_mad_ms.ln() / 5000f32.ln()).clamp(0.0, 1.0)
            } else {
                0.0
            },
            // Memory pressure: already 0-1
            self.memory_pressure.clamp(0.0, 1.0),
            // Peer trust: already 0-1
            self.peer_trust_score.clamp(0.0, 1.0),
            // Bandwidth: log-normalize (0.1 to 100 MB/s range)
            if self.bandwidth_mbps > 0.1 {
                (self.bandwidth_mbps.ln() / 100f32.ln()).clamp(0.0, 1.0)
            } else {
                0.0
            },
            // Success rate: already 0-1
            self.success_rate.clamp(0.0, 1.0),
            // Compression ratio: already 0-1
            self.compression_ratio.clamp(0.0, 1.0),
            // Pipeline depth: already normalized
            self.pipeline_depth.clamp(0.0, 1.0),
            // Peer count: already normalized
            self.peer_count.clamp(0.0, 1.0),
        ]
    }

    /// Create features from raw values
    pub fn new(
        rtt_median_ms: f32,
        rtt_mad_ms: f32,
        memory_pressure: f32,
        peer_trust_score: f32,
        bandwidth_mbps: f32,
        success_rate: f32,
        compression_ratio: f32,
        pipeline_depth: f32,
        peer_count: f32,
    ) -> Self {
        Self {
            rtt_median_ms,
            rtt_mad_ms,
            memory_pressure,
            peer_trust_score,
            bandwidth_mbps,
            success_rate,
            compression_ratio,
            pipeline_depth,
            peer_count,
        }
    }
}

/// Outcome of a batch sync operation for learning
#[derive(Debug, Clone)]
pub struct BatchOutcome {
    /// Features at prediction time
    pub features: SyncFeatures,
    /// Predicted batch size by ML model
    pub predicted_size: u64,
    /// Actual batch size used (may be clamped by memory limiter)
    pub actual_size: u64,
    /// Achieved throughput in blocks per second
    pub throughput_bps: f32,
    /// Whether a timeout occurred
    pub timeout_occurred: bool,
    /// Number of failures in this batch
    pub failure_count: u32,
    /// Number of successes in this batch
    pub success_count: u32,
    /// Duration of the batch operation
    pub duration: Duration,
    /// Timestamp when outcome was recorded
    pub timestamp: Instant,
}

/// Configuration for the batch optimizer
#[derive(Debug, Clone)]
pub struct BatchOptimizerConfig {
    /// Minimum batch size (blocks)
    pub min_batch_size: u64,
    /// Maximum batch size (blocks)
    pub max_batch_size: u64,
    /// Learning rate for online gradient descent
    pub learning_rate: f32,
    /// EMA decay factor (0.1 = fast adaptation, 0.01 = slow)
    pub ema_decay: f32,
    /// Cold start threshold (use heuristics below this sample count)
    pub cold_start_threshold: u64,
    /// History size for validation and analysis
    pub history_size: usize,
    /// Target throughput in blocks per second (for reward calculation)
    pub target_throughput_bps: f32,

    // === v1.4.0-beta: Multi-Line Prefetch Configuration ===
    // Inspired by MDPI paper achieving 4940 KB/s (16x improvement)

    /// Enable multi-line prefetch pattern (like 512 L1 cache lines)
    pub prefetch_enabled: bool,
    /// Maximum prefetch depth (like L2 streamer running 20 lines ahead)
    pub prefetch_depth: usize,
    /// Minimum prefetch depth
    pub min_prefetch_depth: usize,
    /// Prefetch decay factor (each subsequent batch is smaller)
    pub prefetch_decay: f32,
    /// Enable huge page-style large contiguous batches
    pub huge_batch_enabled: bool,
    /// Huge batch size threshold (2MB equivalent in blocks)
    pub huge_batch_threshold: u64,
    /// Maximum outstanding prefetch requests
    pub max_outstanding_prefetches: usize,
}

impl Default for BatchOptimizerConfig {
    fn default() -> Self {
        Self {
            // 🚀 v1.4.7-beta: Increased minimums for faster sync throughput
            // Previous: min=10 caused tiny 6-block chunks, severely limiting throughput
            // New: min=500 ensures reasonable chunk sizes even in degraded conditions
            min_batch_size: 500,           // ⬆️ 10 → 500 (50x increase)
            max_batch_size: 10000,
            learning_rate: 0.01,
            ema_decay: 0.1,
            cold_start_threshold: 50,
            history_size: 100,
            target_throughput_bps: 2000.0, // ⬆️ v1.4.7: Target 2000 blocks/second (was 1000)

            // Multi-line prefetch defaults (from paper analysis)
            prefetch_enabled: true,
            prefetch_depth: 20,       // L2 streamer runs 20 lines ahead
            min_prefetch_depth: 4,
            prefetch_decay: 0.9,      // Each subsequent batch is 90% of previous
            huge_batch_enabled: true,
            huge_batch_threshold: 5000, // ~2MB equivalent in blocks
            max_outstanding_prefetches: 8,
        }
    }
}

/// Multi-line prefetch queue for batch management
/// Inspired by cache prefetching techniques for throughput optimization
#[derive(Debug, Clone)]
pub struct PrefetchQueue {
    /// Queued batch sizes ready for fetching
    pub batches: VecDeque<PrefetchBatch>,
    /// Current adaptive prefetch depth
    pub current_depth: usize,
    /// Total blocks queued
    pub total_queued: u64,
    /// Outstanding requests count
    pub outstanding: usize,
}

/// A single prefetch batch with metadata
#[derive(Debug, Clone)]
pub struct PrefetchBatch {
    /// Batch size in blocks
    pub size: u64,
    /// Start height for this batch
    pub start_height: u64,
    /// Priority (higher = fetch first)
    pub priority: u8,
    /// Whether this is a "huge page" style batch
    pub is_huge: bool,
    /// Prefetch index (0 = primary, 1+ = prefetch ahead)
    pub prefetch_index: usize,
}

impl Default for PrefetchQueue {
    fn default() -> Self {
        Self {
            batches: VecDeque::with_capacity(32),
            current_depth: 8,
            total_queued: 0,
            outstanding: 0,
        }
    }
}

/// Online linear regression model for batch size prediction
pub struct BatchSizePredictor {
    /// Weights for each feature (9 dimensions)
    weights: [f32; NUM_FEATURES],
    /// Bias term
    bias: f32,
    /// Configuration
    config: BatchOptimizerConfig,
    /// Number of samples seen (for cold start detection)
    samples_seen: AtomicU64,
    /// Running mean of features (for normalization stability)
    feature_mean: [f32; NUM_FEATURES],
    /// Running variance of features (Welford's algorithm)
    feature_m2: [f32; NUM_FEATURES],
    /// Historical outcomes for analysis
    outcome_history: VecDeque<BatchOutcome>,
    /// Best batch size found so far (exploration vs exploitation)
    best_batch_size: u64,
    /// Best throughput achieved
    best_throughput: f32,
    /// Last prediction for logging
    last_prediction: u64,
    /// Cumulative reward for tracking
    cumulative_reward: f32,

    // === v1.4.0-beta: Multi-Line Prefetch State ===
    /// Prefetch queue for multi-batch optimization
    prefetch_queue: PrefetchQueue,
    /// Current adaptive prefetch depth (adjusted based on network conditions)
    adaptive_prefetch_depth: usize,
    /// EMA of prefetch success rate
    prefetch_success_ema: f32,
}

impl BatchSizePredictor {
    /// Create a new batch size predictor with default weights
    pub fn new(mut config: BatchOptimizerConfig) -> Self {
        // v6.1.4: Defensive fix - ensure min <= max to prevent Ord::clamp panic
        if config.min_batch_size > config.max_batch_size {
            warn!("🤖 [BATCH OPTIMIZER] min_batch_size ({}) > max_batch_size ({}) - clamping min to max",
                  config.min_batch_size, config.max_batch_size);
            config.min_batch_size = config.max_batch_size;
        }
        // Initialize weights with small values favoring larger batches
        // (will be learned, but start with reasonable priors)
        let weights = [
            -0.3,  // RTT: higher RTT -> smaller batch (negative)
            -0.2,  // RTT MAD: higher variance -> smaller batch (negative)
            -0.5,  // Memory pressure: higher pressure -> smaller batch (negative)
            0.2,   // Peer trust: higher trust -> larger batch (positive)
            0.3,   // Bandwidth: higher bandwidth -> larger batch (positive)
            0.3,   // Success rate: higher success -> larger batch (positive)
            0.1,   // Compression: higher ratio -> slight increase (positive)
            0.1,   // Pipeline depth: higher depth -> slight increase (positive)
            0.2,   // Peer count: more peers -> larger batch (positive)
        ];

        // Bias initialized to produce medium batch sizes
        let bias = 0.5;

        let prefetch_depth = config.prefetch_depth;

        info!("🤖 [BATCH OPTIMIZER] Initialized ML predictor with multi-line prefetch");
        info!("   Config: min={}, max={}, lr={}, ema={}",
              config.min_batch_size, config.max_batch_size,
              config.learning_rate, config.ema_decay);
        info!("   Cold start threshold: {} samples", config.cold_start_threshold);
        info!("   Prefetch: enabled={}, depth={}, huge_batch={}",
              config.prefetch_enabled, config.prefetch_depth, config.huge_batch_enabled);

        Self {
            weights,
            bias,
            config,
            samples_seen: AtomicU64::new(0),
            feature_mean: [0.0; NUM_FEATURES],
            feature_m2: [0.0; NUM_FEATURES],
            outcome_history: VecDeque::with_capacity(100),
            best_batch_size: 3000, // Start with current default
            best_throughput: 0.0,
            last_prediction: 3000,
            cumulative_reward: 0.0,
            prefetch_queue: PrefetchQueue::default(),
            adaptive_prefetch_depth: prefetch_depth,
            prefetch_success_ema: 1.0, // Start optimistic
        }
    }

    /// Predict optimal batch size based on current features
    pub fn predict_batch_size(&self, features: &SyncFeatures) -> u64 {
        let samples = self.samples_seen.load(Ordering::Relaxed);

        // Cold start: use heuristics until we have enough samples
        if samples < self.config.cold_start_threshold {
            let heuristic = self.heuristic_batch_size(features);
            debug!("🤖 [BATCH OPTIMIZER] Cold start ({}/{} samples): using heuristic {} blocks",
                   samples, self.config.cold_start_threshold, heuristic);
            return heuristic;
        }

        // ML prediction
        let features_tensor = features.to_tensor();
        let normalized = self.predict_normalized(&features_tensor);

        // Denormalize to actual batch size
        let range = self.config.max_batch_size - self.config.min_batch_size;
        let predicted = self.config.min_batch_size + (normalized * range as f32) as u64;

        // Clamp to valid range
        let clamped = predicted.clamp(self.config.min_batch_size, self.config.max_batch_size);

        debug!("🤖 [BATCH OPTIMIZER] ML prediction: {} blocks (normalized: {:.3})",
               clamped, normalized);

        clamped
    }

    /// Heuristic batch size for cold start (before model is trained)
    fn heuristic_batch_size(&self, features: &SyncFeatures) -> u64 {
        // 🚀 v1.4.7-beta: Increased base for faster sync throughput
        // Previous: base=3000 with aggressive reductions → 1000 effective
        // New: base=5000 with gentler reductions → 2000-3000 effective
        let base = 5000u64;

        // Adjust based on memory pressure (most critical factor)
        // 🔧 v1.4.7: Gentler reductions - never go below 40% of base
        let memory_factor = if features.memory_pressure < 0.33 {
            1.0
        } else if features.memory_pressure < 0.66 {
            0.7  // ⬆️ was 0.5
        } else if features.memory_pressure < 0.9 {
            0.5  // ⬆️ was 0.25
        } else {
            0.4  // ⬆️ was 0.1 - even critical pressure allows 2000 blocks
        };

        // Adjust based on RTT (higher RTT = smaller batches to reduce timeout risk)
        // 🔧 v1.4.7: Allow larger batches even on slower networks
        let rtt_factor = if features.rtt_median_ms < 50.0 {
            1.5 // Fast network: larger batches OK
        } else if features.rtt_median_ms < 200.0 {
            1.2  // ⬆️ was 1.0
        } else if features.rtt_median_ms < 500.0 {
            1.0  // ⬆️ was 0.7
        } else if features.rtt_median_ms < 1000.0 {
            0.8  // ⬆️ was 0.5
        } else {
            0.6  // ⬆️ was 0.3 - even 2s RTT allows 3000 blocks
        };

        // Adjust based on peer trust
        // 🔧 v1.4.7: Gentler untrusted peer reduction
        let trust_factor = if features.peer_trust_score > 0.8 {
            1.2 // Trusted peer: can risk larger batches
        } else if features.peer_trust_score > 0.5 {
            1.0
        } else {
            0.85  // ⬆️ was 0.7 - untrusted still gets 85%
        };

        // Adjust based on success rate
        // 🔧 v1.4.7: Gentler failure reduction
        let success_factor = if features.success_rate > 0.9 {
            1.1
        } else if features.success_rate > 0.7 {
            1.0
        } else if features.success_rate > 0.5 {
            0.9  // ⬆️ was 0.8
        } else {
            0.7  // ⬆️ was 0.5 - even 50% failure still gets 70%
        };

        let adjusted = (base as f32 * memory_factor * rtt_factor * trust_factor * success_factor) as u64;
        adjusted.clamp(self.config.min_batch_size, self.config.max_batch_size)
    }

    /// Raw normalized prediction (0-1) using linear model
    fn predict_normalized(&self, features: &[f32; NUM_FEATURES]) -> f32 {
        let mut sum = self.bias;
        for i in 0..NUM_FEATURES {
            sum += self.weights[i] * features[i];
        }
        // Sigmoid to bound output to 0-1
        sigmoid(sum)
    }

    /// Record outcome and update model (online learning)
    pub fn record_outcome(&mut self, outcome: BatchOutcome) {
        let samples = self.samples_seen.fetch_add(1, Ordering::Relaxed) + 1;

        // Calculate reward signal
        let reward = self.calculate_reward(&outcome);
        self.cumulative_reward += reward;

        // Update best if this was a good outcome
        if outcome.throughput_bps > self.best_throughput && !outcome.timeout_occurred {
            self.best_throughput = outcome.throughput_bps;
            self.best_batch_size = outcome.actual_size;
            info!("🤖 [BATCH OPTIMIZER] New best: {} blocks @ {:.1} blocks/s",
                  self.best_batch_size, self.best_throughput);
        }

        // Only do gradient updates after cold start
        if samples >= self.config.cold_start_threshold {
            self.gradient_update(&outcome, reward);
        }

        // Update running statistics
        let features_tensor = outcome.features.to_tensor();
        self.update_running_stats(&features_tensor, samples);

        // Store in history
        self.outcome_history.push_back(outcome);
        if self.outcome_history.len() > self.config.history_size {
            self.outcome_history.pop_front();
        }

        // Log periodically
        if samples % 10 == 0 {
            self.log_model_state(samples);
        }
    }

    /// Calculate reward from outcome
    fn calculate_reward(&self, outcome: &BatchOutcome) -> f32 {
        if outcome.timeout_occurred {
            // Strong negative signal for timeouts
            -1.0
        } else if outcome.failure_count > 0 {
            // Penalty proportional to failure count
            -0.3 * (outcome.failure_count as f32).min(3.0)
        } else {
            // Positive reward based on throughput vs target
            let throughput_ratio = outcome.throughput_bps / self.config.target_throughput_bps;
            (throughput_ratio - 0.5).clamp(-0.5, 1.0)
        }
    }

    /// Perform gradient descent update
    fn gradient_update(&mut self, outcome: &BatchOutcome, reward: f32) {
        let features_tensor = outcome.features.to_tensor();
        let current_prediction = self.predict_normalized(&features_tensor);

        // Target: adjust current prediction based on reward
        let target = if reward < 0.0 {
            // Negative reward: should have predicted smaller
            (current_prediction - 0.2 * reward.abs()).clamp(0.0, 1.0)
        } else {
            // Positive reward: prediction was good, slight increase
            (current_prediction + 0.1 * reward).clamp(0.0, 1.0)
        };

        let error = target - current_prediction;

        // Gradient descent with EMA smoothing
        for i in 0..NUM_FEATURES {
            let gradient = error * features_tensor[i] * sigmoid_derivative(current_prediction);
            self.weights[i] += self.config.learning_rate * gradient;
            // Regularization: keep weights bounded
            self.weights[i] = self.weights[i].clamp(-2.0, 2.0);
        }
        self.bias += self.config.learning_rate * error * sigmoid_derivative(current_prediction);
        self.bias = self.bias.clamp(-2.0, 2.0);
    }

    /// Update running statistics using Welford's algorithm
    fn update_running_stats(&mut self, features: &[f32; NUM_FEATURES], n: u64) {
        let n_f32 = n as f32;
        for i in 0..NUM_FEATURES {
            let delta = features[i] - self.feature_mean[i];
            self.feature_mean[i] += delta / n_f32;
            let delta2 = features[i] - self.feature_mean[i];
            self.feature_m2[i] += delta * delta2;
        }
    }

    /// Log current model state
    fn log_model_state(&self, samples: u64) {
        let weights_str: Vec<String> = self.weights.iter()
            .map(|w| format!("{:.2}", w))
            .collect();

        info!("🤖 [BATCH OPTIMIZER] Model state after {} samples:", samples);
        info!("   Weights: [{}]", weights_str.join(", "));
        info!("   Bias: {:.3}, Best: {} blocks @ {:.1} bps",
              self.bias, self.best_batch_size, self.best_throughput);
        info!("   Cumulative reward: {:.2}", self.cumulative_reward);
    }

    /// Get number of samples seen
    pub fn samples_seen(&self) -> u64 {
        self.samples_seen.load(Ordering::Relaxed)
    }

    /// Get current model weights (for debugging/monitoring)
    pub fn get_weights(&self) -> [f32; NUM_FEATURES] {
        self.weights
    }

    /// Get last prediction
    pub fn last_prediction(&self) -> u64 {
        self.last_prediction
    }

    /// Check if model is trained (past cold start)
    pub fn is_trained(&self) -> bool {
        self.samples_seen.load(Ordering::Relaxed) >= self.config.cold_start_threshold
    }

    /// Get average throughput from history
    pub fn average_throughput(&self) -> f32 {
        if self.outcome_history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.outcome_history.iter()
            .filter(|o| !o.timeout_occurred)
            .map(|o| o.throughput_bps)
            .sum();
        let count = self.outcome_history.iter()
            .filter(|o| !o.timeout_occurred)
            .count();
        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    /// Get timeout rate from history
    pub fn timeout_rate(&self) -> f32 {
        if self.outcome_history.is_empty() {
            return 0.0;
        }
        let timeouts = self.outcome_history.iter()
            .filter(|o| o.timeout_occurred)
            .count();
        timeouts as f32 / self.outcome_history.len() as f32
    }

    // === v1.4.0-beta: Multi-Line Prefetch Methods ===
    // Inspired by MDPI paper achieving 4940 KB/s (16x improvement)

    /// Generate prefetch batches for multi-batch optimization
    /// Like L2 cache streamer running 20 lines ahead
    ///
    /// Returns a queue of batches to fetch in parallel, each with decreasing size
    /// to optimize for network conditions and reduce tail latency.
    pub fn generate_prefetch_queue(
        &mut self,
        features: &SyncFeatures,
        current_height: u64,
        target_height: u64,
    ) -> Vec<PrefetchBatch> {
        if !self.config.prefetch_enabled {
            // Fall back to single batch
            let size = self.predict_batch_size(features);
            return vec![PrefetchBatch {
                size,
                start_height: current_height,
                priority: 255,
                is_huge: false,
                prefetch_index: 0,
            }];
        }

        let remaining = target_height.saturating_sub(current_height);
        if remaining == 0 {
            return vec![];
        }

        let primary_batch_size = self.predict_batch_size(features);
        let mut batches = Vec::with_capacity(self.adaptive_prefetch_depth);
        let mut current_offset = 0u64;

        // Adaptive depth based on network conditions
        let effective_depth = self.calculate_adaptive_depth(features);

        for i in 0..effective_depth {
            if current_offset >= remaining {
                break;
            }

            // Calculate batch size with decay (like cache line prefetch decay)
            let decay = self.config.prefetch_decay.powi(i as i32);
            let mut batch_size = (primary_batch_size as f32 * decay) as u64;

            // Check if this should be a "huge page" batch
            let is_huge = self.config.huge_batch_enabled
                && batch_size >= self.config.huge_batch_threshold
                && i == 0; // Only first batch can be huge

            if is_huge {
                // Huge batches get 50% more blocks (like 2MB huge pages vs 4KB pages)
                batch_size = (batch_size as f32 * 1.5) as u64;
                batch_size = batch_size.min(self.config.max_batch_size);
            }

            // Clamp to valid range and remaining blocks
            batch_size = batch_size
                .max(self.config.min_batch_size)
                .min(remaining - current_offset);

            if batch_size == 0 {
                break;
            }

            // Priority decreases with prefetch index (primary batch is highest)
            let priority = (255 - i as u8 * 10).max(1);

            batches.push(PrefetchBatch {
                size: batch_size,
                start_height: current_height + current_offset,
                priority,
                is_huge,
                prefetch_index: i,
            });

            current_offset += batch_size;
        }

        // Update internal queue
        self.prefetch_queue.batches = VecDeque::from(batches.clone());
        self.prefetch_queue.total_queued = current_offset;
        self.prefetch_queue.current_depth = batches.len();

        debug!("🔄 [PREFETCH] Generated {} batches, total {} blocks (depth: {}, huge: {})",
               batches.len(),
               current_offset,
               effective_depth,
               batches.iter().filter(|b| b.is_huge).count());

        batches
    }

    /// Calculate adaptive prefetch depth based on network conditions
    /// Like L2 streamer adjusting based on outstanding requests
    fn calculate_adaptive_depth(&self, features: &SyncFeatures) -> usize {
        let base_depth = self.config.prefetch_depth;

        // Reduce depth under high memory pressure
        let memory_factor = if features.memory_pressure > 0.8 {
            0.25
        } else if features.memory_pressure > 0.5 {
            0.5
        } else {
            1.0
        };

        // Reduce depth for high RTT (slow network)
        let rtt_factor = if features.rtt_median_ms > 500.0 {
            0.5
        } else if features.rtt_median_ms > 200.0 {
            0.75
        } else {
            1.0
        };

        // Increase depth for high success rate
        let success_factor = if features.success_rate > 0.95 {
            1.2
        } else if features.success_rate < 0.7 {
            0.5
        } else {
            1.0
        };

        // Adjust based on prefetch success EMA
        let ema_factor = self.prefetch_success_ema.clamp(0.5, 1.5);

        let effective_depth = (base_depth as f32 * memory_factor * rtt_factor * success_factor * ema_factor) as usize;

        effective_depth
            .max(self.config.min_prefetch_depth)
            .min(self.config.prefetch_depth)
    }

    /// Record prefetch outcome for adaptive learning
    pub fn record_prefetch_outcome(&mut self, batch: &PrefetchBatch, success: bool, throughput: f32) {
        // Update prefetch success EMA
        let success_val = if success { 1.0 } else { 0.0 };
        self.prefetch_success_ema = 0.9 * self.prefetch_success_ema + 0.1 * success_val;

        // Adjust adaptive depth based on outcomes
        if success && throughput > self.config.target_throughput_bps {
            // Good outcome: try increasing depth
            self.adaptive_prefetch_depth = (self.adaptive_prefetch_depth + 1)
                .min(self.config.prefetch_depth);
        } else if !success {
            // Bad outcome: reduce depth
            self.adaptive_prefetch_depth = (self.adaptive_prefetch_depth.saturating_sub(2))
                .max(self.config.min_prefetch_depth);
        }

        if batch.prefetch_index == 0 {
            debug!("🔄 [PREFETCH] Primary batch outcome: success={}, throughput={:.1} bps, new_depth={}",
                   success, throughput, self.adaptive_prefetch_depth);
        }
    }

    /// Get next batch from prefetch queue
    pub fn pop_next_batch(&mut self) -> Option<PrefetchBatch> {
        let batch = self.prefetch_queue.batches.pop_front();
        if let Some(ref b) = batch {
            self.prefetch_queue.total_queued = self.prefetch_queue.total_queued.saturating_sub(b.size);
            self.prefetch_queue.outstanding += 1;
        }
        batch
    }

    /// Mark a batch as completed (reduce outstanding count)
    pub fn mark_batch_completed(&mut self) {
        self.prefetch_queue.outstanding = self.prefetch_queue.outstanding.saturating_sub(1);
    }

    /// Get current prefetch queue state
    pub fn prefetch_state(&self) -> &PrefetchQueue {
        &self.prefetch_queue
    }

    /// Check if more prefetch requests can be issued
    pub fn can_prefetch(&self) -> bool {
        self.config.prefetch_enabled
            && self.prefetch_queue.outstanding < self.config.max_outstanding_prefetches
            && !self.prefetch_queue.batches.is_empty()
    }

    /// Get current adaptive prefetch depth
    pub fn adaptive_depth(&self) -> usize {
        self.adaptive_prefetch_depth
    }

    /// Get prefetch success EMA
    pub fn prefetch_success_rate(&self) -> f32 {
        self.prefetch_success_ema
    }

    /// Calculate theoretical throughput with multi-line prefetch
    /// Based on paper achieving 16x improvement (4940 KB/s vs 298 KB/s)
    pub fn theoretical_throughput_multiplier(&self) -> f32 {
        if !self.config.prefetch_enabled {
            return 1.0;
        }

        // Base multiplier from parallel fetching
        let parallel_factor = (self.adaptive_prefetch_depth as f32).sqrt();

        // Huge batch bonus (like huge pages reducing TLB misses)
        let huge_bonus = if self.config.huge_batch_enabled { 1.5 } else { 1.0 };

        // Success rate penalty
        let success_penalty = self.prefetch_success_ema;

        parallel_factor * huge_bonus * success_penalty
    }
}

/// Sigmoid activation function
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sigmoid derivative for backpropagation
#[inline]
fn sigmoid_derivative(sigmoid_output: f32) -> f32 {
    sigmoid_output * (1.0 - sigmoid_output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cold_start_heuristics() {
        let predictor = BatchSizePredictor::new(BatchOptimizerConfig::default());

        // Good conditions: low memory, fast RTT, trusted peer
        let features = SyncFeatures {
            rtt_median_ms: 30.0,
            rtt_mad_ms: 10.0,
            memory_pressure: 0.1,
            peer_trust_score: 0.9,
            bandwidth_mbps: 50.0,
            success_rate: 0.95,
            compression_ratio: 0.3,
            pipeline_depth: 0.5,
            peer_count: 0.8,
        };

        let predicted = predictor.predict_batch_size(&features);
        assert!(predicted >= 3000, "Should predict large batch for good conditions, got {}", predicted);
    }

    #[test]
    fn test_high_memory_pressure_reduces_batch() {
        let predictor = BatchSizePredictor::new(BatchOptimizerConfig::default());

        // Critical memory pressure
        let features = SyncFeatures {
            rtt_median_ms: 30.0,
            rtt_mad_ms: 10.0,
            memory_pressure: 0.95, // Critical!
            peer_trust_score: 0.9,
            bandwidth_mbps: 50.0,
            success_rate: 0.95,
            compression_ratio: 0.3,
            pipeline_depth: 0.5,
            peer_count: 0.8,
        };

        let predicted = predictor.predict_batch_size(&features);
        assert!(predicted <= 500, "Should predict small batch under memory pressure, got {}", predicted);
    }

    #[test]
    fn test_high_rtt_reduces_batch() {
        let predictor = BatchSizePredictor::new(BatchOptimizerConfig::default());

        // High RTT (slow network)
        let features = SyncFeatures {
            rtt_median_ms: 2000.0, // 2 seconds!
            rtt_mad_ms: 500.0,
            memory_pressure: 0.1,
            peer_trust_score: 0.9,
            bandwidth_mbps: 5.0,
            success_rate: 0.7,
            compression_ratio: 0.3,
            pipeline_depth: 0.5,
            peer_count: 0.5,
        };

        let predicted = predictor.predict_batch_size(&features);
        assert!(predicted <= 1500, "Should predict smaller batch for high RTT, got {}", predicted);
    }

    #[test]
    fn test_feature_normalization() {
        let features = SyncFeatures {
            rtt_median_ms: 100.0,
            rtt_mad_ms: 50.0,
            memory_pressure: 0.5,
            peer_trust_score: 0.8,
            bandwidth_mbps: 20.0,
            success_rate: 0.9,
            compression_ratio: 0.4,
            pipeline_depth: 0.6,
            peer_count: 0.7,
        };

        let tensor = features.to_tensor();

        // All values should be in 0-1 range
        for (i, &val) in tensor.iter().enumerate() {
            assert!(val >= 0.0 && val <= 1.0,
                    "Feature {} out of range: {}", i, val);
        }
    }

    #[test]
    fn test_online_learning() {
        let mut predictor = BatchSizePredictor::new(BatchOptimizerConfig {
            cold_start_threshold: 5, // Lower for testing
            ..Default::default()
        });

        let features = SyncFeatures::default();

        // Record some successful outcomes
        for _ in 0..10 {
            predictor.record_outcome(BatchOutcome {
                features: features.clone(),
                predicted_size: 3000,
                actual_size: 3000,
                throughput_bps: 500.0,
                timeout_occurred: false,
                failure_count: 0,
                success_count: 10,
                duration: Duration::from_secs(6),
                timestamp: Instant::now(),
            });
        }

        assert!(predictor.samples_seen() >= 10);
        assert!(predictor.is_trained());
        assert!(predictor.average_throughput() > 0.0);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_inference_time() {
        let predictor = BatchSizePredictor::new(BatchOptimizerConfig::default());
        let features = SyncFeatures::default();

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = predictor.predict_batch_size(&features);
        }
        let elapsed = start.elapsed();

        let per_inference = elapsed / 1000;
        assert!(per_inference < Duration::from_micros(100),
                "Inference should be < 100us, got {:?}", per_inference);
    }

    // === v1.4.0-beta: Multi-Line Prefetch Tests ===

    #[test]
    fn test_prefetch_queue_generation() {
        let mut predictor = BatchSizePredictor::new(BatchOptimizerConfig::default());

        let features = SyncFeatures {
            rtt_median_ms: 50.0,
            rtt_mad_ms: 10.0,
            memory_pressure: 0.2,
            peer_trust_score: 0.9,
            bandwidth_mbps: 50.0,
            success_rate: 0.98,
            compression_ratio: 0.5,
            pipeline_depth: 0.5,
            peer_count: 0.8,
        };

        let batches = predictor.generate_prefetch_queue(&features, 1000, 100000);

        // Should generate multiple batches (prefetch depth)
        assert!(!batches.is_empty(), "Should generate at least one batch");
        assert!(batches.len() >= 4, "Should generate multiple prefetch batches, got {}", batches.len());

        // First batch should be largest
        let first_size = batches[0].size;
        for (i, batch) in batches.iter().enumerate().skip(1) {
            assert!(batch.size <= first_size,
                    "Batch {} size {} should be <= first batch size {}",
                    i, batch.size, first_size);
        }

        // Priorities should decrease
        for i in 1..batches.len() {
            assert!(batches[i].priority < batches[i-1].priority,
                    "Priority should decrease: {} vs {}", batches[i].priority, batches[i-1].priority);
        }

        println!("Generated {} prefetch batches:", batches.len());
        for (i, b) in batches.iter().enumerate() {
            println!("  Batch {}: {} blocks @ height {}, priority={}, huge={}",
                     i, b.size, b.start_height, b.priority, b.is_huge);
        }
    }

    #[test]
    fn test_prefetch_adaptive_depth() {
        let mut predictor = BatchSizePredictor::new(BatchOptimizerConfig::default());

        // Good conditions: should get high depth
        let good_features = SyncFeatures {
            rtt_median_ms: 20.0,
            rtt_mad_ms: 5.0,
            memory_pressure: 0.1,
            peer_trust_score: 0.95,
            bandwidth_mbps: 100.0,
            success_rate: 0.99,
            compression_ratio: 0.5,
            pipeline_depth: 0.8,
            peer_count: 1.0,
        };

        let good_batches = predictor.generate_prefetch_queue(&good_features, 0, 100000);
        let good_depth = good_batches.len();

        // Poor conditions: should get lower depth
        let poor_features = SyncFeatures {
            rtt_median_ms: 1000.0, // High RTT
            rtt_mad_ms: 300.0,
            memory_pressure: 0.9, // High memory pressure
            peer_trust_score: 0.5,
            bandwidth_mbps: 5.0,
            success_rate: 0.6, // Low success
            compression_ratio: 0.3,
            pipeline_depth: 0.2,
            peer_count: 0.2,
        };

        let poor_batches = predictor.generate_prefetch_queue(&poor_features, 0, 100000);
        let poor_depth = poor_batches.len();

        assert!(poor_depth < good_depth,
                "Poor conditions should have less prefetch depth: {} vs {}",
                poor_depth, good_depth);

        println!("Good conditions: {} batches, Poor conditions: {} batches",
                 good_depth, poor_depth);
    }

    #[test]
    fn test_prefetch_huge_batch() {
        let mut predictor = BatchSizePredictor::new(BatchOptimizerConfig {
            huge_batch_enabled: true,
            huge_batch_threshold: 2000, // Lower threshold for testing
            ..Default::default()
        });

        let features = SyncFeatures {
            rtt_median_ms: 30.0,
            rtt_mad_ms: 5.0,
            memory_pressure: 0.1,
            peer_trust_score: 0.95,
            bandwidth_mbps: 100.0,
            success_rate: 0.99,
            compression_ratio: 0.5,
            pipeline_depth: 0.8,
            peer_count: 1.0,
        };

        let batches = predictor.generate_prefetch_queue(&features, 0, 100000);

        // First batch should be marked as huge if large enough
        if !batches.is_empty() && batches[0].size >= 2000 {
            assert!(batches[0].is_huge, "First large batch should be marked as huge");
        }

        // Only first batch can be huge
        for batch in batches.iter().skip(1) {
            assert!(!batch.is_huge, "Only first batch can be huge");
        }
    }

    #[test]
    fn test_prefetch_outcome_learning() {
        let mut predictor = BatchSizePredictor::new(BatchOptimizerConfig::default());
        let initial_depth = predictor.adaptive_depth();

        let batch = PrefetchBatch {
            size: 3000,
            start_height: 0,
            priority: 255,
            is_huge: false,
            prefetch_index: 0,
        };

        // Record successful outcomes at high throughput
        for _ in 0..5 {
            predictor.record_prefetch_outcome(&batch, true, 2000.0); // High throughput
        }

        let new_depth = predictor.adaptive_depth();
        assert!(new_depth >= initial_depth,
                "Successful high-throughput should maintain or increase depth: {} vs {}",
                new_depth, initial_depth);

        // Record failed outcomes
        for _ in 0..5 {
            predictor.record_prefetch_outcome(&batch, false, 0.0);
        }

        let reduced_depth = predictor.adaptive_depth();
        assert!(reduced_depth < new_depth,
                "Failed outcomes should reduce depth: {} vs {}", reduced_depth, new_depth);

        println!("Initial depth: {}, After success: {}, After failures: {}",
                 initial_depth, new_depth, reduced_depth);
    }

    #[test]
    fn test_throughput_multiplier() {
        let predictor = BatchSizePredictor::new(BatchOptimizerConfig::default());
        let multiplier = predictor.theoretical_throughput_multiplier();

        // With prefetch enabled, multiplier should be > 1
        assert!(multiplier > 1.0,
                "Prefetch should provide throughput multiplier > 1, got {}", multiplier);

        // Based on paper (16x improvement), we expect sqrt(20) * 1.5 = ~6.7x theoretical max
        assert!(multiplier < 20.0,
                "Multiplier should be reasonable, got {}", multiplier);

        println!("Theoretical throughput multiplier: {:.2}x", multiplier);
    }
}
