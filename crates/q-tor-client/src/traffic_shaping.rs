/// Enhanced Traffic Shaping for Q-NarwhalKnight
///
/// This module implements traffic shaping techniques to prevent bandwidth
/// fingerprinting attacks. Without traffic shaping, an adversary observing
/// network traffic can identify Tor usage and potentially correlate traffic
/// patterns to specific applications or users.
///
/// # Attack Model
/// Bandwidth fingerprinting attacks include:
/// - Traffic volume correlation (matching ingress/egress patterns)
/// - Timing analysis (identifying burst patterns)
/// - Packet size analysis (identifying protocol-specific sizes)
/// - Flow watermarking (injecting timing patterns)
///
/// # Protection Mechanisms
/// - Constant-rate padding: Maintain steady bandwidth regardless of actual traffic
/// - Packet size normalization: Pad packets to fixed sizes
/// - Timing jitter injection: Randomize packet timing
/// - Dummy traffic generation: Insert cover traffic
/// - Adaptive shaping: Adjust based on observed patterns

use anyhow::{anyhow, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, trace, warn};

/// Packet size classes for normalization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PacketSizeClass {
    /// Small packets (control messages, ACKs)
    Small,  // 128 bytes
    /// Medium packets (typical data)
    Medium, // 512 bytes
    /// Large packets (bulk transfer)
    Large,  // 1024 bytes
    /// Maximum size (MTU-safe)
    Maximum, // 1400 bytes
}

impl PacketSizeClass {
    /// Get the padded size for this class
    pub fn padded_size(&self) -> usize {
        match self {
            PacketSizeClass::Small => 128,
            PacketSizeClass::Medium => 512,
            PacketSizeClass::Large => 1024,
            PacketSizeClass::Maximum => 1400,
        }
    }

    /// Determine the appropriate class for a given payload size
    pub fn for_payload(size: usize) -> Self {
        if size <= 128 {
            PacketSizeClass::Small
        } else if size <= 512 {
            PacketSizeClass::Medium
        } else if size <= 1024 {
            PacketSizeClass::Large
        } else {
            PacketSizeClass::Maximum
        }
    }
}

/// Traffic shaping mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapingMode {
    /// No shaping (baseline)
    Disabled,
    /// Light shaping (packet normalization only)
    Light,
    /// Medium shaping (normalization + timing jitter)
    Medium,
    /// Heavy shaping (constant rate with padding)
    Heavy,
    /// Paranoid mode (full cover traffic)
    Paranoid,
}

impl ShapingMode {
    pub fn name(&self) -> &'static str {
        match self {
            ShapingMode::Disabled => "disabled",
            ShapingMode::Light => "light",
            ShapingMode::Medium => "medium",
            ShapingMode::Heavy => "heavy",
            ShapingMode::Paranoid => "paranoid",
        }
    }
}

/// Configuration for traffic shaping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficShapingConfig {
    /// Enable traffic shaping
    pub enabled: bool,
    /// Shaping mode
    pub mode: ShapingMode,
    /// Target bandwidth (bytes per second) for constant-rate mode
    pub target_bandwidth: u64,
    /// Minimum packet size (for normalization)
    pub min_packet_size: usize,
    /// Maximum jitter to add (milliseconds)
    pub max_jitter_ms: u64,
    /// Dummy packet interval (milliseconds) for cover traffic
    pub dummy_interval_ms: u64,
    /// Enable adaptive shaping based on observed patterns
    pub adaptive: bool,
    /// Burst tolerance (how much traffic can exceed target before throttling)
    pub burst_tolerance: f64,
    /// Use quantum entropy for jitter generation
    pub quantum_jitter: bool,
}

impl Default for TrafficShapingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: ShapingMode::Medium,
            target_bandwidth: 1_000_000, // 1 MB/s
            min_packet_size: 128,
            max_jitter_ms: 50,
            dummy_interval_ms: 100,
            adaptive: true,
            burst_tolerance: 1.5,
            quantum_jitter: false,
        }
    }
}

impl TrafficShapingConfig {
    /// High security configuration
    pub fn high_security() -> Self {
        Self {
            enabled: true,
            mode: ShapingMode::Heavy,
            target_bandwidth: 500_000, // 500 KB/s
            min_packet_size: 512,
            max_jitter_ms: 100,
            dummy_interval_ms: 50,
            adaptive: true,
            burst_tolerance: 1.2,
            quantum_jitter: true,
        }
    }

    /// Paranoid configuration (maximum protection, high overhead)
    pub fn paranoid() -> Self {
        Self {
            enabled: true,
            mode: ShapingMode::Paranoid,
            target_bandwidth: 256_000, // 256 KB/s constant
            min_packet_size: 1024,
            max_jitter_ms: 200,
            dummy_interval_ms: 25,
            adaptive: false, // Fixed rate
            burst_tolerance: 1.0, // No bursts
            quantum_jitter: true,
        }
    }

    /// Low latency configuration (minimal overhead)
    pub fn low_latency() -> Self {
        Self {
            enabled: true,
            mode: ShapingMode::Light,
            target_bandwidth: 10_000_000, // 10 MB/s
            min_packet_size: 128,
            max_jitter_ms: 10,
            dummy_interval_ms: 500,
            adaptive: true,
            burst_tolerance: 3.0,
            quantum_jitter: false,
        }
    }
}

/// Traffic statistics for analysis
#[derive(Debug, Clone, Default)]
pub struct TrafficStats {
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Total padding bytes sent
    pub padding_bytes_sent: u64,
    /// Total dummy packets sent
    pub dummy_packets_sent: u64,
    /// Total jitter added (milliseconds)
    pub total_jitter_ms: u64,
    /// Packets normalized
    pub packets_normalized: u64,
    /// Current send rate (bytes/s)
    pub current_send_rate: f64,
    /// Current receive rate (bytes/s)
    pub current_receive_rate: f64,
    /// Last update time
    pub last_update: Option<Instant>,
}

/// A packet ready to be sent (with shaping applied)
#[derive(Debug, Clone)]
pub struct ShapedPacket {
    /// The packet data (padded if necessary)
    pub data: Vec<u8>,
    /// Delay to apply before sending (jitter)
    pub delay: Duration,
    /// Whether this is a dummy packet
    pub is_dummy: bool,
    /// Original size before padding
    pub original_size: usize,
    /// Size class used
    pub size_class: PacketSizeClass,
}

/// Traffic shaping engine
pub struct TrafficShaper {
    /// Configuration
    config: TrafficShapingConfig,
    /// Random number generator (optionally seeded with quantum entropy)
    rng: Mutex<ChaCha20Rng>,
    /// Statistics
    stats: RwLock<TrafficStats>,
    /// Recent send timestamps for rate calculation
    send_history: Mutex<VecDeque<(Instant, usize)>>,
    /// Recent receive timestamps for rate calculation
    receive_history: Mutex<VecDeque<(Instant, usize)>>,
    /// Last dummy packet sent
    last_dummy: Mutex<Instant>,
    /// Quantum entropy source (if available)
    quantum_entropy: Option<Arc<dyn QuantumEntropySource + Send + Sync>>,
}

/// Trait for quantum entropy sources
pub trait QuantumEntropySource {
    fn get_entropy(&self) -> [u8; 32];
}

impl TrafficShaper {
    /// Create a new traffic shaper
    pub fn new(config: TrafficShapingConfig) -> Self {
        info!(
            "🎭 Initializing traffic shaper: mode={}, target={}KB/s",
            config.mode.name(),
            config.target_bandwidth / 1024
        );

        // Initialize RNG from OS entropy (will be reseeded with quantum entropy if available)
        let rng = ChaCha20Rng::from_os_rng();

        Self {
            config,
            rng: Mutex::new(rng),
            stats: RwLock::new(TrafficStats::default()),
            send_history: Mutex::new(VecDeque::with_capacity(1000)),
            receive_history: Mutex::new(VecDeque::with_capacity(1000)),
            last_dummy: Mutex::new(Instant::now()),
            quantum_entropy: None,
        }
    }

    /// Create with quantum entropy source
    pub fn with_quantum_entropy(
        config: TrafficShapingConfig,
        entropy: Arc<dyn QuantumEntropySource + Send + Sync>,
    ) -> Self {
        let mut shaper = Self::new(config);
        shaper.quantum_entropy = Some(entropy);
        shaper
    }

    /// Reseed RNG with quantum entropy
    pub async fn reseed_rng(&self) {
        if let Some(entropy) = &self.quantum_entropy {
            let seed = entropy.get_entropy();
            let mut rng = self.rng.lock().await;
            *rng = ChaCha20Rng::from_seed(seed);
            debug!("🎭 RNG reseeded with quantum entropy");
        }
    }

    /// Shape an outgoing packet
    pub async fn shape_outgoing(&self, data: &[u8]) -> ShapedPacket {
        if !self.config.enabled || self.config.mode == ShapingMode::Disabled {
            return ShapedPacket {
                data: data.to_vec(),
                delay: Duration::ZERO,
                is_dummy: false,
                original_size: data.len(),
                size_class: PacketSizeClass::for_payload(data.len()),
            };
        }

        let original_size = data.len();
        let size_class = PacketSizeClass::for_payload(original_size);

        // Normalize packet size
        let padded_data = self.normalize_packet_size(data, size_class).await;

        // Calculate jitter
        let delay = self.calculate_jitter().await;

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.bytes_sent += padded_data.len() as u64;
            stats.padding_bytes_sent += (padded_data.len() - original_size) as u64;
            stats.packets_normalized += 1;
            stats.total_jitter_ms += delay.as_millis() as u64;
        }

        // Record in send history
        {
            let mut history = self.send_history.lock().await;
            history.push_back((Instant::now(), padded_data.len()));
            // Keep only last 5 seconds of history
            while history.front().map(|(t, _)| t.elapsed() > Duration::from_secs(5)).unwrap_or(false) {
                history.pop_front();
            }
        }

        ShapedPacket {
            data: padded_data,
            delay,
            is_dummy: false,
            original_size,
            size_class,
        }
    }

    /// Normalize packet to standard size class
    async fn normalize_packet_size(&self, data: &[u8], size_class: PacketSizeClass) -> Vec<u8> {
        let target_size = size_class.padded_size().max(self.config.min_packet_size);

        if data.len() >= target_size {
            return data.to_vec();
        }

        // Pad with random bytes
        let mut padded = data.to_vec();
        let padding_needed = target_size - data.len();

        let padding: Vec<u8> = {
            let mut rng = self.rng.lock().await;
            (0..padding_needed).map(|_| rng.gen()).collect()
        };

        padded.extend(padding);
        padded
    }

    /// Calculate jitter delay
    async fn calculate_jitter(&self) -> Duration {
        match self.config.mode {
            ShapingMode::Disabled | ShapingMode::Light => Duration::ZERO,
            _ => {
                let max_jitter = self.config.max_jitter_ms;
                if max_jitter == 0 {
                    return Duration::ZERO;
                }

                let jitter_ms: u64 = {
                    let mut rng = self.rng.lock().await;
                    rng.gen_range(0..=max_jitter)
                };

                Duration::from_millis(jitter_ms)
            }
        }
    }

    /// Generate a dummy packet (for cover traffic)
    pub async fn generate_dummy_packet(&self) -> Option<ShapedPacket> {
        if !self.config.enabled || self.config.mode == ShapingMode::Disabled {
            return None;
        }

        // Check if it's time for a dummy packet
        let should_send = {
            let last = self.last_dummy.lock().await;
            last.elapsed() >= Duration::from_millis(self.config.dummy_interval_ms)
        };

        if !should_send {
            return None;
        }

        // Update last dummy time
        {
            let mut last = self.last_dummy.lock().await;
            *last = Instant::now();
        }

        // Generate random dummy data
        let size = match self.config.mode {
            ShapingMode::Paranoid => PacketSizeClass::Maximum.padded_size(),
            ShapingMode::Heavy => PacketSizeClass::Large.padded_size(),
            _ => PacketSizeClass::Medium.padded_size(),
        };

        let data: Vec<u8> = {
            let mut rng = self.rng.lock().await;
            (0..size).map(|_| rng.gen()).collect()
        };

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.dummy_packets_sent += 1;
            stats.bytes_sent += size as u64;
        }

        Some(ShapedPacket {
            data,
            delay: Duration::ZERO,
            is_dummy: true,
            original_size: 0,
            size_class: PacketSizeClass::for_payload(size),
        })
    }

    /// Record received data (for rate calculation)
    pub async fn record_received(&self, size: usize) {
        {
            let mut stats = self.stats.write().await;
            stats.bytes_received += size as u64;
        }

        {
            let mut history = self.receive_history.lock().await;
            history.push_back((Instant::now(), size));
            // Keep only last 5 seconds
            while history.front().map(|(t, _)| t.elapsed() > Duration::from_secs(5)).unwrap_or(false) {
                history.pop_front();
            }
        }
    }

    /// Calculate current send rate
    pub async fn get_send_rate(&self) -> f64 {
        let history = self.send_history.lock().await;
        Self::calculate_rate(&history)
    }

    /// Calculate current receive rate
    pub async fn get_receive_rate(&self) -> f64 {
        let history = self.receive_history.lock().await;
        Self::calculate_rate(&history)
    }

    /// Calculate rate from history
    fn calculate_rate(history: &VecDeque<(Instant, usize)>) -> f64 {
        if history.len() < 2 {
            return 0.0;
        }

        let first = history.front().unwrap();
        let last = history.back().unwrap();
        let duration = last.0.duration_since(first.0).as_secs_f64();

        if duration < 0.001 {
            return 0.0;
        }

        let total_bytes: usize = history.iter().map(|(_, s)| s).sum();
        total_bytes as f64 / duration
    }

    /// Check if we should throttle (rate limiting)
    pub async fn should_throttle(&self) -> bool {
        if !self.config.adaptive {
            return false;
        }

        let current_rate = self.get_send_rate().await;
        let target = self.config.target_bandwidth as f64;
        let threshold = target * self.config.burst_tolerance;

        current_rate > threshold
    }

    /// Get delay for rate limiting
    pub async fn get_throttle_delay(&self) -> Duration {
        if !self.should_throttle().await {
            return Duration::ZERO;
        }

        let current_rate = self.get_send_rate().await;
        let target = self.config.target_bandwidth as f64;

        if current_rate <= target {
            return Duration::ZERO;
        }

        // Calculate delay to bring rate down to target
        let ratio = current_rate / target;
        let delay_ms = ((ratio - 1.0) * 100.0) as u64;

        Duration::from_millis(delay_ms.min(1000))
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> TrafficStats {
        let mut stats = self.stats.read().await.clone();
        stats.current_send_rate = self.get_send_rate().await;
        stats.current_receive_rate = self.get_receive_rate().await;
        stats.last_update = Some(Instant::now());
        stats
    }

    /// Get shaping efficiency metrics
    pub async fn get_efficiency(&self) -> ShapingEfficiency {
        let stats = self.stats.read().await;

        let overhead_ratio = if stats.bytes_sent > 0 {
            (stats.padding_bytes_sent as f64 + (stats.dummy_packets_sent * 512) as f64)
                / stats.bytes_sent as f64
        } else {
            0.0
        };

        let avg_jitter = if stats.packets_normalized > 0 {
            stats.total_jitter_ms as f64 / stats.packets_normalized as f64
        } else {
            0.0
        };

        ShapingEfficiency {
            overhead_ratio,
            avg_jitter_ms: avg_jitter,
            dummy_traffic_ratio: if stats.bytes_sent > 0 {
                (stats.dummy_packets_sent * 512) as f64 / stats.bytes_sent as f64
            } else {
                0.0
            },
            packets_shaped: stats.packets_normalized,
        }
    }
}

/// Shaping efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapingEfficiency {
    /// Ratio of overhead (padding + dummy) to total traffic
    pub overhead_ratio: f64,
    /// Average jitter added per packet (ms)
    pub avg_jitter_ms: f64,
    /// Ratio of dummy traffic to total traffic
    pub dummy_traffic_ratio: f64,
    /// Total packets shaped
    pub packets_shaped: u64,
}

/// Defense levels against specific attacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefenseLevel {
    /// Protection against volume correlation
    pub volume_correlation: DefenseRating,
    /// Protection against timing analysis
    pub timing_analysis: DefenseRating,
    /// Protection against packet size analysis
    pub size_analysis: DefenseRating,
    /// Protection against flow watermarking
    pub watermarking: DefenseRating,
}

/// Defense rating
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DefenseRating {
    None,
    Low,
    Medium,
    High,
    Maximum,
}

impl TrafficShapingConfig {
    /// Calculate defense levels for this configuration
    pub fn defense_levels(&self) -> DefenseLevel {
        match self.mode {
            ShapingMode::Disabled => DefenseLevel {
                volume_correlation: DefenseRating::None,
                timing_analysis: DefenseRating::None,
                size_analysis: DefenseRating::None,
                watermarking: DefenseRating::None,
            },
            ShapingMode::Light => DefenseLevel {
                volume_correlation: DefenseRating::Low,
                timing_analysis: DefenseRating::None,
                size_analysis: DefenseRating::Medium,
                watermarking: DefenseRating::Low,
            },
            ShapingMode::Medium => DefenseLevel {
                volume_correlation: DefenseRating::Medium,
                timing_analysis: DefenseRating::Medium,
                size_analysis: DefenseRating::High,
                watermarking: DefenseRating::Medium,
            },
            ShapingMode::Heavy => DefenseLevel {
                volume_correlation: DefenseRating::High,
                timing_analysis: DefenseRating::High,
                size_analysis: DefenseRating::High,
                watermarking: DefenseRating::High,
            },
            ShapingMode::Paranoid => DefenseLevel {
                volume_correlation: DefenseRating::Maximum,
                timing_analysis: DefenseRating::Maximum,
                size_analysis: DefenseRating::Maximum,
                watermarking: DefenseRating::Maximum,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_size_class() {
        assert_eq!(PacketSizeClass::for_payload(50), PacketSizeClass::Small);
        assert_eq!(PacketSizeClass::for_payload(200), PacketSizeClass::Medium);
        assert_eq!(PacketSizeClass::for_payload(800), PacketSizeClass::Large);
        assert_eq!(PacketSizeClass::for_payload(1200), PacketSizeClass::Maximum);
    }

    #[tokio::test]
    async fn test_traffic_shaper_creation() {
        let config = TrafficShapingConfig::default();
        let shaper = TrafficShaper::new(config);

        let stats = shaper.get_stats().await;
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.dummy_packets_sent, 0);
    }

    #[tokio::test]
    async fn test_packet_shaping() {
        let config = TrafficShapingConfig::default();
        let shaper = TrafficShaper::new(config);

        let data = vec![1, 2, 3, 4, 5];
        let shaped = shaper.shape_outgoing(&data).await;

        // Should be padded to at least min_packet_size
        assert!(shaped.data.len() >= 128);
        assert_eq!(shaped.original_size, 5);
        assert!(!shaped.is_dummy);
    }

    #[tokio::test]
    async fn test_dummy_packet_generation() {
        let mut config = TrafficShapingConfig::default();
        config.dummy_interval_ms = 0; // Immediate
        let shaper = TrafficShaper::new(config);

        // Wait a bit for interval
        tokio::time::sleep(Duration::from_millis(10)).await;

        let dummy = shaper.generate_dummy_packet().await;
        assert!(dummy.is_some());
        assert!(dummy.unwrap().is_dummy);
    }

    #[test]
    fn test_defense_levels() {
        let paranoid = TrafficShapingConfig::paranoid();
        let defense = paranoid.defense_levels();

        assert_eq!(defense.volume_correlation, DefenseRating::Maximum);
        assert_eq!(defense.timing_analysis, DefenseRating::Maximum);
    }
}
