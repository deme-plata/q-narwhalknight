/// 🚀 Project APOLLO Phase 4: SLINGSHOT - Hot Cache Peer Selection (GRAVITY ASSIST)
///
/// Peer momentum tracking for optimal sync performance:
/// - Track which height ranges each peer has recently served
/// - Prefer peers with "hot cache" for adjacent blocks
/// - Like planetary gravity assists, use peer momentum to accelerate sync
///
/// Key insight: Peers that just served blocks N-1000 to N likely have blocks N to N+1000
/// in their OS page cache or RocksDB block cache. Exploiting this cache locality
/// can yield 3-5x performance improvement over random peer selection.
///
/// Metrics tracked per peer:
/// - Last served height range
/// - Cache heat (decays exponentially over time)
/// - Bandwidth velocity (bytes/second historical average)
/// - Latency profile (RTT percentiles)

use dashmap::DashMap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Peer ID type (libp2p PeerId wrapper for convenience)
pub type PeerId = String;

/// Peer momentum state for cache-aware selection
#[derive(Clone, Debug)]
pub struct PeerMomentum {
    /// Peer identifier
    pub peer_id: PeerId,

    /// Last height range this peer served
    pub last_served_range: Range<u64>,

    /// When the last range was served
    pub last_served_at: Instant,

    /// Cache heat score (0.0 - 1.0, decays exponentially)
    /// 1.0 = just served, 0.0 = cold cache
    pub cache_heat: f64,

    /// Historical bandwidth (bytes per second, exponential moving average)
    pub bandwidth_velocity: f64,

    /// Latency samples (ms) - last N measurements
    pub latency_samples: Vec<u32>,

    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,

    /// Total blocks served by this peer
    pub blocks_served: u64,

    /// Total bytes served by this peer
    pub bytes_served: u64,
}

impl PeerMomentum {
    /// Create new peer momentum tracker
    pub fn new(peer_id: PeerId) -> Self {
        Self {
            peer_id,
            last_served_range: 0..0,
            last_served_at: Instant::now(),
            cache_heat: 0.0,
            bandwidth_velocity: 0.0,
            latency_samples: Vec::with_capacity(100),
            success_rate: 0.5, // Start neutral
            blocks_served: 0,
            bytes_served: 0,
        }
    }

    /// Update momentum after successful serve
    pub fn record_serve(
        &mut self,
        range: Range<u64>,
        bytes: u64,
        latency_ms: u32,
    ) {
        let now = Instant::now();
        let blocks = range.end.saturating_sub(range.start);

        // Update range tracking
        self.last_served_range = range;
        self.last_served_at = now;

        // Set cache heat to maximum (will decay)
        self.cache_heat = 1.0;

        // Update bandwidth with exponential moving average (alpha = 0.2)
        let elapsed_secs = latency_ms as f64 / 1000.0;
        if elapsed_secs > 0.0 {
            let current_bandwidth = bytes as f64 / elapsed_secs;
            self.bandwidth_velocity = 0.8 * self.bandwidth_velocity + 0.2 * current_bandwidth;
        }

        // Add latency sample (keep last 100)
        if self.latency_samples.len() >= 100 {
            self.latency_samples.remove(0);
        }
        self.latency_samples.push(latency_ms);

        // Update success rate (EMA alpha = 0.1)
        self.success_rate = 0.9 * self.success_rate + 0.1 * 1.0;

        // Update totals
        self.blocks_served += blocks;
        self.bytes_served += bytes;

        debug!(
            "📊 [GRAVITY ASSIST] Peer {} served {}-{} ({} blocks, {:.1} KB) in {}ms, heat={:.2}, bw={:.1} MB/s",
            self.peer_id,
            self.last_served_range.start,
            self.last_served_range.end,
            blocks,
            bytes as f64 / 1024.0,
            latency_ms,
            self.cache_heat,
            self.bandwidth_velocity / 1_000_000.0
        );
    }

    /// Record a failure
    pub fn record_failure(&mut self) {
        // Reduce success rate (EMA alpha = 0.1)
        self.success_rate = 0.9 * self.success_rate + 0.1 * 0.0;

        // Cool down cache heat faster on failure
        self.cache_heat *= 0.5;
    }

    /// Calculate current cache heat (with time decay)
    /// Half-life: 30 seconds
    pub fn current_heat(&self) -> f64 {
        let elapsed = self.last_served_at.elapsed().as_secs_f64();
        let half_life = 30.0; // seconds
        self.cache_heat * 0.5_f64.powf(elapsed / half_life)
    }

    /// Calculate cache proximity score for a target range
    /// Higher score = peer more likely to have this range in cache
    pub fn cache_proximity(&self, target: &Range<u64>) -> f64 {
        let heat = self.current_heat();
        if heat < 0.01 {
            return 0.0; // Cold cache, no proximity bonus
        }

        // Calculate distance from last served range to target
        let last_end = self.last_served_range.end;
        let target_start = target.start;

        // Ideal: target starts right after last served
        let distance = if target_start >= last_end {
            target_start - last_end
        } else if target_start >= self.last_served_range.start {
            0 // Overlapping range, excellent
        } else {
            self.last_served_range.start - target_start // Before our range
        };

        // Score decreases with distance (optimal is 0)
        // At distance 1000, score is ~0.37 of heat
        let proximity = (-((distance as f64) / 1000.0)).exp();

        heat * proximity
    }

    /// Calculate composite score for peer selection
    pub fn selection_score(&self, target: &Range<u64>) -> f64 {
        let cache_score = self.cache_proximity(target);
        let bandwidth_score = (self.bandwidth_velocity / 100_000_000.0).min(1.0); // Normalize to 100 MB/s max
        let reliability_score = self.success_rate;
        let latency_score = self.latency_score();

        // Weighted combination
        // Cache proximity is king for sequential sync
        0.5 * cache_score
            + 0.25 * bandwidth_score
            + 0.15 * reliability_score
            + 0.10 * latency_score
    }

    /// Calculate latency score (0-1, higher is better/lower latency)
    fn latency_score(&self) -> f64 {
        if self.latency_samples.is_empty() {
            return 0.5; // Neutral
        }

        let avg: f64 = self.latency_samples.iter().map(|&x| x as f64).sum::<f64>()
            / self.latency_samples.len() as f64;

        // Score: 1.0 at 10ms, 0.5 at 100ms, ~0 at 1000ms
        1.0 / (1.0 + avg / 100.0)
    }

    /// Get P50/P90/P99 latency
    pub fn latency_percentiles(&self) -> (u32, u32, u32) {
        if self.latency_samples.is_empty() {
            return (0, 0, 0);
        }

        let mut sorted: Vec<u32> = self.latency_samples.clone();
        sorted.sort_unstable();

        let p50 = sorted[sorted.len() / 2];
        let p90 = sorted[(sorted.len() * 90) / 100];
        let p99 = sorted[(sorted.len() * 99) / 100];

        (p50, p90, p99)
    }
}

/// Peer momentum manager (GRAVITY ASSIST mission control)
pub struct PeerMomentumManager {
    /// Per-peer momentum tracking
    peers: DashMap<PeerId, PeerMomentum>,

    /// Maximum peers to track
    max_peers: usize,

    /// Minimum cache heat threshold for "hot" classification
    hot_threshold: f64,
}

impl PeerMomentumManager {
    /// Create new momentum manager
    pub fn new() -> Self {
        Self {
            peers: DashMap::new(),
            max_peers: 1000,
            hot_threshold: 0.1,
        }
    }

    /// Create with custom settings
    pub fn with_config(max_peers: usize, hot_threshold: f64) -> Self {
        Self {
            peers: DashMap::new(),
            max_peers,
            hot_threshold,
        }
    }

    /// Record successful serve from peer
    pub fn record_serve(
        &self,
        peer_id: &str,
        range: Range<u64>,
        bytes: u64,
        latency_ms: u32,
    ) {
        self.peers
            .entry(peer_id.to_string())
            .or_insert_with(|| PeerMomentum::new(peer_id.to_string()))
            .record_serve(range, bytes, latency_ms);

        // Prune if too many peers
        if self.peers.len() > self.max_peers {
            self.prune_cold_peers();
        }
    }

    /// v8.4.0: Seed a peer's bandwidth_velocity from handshake-reported tier
    /// Called once after handshake completes. Only seeds if no measurements yet (cold start).
    /// This gives gravity-assist an initial bandwidth signal before any actual transfers.
    pub fn seed_bandwidth_from_handshake(&self, peer_id: &str, bandwidth_mbps: u32) {
        if bandwidth_mbps == 0 {
            return; // Unknown bandwidth, don't seed
        }
        // Convert Mbps to bytes/sec, conservative 70% of reported
        let bps = bandwidth_mbps as f64 * 1_000_000.0 / 8.0 * 0.7;
        self.peers
            .entry(peer_id.to_string())
            .and_modify(|m| {
                if m.bandwidth_velocity == 0.0 {
                    // Only seed if no measurements yet (cold start)
                    m.bandwidth_velocity = bps;
                    info!(
                        "🌍 [GRAVITY ASSIST] Seeded peer {} bandwidth from handshake: {} Mbps → {:.1} MB/s",
                        &peer_id[..peer_id.len().min(12)],
                        bandwidth_mbps,
                        bps / 1_000_000.0
                    );
                }
            })
            .or_insert_with(|| {
                let mut m = PeerMomentum::new(peer_id.to_string());
                m.bandwidth_velocity = bps;
                info!(
                    "🌍 [GRAVITY ASSIST] Seeded new peer {} bandwidth from handshake: {} Mbps → {:.1} MB/s",
                    &peer_id[..peer_id.len().min(12)],
                    bandwidth_mbps,
                    bps / 1_000_000.0
                );
                m
            });
    }

    /// Record failure from peer
    pub fn record_failure(&self, peer_id: &str) {
        if let Some(mut momentum) = self.peers.get_mut(peer_id) {
            momentum.record_failure();
        }
    }

    /// Select best peer for target range (GRAVITY ASSIST)
    pub fn select_best_peer(&self, target: &Range<u64>, available: &[&str]) -> Option<String> {
        if available.is_empty() {
            return None;
        }

        let mut best_peer: Option<(String, f64)> = None;

        for peer_id in available {
            let score = if let Some(momentum) = self.peers.get(*peer_id) {
                momentum.selection_score(target)
            } else {
                0.1 // Small base score for unknown peers (give them a chance)
            };

            match &best_peer {
                None => best_peer = Some((peer_id.to_string(), score)),
                Some((_, best_score)) if score > *best_score => {
                    best_peer = Some((peer_id.to_string(), score));
                }
                _ => {}
            }
        }

        if let Some((peer, score)) = &best_peer {
            debug!(
                "🎯 [GRAVITY ASSIST] Selected peer {} with score {:.3} for range {}-{}",
                peer, score, target.start, target.end
            );
        }

        best_peer.map(|(peer, _)| peer)
    }

    /// Get list of "hot" peers (cache heat above threshold)
    pub fn get_hot_peers(&self) -> Vec<(String, f64)> {
        self.peers
            .iter()
            .filter_map(|entry| {
                let heat = entry.value().current_heat();
                if heat >= self.hot_threshold {
                    Some((entry.key().clone(), heat))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get peer momentum for specific peer
    pub fn get_peer_momentum(&self, peer_id: &str) -> Option<PeerMomentum> {
        self.peers.get(peer_id).map(|p| p.clone())
    }

    /// Get all peer stats for monitoring
    pub fn get_all_stats(&self) -> Vec<PeerStats> {
        self.peers
            .iter()
            .map(|entry| {
                let p = entry.value();
                let (p50, p90, p99) = p.latency_percentiles();
                // v8.6.2: Classify bandwidth tier from measured velocity (bytes/s → Mbps)
                let bw_mbps = p.bandwidth_velocity / 1_000_000.0;
                let tier_label = if bw_mbps >= 625.0 {      // 5000 Mbps = 625 MB/s
                    "SUPERNODE"
                } else if bw_mbps >= 62.5 {                 // 500 Mbps = 62.5 MB/s
                    "STANDARD"
                } else if bw_mbps > 0.0 {
                    "FALLBACK"
                } else {
                    "UNKNOWN"
                };
                PeerStats {
                    peer_id: p.peer_id.clone(),
                    cache_heat: p.current_heat(),
                    bandwidth_mbps: bw_mbps,
                    success_rate: p.success_rate,
                    latency_p50: p50,
                    latency_p90: p90,
                    latency_p99: p99,
                    blocks_served: p.blocks_served,
                    bytes_served: p.bytes_served,
                    last_range: format!("{}-{}", p.last_served_range.start, p.last_served_range.end),
                    bandwidth_tier: tier_label.to_string(),
                }
            })
            .collect()
    }

    /// Prune peers with cold cache
    fn prune_cold_peers(&self) {
        let cold_threshold = 0.01;
        let mut to_remove = Vec::new();

        for entry in self.peers.iter() {
            if entry.value().current_heat() < cold_threshold {
                to_remove.push(entry.key().clone());
            }
        }

        for peer_id in to_remove {
            self.peers.remove(&peer_id);
        }

        info!(
            "🧹 [GRAVITY ASSIST] Pruned cold peers, {} remaining",
            self.peers.len()
        );
    }

    /// Get total peers tracked
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Clear all peer data
    pub fn clear(&self) {
        self.peers.clear();
    }
}

impl Default for PeerMomentumManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Peer statistics for monitoring/display
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerStats {
    pub peer_id: String,
    pub cache_heat: f64,
    pub bandwidth_mbps: f64,
    pub success_rate: f64,
    pub latency_p50: u32,
    pub latency_p90: u32,
    pub latency_p99: u32,
    pub blocks_served: u64,
    pub bytes_served: u64,
    pub last_range: String,
    /// v8.6.2: Bandwidth tier classification ("SUPERNODE", "STANDARD", "FALLBACK", "UNKNOWN")
    pub bandwidth_tier: String,
}

/// Request-level peer selection with gravity assist
pub struct GravityAssistedSelector {
    /// Momentum manager
    manager: Arc<PeerMomentumManager>,

    /// Minimum peers to consider (for load balancing)
    min_candidates: usize,

    /// Random factor for exploration (0.0 - 1.0)
    exploration_factor: f64,
}

impl GravityAssistedSelector {
    pub fn new(manager: Arc<PeerMomentumManager>) -> Self {
        Self {
            manager,
            min_candidates: 3,
            exploration_factor: 0.1,
        }
    }

    /// Select peer with gravity assist, with occasional random exploration
    pub fn select(&self, target: &Range<u64>, available: &[&str]) -> Option<String> {
        // Occasionally explore random peer (for discovery)
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.exploration_factor && !available.is_empty() {
            let idx = rng.gen_range(0..available.len());
            return Some(available[idx].to_string());
        }

        // Use gravity assist for optimal selection
        self.manager.select_best_peer(target, available)
    }

    /// Select multiple peers (for parallel requests)
    pub fn select_multiple(
        &self,
        target: &Range<u64>,
        available: &[&str],
        count: usize,
    ) -> Vec<String> {
        if available.len() <= count {
            return available.iter().map(|s| s.to_string()).collect();
        }

        // Score all peers
        let mut scored: Vec<(String, f64)> = available
            .iter()
            .map(|&peer_id| {
                let score = self
                    .manager
                    .peers
                    .get(peer_id)
                    .map(|m| m.selection_score(target))
                    .unwrap_or(0.1);
                (peer_id.to_string(), score)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N
        scored.into_iter().take(count).map(|(p, _)| p).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_momentum_creation() {
        let momentum = PeerMomentum::new("peer1".to_string());
        assert_eq!(momentum.peer_id, "peer1");
        assert_eq!(momentum.cache_heat, 0.0);
        assert_eq!(momentum.blocks_served, 0);
    }

    #[test]
    fn test_record_serve() {
        let mut momentum = PeerMomentum::new("peer1".to_string());
        momentum.record_serve(100..200, 50_000, 50);

        assert_eq!(momentum.cache_heat, 1.0);
        assert_eq!(momentum.blocks_served, 100);
        assert_eq!(momentum.bytes_served, 50_000);
        assert!(!momentum.latency_samples.is_empty());
    }

    #[test]
    fn test_cache_proximity() {
        let mut momentum = PeerMomentum::new("peer1".to_string());
        momentum.record_serve(100..200, 50_000, 50);

        // Adjacent range should have high proximity
        let adjacent = momentum.cache_proximity(&(200..300));
        assert!(adjacent > 0.9);

        // Distant range should have low proximity
        let distant = momentum.cache_proximity(&(10000..11000));
        assert!(distant < 0.1);
    }

    #[test]
    fn test_peer_momentum_manager() {
        let manager = PeerMomentumManager::new();

        manager.record_serve("peer1", 100..200, 50_000, 50);
        manager.record_serve("peer2", 500..600, 50_000, 100);

        // For range 200-300, peer1 should be preferred
        let available = vec!["peer1", "peer2"];
        let selected = manager.select_best_peer(&(200..300), &available);
        assert_eq!(selected, Some("peer1".to_string()));

        // For range 600-700, peer2 should be preferred
        let selected = manager.select_best_peer(&(600..700), &available);
        assert_eq!(selected, Some("peer2".to_string()));
    }

    #[test]
    fn test_heat_decay() {
        let mut momentum = PeerMomentum::new("peer1".to_string());
        momentum.cache_heat = 1.0;
        momentum.last_served_at = Instant::now() - Duration::from_secs(30);

        // After 30 seconds (one half-life), heat should be ~0.5
        let heat = momentum.current_heat();
        assert!((heat - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_hot_peers() {
        let manager = PeerMomentumManager::new();

        manager.record_serve("hot_peer", 100..200, 50_000, 50);

        let hot = manager.get_hot_peers();
        assert!(!hot.is_empty());
        assert_eq!(hot[0].0, "hot_peer");
    }
}
