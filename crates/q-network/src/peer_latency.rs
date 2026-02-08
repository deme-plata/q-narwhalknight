//! Peer Latency Tracker for Gossipsub Mesh Scoring
//! v4.3.0-beta: Track per-peer RTT from libp2p ping events,
//! feed into gossipsub peer scoring for latency-weighted mesh selection.

use std::time::{Duration, Instant};
use dashmap::DashMap;
use libp2p::PeerId;
use tracing::{debug, info};

/// Per-peer latency information with EWMA smoothing
#[derive(Debug, Clone)]
pub struct PeerLatencyInfo {
    /// Exponentially-weighted moving average RTT in milliseconds (alpha=0.3)
    pub ewma_rtt_ms: f64,
    /// Jitter (EWMA of |rtt - ewma|)
    pub jitter_ms: f64,
    /// Number of RTT samples received
    pub sample_count: u64,
    /// Last raw RTT in milliseconds
    pub last_rtt_ms: f64,
    /// Last update timestamp
    pub last_updated: Instant,
}

/// Peer latency tracker using EWMA smoothing
pub struct PeerLatencyTracker {
    peers: DashMap<PeerId, PeerLatencyInfo>,
    /// EWMA smoothing factor (0.3 = moderate smoothing)
    alpha: f64,
}

impl PeerLatencyTracker {
    /// Create a new tracker with default alpha=0.3
    pub fn new() -> Self {
        Self {
            peers: DashMap::new(),
            alpha: 0.3,
        }
    }

    /// Update RTT measurement for a peer
    pub fn update_rtt(&self, peer_id: &PeerId, rtt: Duration) {
        let rtt_ms = rtt.as_secs_f64() * 1000.0;
        let now = Instant::now();

        self.peers.entry(*peer_id)
            .and_modify(|info| {
                let deviation = (rtt_ms - info.ewma_rtt_ms).abs();
                info.ewma_rtt_ms = self.alpha * rtt_ms + (1.0 - self.alpha) * info.ewma_rtt_ms;
                info.jitter_ms = self.alpha * deviation + (1.0 - self.alpha) * info.jitter_ms;
                info.sample_count += 1;
                info.last_rtt_ms = rtt_ms;
                info.last_updated = now;
            })
            .or_insert_with(|| PeerLatencyInfo {
                ewma_rtt_ms: rtt_ms,
                jitter_ms: 0.0,
                sample_count: 1,
                last_rtt_ms: rtt_ms,
                last_updated: now,
            });
    }

    /// Get gossipsub application score for a peer based on latency
    /// Maps RTT to score range [-1000.0, 1000.0]
    /// <10ms ≈ 980, 100ms ≈ 800, 250ms ≈ 500, 500ms+ ≈ 0
    pub fn get_score(&self, peer_id: &PeerId) -> f64 {
        match self.peers.get(peer_id) {
            Some(info) => {
                let normalized = (info.ewma_rtt_ms / 500.0).min(2.0);
                (1000.0 * (1.0 - normalized)).max(-1000.0)
            }
            None => 0.0, // Unknown peer gets neutral score
        }
    }

    /// Get EWMA RTT in milliseconds for a peer
    pub fn get_rtt(&self, peer_id: &PeerId) -> Option<f64> {
        self.peers.get(peer_id).map(|info| info.ewma_rtt_ms)
    }

    /// Get all peer scores for periodic gossipsub updates
    pub fn get_all_scores(&self) -> Vec<(PeerId, f64)> {
        self.peers.iter().map(|entry| {
            let peer_id = *entry.key();
            let normalized = (entry.value().ewma_rtt_ms / 500.0).min(2.0);
            let score = (1000.0 * (1.0 - normalized)).max(-1000.0);
            (peer_id, score)
        }).collect()
    }

    /// Remove a disconnected peer
    pub fn remove_peer(&self, peer_id: &PeerId) {
        self.peers.remove(peer_id);
    }

    /// Get number of tracked peers
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Get full latency info for a peer
    pub fn get_info(&self, peer_id: &PeerId) -> Option<PeerLatencyInfo> {
        self.peers.get(peer_id).map(|r| r.value().clone())
    }
}

impl Default for PeerLatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Bootstrap Health-Check Cache
// ============================================================================

/// Health information for a bootstrap endpoint
#[derive(Debug, Clone)]
pub struct BootstrapHealth {
    pub last_rtt_ms: f64,
    pub success_count: u64,
    pub failure_count: u64,
    pub last_peer_id: Option<String>,
    pub last_checked: Instant,
}

/// Cache of bootstrap endpoint health for ordering
pub struct BootstrapHealthCache {
    endpoints: DashMap<String, BootstrapHealth>,
}

impl BootstrapHealthCache {
    pub fn new() -> Self {
        Self {
            endpoints: DashMap::new(),
        }
    }

    /// Record a successful health check
    pub fn record_success(&self, endpoint: &str, rtt: Duration, peer_id: &str) {
        let rtt_ms = rtt.as_secs_f64() * 1000.0;
        self.endpoints.entry(endpoint.to_string())
            .and_modify(|h| {
                // EWMA smoothing on RTT
                h.last_rtt_ms = 0.3 * rtt_ms + 0.7 * h.last_rtt_ms;
                h.success_count += 1;
                h.last_peer_id = Some(peer_id.to_string());
                h.last_checked = Instant::now();
            })
            .or_insert_with(|| BootstrapHealth {
                last_rtt_ms: rtt_ms,
                success_count: 1,
                failure_count: 0,
                last_peer_id: Some(peer_id.to_string()),
                last_checked: Instant::now(),
            });
    }

    /// Record a failed health check
    pub fn record_failure(&self, endpoint: &str) {
        self.endpoints.entry(endpoint.to_string())
            .and_modify(|h| {
                h.failure_count += 1;
                h.last_checked = Instant::now();
            })
            .or_insert_with(|| BootstrapHealth {
                last_rtt_ms: f64::MAX,
                success_count: 0,
                failure_count: 1,
                last_peer_id: None,
                last_checked: Instant::now(),
            });
    }

    /// Get endpoints sorted by health (lowest RTT + highest success rate first)
    pub fn get_ordered_endpoints(&self, endpoints: &[&str]) -> Vec<String> {
        let mut scored: Vec<(String, f64)> = endpoints.iter().map(|ep| {
            let score = match self.endpoints.get(*ep) {
                Some(h) => {
                    let total = (h.success_count + h.failure_count) as f64;
                    let success_rate = if total > 0.0 { h.success_count as f64 / total } else { 0.5 };
                    // Lower score = better. Combine RTT and failure penalty
                    h.last_rtt_ms + (1.0 - success_rate) * 10000.0
                }
                None => 5000.0, // Unknown endpoints get middle priority
            };
            (ep.to_string(), score)
        }).collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(ep, _)| ep).collect()
    }

    /// Get cached peer ID for an endpoint
    pub fn get_cached_peer_id(&self, endpoint: &str) -> Option<String> {
        self.endpoints.get(endpoint).and_then(|h| h.last_peer_id.clone())
    }
}

impl Default for BootstrapHealthCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma_convergence() {
        let tracker = PeerLatencyTracker::new();
        let peer = PeerId::random();

        // Send 10 samples at 100ms
        for _ in 0..10 {
            tracker.update_rtt(&peer, Duration::from_millis(100));
        }

        let rtt = tracker.get_rtt(&peer).unwrap();
        assert!((rtt - 100.0).abs() < 1.0, "EWMA should converge to 100ms, got {}", rtt);
    }

    #[test]
    fn test_ewma_smoothing() {
        let tracker = PeerLatencyTracker::new();
        let peer = PeerId::random();

        // Establish baseline at 50ms
        for _ in 0..20 {
            tracker.update_rtt(&peer, Duration::from_millis(50));
        }

        // Spike to 200ms - EWMA should not jump to 200
        tracker.update_rtt(&peer, Duration::from_millis(200));
        let rtt = tracker.get_rtt(&peer).unwrap();
        assert!(rtt < 100.0, "EWMA should smooth spike, got {}", rtt);
    }

    #[test]
    fn test_score_mapping() {
        let tracker = PeerLatencyTracker::new();

        let fast_peer = PeerId::random();
        tracker.update_rtt(&fast_peer, Duration::from_millis(10));
        let fast_score = tracker.get_score(&fast_peer);
        assert!(fast_score > 900.0, "10ms peer should score >900, got {}", fast_score);

        let medium_peer = PeerId::random();
        tracker.update_rtt(&medium_peer, Duration::from_millis(250));
        let medium_score = tracker.get_score(&medium_peer);
        assert!(medium_score > 400.0 && medium_score < 600.0,
            "250ms peer should score 400-600, got {}", medium_score);

        let slow_peer = PeerId::random();
        tracker.update_rtt(&slow_peer, Duration::from_millis(500));
        let slow_score = tracker.get_score(&slow_peer);
        assert!(slow_score.abs() < 50.0, "500ms peer should score ~0, got {}", slow_score);
    }

    #[test]
    fn test_jitter_tracking() {
        let tracker = PeerLatencyTracker::new();
        let peer = PeerId::random();

        // Alternating RTTs should produce non-zero jitter
        for i in 0..20 {
            let rtt = if i % 2 == 0 { 50 } else { 150 };
            tracker.update_rtt(&peer, Duration::from_millis(rtt));
        }

        let info = tracker.get_info(&peer).unwrap();
        assert!(info.jitter_ms > 10.0, "Alternating RTTs should produce jitter, got {}", info.jitter_ms);
    }

    #[test]
    fn test_unknown_peer_neutral_score() {
        let tracker = PeerLatencyTracker::new();
        let unknown = PeerId::random();
        assert_eq!(tracker.get_score(&unknown), 0.0);
    }

    #[test]
    fn test_remove_peer() {
        let tracker = PeerLatencyTracker::new();
        let peer = PeerId::random();

        tracker.update_rtt(&peer, Duration::from_millis(100));
        assert_eq!(tracker.peer_count(), 1);

        tracker.remove_peer(&peer);
        assert_eq!(tracker.peer_count(), 0);
        assert!(tracker.get_rtt(&peer).is_none());
    }

    #[test]
    fn test_bootstrap_health_ordering() {
        let cache = BootstrapHealthCache::new();

        // Fast endpoint
        cache.record_success("http://fast:8080", Duration::from_millis(20), "peer1");
        // Slow endpoint
        cache.record_success("http://slow:8080", Duration::from_millis(200), "peer2");
        // Failed endpoint
        cache.record_failure("http://down:8080");

        let ordered = cache.get_ordered_endpoints(&[
            "http://down:8080",
            "http://slow:8080",
            "http://fast:8080",
        ]);

        assert_eq!(ordered[0], "http://fast:8080");
        assert_eq!(ordered[1], "http://slow:8080");
        assert_eq!(ordered[2], "http://down:8080");
    }
}
