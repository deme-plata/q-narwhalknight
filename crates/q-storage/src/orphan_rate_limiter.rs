//! Orphan Block Rate Limiter - DAG Spam Attack Prevention (v1.4.5-beta)
//!
//! Prevents DoS attacks where malicious peers spam blocks with missing parents
//! to consume resources (storage, CPU for validation, network bandwidth).
//!
//! ## Attack Vectors Mitigated:
//!
//! 1. **Orphan Flood Attack**: Spamming blocks with non-existent parents
//!    - Forces node to store blocks in pending queue
//!    - Wastes memory and CPU for validation
//!    - Prevents legitimate blocks from being processed
//!
//! 2. **Parent Withholding Attack**: Sending child blocks before parents
//!    - Creates artificial orphans that fill pending queue
//!    - Then never sending the parent blocks
//!
//! 3. **Circular Parent Attack**: Creating blocks that reference each other
//!    - Infinite validation loops
//!    - Cycle detection in causal validator helps, but rate limiting is first defense
//!
//! ## Defense Strategy:
//!
//! 1. **Per-peer sliding window rate limiting**
//!    - Track orphans per peer in 60-second windows
//!    - Threshold: 10 orphans/minute = warning, 50/minute = temporary ban
//!
//! 2. **Global orphan queue limits**
//!    - Maximum 10,000 pending orphans total
//!    - FIFO eviction when exceeded
//!
//! 3. **Integration with PeerTrustRegistry**
//!    - Excessive orphans reduce peer trust score
//!    - Banned peers are disconnected

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Orphan event for rate tracking
#[derive(Debug, Clone)]
struct OrphanEvent {
    /// When this orphan was received
    timestamp: Instant,
    /// Block hash (for deduplication)
    block_hash: String,
    /// Missing parent count
    missing_parents: usize,
}

/// Per-peer orphan tracking
#[derive(Debug)]
struct PeerOrphanStats {
    /// Sliding window of orphan events
    recent_orphans: VecDeque<OrphanEvent>,
    /// Total orphans ever received from this peer
    total_orphans: u64,
    /// Total valid blocks received (for ratio calculation)
    total_valid: u64,
    /// When peer was last warned
    last_warning: Option<Instant>,
    /// When peer was banned (temporary)
    banned_until: Option<Instant>,
}

impl Default for PeerOrphanStats {
    fn default() -> Self {
        Self {
            recent_orphans: VecDeque::with_capacity(100),
            total_orphans: 0,
            total_valid: 0,
            last_warning: None,
            banned_until: None,
        }
    }
}

/// Rate limiting thresholds
#[derive(Debug, Clone)]
pub struct OrphanRateLimits {
    /// Window size for rate calculation
    pub window_duration: Duration,
    /// Orphans per window before warning
    pub warning_threshold: usize,
    /// Orphans per window before temporary ban
    pub ban_threshold: usize,
    /// Duration of temporary ban
    pub ban_duration: Duration,
    /// Maximum global pending orphans
    pub max_global_orphans: usize,
    /// Trust penalty per orphan
    pub trust_penalty_per_orphan: f64,
}

impl Default for OrphanRateLimits {
    fn default() -> Self {
        Self {
            window_duration: Duration::from_secs(60),      // 1 minute window
            warning_threshold: 10,                          // 10 orphans/minute = warning
            ban_threshold: 50,                              // 50 orphans/minute = ban
            ban_duration: Duration::from_secs(300),        // 5 minute ban
            max_global_orphans: 10_000,                     // Global limit
            trust_penalty_per_orphan: 0.01,                // -1% trust per orphan
        }
    }
}

/// Orphan rate limiting result
#[derive(Debug, Clone, PartialEq)]
pub enum OrphanRateResult {
    /// Block accepted (within limits)
    Accepted,
    /// Rate limit warning issued
    Warning { orphans_per_minute: f64 },
    /// Peer temporarily banned
    Banned { until: Instant, reason: String },
    /// Global queue full
    GlobalQueueFull { current_size: usize },
}

/// Orphan rate limiter for DAG spam prevention
pub struct OrphanRateLimiter {
    /// Per-peer statistics
    peer_stats: HashMap<String, PeerOrphanStats>,
    /// Rate limiting configuration
    limits: OrphanRateLimits,
    /// Global orphan count (across all peers)
    global_orphan_count: usize,
    /// Global orphan hashes (for FIFO eviction)
    global_orphan_queue: VecDeque<(String, String)>, // (block_hash, peer_id)
}

impl OrphanRateLimiter {
    /// Create new rate limiter with default limits
    pub fn new() -> Self {
        Self {
            peer_stats: HashMap::new(),
            limits: OrphanRateLimits::default(),
            global_orphan_count: 0,
            global_orphan_queue: VecDeque::with_capacity(10_000),
        }
    }

    /// Create with custom limits
    pub fn with_limits(limits: OrphanRateLimits) -> Self {
        Self {
            peer_stats: HashMap::new(),
            limits,
            global_orphan_count: 0,
            global_orphan_queue: VecDeque::with_capacity(10_000),
        }
    }

    /// Record an orphan block from a peer
    ///
    /// Returns rate limiting decision and optional trust penalty
    pub fn record_orphan(
        &mut self,
        peer_id: &str,
        block_hash: &str,
        missing_parents: usize,
    ) -> (OrphanRateResult, f64) {
        let now = Instant::now();

        // Check global queue limit
        if self.global_orphan_count >= self.limits.max_global_orphans {
            // Evict oldest orphan (FIFO)
            if let Some((old_hash, old_peer)) = self.global_orphan_queue.pop_front() {
                if let Some(stats) = self.peer_stats.get_mut(&old_peer) {
                    stats.recent_orphans.retain(|e| e.block_hash != old_hash);
                }
                self.global_orphan_count = self.global_orphan_count.saturating_sub(1);
            }
        }

        // Get or create peer stats
        let stats = self.peer_stats.entry(peer_id.to_string()).or_default();

        // Check if peer is currently banned
        if let Some(banned_until) = stats.banned_until {
            if now < banned_until {
                return (
                    OrphanRateResult::Banned {
                        until: banned_until,
                        reason: format!("Excessive orphan rate from peer {}", &peer_id[..8.min(peer_id.len())]),
                    },
                    0.0, // No additional penalty while banned
                );
            } else {
                // Ban expired
                stats.banned_until = None;
            }
        }

        // Prune old events outside window
        let cutoff = now - self.limits.window_duration;
        stats.recent_orphans.retain(|e| e.timestamp > cutoff);

        // Record new orphan event
        let event = OrphanEvent {
            timestamp: now,
            block_hash: block_hash.to_string(),
            missing_parents,
        };
        stats.recent_orphans.push_back(event);
        stats.total_orphans += 1;
        self.global_orphan_count += 1;
        self.global_orphan_queue.push_back((block_hash.to_string(), peer_id.to_string()));

        // Calculate current rate
        let orphan_count = stats.recent_orphans.len();
        let window_secs = self.limits.window_duration.as_secs_f64();
        let orphans_per_minute = (orphan_count as f64 / window_secs) * 60.0;

        // Calculate trust penalty
        let trust_penalty = self.limits.trust_penalty_per_orphan * missing_parents as f64;

        // Check thresholds
        if orphan_count >= self.limits.ban_threshold {
            // Temporary ban
            let ban_until = now + self.limits.ban_duration;
            stats.banned_until = Some(ban_until);

            warn!(
                "🚫 [ORPHAN RATE] Banning peer {} for {} seconds - {} orphans in last {:?}",
                &peer_id[..8.min(peer_id.len())],
                self.limits.ban_duration.as_secs(),
                orphan_count,
                self.limits.window_duration
            );

            return (
                OrphanRateResult::Banned {
                    until: ban_until,
                    reason: format!(
                        "Exceeded {} orphans/minute (current: {:.1})",
                        self.limits.ban_threshold, orphans_per_minute
                    ),
                },
                trust_penalty * 10.0, // Extra penalty for triggering ban
            );
        } else if orphan_count >= self.limits.warning_threshold {
            // Issue warning (at most once per 30 seconds)
            let should_warn = stats.last_warning
                .map(|t| now.duration_since(t) > Duration::from_secs(30))
                .unwrap_or(true);

            if should_warn {
                stats.last_warning = Some(now);
                info!(
                    "⚠️  [ORPHAN RATE] Warning peer {} - {} orphans in last {:?} ({:.1}/min)",
                    &peer_id[..8.min(peer_id.len())],
                    orphan_count,
                    self.limits.window_duration,
                    orphans_per_minute
                );
            }

            return (
                OrphanRateResult::Warning { orphans_per_minute },
                trust_penalty * 2.0, // Extra penalty for warning threshold
            );
        }

        // Normal operation
        debug!(
            "📦 [ORPHAN RATE] Recorded orphan from {} - {}/{} in window",
            &peer_id[..8.min(peer_id.len())],
            orphan_count,
            self.limits.warning_threshold
        );

        (OrphanRateResult::Accepted, trust_penalty)
    }

    /// Record a valid block from a peer (for ratio tracking)
    pub fn record_valid_block(&mut self, peer_id: &str) {
        let stats = self.peer_stats.entry(peer_id.to_string()).or_default();
        stats.total_valid += 1;
    }

    /// Get orphan/valid ratio for a peer
    pub fn get_orphan_ratio(&self, peer_id: &str) -> Option<f64> {
        self.peer_stats.get(peer_id).map(|stats| {
            if stats.total_valid == 0 {
                if stats.total_orphans == 0 {
                    0.0
                } else {
                    1.0 // All orphans, no valid
                }
            } else {
                stats.total_orphans as f64 / (stats.total_orphans + stats.total_valid) as f64
            }
        })
    }

    /// Check if peer is currently banned
    pub fn is_banned(&self, peer_id: &str) -> bool {
        self.peer_stats.get(peer_id).map(|stats| {
            stats.banned_until.map(|t| Instant::now() < t).unwrap_or(false)
        }).unwrap_or(false)
    }

    /// Get current orphan rate for a peer (orphans per minute)
    pub fn get_current_rate(&self, peer_id: &str) -> f64 {
        self.peer_stats.get(peer_id).map(|stats| {
            let now = Instant::now();
            let cutoff = now - self.limits.window_duration;
            let count = stats.recent_orphans.iter().filter(|e| e.timestamp > cutoff).count();
            (count as f64 / self.limits.window_duration.as_secs_f64()) * 60.0
        }).unwrap_or(0.0)
    }

    /// Get global orphan queue size
    pub fn global_queue_size(&self) -> usize {
        self.global_orphan_count
    }

    /// Clear expired bans
    pub fn clear_expired_bans(&mut self) {
        let now = Instant::now();
        for stats in self.peer_stats.values_mut() {
            if stats.banned_until.map(|t| now >= t).unwrap_or(false) {
                stats.banned_until = None;
            }
        }
    }

    /// Get statistics for monitoring
    pub fn get_stats(&self) -> OrphanRateLimiterStats {
        let now = Instant::now();
        let cutoff = now - self.limits.window_duration;

        let banned_peers = self.peer_stats.values()
            .filter(|s| s.banned_until.map(|t| now < t).unwrap_or(false))
            .count();

        let warned_peers = self.peer_stats.values()
            .filter(|s| s.recent_orphans.len() >= self.limits.warning_threshold)
            .count();

        let total_recent_orphans: usize = self.peer_stats.values()
            .map(|s| s.recent_orphans.iter().filter(|e| e.timestamp > cutoff).count())
            .sum();

        OrphanRateLimiterStats {
            tracked_peers: self.peer_stats.len(),
            banned_peers,
            warned_peers,
            global_orphan_count: self.global_orphan_count,
            recent_orphans_in_window: total_recent_orphans,
        }
    }
}

impl Default for OrphanRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for monitoring orphan rate limiter
#[derive(Debug, Clone)]
pub struct OrphanRateLimiterStats {
    /// Number of peers being tracked
    pub tracked_peers: usize,
    /// Number of currently banned peers
    pub banned_peers: usize,
    /// Number of peers at warning level
    pub warned_peers: usize,
    /// Total orphans in global queue
    pub global_orphan_count: usize,
    /// Orphans received in current window
    pub recent_orphans_in_window: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_orphan_rate() {
        let mut limiter = OrphanRateLimiter::new();

        // A few orphans should be accepted
        for i in 0..5 {
            let (result, penalty) = limiter.record_orphan(
                "peer1",
                &format!("block_{}", i),
                1,
            );
            assert_eq!(result, OrphanRateResult::Accepted);
            assert!(penalty < 0.1);
        }

        let stats = limiter.get_stats();
        assert_eq!(stats.banned_peers, 0);
        assert_eq!(stats.warned_peers, 0);
    }

    #[test]
    fn test_warning_threshold() {
        let mut limiter = OrphanRateLimiter::new();

        // Trigger warning threshold
        for i in 0..15 {
            let (result, _) = limiter.record_orphan(
                "peer2",
                &format!("block_{}", i),
                1,
            );
            if i >= 10 {
                assert!(matches!(result, OrphanRateResult::Warning { .. }));
            }
        }

        let stats = limiter.get_stats();
        assert_eq!(stats.warned_peers, 1);
        assert_eq!(stats.banned_peers, 0);
    }

    #[test]
    fn test_ban_threshold() {
        let mut limiter = OrphanRateLimiter::new();

        // Trigger ban threshold
        for i in 0..60 {
            let (result, _) = limiter.record_orphan(
                "peer3",
                &format!("block_{}", i),
                1,
            );
            if i >= 50 {
                assert!(matches!(result, OrphanRateResult::Banned { .. }));
            }
        }

        assert!(limiter.is_banned("peer3"));

        let stats = limiter.get_stats();
        assert_eq!(stats.banned_peers, 1);
    }

    #[test]
    fn test_orphan_ratio() {
        let mut limiter = OrphanRateLimiter::new();

        // 10 valid, 5 orphans = 33% orphan ratio
        for _ in 0..10 {
            limiter.record_valid_block("peer4");
        }
        for i in 0..5 {
            limiter.record_orphan("peer4", &format!("orphan_{}", i), 1);
        }

        let ratio = limiter.get_orphan_ratio("peer4").unwrap();
        assert!((ratio - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_global_queue_limit() {
        let mut limiter = OrphanRateLimiter::with_limits(OrphanRateLimits {
            max_global_orphans: 100,
            ..Default::default()
        });

        // Exceed global limit
        for i in 0..150 {
            limiter.record_orphan(&format!("peer_{}", i % 10), &format!("block_{}", i), 1);
        }

        // Queue should be capped at limit
        assert!(limiter.global_queue_size() <= 100);
    }
}
