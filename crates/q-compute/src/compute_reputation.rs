#![allow(dead_code)]
//! # Issue #022: Node Reputation for Compute Task Assignment
//!
//! Tracks per-peer reputation across 6 weighted dimensions to enable
//! intelligent compute task routing. Peers with consistently good
//! performance get more tasks; peers that fail or timeout get demoted.
//!
//! ## Dimensions & Weights
//!
//! | Dimension            | Weight | Description                          |
//! |----------------------|--------|--------------------------------------|
//! | Task success rate    | 30%    | Fraction of tasks completed ok       |
//! | Average latency      | 20%    | EMA of task completion latency       |
//! | Uptime (24h)         | 15%    | Fraction of last 24h peer was online |
//! | Capacity honesty     | 15%    | Announced vs actual resources        |
//! | Result quality       | 10%    | Quality score of returned results    |
//! | Payment reliability  | 10%    | Fraction of payments settled on time |

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Weight for task success rate dimension (30%).
const W_SUCCESS_RATE: f64 = 0.30;
/// Weight for average latency dimension (20%).
const W_LATENCY: f64 = 0.20;
/// Weight for uptime dimension (15%).
const W_UPTIME: f64 = 0.15;
/// Weight for capacity honesty dimension (15%).
const W_CAPACITY_HONESTY: f64 = 0.15;
/// Weight for result quality dimension (10%).
const W_RESULT_QUALITY: f64 = 0.10;
/// Weight for payment reliability dimension (10%).
const W_PAYMENT_RELIABILITY: f64 = 0.10;

/// Consecutive failures before auto-demotion (excluded for 10 min).
const DEMOTION_FAILURE_THRESHOLD: u64 = 3;

/// Duration of exclusion after demotion, in milliseconds (10 minutes).
const EXCLUSION_DURATION_MS: u64 = 10 * 60 * 1000;

/// Consecutive successes required for auto-promotion.
const PROMOTION_SUCCESS_THRESHOLD: u64 = 100;

/// Exponential moving average alpha for latency updates.
/// Gives ~86% weight to the last 6 observations.
const LATENCY_EMA_ALPHA: f64 = 0.25;

/// Default neutral score for a brand-new peer.
const NEUTRAL_SCORE: f64 = 0.5;

/// Maximum expected latency in ms — used to normalise the latency dimension.
/// Anything at or above this is scored 0.0 for the latency component.
const MAX_EXPECTED_LATENCY_MS: f64 = 10_000.0;

/// Per-tick multiplicative decay factor applied to consecutive counters.
/// Called periodically to ensure old data does not dominate forever.
const DECAY_FACTOR: f64 = 0.95;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The 6-dimensional reputation vector for a single peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeReputation {
    /// Fraction of tasks completed successfully (0.0..1.0).
    pub task_success_rate: f64,
    /// Exponential moving average of task latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Fraction of the last 24 hours the peer was reachable (0.0..1.0).
    pub uptime_24h: f64,
    /// How closely announced capacity matches actual (0.0..1.0).
    pub capacity_honesty: f64,
    /// Quality score of returned compute results (0.0..1.0).
    pub result_quality: f64,
    /// Fraction of payments settled on time (0.0..1.0).
    pub payment_reliability: f64,
}

impl Default for ComputeReputation {
    fn default() -> Self {
        Self {
            task_success_rate: NEUTRAL_SCORE,
            avg_latency_ms: 0.0,
            uptime_24h: NEUTRAL_SCORE,
            capacity_honesty: NEUTRAL_SCORE,
            result_quality: NEUTRAL_SCORE,
            payment_reliability: NEUTRAL_SCORE,
        }
    }
}

impl ComputeReputation {
    /// Compute the weighted composite score in 0.0..1.0.
    pub fn composite_score(&self) -> f64 {
        let latency_score = if self.avg_latency_ms <= 0.0 {
            // No data yet — neutral.
            NEUTRAL_SCORE
        } else {
            (1.0 - (self.avg_latency_ms / MAX_EXPECTED_LATENCY_MS)).clamp(0.0, 1.0)
        };

        let raw = self.task_success_rate * W_SUCCESS_RATE
            + latency_score * W_LATENCY
            + self.uptime_24h * W_UPTIME
            + self.capacity_honesty * W_CAPACITY_HONESTY
            + self.result_quality * W_RESULT_QUALITY
            + self.payment_reliability * W_PAYMENT_RELIABILITY;

        raw.clamp(0.0, 1.0)
    }
}

/// Operational history for a single peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerComputeHistory {
    pub total_tasks: u64,
    pub total_successes: u64,
    pub total_failures: u64,
    pub total_timeouts: u64,
    pub consecutive_failures: u64,
    pub consecutive_successes: u64,
    /// Exponential moving average of latency in ms.
    pub avg_latency_ms: f64,
    /// Unix timestamp (ms) of last failure, 0 if never failed.
    pub last_failure_time: u64,
    /// Unix timestamp (ms) until which this peer is excluded. 0 = not excluded.
    pub excluded_until: u64,
    /// Unix timestamp (ms) of last update.
    pub last_updated_ms: u64,
}

impl Default for PeerComputeHistory {
    fn default() -> Self {
        Self {
            total_tasks: 0,
            total_successes: 0,
            total_failures: 0,
            total_timeouts: 0,
            consecutive_failures: 0,
            consecutive_successes: 0,
            avg_latency_ms: 0.0,
            last_failure_time: 0,
            excluded_until: 0,
            last_updated_ms: now_ms(),
        }
    }
}

/// Aggregate statistics across all tracked peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationStats {
    /// Total number of tracked peers.
    pub total_peers: usize,
    /// Peers currently excluded (demoted).
    pub excluded_peers: usize,
    /// Peers above 0.8 composite score.
    pub high_reputation_peers: usize,
    /// Peers below 0.3 composite score.
    pub low_reputation_peers: usize,
    /// Average composite score across all peers.
    pub avg_score: f64,
}

// ---------------------------------------------------------------------------
// Manager
// ---------------------------------------------------------------------------

/// Thread-safe manager holding reputation data for all known compute peers.
#[derive(Clone)]
pub struct ComputeReputationManager {
    inner: Arc<RwLock<ManagerInner>>,
}

struct ManagerInner {
    reputations: HashMap<String, ComputeReputation>,
    histories: HashMap<String, PeerComputeHistory>,
}

impl ComputeReputationManager {
    /// Create a new, empty manager.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(ManagerInner {
                reputations: HashMap::new(),
                histories: HashMap::new(),
            })),
        }
    }

    // -- mutation helpers --------------------------------------------------

    /// Ensure both maps have an entry for `peer_id`. Returns nothing; callers
    /// already hold the write-lock.
    fn ensure_peer(inner: &mut ManagerInner, peer_id: &str) {
        inner
            .reputations
            .entry(peer_id.to_string())
            .or_insert_with(ComputeReputation::default);
        inner
            .histories
            .entry(peer_id.to_string())
            .or_insert_with(PeerComputeHistory::default);
    }

    /// Recompute `task_success_rate` from history counters.
    fn refresh_success_rate(inner: &mut ManagerInner, peer_id: &str) {
        if let Some(h) = inner.histories.get(peer_id) {
            if h.total_tasks > 0 {
                let rate = h.total_successes as f64 / h.total_tasks as f64;
                if let Some(r) = inner.reputations.get_mut(peer_id) {
                    r.task_success_rate = rate;
                }
            }
        }
    }

    /// Sync EMA latency from history into the reputation struct.
    fn refresh_latency(inner: &mut ManagerInner, peer_id: &str) {
        if let Some(h) = inner.histories.get(peer_id) {
            if let Some(r) = inner.reputations.get_mut(peer_id) {
                r.avg_latency_ms = h.avg_latency_ms;
            }
        }
    }

    // -- public API --------------------------------------------------------

    /// Record a successful task completion for `peer_id`.
    pub fn record_task_success(&self, peer_id: &str, latency_ms: u64) {
        let mut inner = self.inner.write();
        Self::ensure_peer(&mut inner, peer_id);

        if let Some(h) = inner.histories.get_mut(peer_id) {
            h.total_tasks += 1;
            h.total_successes += 1;
            h.consecutive_successes += 1;
            h.consecutive_failures = 0;
            h.last_updated_ms = now_ms();

            // Update EMA latency.
            let lat = latency_ms as f64;
            if h.avg_latency_ms <= 0.0 {
                h.avg_latency_ms = lat;
            } else {
                h.avg_latency_ms =
                    LATENCY_EMA_ALPHA * lat + (1.0 - LATENCY_EMA_ALPHA) * h.avg_latency_ms;
            }
        }

        Self::refresh_success_rate(&mut inner, peer_id);
        Self::refresh_latency(&mut inner, peer_id);

        debug!(peer_id, latency_ms, "compute reputation: task success recorded");
    }

    /// Record a task failure (peer returned an error or bad result).
    pub fn record_task_failure(&self, peer_id: &str) {
        let mut inner = self.inner.write();
        Self::ensure_peer(&mut inner, peer_id);

        let now = now_ms();
        if let Some(h) = inner.histories.get_mut(peer_id) {
            h.total_tasks += 1;
            h.total_failures += 1;
            h.consecutive_failures += 1;
            h.consecutive_successes = 0;
            h.last_failure_time = now;
            h.last_updated_ms = now;

            if h.consecutive_failures >= DEMOTION_FAILURE_THRESHOLD {
                h.excluded_until = now + EXCLUSION_DURATION_MS;
                warn!(
                    peer_id,
                    consecutive_failures = h.consecutive_failures,
                    excluded_until = h.excluded_until,
                    "compute reputation: peer auto-demoted after consecutive failures"
                );
            }
        }

        Self::refresh_success_rate(&mut inner, peer_id);
    }

    /// Record a task timeout — treated as worse than a normal failure.
    /// Timeouts count as both a failure AND a timeout, and increment
    /// consecutive failures by 2 instead of 1.
    pub fn record_task_timeout(&self, peer_id: &str) {
        let mut inner = self.inner.write();
        Self::ensure_peer(&mut inner, peer_id);

        let now = now_ms();
        if let Some(h) = inner.histories.get_mut(peer_id) {
            // Timeouts count as 2 tasks / 2 failures so the success rate
            // drops faster than a regular failure.
            h.total_tasks += 2;
            h.total_failures += 2;
            h.total_timeouts += 1;
            // Also double-increment consecutive failures for faster demotion.
            h.consecutive_failures += 2;
            h.consecutive_successes = 0;
            h.last_failure_time = now;
            h.last_updated_ms = now;

            if h.consecutive_failures >= DEMOTION_FAILURE_THRESHOLD {
                h.excluded_until = now + EXCLUSION_DURATION_MS;
                warn!(
                    peer_id,
                    consecutive_failures = h.consecutive_failures,
                    "compute reputation: peer auto-demoted after timeout(s)"
                );
            }
        }

        Self::refresh_success_rate(&mut inner, peer_id);
    }

    /// Update the uptime fraction for a peer (called by the health monitor).
    pub fn update_uptime(&self, peer_id: &str, uptime_fraction: f64) {
        let mut inner = self.inner.write();
        Self::ensure_peer(&mut inner, peer_id);
        if let Some(r) = inner.reputations.get_mut(peer_id) {
            r.uptime_24h = uptime_fraction.clamp(0.0, 1.0);
        }
    }

    /// Update the capacity honesty score for a peer.
    pub fn update_capacity_honesty(&self, peer_id: &str, score: f64) {
        let mut inner = self.inner.write();
        Self::ensure_peer(&mut inner, peer_id);
        if let Some(r) = inner.reputations.get_mut(peer_id) {
            r.capacity_honesty = score.clamp(0.0, 1.0);
        }
    }

    /// Update the result quality score for a peer.
    pub fn update_result_quality(&self, peer_id: &str, score: f64) {
        let mut inner = self.inner.write();
        Self::ensure_peer(&mut inner, peer_id);
        if let Some(r) = inner.reputations.get_mut(peer_id) {
            r.result_quality = score.clamp(0.0, 1.0);
        }
    }

    /// Update the payment reliability score for a peer.
    pub fn update_payment_reliability(&self, peer_id: &str, score: f64) {
        let mut inner = self.inner.write();
        Self::ensure_peer(&mut inner, peer_id);
        if let Some(r) = inner.reputations.get_mut(peer_id) {
            r.payment_reliability = score.clamp(0.0, 1.0);
        }
    }

    /// Get the composite score for a peer (0.0..1.0).
    /// Returns [`NEUTRAL_SCORE`] if the peer is unknown.
    pub fn get_score(&self, peer_id: &str) -> f64 {
        let inner = self.inner.read();
        inner
            .reputations
            .get(peer_id)
            .map(|r| r.composite_score())
            .unwrap_or(NEUTRAL_SCORE)
    }

    /// Return the top `n` peers whose composite score is >= `min_score`,
    /// sorted descending by score. Excluded peers are filtered out.
    pub fn get_best_peers(&self, n: usize, min_score: f64) -> Vec<(String, f64)> {
        let inner = self.inner.read();
        let now = now_ms();

        let mut scored: Vec<(String, f64)> = inner
            .reputations
            .iter()
            .filter(|(pid, _)| {
                // Filter out currently-excluded peers.
                inner
                    .histories
                    .get(*pid)
                    .map(|h| h.excluded_until < now)
                    .unwrap_or(true)
            })
            .map(|(pid, r)| (pid.clone(), r.composite_score()))
            .filter(|(_, score)| *score >= min_score)
            .collect();

        // Sort descending by score, ties broken by peer_id for determinism.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        scored.truncate(n);
        scored
    }

    /// Check if the peer should be auto-demoted (3+ consecutive failures).
    /// Returns `true` if the peer is currently excluded.
    pub fn check_auto_demotion(&self, peer_id: &str) -> bool {
        let inner = self.inner.read();
        let now = now_ms();
        inner
            .histories
            .get(peer_id)
            .map(|h| {
                h.consecutive_failures >= DEMOTION_FAILURE_THRESHOLD && h.excluded_until >= now
            })
            .unwrap_or(false)
    }

    /// Check if the peer qualifies for auto-promotion (100+ consecutive
    /// successes). Returns `true` if the threshold is met.
    pub fn check_auto_promotion(&self, peer_id: &str) -> bool {
        let inner = self.inner.read();
        inner
            .histories
            .get(peer_id)
            .map(|h| h.consecutive_successes >= PROMOTION_SUCCESS_THRESHOLD)
            .unwrap_or(false)
    }

    /// Decay all scores to reduce the influence of old data.
    /// Should be called on a periodic timer (e.g. every 5 minutes).
    pub fn decay_scores(&self) {
        let mut inner = self.inner.write();
        for (_pid, rep) in inner.reputations.iter_mut() {
            // Pull each dimension toward neutral by DECAY_FACTOR.
            rep.task_success_rate =
                NEUTRAL_SCORE + (rep.task_success_rate - NEUTRAL_SCORE) * DECAY_FACTOR;
            rep.uptime_24h = NEUTRAL_SCORE + (rep.uptime_24h - NEUTRAL_SCORE) * DECAY_FACTOR;
            rep.capacity_honesty =
                NEUTRAL_SCORE + (rep.capacity_honesty - NEUTRAL_SCORE) * DECAY_FACTOR;
            rep.result_quality =
                NEUTRAL_SCORE + (rep.result_quality - NEUTRAL_SCORE) * DECAY_FACTOR;
            rep.payment_reliability =
                NEUTRAL_SCORE + (rep.payment_reliability - NEUTRAL_SCORE) * DECAY_FACTOR;
            // Latency EMA decays toward 0 (no data).
            rep.avg_latency_ms *= DECAY_FACTOR;
        }
        info!(peers = inner.reputations.len(), "compute reputation: decay applied");
    }

    /// Return aggregate statistics across all tracked peers.
    pub fn stats(&self) -> ReputationStats {
        let inner = self.inner.read();
        let now = now_ms();
        let total_peers = inner.reputations.len();

        let excluded_peers = inner
            .histories
            .values()
            .filter(|h| h.excluded_until >= now)
            .count();

        let scores: Vec<f64> = inner.reputations.values().map(|r| r.composite_score()).collect();

        let avg_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        let high_reputation_peers = scores.iter().filter(|s| **s >= 0.8).count();
        let low_reputation_peers = scores.iter().filter(|s| **s < 0.3).count();

        ReputationStats {
            total_peers,
            excluded_peers,
            high_reputation_peers,
            low_reputation_peers,
            avg_score,
        }
    }

    /// Get the full reputation struct for a peer, if known.
    pub fn get_reputation(&self, peer_id: &str) -> Option<ComputeReputation> {
        let inner = self.inner.read();
        inner.reputations.get(peer_id).cloned()
    }

    /// Get the full history struct for a peer, if known.
    pub fn get_history(&self, peer_id: &str) -> Option<PeerComputeHistory> {
        let inner = self.inner.read();
        inner.histories.get(peer_id).cloned()
    }
}

impl Default for ComputeReputationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Current Unix time in milliseconds.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mgr() -> ComputeReputationManager {
        ComputeReputationManager::new()
    }

    #[test]
    fn test_initial_score_neutral() {
        let m = mgr();
        let score = m.get_score("peer-a");
        assert!(
            (score - NEUTRAL_SCORE).abs() < 1e-9,
            "unknown peer should have neutral score {NEUTRAL_SCORE}, got {score}"
        );
    }

    #[test]
    fn test_success_improves_score() {
        let m = mgr();
        let before = m.get_score("peer-a");
        for _ in 0..20 {
            m.record_task_success("peer-a", 50);
        }
        let after = m.get_score("peer-a");
        assert!(
            after > before,
            "score should improve after successes: before={before}, after={after}"
        );
    }

    #[test]
    fn test_failure_reduces_score() {
        let m = mgr();
        // Seed with some successes first so the score is non-minimal.
        for _ in 0..10 {
            m.record_task_success("peer-a", 100);
        }
        let before = m.get_score("peer-a");
        for _ in 0..10 {
            m.record_task_failure("peer-a");
        }
        let after = m.get_score("peer-a");
        assert!(
            after < before,
            "score should decrease after failures: before={before}, after={after}"
        );
    }

    #[test]
    fn test_timeout_worse_than_failure() {
        let m1 = mgr();
        let m2 = mgr();

        // Both start with 10 successes.
        for _ in 0..10 {
            m1.record_task_success("peer-a", 100);
            m2.record_task_success("peer-a", 100);
        }

        // m1 records 5 failures, m2 records 5 timeouts.
        for _ in 0..5 {
            m1.record_task_failure("peer-a");
            m2.record_task_timeout("peer-a");
        }

        let score_fail = m1.get_score("peer-a");
        let score_timeout = m2.get_score("peer-a");
        assert!(
            score_timeout < score_fail,
            "timeout should produce lower score than failure: timeout={score_timeout}, failure={score_fail}"
        );
    }

    #[test]
    fn test_auto_demotion_after_3_failures() {
        let m = mgr();
        assert!(!m.check_auto_demotion("peer-a"));

        // 3 consecutive failures should trigger demotion.
        for _ in 0..3 {
            m.record_task_failure("peer-a");
        }
        assert!(
            m.check_auto_demotion("peer-a"),
            "peer should be demoted after 3 consecutive failures"
        );
    }

    #[test]
    fn test_auto_promotion_after_100_successes() {
        let m = mgr();
        assert!(!m.check_auto_promotion("peer-a"));

        for _ in 0..99 {
            m.record_task_success("peer-a", 50);
        }
        assert!(
            !m.check_auto_promotion("peer-a"),
            "99 successes should not yet qualify for promotion"
        );

        m.record_task_success("peer-a", 50);
        assert!(
            m.check_auto_promotion("peer-a"),
            "100 consecutive successes should qualify for promotion"
        );
    }

    #[test]
    fn test_get_best_peers_sorted() {
        let m = mgr();

        // Give peer-b more successes than peer-a.
        for _ in 0..5 {
            m.record_task_success("peer-a", 200);
        }
        for _ in 0..50 {
            m.record_task_success("peer-b", 50);
        }

        let best = m.get_best_peers(10, 0.0);
        assert!(best.len() >= 2, "should have at least 2 peers");
        assert_eq!(
            best[0].0, "peer-b",
            "peer-b should be ranked higher (more successes, lower latency)"
        );
    }

    #[test]
    fn test_get_best_peers_min_score_filter() {
        let m = mgr();
        // peer-a gets many failures → low score.
        for _ in 0..50 {
            m.record_task_failure("peer-a");
        }
        // peer-b gets many successes → high score.
        for _ in 0..50 {
            m.record_task_success("peer-b", 50);
        }

        let best = m.get_best_peers(10, 0.6);
        let peer_ids: Vec<&str> = best.iter().map(|(p, _)| p.as_str()).collect();
        assert!(
            !peer_ids.contains(&"peer-a"),
            "peer-a with low score should be filtered out"
        );
        assert!(
            peer_ids.contains(&"peer-b"),
            "peer-b with high score should be included"
        );
    }

    #[test]
    fn test_decay_reduces_old_scores() {
        let m = mgr();
        // Build up a high score.
        for _ in 0..50 {
            m.record_task_success("peer-a", 50);
        }
        let before = m.get_score("peer-a");

        // Apply decay multiple times.
        for _ in 0..20 {
            m.decay_scores();
        }
        let after = m.get_score("peer-a");
        assert!(
            after < before,
            "score should decrease after repeated decay: before={before}, after={after}"
        );
        // Should trend toward neutral.
        assert!(
            (after - NEUTRAL_SCORE).abs() < (before - NEUTRAL_SCORE).abs(),
            "decayed score should be closer to neutral than before"
        );
    }

    #[test]
    fn test_excluded_peer_not_in_best_peers() {
        let m = mgr();
        // peer-a gets successes then gets demoted.
        for _ in 0..10 {
            m.record_task_success("peer-a", 50);
        }
        for _ in 0..3 {
            m.record_task_failure("peer-a");
        }
        assert!(m.check_auto_demotion("peer-a"));

        // peer-b is healthy.
        for _ in 0..10 {
            m.record_task_success("peer-b", 50);
        }

        let best = m.get_best_peers(10, 0.0);
        let peer_ids: Vec<&str> = best.iter().map(|(p, _)| p.as_str()).collect();
        assert!(
            !peer_ids.contains(&"peer-a"),
            "excluded/demoted peer should not appear in best peers"
        );
        assert!(
            peer_ids.contains(&"peer-b"),
            "healthy peer should appear in best peers"
        );
    }

    #[test]
    fn test_score_is_bounded_0_to_1() {
        let m = mgr();

        // Drive score as low as possible.
        for _ in 0..1000 {
            m.record_task_failure("peer-low");
        }
        // Also set dimensions to 0.
        m.update_uptime("peer-low", 0.0);
        m.update_capacity_honesty("peer-low", 0.0);
        m.update_result_quality("peer-low", 0.0);
        m.update_payment_reliability("peer-low", 0.0);

        let low = m.get_score("peer-low");
        assert!(
            low >= 0.0 && low <= 1.0,
            "score must be in [0,1], got {low}"
        );

        // Drive score as high as possible.
        for _ in 0..1000 {
            m.record_task_success("peer-high", 1);
        }
        m.update_uptime("peer-high", 1.0);
        m.update_capacity_honesty("peer-high", 1.0);
        m.update_result_quality("peer-high", 1.0);
        m.update_payment_reliability("peer-high", 1.0);

        let high = m.get_score("peer-high");
        assert!(
            high >= 0.0 && high <= 1.0,
            "score must be in [0,1], got {high}"
        );
    }

    #[test]
    fn test_stats_reflect_state() {
        let m = mgr();

        let empty_stats = m.stats();
        assert_eq!(empty_stats.total_peers, 0);
        assert_eq!(empty_stats.excluded_peers, 0);

        // Add some peers.
        for _ in 0..50 {
            m.record_task_success("peer-good", 30);
        }
        m.update_uptime("peer-good", 1.0);
        m.update_capacity_honesty("peer-good", 1.0);
        m.update_result_quality("peer-good", 1.0);
        m.update_payment_reliability("peer-good", 1.0);

        for _ in 0..50 {
            m.record_task_failure("peer-bad");
        }
        m.update_uptime("peer-bad", 0.0);
        m.update_capacity_honesty("peer-bad", 0.0);
        m.update_result_quality("peer-bad", 0.0);
        m.update_payment_reliability("peer-bad", 0.0);

        let stats = m.stats();
        assert_eq!(stats.total_peers, 2);
        assert!(
            stats.excluded_peers >= 1,
            "peer-bad should be excluded (50 consecutive failures)"
        );
        assert!(
            stats.high_reputation_peers >= 1,
            "peer-good should be high-reputation"
        );
        assert!(
            stats.low_reputation_peers >= 1,
            "peer-bad should be low-reputation"
        );
        assert!(
            stats.avg_score > 0.0,
            "average score should be positive with 2 peers"
        );
    }
}
