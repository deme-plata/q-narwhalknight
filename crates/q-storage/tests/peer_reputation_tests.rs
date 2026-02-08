//! Peer Reputation Decentralization Tests
//!
//! Tests to ensure the peer reputation system properly tracks and scores
//! peer behavior for decentralized network health.
//!
//! CRITICAL SCENARIOS TESTED:
//! 1. Reputation scoring for good/bad behavior
//! 2. False height claim penalties
//! 3. Trust level transitions
//! 4. Peer banning and recovery
//! 5. Delivery tracking
//!
//! Run with: cargo test --package q-storage --test peer_reputation_tests

use std::collections::HashMap;
use std::sync::{Arc, RwLock, atomic::{AtomicU64, Ordering}};

// ============================================================================
// MOCK STRUCTURES FOR PEER REPUTATION TESTING
// ============================================================================

/// Trust level for a peer
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    Unknown = 0,
    Suspicious = 1,
    Normal = 2,
    Trusted = 3,
    Allowlisted = 4,
}

/// Peer behavior event type
#[derive(Debug, Clone, Copy)]
pub enum PeerEvent {
    SuccessfulDelivery,
    FailedDelivery,
    FalseHeightClaim,
    ValidBlock,
    InvalidBlock,
    FastResponse,
    SlowResponse,
    ConnectionDropped,
}

/// Peer reputation data
#[derive(Debug, Clone)]
pub struct PeerReputation {
    pub peer_id: [u8; 32],
    pub trust_level: TrustLevel,
    pub reputation_score: i64, // Can go negative
    pub successful_deliveries: u64,
    pub failed_deliveries: u64,
    pub false_height_claims: u64,
    pub valid_blocks: u64,
    pub invalid_blocks: u64,
    pub average_response_ms: u64,
    pub last_seen: u64,
    pub banned_until: Option<u64>,
}

impl PeerReputation {
    pub fn new(peer_id: [u8; 32]) -> Self {
        Self {
            peer_id,
            trust_level: TrustLevel::Unknown,
            reputation_score: 0,
            successful_deliveries: 0,
            failed_deliveries: 0,
            false_height_claims: 0,
            valid_blocks: 0,
            invalid_blocks: 0,
            average_response_ms: 0,
            last_seen: 0,
            banned_until: None,
        }
    }
}

/// Reputation manager for all peers
pub struct ReputationManager {
    peers: RwLock<HashMap<[u8; 32], PeerReputation>>,
    current_time: AtomicU64,
    // Thresholds
    ban_threshold: i64,
    trusted_threshold: i64,
    suspicious_threshold: i64,
    // Scoring weights
    successful_delivery_score: i64,
    failed_delivery_score: i64,
    false_height_score: i64,
    valid_block_score: i64,
    invalid_block_score: i64,
    // Ban duration in seconds
    ban_duration: u64,
}

impl ReputationManager {
    pub fn new() -> Self {
        Self {
            peers: RwLock::new(HashMap::new()),
            current_time: AtomicU64::new(0),
            ban_threshold: -100,
            trusted_threshold: 100,
            suspicious_threshold: -20,
            successful_delivery_score: 1,
            failed_delivery_score: -5,
            false_height_score: -50,
            valid_block_score: 5,
            invalid_block_score: -25,
            ban_duration: 3600, // 1 hour
        }
    }

    pub fn set_time(&self, time: u64) {
        self.current_time.store(time, Ordering::SeqCst);
    }

    /// Register a new peer
    pub fn register_peer(&self, peer_id: [u8; 32]) {
        let mut peers = self.peers.write().unwrap();
        if !peers.contains_key(&peer_id) {
            peers.insert(peer_id, PeerReputation::new(peer_id));
        }
    }

    /// Record a peer event and update reputation
    pub fn record_event(&self, peer_id: [u8; 32], event: PeerEvent) -> Result<(), String> {
        let mut peers = self.peers.write().unwrap();
        let current_time = self.current_time.load(Ordering::SeqCst);

        let peer = peers
            .get_mut(&peer_id)
            .ok_or("PEER_NOT_FOUND")?;

        // Check if peer is banned
        if let Some(banned_until) = peer.banned_until {
            if current_time < banned_until {
                return Err(format!(
                    "PEER_BANNED: {} seconds remaining",
                    banned_until - current_time
                ));
            } else {
                // Ban expired, reset
                peer.banned_until = None;
                peer.reputation_score = 0;
                peer.trust_level = TrustLevel::Unknown;
            }
        }

        peer.last_seen = current_time;

        // Update stats and score based on event
        match event {
            PeerEvent::SuccessfulDelivery => {
                peer.successful_deliveries += 1;
                peer.reputation_score += self.successful_delivery_score;
            }
            PeerEvent::FailedDelivery => {
                peer.failed_deliveries += 1;
                peer.reputation_score += self.failed_delivery_score;
            }
            PeerEvent::FalseHeightClaim => {
                peer.false_height_claims += 1;
                peer.reputation_score += self.false_height_score;
            }
            PeerEvent::ValidBlock => {
                peer.valid_blocks += 1;
                peer.reputation_score += self.valid_block_score;
            }
            PeerEvent::InvalidBlock => {
                peer.invalid_blocks += 1;
                peer.reputation_score += self.invalid_block_score;
            }
            PeerEvent::FastResponse => {
                peer.reputation_score += 1;
            }
            PeerEvent::SlowResponse => {
                peer.reputation_score -= 1;
            }
            PeerEvent::ConnectionDropped => {
                peer.reputation_score -= 2;
            }
        }

        // Update trust level
        self.update_trust_level(peer);

        // Check for ban
        if peer.reputation_score <= self.ban_threshold {
            peer.banned_until = Some(current_time + self.ban_duration);
            peer.trust_level = TrustLevel::Suspicious;
        }

        Ok(())
    }

    fn update_trust_level(&self, peer: &mut PeerReputation) {
        // Don't change allowlisted peers
        if peer.trust_level == TrustLevel::Allowlisted {
            return;
        }

        peer.trust_level = if peer.reputation_score >= self.trusted_threshold {
            TrustLevel::Trusted
        } else if peer.reputation_score <= self.suspicious_threshold {
            TrustLevel::Suspicious
        } else if peer.successful_deliveries > 0 || peer.valid_blocks > 0 {
            TrustLevel::Normal
        } else {
            TrustLevel::Unknown
        };
    }

    /// Set peer as allowlisted (bootstrap nodes, etc.)
    pub fn allowlist_peer(&self, peer_id: [u8; 32]) {
        let mut peers = self.peers.write().unwrap();
        if let Some(peer) = peers.get_mut(&peer_id) {
            peer.trust_level = TrustLevel::Allowlisted;
            peer.reputation_score = self.trusted_threshold;
        }
    }

    /// Get peer reputation
    pub fn get_reputation(&self, peer_id: &[u8; 32]) -> Option<PeerReputation> {
        self.peers.read().unwrap().get(peer_id).cloned()
    }

    /// Check if peer is banned
    pub fn is_banned(&self, peer_id: &[u8; 32]) -> bool {
        let current_time = self.current_time.load(Ordering::SeqCst);
        self.peers
            .read()
            .unwrap()
            .get(peer_id)
            .map(|p| p.banned_until.map(|b| current_time < b).unwrap_or(false))
            .unwrap_or(false)
    }

    /// Get trust level
    pub fn get_trust_level(&self, peer_id: &[u8; 32]) -> TrustLevel {
        self.peers
            .read()
            .unwrap()
            .get(peer_id)
            .map(|p| p.trust_level)
            .unwrap_or(TrustLevel::Unknown)
    }

    /// Get all peers with trust level at or above threshold
    pub fn get_trusted_peers(&self, min_level: TrustLevel) -> Vec<[u8; 32]> {
        self.peers
            .read()
            .unwrap()
            .iter()
            .filter(|(_, p)| p.trust_level >= min_level)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get peer count by trust level
    pub fn count_by_trust_level(&self, level: TrustLevel) -> usize {
        self.peers
            .read()
            .unwrap()
            .values()
            .filter(|p| p.trust_level == level)
            .count()
    }

    /// Calculate delivery success rate
    pub fn delivery_success_rate(&self, peer_id: &[u8; 32]) -> Option<f64> {
        self.peers.read().unwrap().get(peer_id).map(|p| {
            let total = p.successful_deliveries + p.failed_deliveries;
            if total == 0 {
                0.0
            } else {
                p.successful_deliveries as f64 / total as f64
            }
        })
    }
}

// ============================================================================
// BASIC REPUTATION TESTS
// ============================================================================

#[test]
fn test_register_peer() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);

    let rep = manager.get_reputation(&peer).unwrap();
    assert_eq!(rep.trust_level, TrustLevel::Unknown);
    assert_eq!(rep.reputation_score, 0);
}

#[test]
fn test_successful_delivery_increases_score() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);
    manager.record_event(peer, PeerEvent::SuccessfulDelivery).unwrap();

    let rep = manager.get_reputation(&peer).unwrap();
    assert!(rep.reputation_score > 0);
    assert_eq!(rep.successful_deliveries, 1);
}

#[test]
fn test_failed_delivery_decreases_score() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);
    manager.record_event(peer, PeerEvent::FailedDelivery).unwrap();

    let rep = manager.get_reputation(&peer).unwrap();
    assert!(rep.reputation_score < 0);
    assert_eq!(rep.failed_deliveries, 1);
}

// ============================================================================
// FALSE HEIGHT CLAIM TESTS
// ============================================================================

#[test]
fn test_false_height_claim_severe_penalty() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);

    // Build up some reputation first
    for _ in 0..50 {
        manager.record_event(peer, PeerEvent::SuccessfulDelivery).unwrap();
    }

    let rep_before = manager.get_reputation(&peer).unwrap();
    assert!(rep_before.reputation_score > 0);

    // One false height claim
    manager.record_event(peer, PeerEvent::FalseHeightClaim).unwrap();

    let rep_after = manager.get_reputation(&peer).unwrap();
    assert!(rep_after.reputation_score < rep_before.reputation_score);
    assert_eq!(rep_after.false_height_claims, 1);
}

#[test]
fn test_multiple_false_claims_leads_to_ban() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);
    manager.set_time(1000);

    // Multiple false height claims should lead to ban
    for _ in 0..3 {
        let _ = manager.record_event(peer, PeerEvent::FalseHeightClaim);
    }

    assert!(manager.is_banned(&peer));
}

// ============================================================================
// TRUST LEVEL TESTS
// ============================================================================

#[test]
fn test_trust_level_upgrades_to_normal() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);
    manager.record_event(peer, PeerEvent::SuccessfulDelivery).unwrap();

    let level = manager.get_trust_level(&peer);
    assert_eq!(level, TrustLevel::Normal);
}

#[test]
fn test_trust_level_upgrades_to_trusted() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);

    // Many valid blocks should make trusted
    for _ in 0..25 {
        manager.record_event(peer, PeerEvent::ValidBlock).unwrap();
    }

    let level = manager.get_trust_level(&peer);
    assert_eq!(level, TrustLevel::Trusted);
}

#[test]
fn test_trust_level_downgrades_to_suspicious() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);

    // Many failures should make suspicious
    for _ in 0..10 {
        manager.record_event(peer, PeerEvent::FailedDelivery).unwrap();
    }

    let level = manager.get_trust_level(&peer);
    assert_eq!(level, TrustLevel::Suspicious);
}

#[test]
fn test_allowlisted_peer_stays_trusted() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);
    manager.allowlist_peer(peer);

    // Even bad events shouldn't change allowlist status
    manager.record_event(peer, PeerEvent::FailedDelivery).unwrap();
    manager.record_event(peer, PeerEvent::FailedDelivery).unwrap();

    let level = manager.get_trust_level(&peer);
    assert_eq!(level, TrustLevel::Allowlisted);
}

// ============================================================================
// BANNING TESTS
// ============================================================================

#[test]
fn test_peer_gets_banned() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);
    manager.set_time(1000);

    // Bad behavior leads to ban
    for _ in 0..30 {
        let _ = manager.record_event(peer, PeerEvent::FailedDelivery);
    }

    assert!(manager.is_banned(&peer));
}

#[test]
fn test_banned_peer_events_rejected() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);
    manager.set_time(1000);

    // Get banned
    for _ in 0..30 {
        let _ = manager.record_event(peer, PeerEvent::FailedDelivery);
    }

    // Try to record event while banned
    let result = manager.record_event(peer, PeerEvent::SuccessfulDelivery);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("PEER_BANNED"));
}

#[test]
fn test_ban_expires() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);
    manager.set_time(1000);

    // Get banned
    for _ in 0..30 {
        let _ = manager.record_event(peer, PeerEvent::FailedDelivery);
    }

    assert!(manager.is_banned(&peer));

    // Fast forward past ban duration
    manager.set_time(1000 + 3601);

    // Should be unbanned now
    assert!(!manager.is_banned(&peer));

    // Can record events again
    let result = manager.record_event(peer, PeerEvent::SuccessfulDelivery);
    assert!(result.is_ok());

    // Score should be reset
    let rep = manager.get_reputation(&peer).unwrap();
    assert!(rep.reputation_score >= 0);
}

// ============================================================================
// DELIVERY RATE TESTS
// ============================================================================

#[test]
fn test_delivery_success_rate() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);

    // 8 successes, 2 failures = 80% success rate
    for _ in 0..8 {
        manager.record_event(peer, PeerEvent::SuccessfulDelivery).unwrap();
    }
    for _ in 0..2 {
        manager.record_event(peer, PeerEvent::FailedDelivery).unwrap();
    }

    let rate = manager.delivery_success_rate(&peer).unwrap();
    assert!((rate - 0.8).abs() < 0.01);
}

#[test]
fn test_delivery_rate_no_deliveries() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);

    let rate = manager.delivery_success_rate(&peer).unwrap();
    assert_eq!(rate, 0.0);
}

// ============================================================================
// PEER QUERY TESTS
// ============================================================================

#[test]
fn test_get_trusted_peers() {
    let manager = ReputationManager::new();

    // Register peers with different trust levels
    for i in 0..5 {
        let peer = [i as u8; 32];
        manager.register_peer(peer);

        // Make some trusted
        if i < 3 {
            for _ in 0..25 {
                manager.record_event(peer, PeerEvent::ValidBlock).unwrap();
            }
        }
    }

    let trusted = manager.get_trusted_peers(TrustLevel::Trusted);
    assert_eq!(trusted.len(), 3);
}

#[test]
fn test_count_by_trust_level() {
    let manager = ReputationManager::new();

    for i in 0..10 {
        let peer = [i as u8; 32];
        manager.register_peer(peer);
    }

    let unknown_count = manager.count_by_trust_level(TrustLevel::Unknown);
    assert_eq!(unknown_count, 10);
}

// ============================================================================
// BLOCK VALIDATION TESTS
// ============================================================================

#[test]
fn test_valid_block_increases_reputation() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);
    let initial = manager.get_reputation(&peer).unwrap().reputation_score;

    manager.record_event(peer, PeerEvent::ValidBlock).unwrap();

    let final_score = manager.get_reputation(&peer).unwrap().reputation_score;
    assert!(final_score > initial);
}

#[test]
fn test_invalid_block_decreases_reputation() {
    let manager = ReputationManager::new();
    let peer = [1u8; 32];

    manager.register_peer(peer);

    // Build some reputation first
    for _ in 0..10 {
        manager.record_event(peer, PeerEvent::ValidBlock).unwrap();
    }

    let initial = manager.get_reputation(&peer).unwrap().reputation_score;

    manager.record_event(peer, PeerEvent::InvalidBlock).unwrap();

    let final_score = manager.get_reputation(&peer).unwrap().reputation_score;
    assert!(final_score < initial);
}
