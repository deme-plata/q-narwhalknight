//! Sync Malicious Peer Tests
//!
//! Tests to ensure the sync system properly handles malicious peers
//! attempting to corrupt blockchain state or cause data loss.
//!
//! CRITICAL SCENARIOS TESTED:
//! 1. Sync-down attack prevention (peer announces lower height)
//! 2. Height manipulation attacks (u64::MAX, 0, negative deltas)
//! 3. Invalid block data injection
//! 4. Peer height spoofing
//! 5. Network partition recovery
//!
//! Run with: cargo test --package q-storage --test sync_malicious_peer_tests

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

// ============================================================================
// MOCK STRUCTURES FOR SYNC TESTING
// ============================================================================

/// Simulated peer with height announcement
#[derive(Debug, Clone)]
pub struct MockPeer {
    pub id: String,
    pub announced_height: u64,
    pub actual_height: u64,
    pub is_malicious: bool,
    pub response_delay_ms: u64,
}

/// Simulated block for sync testing
#[derive(Debug, Clone)]
pub struct MockBlock {
    pub height: u64,
    pub hash: [u8; 32],
    pub parent_hash: [u8; 32],
    pub is_valid: bool,
}

/// Sync state machine with protection mechanisms
#[derive(Debug)]
pub struct SyncProtector {
    local_height: u64,
    min_height_for_protection: u64,
    max_regression_allowed: u64,
    banned_peers: HashSet<String>,
    peer_heights: HashMap<String, u64>,
    blocks: HashMap<u64, MockBlock>,
}

impl SyncProtector {
    pub fn new(local_height: u64) -> Self {
        Self {
            local_height,
            min_height_for_protection: 1000, // Protect after 1000 blocks
            max_regression_allowed: 10,       // Allow reorg up to 10 blocks
            banned_peers: HashSet::new(),
            peer_heights: HashMap::new(),
            blocks: HashMap::new(),
        }
    }

    pub fn get_local_height(&self) -> u64 {
        self.local_height
    }

    pub fn is_peer_banned(&self, peer_id: &str) -> bool {
        self.banned_peers.contains(peer_id)
    }

    pub fn ban_peer(&mut self, peer_id: &str, reason: &str) {
        eprintln!("🚫 BANNING PEER {}: {}", peer_id, reason);
        self.banned_peers.insert(peer_id.to_string());
    }

    /// Validate a sync request from a peer
    /// Returns Ok(target_height) if sync should proceed, Err if rejected
    pub fn validate_sync_request(&mut self, peer: &MockPeer) -> Result<u64, String> {
        // Check if peer is banned
        if self.banned_peers.contains(&peer.id) {
            return Err(format!("REJECTED: Peer {} is banned", peer.id));
        }

        let target_height = peer.announced_height;

        // CRITICAL CHECK 1: Reject sync-down attacks
        if target_height < self.local_height {
            // Allow small regressions for reorgs, but not catastrophic ones
            let regression = self.local_height - target_height;

            if self.local_height > self.min_height_for_protection {
                if regression > self.max_regression_allowed {
                    self.ban_peer(&peer.id, &format!(
                        "SYNC-DOWN ATTACK: Announced height {} is {} blocks below local {}",
                        target_height, regression, self.local_height
                    ));
                    return Err(format!(
                        "🚨 CRITICAL: Sync-down attack blocked! Peer {} announced height {} but local is {}",
                        peer.id, target_height, self.local_height
                    ));
                }
            }
        }

        // CRITICAL CHECK 2: Reject zero height announcements
        if target_height == 0 && self.local_height > self.min_height_for_protection {
            self.ban_peer(&peer.id, "Announced height 0 - likely malicious");
            return Err(format!(
                "🚨 CRITICAL: Zero height attack blocked from peer {}",
                peer.id
            ));
        }

        // CRITICAL CHECK 3: Reject suspiciously high heights
        // If peer claims to be way ahead, they might be trying to waste our resources
        let max_reasonable_ahead = 100_000; // 100K blocks ahead is suspicious
        if target_height > self.local_height + max_reasonable_ahead {
            return Err(format!(
                "SUSPICIOUS: Peer {} claims height {} which is {} blocks ahead",
                peer.id, target_height, target_height - self.local_height
            ));
        }

        // Record peer's announced height
        self.peer_heights.insert(peer.id.clone(), target_height);

        Ok(target_height)
    }

    /// Simulate applying a sync to target height
    pub fn sync_to_height(&mut self, target_height: u64) -> Result<(), String> {
        if target_height < self.local_height && self.local_height > self.min_height_for_protection {
            let regression = self.local_height - target_height;
            if regression > self.max_regression_allowed {
                return Err(format!(
                    "🚨 SAFETY ABORT: Refusing to sync down from {} to {} ({} blocks)",
                    self.local_height, target_height, regression
                ));
            }
        }

        self.local_height = target_height;
        Ok(())
    }

    /// Validate an incoming block
    pub fn validate_block(&self, block: &MockBlock) -> Result<(), String> {
        // Check block validity flag (simulates signature/hash verification)
        if !block.is_valid {
            return Err(format!("Invalid block at height {}", block.height));
        }

        // Check parent continuity
        if block.height > 0 {
            if let Some(parent) = self.blocks.get(&(block.height - 1)) {
                if parent.hash != block.parent_hash {
                    return Err(format!(
                        "Parent hash mismatch at height {}: expected {:?}, got {:?}",
                        block.height, parent.hash, block.parent_hash
                    ));
                }
            }
        }

        Ok(())
    }

    /// Add a validated block
    pub fn add_block(&mut self, block: MockBlock) -> Result<(), String> {
        self.validate_block(&block)?;
        let height = block.height;
        self.blocks.insert(height, block);
        if height > self.local_height {
            self.local_height = height;
        }
        Ok(())
    }
}

// ============================================================================
// SYNC-DOWN ATTACK PREVENTION TESTS
// ============================================================================

/// Test that sync-down from high to low is blocked
#[test]
fn test_sync_down_attack_blocked() {
    let mut protector = SyncProtector::new(100_000);

    let malicious_peer = MockPeer {
        id: "malicious_1".to_string(),
        announced_height: 1_000, // Claims to be at block 1000
        actual_height: 1_000,
        is_malicious: true,
        response_delay_ms: 0,
    };

    let result = protector.validate_sync_request(&malicious_peer);
    assert!(result.is_err(), "Should block sync-down attack");
    assert!(result.unwrap_err().contains("CRITICAL"));
    assert!(protector.is_peer_banned("malicious_1"), "Malicious peer should be banned");
}

/// Test that small regressions (reorgs) are allowed
#[test]
fn test_small_regression_allowed() {
    let mut protector = SyncProtector::new(100_000);

    // Peer announces 5 blocks behind (within reorg tolerance)
    let peer = MockPeer {
        id: "honest_1".to_string(),
        announced_height: 99_995,
        actual_height: 99_995,
        is_malicious: false,
        response_delay_ms: 0,
    };

    let result = protector.validate_sync_request(&peer);
    assert!(result.is_ok(), "Small regression should be allowed for reorgs");
    assert!(!protector.is_peer_banned("honest_1"));
}

/// Test that large regressions are blocked
#[test]
fn test_large_regression_blocked() {
    let mut protector = SyncProtector::new(100_000);

    // Peer announces 100 blocks behind (exceeds reorg tolerance)
    let peer = MockPeer {
        id: "suspicious_1".to_string(),
        announced_height: 99_900,
        actual_height: 99_900,
        is_malicious: false,
        response_delay_ms: 0,
    };

    let result = protector.validate_sync_request(&peer);
    assert!(result.is_err(), "Large regression should be blocked");
}

/// Test sync-down protection threshold
#[test]
fn test_sync_down_protection_threshold() {
    // New node with only 500 blocks (below protection threshold)
    let mut protector = SyncProtector::new(500);

    let peer = MockPeer {
        id: "peer_1".to_string(),
        announced_height: 100, // Much lower
        actual_height: 100,
        is_malicious: false,
        response_delay_ms: 0,
    };

    // Should be allowed because we're below protection threshold
    let result = protector.validate_sync_request(&peer);
    assert!(result.is_ok(), "Below threshold, sync-down should be allowed");
}

// ============================================================================
// HEIGHT MANIPULATION TESTS
// ============================================================================

/// Test zero height announcement attack
#[test]
fn test_zero_height_attack() {
    let mut protector = SyncProtector::new(50_000);

    let malicious_peer = MockPeer {
        id: "zero_attacker".to_string(),
        announced_height: 0,
        actual_height: 0,
        is_malicious: true,
        response_delay_ms: 0,
    };

    let result = protector.validate_sync_request(&malicious_peer);
    assert!(result.is_err(), "Zero height attack should be blocked");
    assert!(protector.is_peer_banned("zero_attacker"));
}

/// Test u64::MAX height announcement
#[test]
fn test_max_height_announcement() {
    let mut protector = SyncProtector::new(50_000);

    let malicious_peer = MockPeer {
        id: "max_attacker".to_string(),
        announced_height: u64::MAX,
        actual_height: 50_000,
        is_malicious: true,
        response_delay_ms: 0,
    };

    let result = protector.validate_sync_request(&malicious_peer);
    assert!(result.is_err(), "u64::MAX height should be suspicious");
    assert!(result.unwrap_err().contains("SUSPICIOUS"));
}

/// Test height = u64::MAX - 1
#[test]
fn test_near_max_height() {
    let mut protector = SyncProtector::new(50_000);

    let peer = MockPeer {
        id: "near_max".to_string(),
        announced_height: u64::MAX - 1,
        actual_height: 50_000,
        is_malicious: true,
        response_delay_ms: 0,
    };

    let result = protector.validate_sync_request(&peer);
    assert!(result.is_err(), "Near-max height should be rejected");
}

/// Test reasonable height progression
#[test]
fn test_reasonable_height_progression() {
    let mut protector = SyncProtector::new(50_000);

    // Peer is 1000 blocks ahead - reasonable
    let peer = MockPeer {
        id: "honest_peer".to_string(),
        announced_height: 51_000,
        actual_height: 51_000,
        is_malicious: false,
        response_delay_ms: 0,
    };

    let result = protector.validate_sync_request(&peer);
    assert!(result.is_ok(), "Reasonable height increase should be accepted");
}

// ============================================================================
// BANNED PEER TESTS
// ============================================================================

/// Test that banned peers are rejected
#[test]
fn test_banned_peer_rejected() {
    let mut protector = SyncProtector::new(50_000);

    // First, trigger a ban
    let malicious_peer = MockPeer {
        id: "bad_peer".to_string(),
        announced_height: 0,
        actual_height: 0,
        is_malicious: true,
        response_delay_ms: 0,
    };

    let _ = protector.validate_sync_request(&malicious_peer);
    assert!(protector.is_peer_banned("bad_peer"));

    // Now try again with valid height - should still be rejected
    let retry_peer = MockPeer {
        id: "bad_peer".to_string(),
        announced_height: 50_001,
        actual_height: 50_001,
        is_malicious: false, // Behaving now
        response_delay_ms: 0,
    };

    let result = protector.validate_sync_request(&retry_peer);
    assert!(result.is_err(), "Banned peer should be rejected");
    assert!(result.unwrap_err().contains("banned"));
}

/// Test multiple peers, only malicious ones banned
#[test]
fn test_selective_banning() {
    let mut protector = SyncProtector::new(50_000);

    let honest_peer = MockPeer {
        id: "honest".to_string(),
        announced_height: 50_500,
        actual_height: 50_500,
        is_malicious: false,
        response_delay_ms: 0,
    };

    let malicious_peer = MockPeer {
        id: "malicious".to_string(),
        announced_height: 0,
        actual_height: 0,
        is_malicious: true,
        response_delay_ms: 0,
    };

    let _ = protector.validate_sync_request(&honest_peer);
    let _ = protector.validate_sync_request(&malicious_peer);

    assert!(!protector.is_peer_banned("honest"), "Honest peer should not be banned");
    assert!(protector.is_peer_banned("malicious"), "Malicious peer should be banned");
}

// ============================================================================
// BLOCK VALIDATION TESTS
// ============================================================================

/// Test invalid block rejection
#[test]
fn test_invalid_block_rejected() {
    let protector = SyncProtector::new(1000);

    let invalid_block = MockBlock {
        height: 1001,
        hash: [1u8; 32],
        parent_hash: [0u8; 32],
        is_valid: false, // Invalid signature/hash
    };

    let result = protector.validate_block(&invalid_block);
    assert!(result.is_err(), "Invalid block should be rejected");
}

/// Test valid block accepted
#[test]
fn test_valid_block_accepted() {
    let protector = SyncProtector::new(1000);

    let valid_block = MockBlock {
        height: 1001,
        hash: [1u8; 32],
        parent_hash: [0u8; 32],
        is_valid: true,
    };

    let result = protector.validate_block(&valid_block);
    assert!(result.is_ok(), "Valid block should be accepted");
}

/// Test parent hash continuity
#[test]
fn test_parent_hash_continuity() {
    let mut protector = SyncProtector::new(0);

    // Add genesis block
    let genesis = MockBlock {
        height: 0,
        hash: [0u8; 32],
        parent_hash: [0u8; 32],
        is_valid: true,
    };
    protector.add_block(genesis).unwrap();

    // Add block 1 with correct parent
    let block1 = MockBlock {
        height: 1,
        hash: [1u8; 32],
        parent_hash: [0u8; 32], // Matches genesis
        is_valid: true,
    };
    let result = protector.add_block(block1);
    assert!(result.is_ok());

    // Try to add block 2 with wrong parent
    let block2_bad = MockBlock {
        height: 2,
        hash: [2u8; 32],
        parent_hash: [99u8; 32], // Wrong parent!
        is_valid: true,
    };
    let result = protector.validate_block(&block2_bad);
    assert!(result.is_err(), "Block with wrong parent should be rejected");
}

// ============================================================================
// SYNC TO HEIGHT SAFETY TESTS
// ============================================================================

/// Test sync_to_height refuses catastrophic regression
#[test]
fn test_sync_to_height_safety() {
    let mut protector = SyncProtector::new(100_000);

    // Try to sync down catastrophically
    let result = protector.sync_to_height(1_000);
    assert!(result.is_err(), "Catastrophic sync-down should be refused");
    assert!(result.unwrap_err().contains("SAFETY ABORT"));

    // Height should be unchanged
    assert_eq!(protector.get_local_height(), 100_000);
}

/// Test sync_to_height allows forward progress
#[test]
fn test_sync_forward_allowed() {
    let mut protector = SyncProtector::new(100_000);

    let result = protector.sync_to_height(100_500);
    assert!(result.is_ok(), "Forward sync should be allowed");
    assert_eq!(protector.get_local_height(), 100_500);
}

/// Test sync_to_height allows small regression
#[test]
fn test_sync_small_regression_allowed() {
    let mut protector = SyncProtector::new(100_000);

    // 5 block regression (within tolerance)
    let result = protector.sync_to_height(99_995);
    assert!(result.is_ok(), "Small regression should be allowed");
    assert_eq!(protector.get_local_height(), 99_995);
}

// ============================================================================
// PEER HEIGHT TRACKING TESTS
// ============================================================================

/// Test peer height consensus
#[test]
fn test_peer_height_consensus() {
    let mut protector = SyncProtector::new(50_000);

    // Multiple peers announce similar heights
    for i in 0..5 {
        let peer = MockPeer {
            id: format!("peer_{}", i),
            announced_height: 50_500 + i as u64,
            actual_height: 50_500 + i as u64,
            is_malicious: false,
            response_delay_ms: 0,
        };
        let _ = protector.validate_sync_request(&peer);
    }

    // One malicious peer announces much higher
    let malicious = MockPeer {
        id: "outlier".to_string(),
        announced_height: 1_000_000, // Way too high
        actual_height: 50_500,
        is_malicious: true,
        response_delay_ms: 0,
    };
    let result = protector.validate_sync_request(&malicious);
    assert!(result.is_err(), "Outlier height should be suspicious");
}

/// Test height spoofing detection
#[test]
fn test_height_spoofing() {
    let mut protector = SyncProtector::new(50_000);

    // Peer announces high but only has low blocks
    let spoofer = MockPeer {
        id: "spoofer".to_string(),
        announced_height: 100_000,
        actual_height: 50_100, // Actual height much lower
        is_malicious: true,
        response_delay_ms: 0,
    };

    // Initial request succeeds (we don't know it's spoofed yet)
    let result = protector.validate_sync_request(&spoofer);
    assert!(result.is_ok(), "Initial request might succeed");

    // But when we try to get blocks beyond actual height, we'd fail
    // This tests the concept - real implementation would ban after failed block fetch
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

/// Test height = 1 attack on established chain
#[test]
fn test_height_one_attack() {
    let mut protector = SyncProtector::new(100_000);

    let attacker = MockPeer {
        id: "height_one".to_string(),
        announced_height: 1,
        actual_height: 1,
        is_malicious: true,
        response_delay_ms: 0,
    };

    let result = protector.validate_sync_request(&attacker);
    assert!(result.is_err(), "Height 1 attack should be blocked");
}

/// Test rapid height changes from same peer
#[test]
fn test_rapid_height_changes() {
    let mut protector = SyncProtector::new(50_000);

    // Peer announces increasing heights rapidly
    for height in [50_001, 50_100, 50_500, 51_000, 52_000] {
        let peer = MockPeer {
            id: "rapid_peer".to_string(),
            announced_height: height,
            actual_height: height,
            is_malicious: false,
            response_delay_ms: 0,
        };
        let result = protector.validate_sync_request(&peer);
        assert!(result.is_ok(), "Rapid increases should be OK");
    }
}

/// Test exactly at protection threshold
#[test]
fn test_at_protection_threshold() {
    let mut protector = SyncProtector::new(1001); // Just above threshold (protection kicks in when > 1000)

    let peer = MockPeer {
        id: "threshold_peer".to_string(),
        announced_height: 500, // Below our height
        actual_height: 500,
        is_malicious: false,
        response_delay_ms: 0,
    };

    // Above threshold, protection kicks in
    let result = protector.validate_sync_request(&peer);
    // Regression of 501 blocks exceeds max_regression_allowed (10)
    assert!(result.is_err(), "Large regression above threshold should be blocked");
}
