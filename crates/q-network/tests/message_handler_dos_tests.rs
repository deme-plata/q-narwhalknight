//! Message Handler DoS Protection Tests
//!
//! Tests for denial-of-service attack protection in the P2P message handler.
//!
//! CRITICAL SCENARIOS TESTED:
//! 1. Billion-round sync request rejection
//! 2. Malformed message rejection
//! 3. Cache exhaustion attack prevention
//! 4. Rapid message flooding protection
//! 5. Oversized message rejection
//! 6. Invalid peer identity handling
//!
//! Run with: cargo test --package q-network --test message_handler_dos_tests

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ============================================================================
// MOCK STRUCTURES FOR DOS TESTING
// ============================================================================

/// Message types that can be received
#[derive(Debug, Clone, PartialEq)]
pub enum MessageType {
    SyncRequest { start_height: u64, end_height: u64 },
    BlockAnnouncement { height: u64, hash: [u8; 32] },
    PeerHeightUpdate { peer_id: [u8; 32], height: u64 },
    VertexBroadcast { data: Vec<u8> },
    TurboSyncRequest { heights: Vec<u64> },
    Unknown { type_id: u8 },
}

/// Rate limiter for message flooding protection
pub struct RateLimiter {
    windows: Mutex<HashMap<[u8; 32], Vec<Instant>>>,
    max_messages_per_window: usize,
    window_duration: Duration,
}

impl RateLimiter {
    pub fn new(max_messages: usize, window: Duration) -> Self {
        Self {
            windows: Mutex::new(HashMap::new()),
            max_messages_per_window: max_messages,
            window_duration: window,
        }
    }

    /// Check if peer should be rate limited
    pub fn check_rate_limit(&self, peer_id: &[u8; 32]) -> Result<(), String> {
        let mut windows = self.windows.lock().unwrap();
        let now = Instant::now();

        let timestamps = windows.entry(*peer_id).or_insert_with(Vec::new);

        // Remove old timestamps outside window
        timestamps.retain(|t| now.duration_since(*t) < self.window_duration);

        if timestamps.len() >= self.max_messages_per_window {
            return Err(format!(
                "RATE_LIMITED: Peer {:?} exceeded {} messages in {:?}",
                hex::encode(&peer_id[..4]),
                self.max_messages_per_window,
                self.window_duration
            ));
        }

        timestamps.push(now);
        Ok(())
    }

    pub fn get_message_count(&self, peer_id: &[u8; 32]) -> usize {
        let windows = self.windows.lock().unwrap();
        windows.get(peer_id).map(|v| v.len()).unwrap_or(0)
    }
}

/// Message size validator
pub struct MessageValidator {
    max_message_size: usize,
    max_sync_range: u64,
    max_turbo_sync_heights: usize,
    max_vertex_size: usize,
}

impl MessageValidator {
    pub fn new() -> Self {
        Self {
            max_message_size: 10 * 1024 * 1024, // 10 MB
            max_sync_range: 100_000,            // Max blocks per request
            max_turbo_sync_heights: 1_000,      // Max heights in turbo sync
            max_vertex_size: 1024 * 1024,       // 1 MB per vertex
        }
    }

    /// Validate incoming message
    pub fn validate_message(&self, msg: &MessageType, raw_size: usize) -> Result<(), String> {
        // Check raw message size
        if raw_size > self.max_message_size {
            return Err(format!(
                "OVERSIZED_MESSAGE: {} bytes exceeds max {}",
                raw_size, self.max_message_size
            ));
        }

        match msg {
            MessageType::SyncRequest {
                start_height,
                end_height,
            } => {
                // Validate sync range
                if end_height < start_height {
                    return Err("INVALID_SYNC_RANGE: end_height < start_height".to_string());
                }

                let range = end_height.saturating_sub(*start_height);

                // CRITICAL: Check for billion-round attacks FIRST (before normal range check)
                if range > 1_000_000_000 {
                    return Err(
                        "BILLION_ROUND_ATTACK: Refusing absurd sync range request".to_string(),
                    );
                }

                // Check against normal max range
                if range > self.max_sync_range {
                    return Err(format!(
                        "EXCESSIVE_SYNC_RANGE: {} blocks requested (max: {})",
                        range, self.max_sync_range
                    ));
                }
            }

            MessageType::TurboSyncRequest { heights } => {
                if heights.len() > self.max_turbo_sync_heights {
                    return Err(format!(
                        "EXCESSIVE_TURBO_SYNC: {} heights requested (max: {})",
                        heights.len(),
                        self.max_turbo_sync_heights
                    ));
                }
            }

            MessageType::VertexBroadcast { data } => {
                if data.len() > self.max_vertex_size {
                    return Err(format!(
                        "OVERSIZED_VERTEX: {} bytes (max: {})",
                        data.len(),
                        self.max_vertex_size
                    ));
                }
            }

            MessageType::Unknown { type_id } => {
                return Err(format!("UNKNOWN_MESSAGE_TYPE: type_id={}", type_id));
            }

            _ => {}
        }

        Ok(())
    }
}

/// Cache with size limits to prevent exhaustion
pub struct BoundedCache<K, V> {
    data: Mutex<HashMap<K, (V, Instant)>>,
    max_entries: usize,
    max_memory: usize,
    current_memory: AtomicUsize,
    evicted_count: AtomicU64,
}

impl<K: Eq + std::hash::Hash + Clone, V: Clone> BoundedCache<K, V> {
    pub fn new(max_entries: usize, max_memory: usize) -> Self {
        Self {
            data: Mutex::new(HashMap::new()),
            max_entries,
            max_memory,
            current_memory: AtomicUsize::new(0),
            evicted_count: AtomicU64::new(0),
        }
    }

    pub fn insert(&self, key: K, value: V, size: usize) -> Result<(), String> {
        // Check memory limit
        if self.current_memory.load(Ordering::Relaxed) + size > self.max_memory {
            return Err(format!(
                "CACHE_MEMORY_EXHAUSTED: Would exceed {} byte limit",
                self.max_memory
            ));
        }

        let mut data = self.data.lock().unwrap();

        // Check entry limit
        if data.len() >= self.max_entries && !data.contains_key(&key) {
            // Evict oldest entry
            if let Some(oldest_key) = data
                .iter()
                .min_by_key(|(_, (_, time))| *time)
                .map(|(k, _)| k.clone())
            {
                data.remove(&oldest_key);
                self.evicted_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        data.insert(key, (value, Instant::now()));
        self.current_memory.fetch_add(size, Ordering::Relaxed);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.data.lock().unwrap().len()
    }

    pub fn evicted_count(&self) -> u64 {
        self.evicted_count.load(Ordering::Relaxed)
    }
}

/// Peer identity validator
pub struct PeerValidator {
    banned_peers: Mutex<HashMap<[u8; 32], (Instant, String)>>,
    ban_duration: Duration,
    violation_counts: Mutex<HashMap<[u8; 32], u32>>,
    max_violations: u32,
}

impl PeerValidator {
    pub fn new(ban_duration: Duration, max_violations: u32) -> Self {
        Self {
            banned_peers: Mutex::new(HashMap::new()),
            ban_duration,
            violation_counts: Mutex::new(HashMap::new()),
            max_violations,
        }
    }

    /// Check if peer is banned
    pub fn is_banned(&self, peer_id: &[u8; 32]) -> bool {
        let banned = self.banned_peers.lock().unwrap();
        if let Some((ban_time, _)) = banned.get(peer_id) {
            if ban_time.elapsed() < self.ban_duration {
                return true;
            }
        }
        false
    }

    /// Record a violation and potentially ban
    pub fn record_violation(&self, peer_id: &[u8; 32], reason: &str) -> bool {
        let mut violations = self.violation_counts.lock().unwrap();
        let count = violations.entry(*peer_id).or_insert(0);
        *count += 1;

        if *count >= self.max_violations {
            let mut banned = self.banned_peers.lock().unwrap();
            banned.insert(*peer_id, (Instant::now(), reason.to_string()));
            return true; // Peer was banned
        }
        false
    }

    pub fn get_violation_count(&self, peer_id: &[u8; 32]) -> u32 {
        let violations = self.violation_counts.lock().unwrap();
        *violations.get(peer_id).unwrap_or(&0)
    }
}

/// Complete message handler with DoS protection
pub struct MessageHandler {
    rate_limiter: RateLimiter,
    validator: MessageValidator,
    peer_validator: PeerValidator,
    processed_count: AtomicU64,
    rejected_count: AtomicU64,
}

impl MessageHandler {
    pub fn new() -> Self {
        Self {
            rate_limiter: RateLimiter::new(100, Duration::from_secs(1)), // 100 msg/sec
            validator: MessageValidator::new(),
            peer_validator: PeerValidator::new(Duration::from_secs(300), 5), // 5 min ban after 5 violations
            processed_count: AtomicU64::new(0),
            rejected_count: AtomicU64::new(0),
        }
    }

    /// Handle incoming message with full DoS protection
    pub fn handle_message(
        &self,
        peer_id: &[u8; 32],
        msg: MessageType,
        raw_size: usize,
    ) -> Result<(), String> {
        // 1. Check if peer is banned
        if self.peer_validator.is_banned(peer_id) {
            self.rejected_count.fetch_add(1, Ordering::Relaxed);
            return Err("BANNED_PEER: Connection from banned peer rejected".to_string());
        }

        // 2. Check rate limit
        if let Err(e) = self.rate_limiter.check_rate_limit(peer_id) {
            self.peer_validator.record_violation(peer_id, "rate_limit");
            self.rejected_count.fetch_add(1, Ordering::Relaxed);
            return Err(e);
        }

        // 3. Validate message
        if let Err(e) = self.validator.validate_message(&msg, raw_size) {
            let was_banned = self.peer_validator.record_violation(peer_id, &e);
            self.rejected_count.fetch_add(1, Ordering::Relaxed);
            if was_banned {
                return Err(format!("{} - PEER BANNED", e));
            }
            return Err(e);
        }

        // Message accepted
        self.processed_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn processed_count(&self) -> u64 {
        self.processed_count.load(Ordering::Relaxed)
    }

    pub fn rejected_count(&self) -> u64 {
        self.rejected_count.load(Ordering::Relaxed)
    }
}

// ============================================================================
// RATE LIMITING TESTS
// ============================================================================

/// Test basic rate limiting
#[test]
fn test_rate_limit_basic() {
    let limiter = RateLimiter::new(5, Duration::from_secs(1));
    let peer = [1u8; 32];

    // First 5 should pass
    for _ in 0..5 {
        assert!(limiter.check_rate_limit(&peer).is_ok());
    }

    // 6th should fail
    let result = limiter.check_rate_limit(&peer);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("RATE_LIMITED"));
}

/// Test rate limit window expiration
#[test]
fn test_rate_limit_window_expiration() {
    let limiter = RateLimiter::new(2, Duration::from_millis(50));
    let peer = [1u8; 32];

    // Use up the limit
    limiter.check_rate_limit(&peer).unwrap();
    limiter.check_rate_limit(&peer).unwrap();

    // Should be rate limited
    assert!(limiter.check_rate_limit(&peer).is_err());

    // Wait for window to expire
    std::thread::sleep(Duration::from_millis(60));

    // Should be allowed again
    assert!(limiter.check_rate_limit(&peer).is_ok());
}

/// Test independent rate limits per peer
#[test]
fn test_rate_limit_per_peer() {
    let limiter = RateLimiter::new(2, Duration::from_secs(1));
    let peer1 = [1u8; 32];
    let peer2 = [2u8; 32];

    // Exhaust peer1's limit
    limiter.check_rate_limit(&peer1).unwrap();
    limiter.check_rate_limit(&peer1).unwrap();
    assert!(limiter.check_rate_limit(&peer1).is_err());

    // peer2 should still be allowed
    assert!(limiter.check_rate_limit(&peer2).is_ok());
}

// ============================================================================
// MESSAGE VALIDATION TESTS
// ============================================================================

/// Test billion-round sync attack rejection
#[test]
fn test_billion_round_sync_attack_rejected() {
    let validator = MessageValidator::new();

    let msg = MessageType::SyncRequest {
        start_height: 0,
        end_height: 2_000_000_000, // 2 billion blocks
    };

    let result = validator.validate_message(&msg, 100);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("BILLION_ROUND_ATTACK"));
}

/// Test excessive sync range rejection
#[test]
fn test_excessive_sync_range_rejected() {
    let validator = MessageValidator::new();

    let msg = MessageType::SyncRequest {
        start_height: 0,
        end_height: 500_000, // 500k blocks (exceeds 100k max)
    };

    let result = validator.validate_message(&msg, 100);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("EXCESSIVE_SYNC_RANGE"));
}

/// Test invalid sync range (end < start)
#[test]
fn test_invalid_sync_range_rejected() {
    let validator = MessageValidator::new();

    let msg = MessageType::SyncRequest {
        start_height: 1000,
        end_height: 500, // Invalid: end < start
    };

    let result = validator.validate_message(&msg, 100);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("INVALID_SYNC_RANGE"));
}

/// Test valid sync request accepted
#[test]
fn test_valid_sync_request_accepted() {
    let validator = MessageValidator::new();

    let msg = MessageType::SyncRequest {
        start_height: 0,
        end_height: 1000, // 1000 blocks - valid
    };

    let result = validator.validate_message(&msg, 100);
    assert!(result.is_ok());
}

/// Test oversized message rejection
#[test]
fn test_oversized_message_rejected() {
    let validator = MessageValidator::new();

    let msg = MessageType::BlockAnnouncement {
        height: 100,
        hash: [0u8; 32],
    };

    // 15 MB message - exceeds 10 MB limit
    let result = validator.validate_message(&msg, 15 * 1024 * 1024);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("OVERSIZED_MESSAGE"));
}

/// Test oversized vertex rejection
#[test]
fn test_oversized_vertex_rejected() {
    let validator = MessageValidator::new();

    let msg = MessageType::VertexBroadcast {
        data: vec![0u8; 2 * 1024 * 1024], // 2 MB vertex
    };

    let result = validator.validate_message(&msg, 2 * 1024 * 1024);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("OVERSIZED_VERTEX"));
}

/// Test excessive turbo sync heights rejection
#[test]
fn test_excessive_turbo_sync_heights_rejected() {
    let validator = MessageValidator::new();

    let msg = MessageType::TurboSyncRequest {
        heights: (0..2000).collect(), // 2000 heights - exceeds 1000 max
    };

    let result = validator.validate_message(&msg, 8000);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("EXCESSIVE_TURBO_SYNC"));
}

/// Test unknown message type rejection
#[test]
fn test_unknown_message_type_rejected() {
    let validator = MessageValidator::new();

    let msg = MessageType::Unknown { type_id: 255 };

    let result = validator.validate_message(&msg, 100);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("UNKNOWN_MESSAGE_TYPE"));
}

// ============================================================================
// CACHE EXHAUSTION TESTS
// ============================================================================

/// Test cache entry limit enforcement
#[test]
fn test_cache_entry_limit() {
    let cache: BoundedCache<u64, Vec<u8>> = BoundedCache::new(3, 1024 * 1024);

    // Insert 3 entries
    cache.insert(1, vec![1], 10).unwrap();
    cache.insert(2, vec![2], 10).unwrap();
    cache.insert(3, vec![3], 10).unwrap();

    assert_eq!(cache.len(), 3);

    // 4th entry should cause eviction
    cache.insert(4, vec![4], 10).unwrap();
    assert_eq!(cache.len(), 3);
    assert_eq!(cache.evicted_count(), 1);
}

/// Test cache memory limit enforcement
#[test]
fn test_cache_memory_limit() {
    let cache: BoundedCache<u64, Vec<u8>> = BoundedCache::new(100, 1000); // 1000 bytes max

    // Insert entries up to limit
    cache.insert(1, vec![1; 400], 400).unwrap();
    cache.insert(2, vec![2; 400], 400).unwrap();

    // This should fail - would exceed memory limit
    let result = cache.insert(3, vec![3; 400], 400);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("CACHE_MEMORY_EXHAUSTED"));
}

// ============================================================================
// PEER BANNING TESTS
// ============================================================================

/// Test peer gets banned after violations
#[test]
fn test_peer_banned_after_violations() {
    let validator = PeerValidator::new(Duration::from_secs(300), 3);
    let peer = [1u8; 32];

    assert!(!validator.is_banned(&peer));

    // Record violations
    validator.record_violation(&peer, "test1");
    assert!(!validator.is_banned(&peer));

    validator.record_violation(&peer, "test2");
    assert!(!validator.is_banned(&peer));

    // Third violation should trigger ban
    let was_banned = validator.record_violation(&peer, "test3");
    assert!(was_banned);
    assert!(validator.is_banned(&peer));
}

/// Test violation count tracking
#[test]
fn test_violation_count_tracking() {
    let validator = PeerValidator::new(Duration::from_secs(300), 10);
    let peer = [1u8; 32];

    assert_eq!(validator.get_violation_count(&peer), 0);

    validator.record_violation(&peer, "test");
    assert_eq!(validator.get_violation_count(&peer), 1);

    validator.record_violation(&peer, "test");
    assert_eq!(validator.get_violation_count(&peer), 2);
}

// ============================================================================
// INTEGRATED MESSAGE HANDLER TESTS
// ============================================================================

/// Test message handler rejects banned peer
#[test]
fn test_handler_rejects_banned_peer() {
    let handler = MessageHandler::new();
    let peer = [1u8; 32];

    // Generate violations to get banned
    for _ in 0..5 {
        let msg = MessageType::Unknown { type_id: 255 };
        let _ = handler.handle_message(&peer, msg, 100);
    }

    // Now should be banned
    let msg = MessageType::BlockAnnouncement {
        height: 100,
        hash: [0u8; 32],
    };
    let result = handler.handle_message(&peer, msg, 100);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("BANNED_PEER"));
}

/// Test message handler processes valid messages
#[test]
fn test_handler_processes_valid_messages() {
    let handler = MessageHandler::new();
    let peer = [1u8; 32];

    let msg = MessageType::BlockAnnouncement {
        height: 100,
        hash: [0u8; 32],
    };

    let result = handler.handle_message(&peer, msg, 100);
    assert!(result.is_ok());
    assert_eq!(handler.processed_count(), 1);
    assert_eq!(handler.rejected_count(), 0);
}

/// Test rapid message flooding gets rate limited
#[test]
fn test_rapid_flooding_rate_limited() {
    let handler = MessageHandler::new();
    let peer = [1u8; 32];

    let mut accepted = 0;
    let mut rejected = 0;

    // Send 150 messages rapidly (limit is 100/sec)
    for _ in 0..150 {
        let msg = MessageType::BlockAnnouncement {
            height: 100,
            hash: [0u8; 32],
        };
        match handler.handle_message(&peer, msg, 100) {
            Ok(_) => accepted += 1,
            Err(_) => rejected += 1,
        }
    }

    // Should have accepted ~100 and rejected ~50
    assert!(accepted <= 100);
    assert!(rejected >= 50);
    assert_eq!(handler.processed_count(), accepted as u64);
}

/// Test handler tracks statistics correctly
#[test]
fn test_handler_statistics() {
    let handler = MessageHandler::new();
    let peer = [1u8; 32];

    // Send valid message
    let msg1 = MessageType::BlockAnnouncement {
        height: 100,
        hash: [0u8; 32],
    };
    handler.handle_message(&peer, msg1, 100).unwrap();

    // Send invalid message
    let msg2 = MessageType::Unknown { type_id: 255 };
    let _ = handler.handle_message(&peer, msg2, 100);

    assert_eq!(handler.processed_count(), 1);
    assert_eq!(handler.rejected_count(), 1);
}

/// Test handler enforces all protections together
#[test]
fn test_combined_protection() {
    let handler = MessageHandler::new();
    let attacker = [99u8; 32];

    // Try various attacks
    let attacks = vec![
        // Billion-round attack
        MessageType::SyncRequest {
            start_height: 0,
            end_height: 5_000_000_000,
        },
        // Unknown message type
        MessageType::Unknown { type_id: 200 },
        // Excessive turbo sync
        MessageType::TurboSyncRequest {
            heights: (0..5000).collect(),
        },
    ];

    for attack in attacks {
        let result = handler.handle_message(&attacker, attack, 100);
        assert!(result.is_err());
    }

    // After enough violations, peer should be banned
    assert!(handler.rejected_count() >= 3);
}
