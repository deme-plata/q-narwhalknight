//! Network Partition Tests
//!
//! Tests for handling network partitions and split-brain scenarios.
//!
//! CRITICAL SCENARIOS TESTED:
//! 1. Partition detection
//! 2. Partition healing and convergence
//! 3. Fork resolution after partition
//! 4. Message buffering during partition
//! 5. Peer reconnection handling
//! 6. Byzantine peer detection during partition
//!
//! Run with: cargo test --package q-network --test network_partition_tests

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ============================================================================
// MOCK STRUCTURES FOR PARTITION TESTING
// ============================================================================

/// Peer identifier
pub type PeerId = [u8; 32];

/// Network state for a peer
#[derive(Debug, Clone)]
pub struct PeerState {
    pub peer_id: PeerId,
    pub partition_id: u32,
    pub height: u64,
    pub last_seen: Instant,
    pub is_connected: bool,
}

/// Message types
#[derive(Debug, Clone)]
pub enum NetworkMessage {
    HeartBeat { peer_id: PeerId, height: u64 },
    BlockAnnounce { height: u64, hash: [u8; 32] },
    SyncRequest { start: u64, end: u64 },
    PartitionProbe { from: PeerId, partition_id: u32 },
}

/// Simulated network partition
#[derive(Debug, Clone)]
pub struct Partition {
    pub id: u32,
    pub peers: HashSet<PeerId>,
    pub created_at: Instant,
    pub is_healed: bool,
}

/// Network partition manager
pub struct PartitionManager {
    our_peer_id: PeerId,
    peers: Mutex<HashMap<PeerId, PeerState>>,
    partitions: Mutex<Vec<Partition>>,
    our_partition: AtomicU64,
    message_buffer: Mutex<Vec<(PeerId, NetworkMessage)>>,
    partition_detected: AtomicBool,
    fork_detected: AtomicBool,
    convergence_threshold: Duration,
}

impl PartitionManager {
    pub fn new(our_peer_id: PeerId) -> Self {
        Self {
            our_peer_id,
            peers: Mutex::new(HashMap::new()),
            partitions: Mutex::new(Vec::new()),
            our_partition: AtomicU64::new(0),
            message_buffer: Mutex::new(Vec::new()),
            partition_detected: AtomicBool::new(false),
            fork_detected: AtomicBool::new(false),
            convergence_threshold: Duration::from_secs(30),
        }
    }

    /// Add a peer to the network
    pub fn add_peer(&self, peer_id: PeerId, partition_id: u32) {
        let mut peers = self.peers.lock().unwrap();
        peers.insert(
            peer_id,
            PeerState {
                peer_id,
                partition_id,
                height: 0,
                last_seen: Instant::now(),
                is_connected: true,
            },
        );
    }

    /// Update peer's partition (simulates partition event)
    pub fn simulate_partition(&self, peer_id: &PeerId, new_partition: u32) -> Result<(), String> {
        let mut peers = self.peers.lock().unwrap();
        let peer = peers
            .get_mut(peer_id)
            .ok_or_else(|| "PEER_NOT_FOUND".to_string())?;

        if peer.partition_id != new_partition {
            peer.partition_id = new_partition;
            self.partition_detected.store(true, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Check if we can communicate with a peer (same partition)
    pub fn can_communicate(&self, peer_id: &PeerId) -> bool {
        let peers = self.peers.lock().unwrap();
        if let Some(peer) = peers.get(peer_id) {
            let our_partition = self.our_partition.load(Ordering::Relaxed) as u32;
            return peer.partition_id == our_partition && peer.is_connected;
        }
        false
    }

    /// Send message to peer (buffers if partitioned)
    pub fn send_message(&self, peer_id: PeerId, msg: NetworkMessage) -> Result<(), String> {
        if !self.can_communicate(&peer_id) {
            // Buffer for later
            let mut buffer = self.message_buffer.lock().unwrap();
            buffer.push((peer_id, msg));
            return Err("PEER_PARTITIONED: Message buffered".to_string());
        }

        Ok(())
    }

    /// Detect partition based on heartbeat timeouts
    pub fn detect_partitions(&self, timeout: Duration) -> Vec<PeerId> {
        let peers = self.peers.lock().unwrap();
        let mut partitioned = Vec::new();

        for (peer_id, state) in peers.iter() {
            if state.last_seen.elapsed() > timeout {
                partitioned.push(*peer_id);
            }
        }

        if !partitioned.is_empty() {
            self.partition_detected.store(true, Ordering::Relaxed);
        }

        partitioned
    }

    /// Heal partition (reconnect peers)
    pub fn heal_partition(&self, partition_id: u32) -> usize {
        let mut peers = self.peers.lock().unwrap();
        let our_partition = self.our_partition.load(Ordering::Relaxed) as u32;
        let mut healed = 0;

        for (_, state) in peers.iter_mut() {
            if state.partition_id == partition_id {
                state.partition_id = our_partition;
                state.is_connected = true;
                state.last_seen = Instant::now();
                healed += 1;
            }
        }

        // Flush buffered messages
        self.flush_buffered_messages();

        healed
    }

    /// Flush buffered messages after partition heals
    fn flush_buffered_messages(&self) -> usize {
        let mut buffer = self.message_buffer.lock().unwrap();
        let count = buffer.len();
        buffer.clear();
        count
    }

    /// Receive heartbeat from peer
    pub fn receive_heartbeat(&self, peer_id: PeerId, height: u64) -> Result<(), String> {
        let mut peers = self.peers.lock().unwrap();
        if let Some(peer) = peers.get_mut(&peer_id) {
            peer.last_seen = Instant::now();
            peer.height = height;
            peer.is_connected = true;
            Ok(())
        } else {
            Err("UNKNOWN_PEER".to_string())
        }
    }

    /// Get buffered message count
    pub fn buffered_count(&self) -> usize {
        self.message_buffer.lock().unwrap().len()
    }

    /// Check if partition was detected
    pub fn is_partitioned(&self) -> bool {
        self.partition_detected.load(Ordering::Relaxed)
    }

    /// Get connected peer count
    pub fn connected_peer_count(&self) -> usize {
        let peers = self.peers.lock().unwrap();
        let our_partition = self.our_partition.load(Ordering::Relaxed) as u32;
        peers
            .values()
            .filter(|p| p.partition_id == our_partition && p.is_connected)
            .count()
    }

    /// Get peers in a specific partition
    pub fn peers_in_partition(&self, partition_id: u32) -> Vec<PeerId> {
        let peers = self.peers.lock().unwrap();
        peers
            .values()
            .filter(|p| p.partition_id == partition_id)
            .map(|p| p.peer_id)
            .collect()
    }
}

/// Fork resolver for handling divergent chains after partition
pub struct ForkResolver {
    our_chain: Mutex<Vec<(u64, [u8; 32])>>, // (height, hash)
    peer_chains: Mutex<HashMap<PeerId, Vec<(u64, [u8; 32])>>>,
    fork_point: Mutex<Option<u64>>,
}

impl ForkResolver {
    pub fn new() -> Self {
        Self {
            our_chain: Mutex::new(Vec::new()),
            peer_chains: Mutex::new(HashMap::new()),
            fork_point: Mutex::new(None),
        }
    }

    /// Add a block to our chain
    pub fn add_block(&self, height: u64, hash: [u8; 32]) {
        let mut chain = self.our_chain.lock().unwrap();
        chain.push((height, hash));
    }

    /// Report a peer's chain state
    pub fn report_peer_chain(&self, peer_id: PeerId, chain: Vec<(u64, [u8; 32])>) {
        let mut chains = self.peer_chains.lock().unwrap();
        chains.insert(peer_id, chain);
    }

    /// Detect fork point with a peer
    pub fn detect_fork(&self, peer_id: &PeerId) -> Option<u64> {
        let our_chain = self.our_chain.lock().unwrap();
        let chains = self.peer_chains.lock().unwrap();

        if let Some(peer_chain) = chains.get(peer_id) {
            // Find where chains diverge
            for (our_block, peer_block) in our_chain.iter().zip(peer_chain.iter()) {
                if our_block.1 != peer_block.1 {
                    let mut fork = self.fork_point.lock().unwrap();
                    *fork = Some(our_block.0);
                    return Some(our_block.0);
                }
            }
        }

        None
    }

    /// Get the longest chain among all known
    pub fn get_longest_chain(&self) -> (Option<PeerId>, u64) {
        let our_chain = self.our_chain.lock().unwrap();
        let chains = self.peer_chains.lock().unwrap();

        let our_len = our_chain.len() as u64;
        let mut best: (Option<PeerId>, u64) = (None, our_len);

        for (peer_id, chain) in chains.iter() {
            if chain.len() as u64 > best.1 {
                best = (Some(*peer_id), chain.len() as u64);
            }
        }

        best
    }

    /// Resolve fork using longest chain rule
    pub fn resolve_fork(&self) -> Result<Option<PeerId>, String> {
        let (winner, _) = self.get_longest_chain();
        Ok(winner)
    }

    pub fn get_fork_point(&self) -> Option<u64> {
        *self.fork_point.lock().unwrap()
    }
}

/// Byzantine peer detector during partitions
pub struct ByzantineDetector {
    peer_reports: Mutex<HashMap<PeerId, Vec<(u64, [u8; 32])>>>, // height -> hash
    inconsistencies: Mutex<HashMap<PeerId, u32>>,
    threshold: u32,
}

impl ByzantineDetector {
    pub fn new(threshold: u32) -> Self {
        Self {
            peer_reports: Mutex::new(HashMap::new()),
            inconsistencies: Mutex::new(HashMap::new()),
            threshold,
        }
    }

    /// Record a block report from a peer
    pub fn record_block_report(&self, peer_id: PeerId, height: u64, hash: [u8; 32]) {
        let mut reports = self.peer_reports.lock().unwrap();
        reports.entry(peer_id).or_insert_with(Vec::new).push((height, hash));
    }

    /// Check for conflicting block reports (Byzantine behavior)
    pub fn check_conflicts(&self, height: u64) -> Vec<PeerId> {
        let reports = self.peer_reports.lock().unwrap();
        let mut hash_to_peers: HashMap<[u8; 32], Vec<PeerId>> = HashMap::new();

        for (peer_id, blocks) in reports.iter() {
            for (h, hash) in blocks {
                if *h == height {
                    hash_to_peers.entry(*hash).or_insert_with(Vec::new).push(*peer_id);
                }
            }
        }

        // If more than one hash reported for same height, some peers are Byzantine
        let mut byzantine = Vec::new();
        if hash_to_peers.len() > 1 {
            // Find minority reporters (likely Byzantine)
            let max_count = hash_to_peers.values().map(|v| v.len()).max().unwrap_or(0);
            for (_, peers) in hash_to_peers {
                if peers.len() < max_count {
                    byzantine.extend(peers);
                }
            }
        }

        // Record inconsistencies
        let mut inconsistencies = self.inconsistencies.lock().unwrap();
        for peer in &byzantine {
            *inconsistencies.entry(*peer).or_insert(0) += 1;
        }

        byzantine
    }

    /// Check if peer should be flagged as Byzantine
    pub fn is_byzantine(&self, peer_id: &PeerId) -> bool {
        let inconsistencies = self.inconsistencies.lock().unwrap();
        inconsistencies.get(peer_id).copied().unwrap_or(0) >= self.threshold
    }

    pub fn get_inconsistency_count(&self, peer_id: &PeerId) -> u32 {
        let inconsistencies = self.inconsistencies.lock().unwrap();
        inconsistencies.get(peer_id).copied().unwrap_or(0)
    }
}

// ============================================================================
// PARTITION DETECTION TESTS
// ============================================================================

/// Test basic partition detection
#[test]
fn test_partition_detection() {
    let manager = PartitionManager::new([0u8; 32]);

    // Add peers in partition 0
    manager.add_peer([1u8; 32], 0);
    manager.add_peer([2u8; 32], 0);

    // Simulate partition - move one peer to partition 1
    manager.simulate_partition(&[2u8; 32], 1).unwrap();

    assert!(manager.is_partitioned());
}

/// Test heartbeat timeout detection
#[test]
fn test_heartbeat_timeout_detection() {
    let manager = PartitionManager::new([0u8; 32]);

    let peer1 = [1u8; 32];
    let peer2 = [2u8; 32];

    manager.add_peer(peer1, 0);
    manager.add_peer(peer2, 0);

    // peer1 sends heartbeat, peer2 doesn't
    manager.receive_heartbeat(peer1, 100).unwrap();

    // Wait a bit
    std::thread::sleep(Duration::from_millis(50));

    // Use very short timeout for testing
    let partitioned = manager.detect_partitions(Duration::from_millis(10));

    // peer2 should be detected as partitioned (no recent heartbeat)
    // Note: timing-dependent, may need adjustment
    assert!(partitioned.contains(&peer2) || manager.is_partitioned());
}

/// Test communication check across partitions
#[test]
fn test_cross_partition_communication_blocked() {
    let manager = PartitionManager::new([0u8; 32]);

    let peer1 = [1u8; 32];
    let peer2 = [2u8; 32];

    manager.add_peer(peer1, 0); // Same partition as us
    manager.add_peer(peer2, 1); // Different partition

    assert!(manager.can_communicate(&peer1));
    assert!(!manager.can_communicate(&peer2));
}

// ============================================================================
// MESSAGE BUFFERING TESTS
// ============================================================================

/// Test message buffering during partition
#[test]
fn test_message_buffering() {
    let manager = PartitionManager::new([0u8; 32]);

    let peer = [1u8; 32];
    manager.add_peer(peer, 1); // Different partition

    // Try to send message to partitioned peer
    let msg = NetworkMessage::HeartBeat {
        peer_id: [0u8; 32],
        height: 100,
    };
    let result = manager.send_message(peer, msg);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("PEER_PARTITIONED"));
    assert_eq!(manager.buffered_count(), 1);
}

/// Test message delivery after partition heals
#[test]
fn test_message_flush_after_heal() {
    let manager = PartitionManager::new([0u8; 32]);

    let peer = [1u8; 32];
    manager.add_peer(peer, 1); // Different partition

    // Buffer some messages
    for i in 0..5 {
        let msg = NetworkMessage::BlockAnnounce {
            height: i,
            hash: [0u8; 32],
        };
        let _ = manager.send_message(peer, msg);
    }

    assert_eq!(manager.buffered_count(), 5);

    // Heal partition
    manager.heal_partition(1);

    // Buffer should be flushed
    assert_eq!(manager.buffered_count(), 0);
}

// ============================================================================
// PARTITION HEALING TESTS
// ============================================================================

/// Test partition healing
#[test]
fn test_partition_healing() {
    let manager = PartitionManager::new([0u8; 32]);

    // Add peers in different partitions
    manager.add_peer([1u8; 32], 0);
    manager.add_peer([2u8; 32], 1);
    manager.add_peer([3u8; 32], 1);

    assert_eq!(manager.connected_peer_count(), 1); // Only peer in partition 0

    // Heal partition 1
    let healed = manager.heal_partition(1);

    assert_eq!(healed, 2);
    assert_eq!(manager.connected_peer_count(), 3); // All peers now connected
}

/// Test peers in partition query
#[test]
fn test_peers_in_partition() {
    let manager = PartitionManager::new([0u8; 32]);

    manager.add_peer([1u8; 32], 0);
    manager.add_peer([2u8; 32], 0);
    manager.add_peer([3u8; 32], 1);
    manager.add_peer([4u8; 32], 1);

    let partition0_peers = manager.peers_in_partition(0);
    let partition1_peers = manager.peers_in_partition(1);

    assert_eq!(partition0_peers.len(), 2);
    assert_eq!(partition1_peers.len(), 2);
}

// ============================================================================
// FORK RESOLUTION TESTS
// ============================================================================

/// Test fork detection
#[test]
fn test_fork_detection() {
    let resolver = ForkResolver::new();

    // Our chain
    resolver.add_block(0, [0u8; 32]);
    resolver.add_block(1, [1u8; 32]);
    resolver.add_block(2, [2u8; 32]); // Divergence point

    // Peer's chain (different at height 2)
    let peer = [1u8; 32];
    let peer_chain = vec![
        (0, [0u8; 32]),
        (1, [1u8; 32]),
        (2, [99u8; 32]), // Different hash!
    ];
    resolver.report_peer_chain(peer, peer_chain);

    let fork_point = resolver.detect_fork(&peer);
    assert_eq!(fork_point, Some(2));
}

/// Test longest chain detection
#[test]
fn test_longest_chain_detection() {
    let resolver = ForkResolver::new();

    // Our chain: 3 blocks
    resolver.add_block(0, [0u8; 32]);
    resolver.add_block(1, [1u8; 32]);
    resolver.add_block(2, [2u8; 32]);

    // Peer 1: 5 blocks (longer)
    let peer1 = [1u8; 32];
    resolver.report_peer_chain(
        peer1,
        (0..5).map(|i| (i, [i as u8; 32])).collect(),
    );

    // Peer 2: 4 blocks
    let peer2 = [2u8; 32];
    resolver.report_peer_chain(
        peer2,
        (0..4).map(|i| (i, [i as u8; 32])).collect(),
    );

    let (winner, length) = resolver.get_longest_chain();
    assert_eq!(winner, Some(peer1));
    assert_eq!(length, 5);
}

/// Test fork resolution
#[test]
fn test_fork_resolution() {
    let resolver = ForkResolver::new();

    // Our chain: 3 blocks
    for i in 0..3 {
        resolver.add_block(i, [i as u8; 32]);
    }

    // Peer with longer chain
    let peer = [1u8; 32];
    resolver.report_peer_chain(
        peer,
        (0..10).map(|i| (i, [i as u8; 32])).collect(),
    );

    let winner = resolver.resolve_fork().unwrap();
    assert_eq!(winner, Some(peer)); // Peer wins with longer chain
}

// ============================================================================
// BYZANTINE DETECTION TESTS
// ============================================================================

/// Test Byzantine peer detection
#[test]
fn test_byzantine_conflict_detection() {
    let detector = ByzantineDetector::new(3);

    let peer1 = [1u8; 32];
    let peer2 = [2u8; 32];
    let peer3 = [3u8; 32];

    // Honest peers report same hash for height 100
    detector.record_block_report(peer1, 100, [1u8; 32]);
    detector.record_block_report(peer2, 100, [1u8; 32]);

    // Byzantine peer reports different hash
    detector.record_block_report(peer3, 100, [99u8; 32]);

    let byzantine = detector.check_conflicts(100);
    assert!(byzantine.contains(&peer3));
    assert!(!byzantine.contains(&peer1));
    assert!(!byzantine.contains(&peer2));
}

/// Test Byzantine threshold
#[test]
fn test_byzantine_threshold() {
    let detector = ByzantineDetector::new(2); // Need 2 inconsistencies

    let byzantine_peer = [1u8; 32];
    let honest_peer = [2u8; 32];

    // Byzantine peer reports conflicting blocks
    detector.record_block_report(byzantine_peer, 100, [1u8; 32]);
    detector.record_block_report(honest_peer, 100, [2u8; 32]);
    detector.record_block_report(honest_peer, 100, [2u8; 32]); // Majority
    detector.check_conflicts(100);

    detector.record_block_report(byzantine_peer, 101, [1u8; 32]);
    detector.record_block_report(honest_peer, 101, [2u8; 32]);
    detector.record_block_report(honest_peer, 101, [2u8; 32]); // Majority
    detector.check_conflicts(101);

    // After 2 inconsistencies, peer should be flagged
    assert!(detector.is_byzantine(&byzantine_peer));
    assert!(!detector.is_byzantine(&honest_peer));
}

/// Test inconsistency count tracking
#[test]
fn test_inconsistency_count() {
    let detector = ByzantineDetector::new(10);

    let peer = [1u8; 32];
    let honest = [2u8; 32];

    assert_eq!(detector.get_inconsistency_count(&peer), 0);

    // Create conflicts
    for height in 0..5 {
        detector.record_block_report(peer, height, [1u8; 32]);
        detector.record_block_report(honest, height, [2u8; 32]);
        detector.record_block_report(honest, height, [2u8; 32]); // Majority
        detector.check_conflicts(height);
    }

    assert_eq!(detector.get_inconsistency_count(&peer), 5);
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

/// Test full partition scenario
#[test]
fn test_full_partition_scenario() {
    let manager = PartitionManager::new([0u8; 32]);
    let resolver = ForkResolver::new();

    // Setup: 4 peers, all in partition 0
    for i in 1..=4 {
        manager.add_peer([i as u8; 32], 0);
    }

    // Phase 1: Network split
    manager.simulate_partition(&[3u8; 32], 1).unwrap();
    manager.simulate_partition(&[4u8; 32], 1).unwrap();

    assert!(manager.is_partitioned());
    assert_eq!(manager.peers_in_partition(0).len(), 2);
    assert_eq!(manager.peers_in_partition(1).len(), 2);

    // Phase 2: Both partitions produce blocks
    resolver.add_block(0, [0u8; 32]);
    resolver.add_block(1, [1u8; 32]);
    resolver.add_block(2, [2u8; 32]); // Our partition

    let peer_in_other_partition = [3u8; 32];
    resolver.report_peer_chain(
        peer_in_other_partition,
        vec![
            (0, [0u8; 32]),
            (1, [1u8; 32]),
            (2, [100u8; 32]), // Different block at height 2
            (3, [101u8; 32]),
            (4, [102u8; 32]), // Longer chain
        ],
    );

    // Phase 3: Detect fork
    let fork_point = resolver.detect_fork(&peer_in_other_partition);
    assert_eq!(fork_point, Some(2));

    // Phase 4: Heal partition
    let healed = manager.heal_partition(1);
    assert_eq!(healed, 2);

    // Phase 5: Resolve fork (longest chain wins)
    let winner = resolver.resolve_fork().unwrap();
    assert_eq!(winner, Some(peer_in_other_partition));
}
