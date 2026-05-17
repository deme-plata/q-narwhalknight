//! CRDT-based Distributed PPLNS State Machine
//!
//! Uses Conflict-free Replicated Data Types (CRDTs) for:
//! - Automatic state synchronization across pool nodes
//! - No coordination required for merges
//! - Eventually consistent reward calculations

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

use super::{DistributedShare, PeerIdBytes, ShareId};
use crate::pplns::RewardEntry;

/// Hash of PPLNS state for quick comparison
pub type PPLNSStateHash = [u8; 32];

/// CRDT-based PPLNS state that merges automatically
///
/// Uses G-Counters for difficulty tracking and G-Set for shares,
/// ensuring convergence regardless of message ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedPPLNS {
    /// Worker difficulty contributions (G-Counter per worker)
    /// Key: wallet_address, Value: map of (node_id -> difficulty contributed via that node)
    worker_difficulty: BTreeMap<String, GCounter>,

    /// Shares in current window (G-Set - grow only)
    /// Once a share is added, it's never removed (until window slides)
    window_shares: GSet<ShareId>,

    /// Current round number (last block height found)
    pub round_number: u64,

    /// Network difficulty at round start
    pub network_difficulty: f64,

    /// Window size multiplier (N factor)
    pub n_factor: f64,

    /// Vector clock for state versioning
    vclock: VectorClock,

    /// Total difficulty in window (cached)
    cached_total_difficulty: f64,

    /// Last state hash (for quick comparison)
    cached_state_hash: Option<PPLNSStateHash>,
}

/// G-Counter (Grow-only Counter) CRDT
///
/// Each node tracks its own increment count, and the total
/// is the sum across all nodes. Merging takes the max per node.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GCounter {
    /// Counts per node: node_id -> count
    counts: BTreeMap<PeerIdBytes, u64>,
}

impl GCounter {
    /// Create new empty counter
    pub fn new() -> Self {
        Self {
            counts: BTreeMap::new(),
        }
    }

    /// Increment counter for a specific node
    pub fn increment(&mut self, node_id: PeerIdBytes, amount: u64) {
        *self.counts.entry(node_id).or_insert(0) += amount;
    }

    /// Get total count across all nodes
    pub fn read(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Merge with another G-Counter (takes max per node)
    pub fn merge(&mut self, other: &Self) {
        for (node_id, &count) in &other.counts {
            let entry = self.counts.entry(*node_id).or_insert(0);
            *entry = (*entry).max(count);
        }
    }
}

/// G-Set (Grow-only Set) CRDT
///
/// Elements can only be added, never removed.
/// Merging is simple set union.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GSet<T: Ord + Clone + std::hash::Hash> {
    elements: HashSet<T>,
}

impl<T: Ord + Clone + std::hash::Hash> GSet<T> {
    /// Create new empty set
    pub fn new() -> Self {
        Self {
            elements: HashSet::new(),
        }
    }

    /// Add element to set
    pub fn add(&mut self, element: T) {
        self.elements.insert(element);
    }

    /// Check if element exists
    pub fn contains(&self, element: &T) -> bool {
        self.elements.contains(element)
    }

    /// Get number of elements
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Merge with another G-Set (union)
    pub fn merge(&mut self, other: &Self) {
        self.elements.extend(other.elements.iter().cloned());
    }

    /// Iterate over elements
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.elements.iter()
    }
}

/// Vector Clock for causality tracking
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorClock {
    /// Logical timestamps per node
    clocks: BTreeMap<PeerIdBytes, u64>,
}

impl VectorClock {
    /// Create new vector clock
    pub fn new() -> Self {
        Self {
            clocks: BTreeMap::new(),
        }
    }

    /// Increment clock for a node
    pub fn increment(&mut self, node_id: PeerIdBytes) {
        *self.clocks.entry(node_id).or_insert(0) += 1;
    }

    /// Merge with another vector clock (takes max per node)
    pub fn merge(&mut self, other: &Self) {
        for (node_id, &clock) in &other.clocks {
            let entry = self.clocks.entry(*node_id).or_insert(0);
            *entry = (*entry).max(clock);
        }
    }

    /// Check if this clock happened-before another
    pub fn happened_before(&self, other: &Self) -> bool {
        let mut dominated = false;
        for (node_id, &our_clock) in &self.clocks {
            let their_clock = other.clocks.get(node_id).copied().unwrap_or(0);
            if our_clock > their_clock {
                return false; // We're not dominated
            }
            if our_clock < their_clock {
                dominated = true;
            }
        }
        // Check nodes they have that we don't
        for (node_id, &their_clock) in &other.clocks {
            if !self.clocks.contains_key(node_id) && their_clock > 0 {
                dominated = true;
            }
        }
        dominated
    }
}

impl DistributedPPLNS {
    /// Create new PPLNS state
    pub fn new(n_factor: f64) -> Self {
        Self {
            worker_difficulty: BTreeMap::new(),
            window_shares: GSet::new(),
            round_number: 0,
            network_difficulty: 1.0,
            n_factor,
            vclock: VectorClock::new(),
            cached_total_difficulty: 0.0,
            cached_state_hash: None,
        }
    }

    /// Merge state from another node (CRDT merge is commutative + idempotent)
    pub fn merge(&mut self, other: &Self) {
        // Merge worker difficulty G-Counters
        for (wallet, counter) in &other.worker_difficulty {
            self.worker_difficulty
                .entry(wallet.clone())
                .or_insert_with(GCounter::new)
                .merge(counter);
        }

        // Merge share G-Set
        self.window_shares.merge(&other.window_shares);

        // Vector clock merge
        self.vclock.merge(&other.vclock);

        // Round number takes max (monotonically increasing)
        self.round_number = self.round_number.max(other.round_number);

        // Network difficulty from higher round
        if other.round_number > self.round_number {
            self.network_difficulty = other.network_difficulty;
        }

        // Invalidate cache
        self.invalidate_cache();
    }

    /// Add share to PPLNS window
    pub fn add_share(&mut self, node_id: PeerIdBytes, share: &DistributedShare) {
        // Check for duplicate
        if self.window_shares.contains(&share.share_id) {
            return;
        }

        // Increment worker's difficulty counter
        self.worker_difficulty
            .entry(share.wallet_address().to_string())
            .or_insert_with(GCounter::new)
            .increment(node_id, (share.difficulty * 1_000_000.0) as u64);

        // Add to share set
        self.window_shares.add(share.share_id);

        // Increment our vector clock
        self.vclock.increment(node_id);

        // Invalidate cache
        self.invalidate_cache();
    }

    /// Start new round (block was found)
    pub fn new_round(&mut self, block_height: u64, network_difficulty: f64) {
        self.round_number = block_height;
        self.network_difficulty = network_difficulty;

        // Note: In PPLNS, we don't clear shares on new round
        // The window slides based on difficulty sum, not rounds

        // Invalidate cache
        self.invalidate_cache();
    }

    /// Get total difficulty in window
    pub fn total_difficulty(&mut self) -> f64 {
        if self.cached_total_difficulty == 0.0 {
            self.cached_total_difficulty = self
                .worker_difficulty
                .values()
                .map(|c| c.read() as f64 / 1_000_000.0)
                .sum();
        }
        self.cached_total_difficulty
    }

    /// Get worker statistics
    pub fn worker_stats(&self) -> Vec<(String, f64, f64)> {
        let total: f64 = self
            .worker_difficulty
            .values()
            .map(|c| c.read() as f64)
            .sum();

        self.worker_difficulty
            .iter()
            .map(|(wallet, counter)| {
                let difficulty = counter.read() as f64 / 1_000_000.0;
                let proportion = if total > 0.0 {
                    (counter.read() as f64) / total
                } else {
                    0.0
                };
                (wallet.clone(), difficulty, proportion)
            })
            .collect()
    }

    /// Calculate rewards (deterministic from CRDT state)
    pub fn calculate_rewards(&self, block_reward: u64) -> Vec<RewardEntry> {
        let total_difficulty: u64 = self
            .worker_difficulty
            .values()
            .map(|c| c.read())
            .sum();

        if total_difficulty == 0 {
            return vec![];
        }

        // Dev fee: 1%, Pool fee: 1.5%
        let dev_fee = block_reward * 100 / 10_000;
        let pool_fee = block_reward * 150 / 10_000;
        let miner_rewards = block_reward - dev_fee - pool_fee;

        self.worker_difficulty
            .iter()
            .map(|(wallet, counter)| {
                let difficulty = counter.read();
                let proportion = difficulty as f64 / total_difficulty as f64;
                let amount = (miner_rewards as f64 * proportion) as u64;
                RewardEntry {
                    worker_id: crate::worker::WorkerId::new(wallet, ""),
                    wallet_address: wallet.clone(),
                    amount,
                    proportion,
                    difficulty_contribution: difficulty as f64 / 1_000_000.0,
                }
            })
            .filter(|r| r.amount > 0)
            .collect()
    }

    /// Compute state hash for comparison
    pub fn state_hash(&mut self) -> PPLNSStateHash {
        if let Some(hash) = self.cached_state_hash {
            return hash;
        }

        let mut hasher = Hasher::new();

        // Hash round info
        hasher.update(&self.round_number.to_le_bytes());
        hasher.update(&self.network_difficulty.to_le_bytes());

        // Hash worker difficulties (sorted order for determinism)
        for (wallet, counter) in &self.worker_difficulty {
            hasher.update(wallet.as_bytes());
            hasher.update(&counter.read().to_le_bytes());
        }

        // Hash share count
        hasher.update(&(self.window_shares.len() as u64).to_le_bytes());

        let hash = *hasher.finalize().as_bytes();
        self.cached_state_hash = Some(hash);
        hash
    }

    /// Get share count
    pub fn share_count(&self) -> usize {
        self.window_shares.len()
    }

    /// Get worker count
    pub fn worker_count(&self) -> usize {
        self.worker_difficulty.len()
    }

    /// Check if share is already in window
    pub fn has_share(&self, share_id: &ShareId) -> bool {
        self.window_shares.contains(share_id)
    }

    /// Invalidate cached values
    fn invalidate_cache(&mut self) {
        self.cached_total_difficulty = 0.0;
        self.cached_state_hash = None;
    }

    /// Trim window to N * network_difficulty
    /// Called periodically to prevent unbounded growth
    pub fn trim_window(&mut self, max_shares: usize) {
        // G-Set doesn't support removal, so we'd need to
        // swap to a new state when window slides significantly.
        // For now, just track that we might need to start fresh.
        if self.window_shares.len() > max_shares * 2 {
            tracing::warn!(
                "PPLNS window exceeds {} shares, consider round reset",
                max_shares
            );
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        bincode::deserialize(data).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcounter_increment_and_merge() {
        let node_a = [1u8; 32];
        let node_b = [2u8; 32];

        let mut counter_a = GCounter::new();
        counter_a.increment(node_a, 100);
        counter_a.increment(node_a, 50);

        let mut counter_b = GCounter::new();
        counter_b.increment(node_b, 75);

        assert_eq!(counter_a.read(), 150);
        assert_eq!(counter_b.read(), 75);

        // Merge
        counter_a.merge(&counter_b);
        assert_eq!(counter_a.read(), 225); // 150 + 75
    }

    #[test]
    fn test_gcounter_merge_idempotent() {
        let node_a = [1u8; 32];

        let mut counter = GCounter::new();
        counter.increment(node_a, 100);

        let clone = counter.clone();
        counter.merge(&clone);
        counter.merge(&clone);

        assert_eq!(counter.read(), 100); // Still 100, merge is idempotent
    }

    #[test]
    fn test_gset_operations() {
        let mut set: GSet<[u8; 32]> = GSet::new();

        let item1 = [1u8; 32];
        let item2 = [2u8; 32];

        set.add(item1);
        set.add(item2);
        set.add(item1); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&item1));
        assert!(set.contains(&item2));
    }

    #[test]
    fn test_distributed_pplns_merge() {
        let node_a = [1u8; 32];
        let node_b = [2u8; 32];

        let mut pplns_a = DistributedPPLNS::new(2.0);
        let mut pplns_b = DistributedPPLNS::new(2.0);

        // Simulate shares on different nodes
        let share_id_1 = [1u8; 32];
        let share_id_2 = [2u8; 32];

        // Add to node A
        pplns_a
            .worker_difficulty
            .entry("wallet1".to_string())
            .or_insert_with(GCounter::new)
            .increment(node_a, 1_000_000);
        pplns_a.window_shares.add(share_id_1);

        // Add to node B
        pplns_b
            .worker_difficulty
            .entry("wallet2".to_string())
            .or_insert_with(GCounter::new)
            .increment(node_b, 2_000_000);
        pplns_b.window_shares.add(share_id_2);

        // Merge
        pplns_a.merge(&pplns_b);

        // Should have both wallets and both shares
        assert_eq!(pplns_a.worker_count(), 2);
        assert_eq!(pplns_a.share_count(), 2);
    }

    #[test]
    fn test_reward_calculation() {
        let node_a = [1u8; 32];

        let mut pplns = DistributedPPLNS::new(2.0);

        // Add workers with different difficulties
        pplns
            .worker_difficulty
            .entry("wallet1".to_string())
            .or_insert_with(GCounter::new)
            .increment(node_a, 7_000_000); // 70%

        pplns
            .worker_difficulty
            .entry("wallet2".to_string())
            .or_insert_with(GCounter::new)
            .increment(node_a, 3_000_000); // 30%

        let rewards = pplns.calculate_rewards(1_000_000_000); // 1 QUG

        assert_eq!(rewards.len(), 2);

        // After fees (2.5%), miner rewards = 975_000_000
        let total_rewards: u64 = rewards.iter().map(|r| r.amount).sum();
        assert!(total_rewards > 970_000_000 && total_rewards < 980_000_000);
    }
}
