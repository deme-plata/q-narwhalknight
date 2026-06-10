//! Distributed Share Ledger with CRDT Consistency
//!
//! v2.3.5-beta: Trustless share tracking with Merkle proofs for decentralized mining.
//!
//! ## Design Principles
//!
//! 1. **Sub-50ms finality preserved**: Rewards are credited INSTANTLY via the fast path.
//!    This ledger runs ASYNCHRONOUSLY for audit/proof generation only.
//!
//! 2. **CRDT-based**: Uses G-Set for append-only share log. No coordination needed.
//!
//! 3. **Merkle proofs**: Miners can prove their share was recorded.
//!
//! 4. **Gossipsub sync**: Share announcements propagate to all pool nodes.

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use super::merkle::{MerkleProof, MerkleTree};
use super::{PeerIdBytes, ShareId};

/// Share ledger entry - immutable record of a mining share
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ShareLedgerEntry {
    /// Unique share identifier (hash of share content)
    pub share_id: ShareId,

    /// Worker wallet address
    pub worker: String,

    /// Share difficulty (work done)
    pub difficulty: u64,  // Using fixed-point for CRDT compatibility

    /// Timestamp when share was submitted (ms)
    pub timestamp_ms: u64,

    /// Pool node that received this share
    pub receiving_node: PeerIdBytes,

    /// Block height at time of submission
    pub block_height: u64,

    /// PPLNS window ID this share belongs to
    pub window_id: u64,

    /// Hash of the block template this share was for
    pub block_template_hash: [u8; 32],

    /// Solution nonce
    pub nonce: u64,

    /// VDF proof hash (for anti-grinding verification)
    pub vdf_proof_hash: [u8; 32],
}

impl ShareLedgerEntry {
    /// Create new share ledger entry
    pub fn new(
        worker: String,
        difficulty: u64,
        receiving_node: PeerIdBytes,
        block_height: u64,
        window_id: u64,
        block_template_hash: [u8; 32],
        nonce: u64,
        vdf_proof_hash: [u8; 32],
    ) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Generate share ID as hash of all content
        let share_id = Self::compute_share_id(
            &worker,
            difficulty,
            timestamp_ms,
            &receiving_node,
            block_height,
            nonce,
        );

        Self {
            share_id,
            worker,
            difficulty,
            timestamp_ms,
            receiving_node,
            block_height,
            window_id,
            block_template_hash,
            nonce,
            vdf_proof_hash,
        }
    }

    /// Compute share ID from content
    fn compute_share_id(
        worker: &str,
        difficulty: u64,
        timestamp_ms: u64,
        receiving_node: &PeerIdBytes,
        block_height: u64,
        nonce: u64,
    ) -> ShareId {
        let mut hasher = Hasher::new();
        hasher.update(worker.as_bytes());
        hasher.update(&difficulty.to_le_bytes());
        hasher.update(&timestamp_ms.to_le_bytes());
        hasher.update(receiving_node);
        hasher.update(&block_height.to_le_bytes());
        hasher.update(&nonce.to_le_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Get leaf hash for Merkle tree
    pub fn merkle_leaf_hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&self.share_id);
        hasher.update(&self.difficulty.to_le_bytes());
        hasher.update(self.worker.as_bytes());
        hasher.update(&self.window_id.to_le_bytes());
        *hasher.finalize().as_bytes()
    }
}

/// G-Set (Grow-only Set) CRDT for shares
///
/// Shares can only be added, never removed.
/// Merging is simple set union - no coordination needed.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ShareGSet {
    shares: HashSet<ShareId>,
    entries: BTreeMap<ShareId, ShareLedgerEntry>,
}

impl ShareGSet {
    /// Create new empty share set
    pub fn new() -> Self {
        Self {
            shares: HashSet::new(),
            entries: BTreeMap::new(),
        }
    }

    /// Add a share (idempotent)
    pub fn insert(&mut self, entry: ShareLedgerEntry) -> bool {
        if self.shares.contains(&entry.share_id) {
            return false; // Already exists
        }
        self.shares.insert(entry.share_id);
        self.entries.insert(entry.share_id, entry);
        true
    }

    /// Check if share exists
    pub fn contains(&self, share_id: &ShareId) -> bool {
        self.shares.contains(share_id)
    }

    /// Get share by ID
    pub fn get(&self, share_id: &ShareId) -> Option<&ShareLedgerEntry> {
        self.entries.get(share_id)
    }

    /// Get all shares in window
    pub fn shares_in_window(&self, window_id: u64) -> Vec<&ShareLedgerEntry> {
        self.entries
            .values()
            .filter(|e| e.window_id == window_id)
            .collect()
    }

    /// Get shares for worker
    pub fn shares_for_worker(&self, worker: &str) -> Vec<&ShareLedgerEntry> {
        self.entries
            .values()
            .filter(|e| e.worker == worker)
            .collect()
    }

    /// Merge with another G-Set (CRDT merge - set union)
    pub fn merge(&mut self, other: &Self) {
        for (share_id, entry) in &other.entries {
            if !self.shares.contains(share_id) {
                self.shares.insert(*share_id);
                self.entries.insert(*share_id, entry.clone());
            }
        }
    }

    /// Get total number of shares
    pub fn len(&self) -> usize {
        self.shares.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.shares.is_empty()
    }

    /// Get total difficulty for a worker in a window
    pub fn worker_difficulty_in_window(&self, worker: &str, window_id: u64) -> u64 {
        self.entries
            .values()
            .filter(|e| e.worker == worker && e.window_id == window_id)
            .map(|e| e.difficulty)
            .sum()
    }

    /// Get total difficulty in window
    pub fn total_difficulty_in_window(&self, window_id: u64) -> u64 {
        self.entries
            .values()
            .filter(|e| e.window_id == window_id)
            .map(|e| e.difficulty)
            .sum()
    }

    /// Get all entries as sorted vec (for Merkle tree)
    pub fn sorted_entries(&self) -> Vec<&ShareLedgerEntry> {
        let mut entries: Vec<_> = self.entries.values().collect();
        entries.sort_by_key(|e| (e.window_id, e.timestamp_ms, e.share_id));
        entries
    }
}

/// Share ledger checkpoint - periodic snapshot for consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareLedgerCheckpoint {
    /// Checkpoint ID (sequential)
    pub checkpoint_id: u64,

    /// Window ID this checkpoint covers
    pub window_id: u64,

    /// Merkle root of all shares in window
    pub merkle_root: [u8; 32],

    /// Total shares in this checkpoint
    pub total_shares: u64,

    /// Total difficulty in this checkpoint
    pub total_difficulty: u64,

    /// Timestamp of checkpoint creation
    pub timestamp_ms: u64,

    /// Block height at checkpoint
    pub block_height: u64,

    /// Signatures from pool nodes (2/3+ for consensus)
    pub node_signatures: Vec<CheckpointSignature>,
}

/// Node signature for a checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSignature {
    /// Signing node's peer ID
    pub node_id: PeerIdBytes,

    /// Ed25519/Dilithium5 signature
    pub signature: Vec<u8>,

    /// Public key for verification
    pub public_key: Vec<u8>,

    /// Timestamp of signature
    pub timestamp_ms: u64,
}

/// Distributed Share Ledger
///
/// CRDT-based share tracking with Merkle proofs and checkpoint consensus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareLedger {
    /// All shares (CRDT G-Set)
    shares: ShareGSet,

    /// Merkle trees per window (generated lazily)
    #[serde(skip)]
    merkle_trees: BTreeMap<u64, MerkleTree>,

    /// Finalized checkpoints
    checkpoints: BTreeMap<u64, ShareLedgerCheckpoint>,

    /// Current window ID
    pub current_window_id: u64,

    /// Checkpoint interval (shares per checkpoint)
    pub checkpoint_interval: u64,

    /// Shares since last checkpoint
    shares_since_checkpoint: u64,

    /// Last checkpoint merkle root
    last_checkpoint_root: Option<[u8; 32]>,
}

impl ShareLedger {
    /// Create new share ledger
    pub fn new(checkpoint_interval: u64) -> Self {
        Self {
            shares: ShareGSet::new(),
            merkle_trees: BTreeMap::new(),
            checkpoints: BTreeMap::new(),
            current_window_id: 0,
            checkpoint_interval,
            shares_since_checkpoint: 0,
            last_checkpoint_root: None,
        }
    }

    /// Add a share to the ledger (async-safe, non-blocking)
    ///
    /// This is called AFTER the reward has been credited (fast path).
    /// It never blocks the reward crediting.
    pub fn add_share(&mut self, entry: ShareLedgerEntry) -> bool {
        let added = self.shares.insert(entry);
        if added {
            self.shares_since_checkpoint += 1;
            // Invalidate cached Merkle tree for this window
            self.merkle_trees.remove(&self.current_window_id);
        }
        added
    }

    /// Check if share exists
    pub fn contains_share(&self, share_id: &ShareId) -> bool {
        self.shares.contains(share_id)
    }

    /// Get share by ID
    pub fn get_share(&self, share_id: &ShareId) -> Option<&ShareLedgerEntry> {
        self.shares.get(share_id)
    }

    /// Get shares for a worker
    pub fn get_worker_shares(&self, worker: &str) -> Vec<&ShareLedgerEntry> {
        self.shares.shares_for_worker(worker)
    }

    /// Get shares in current window
    pub fn get_current_window_shares(&self) -> Vec<&ShareLedgerEntry> {
        self.shares.shares_in_window(self.current_window_id)
    }

    /// Get worker difficulty in current window
    pub fn get_worker_difficulty(&self, worker: &str) -> u64 {
        self.shares.worker_difficulty_in_window(worker, self.current_window_id)
    }

    /// Get total difficulty in current window
    pub fn get_total_difficulty(&self) -> u64 {
        self.shares.total_difficulty_in_window(self.current_window_id)
    }

    /// Build Merkle tree for a window (lazy/cached)
    pub fn get_merkle_tree(&mut self, window_id: u64) -> &MerkleTree {
        if !self.merkle_trees.contains_key(&window_id) {
            let shares = self.shares.shares_in_window(window_id);
            let leaves: Vec<[u8; 32]> = shares.iter().map(|s| s.merkle_leaf_hash()).collect();
            let tree = MerkleTree::build(&leaves);
            self.merkle_trees.insert(window_id, tree);
        }
        self.merkle_trees.get(&window_id).unwrap()
    }

    /// Generate Merkle proof for a share
    pub fn generate_proof(&mut self, share_id: &ShareId) -> Option<MerkleProof> {
        let entry = self.shares.get(share_id)?;
        let window_id = entry.window_id;

        // Find index of this share in sorted order
        let shares = self.shares.shares_in_window(window_id);
        let mut sorted: Vec<_> = shares.iter().collect();
        sorted.sort_by_key(|e| (e.timestamp_ms, e.share_id));

        let index = sorted.iter().position(|s| s.share_id == *share_id)?;

        let tree = self.get_merkle_tree(window_id);
        tree.generate_proof(index)
    }

    /// Get current Merkle root for window
    pub fn get_merkle_root(&mut self, window_id: u64) -> [u8; 32] {
        self.get_merkle_tree(window_id).root()
    }

    /// Check if checkpoint is needed
    pub fn needs_checkpoint(&self) -> bool {
        self.shares_since_checkpoint >= self.checkpoint_interval
    }

    /// Generate checkpoint (call when consensus reached)
    pub fn create_checkpoint(&mut self, block_height: u64) -> ShareLedgerCheckpoint {
        let window_id = self.current_window_id;
        let merkle_root = self.get_merkle_root(window_id);
        let total_shares = self.shares.shares_in_window(window_id).len() as u64;
        let total_difficulty = self.get_total_difficulty();
        let checkpoint_id = self.checkpoints.len() as u64;

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let checkpoint = ShareLedgerCheckpoint {
            checkpoint_id,
            window_id,
            merkle_root,
            total_shares,
            total_difficulty,
            timestamp_ms,
            block_height,
            node_signatures: Vec::new(), // Signatures added via consensus
        };

        self.last_checkpoint_root = Some(merkle_root);
        self.shares_since_checkpoint = 0;

        checkpoint
    }

    /// Add signed checkpoint after consensus
    pub fn add_checkpoint(&mut self, checkpoint: ShareLedgerCheckpoint) {
        self.checkpoints.insert(checkpoint.checkpoint_id, checkpoint);
    }

    /// Get checkpoint by ID
    pub fn get_checkpoint(&self, checkpoint_id: u64) -> Option<&ShareLedgerCheckpoint> {
        self.checkpoints.get(&checkpoint_id)
    }

    /// Advance to new window (e.g., after block found)
    pub fn advance_window(&mut self) {
        self.current_window_id += 1;
        self.shares_since_checkpoint = 0;
    }

    /// Merge with another ledger (CRDT merge)
    pub fn merge(&mut self, other: &Self) {
        self.shares.merge(&other.shares);

        // Merge checkpoints (take ones we don't have)
        for (id, checkpoint) in &other.checkpoints {
            if !self.checkpoints.contains_key(id) {
                self.checkpoints.insert(*id, checkpoint.clone());
            }
        }

        // Update window ID to max
        self.current_window_id = self.current_window_id.max(other.current_window_id);

        // Invalidate Merkle caches
        self.merkle_trees.clear();
    }

    /// Get state hash for quick comparison
    pub fn state_hash(&mut self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&self.current_window_id.to_le_bytes());
        hasher.update(&(self.shares.len() as u64).to_le_bytes());
        hasher.update(&self.get_merkle_root(self.current_window_id));
        *hasher.finalize().as_bytes()
    }

    /// Get ledger stats
    pub fn stats(&self) -> ShareLedgerStats {
        ShareLedgerStats {
            total_shares: self.shares.len() as u64,
            current_window_id: self.current_window_id,
            shares_in_current_window: self.shares.shares_in_window(self.current_window_id).len() as u64,
            total_checkpoints: self.checkpoints.len() as u64,
            shares_since_checkpoint: self.shares_since_checkpoint,
        }
    }
}

/// Share ledger statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareLedgerStats {
    pub total_shares: u64,
    pub current_window_id: u64,
    pub shares_in_current_window: u64,
    pub total_checkpoints: u64,
    pub shares_since_checkpoint: u64,
}

/// Gossipsub message types for share ledger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShareLedgerMessage {
    /// New share announcement (high volume)
    ShareAnnouncement(ShareLedgerEntry),

    /// Checkpoint announcement (periodic)
    CheckpointAnnouncement(ShareLedgerCheckpoint),

    /// Checkpoint vote (for consensus)
    CheckpointVote {
        checkpoint_id: u64,
        merkle_root: [u8; 32],
        signature: CheckpointSignature,
    },

    /// State sync request
    StateSyncRequest {
        from_window_id: u64,
        requester: PeerIdBytes,
    },

    /// State sync response (paginated)
    StateSyncResponse {
        window_id: u64,
        shares: Vec<ShareLedgerEntry>,
        has_more: bool,
        page: u64,
    },

    /// Merkle proof request
    MerkleProofRequest {
        share_id: ShareId,
        requester: PeerIdBytes,
    },

    /// Merkle proof response
    MerkleProofResponse {
        share_id: ShareId,
        proof: MerkleProof,
        checkpoint_id: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_share_gset_insert() {
        let mut gset = ShareGSet::new();

        let entry = ShareLedgerEntry::new(
            "qnk1234567890abcdef".to_string(),
            1000,
            [0u8; 32],
            100,
            1,
            [0u8; 32],
            12345,
            [0u8; 32],
        );

        assert!(gset.insert(entry.clone()));
        assert!(!gset.insert(entry)); // Duplicate
        assert_eq!(gset.len(), 1);
    }

    #[test]
    fn test_share_gset_merge() {
        let mut gset1 = ShareGSet::new();
        let mut gset2 = ShareGSet::new();

        let entry1 = ShareLedgerEntry::new(
            "worker1".to_string(), 1000, [0u8; 32], 100, 1, [0u8; 32], 1, [0u8; 32],
        );
        let entry2 = ShareLedgerEntry::new(
            "worker2".to_string(), 2000, [0u8; 32], 100, 1, [0u8; 32], 2, [0u8; 32],
        );

        gset1.insert(entry1);
        gset2.insert(entry2);

        gset1.merge(&gset2);

        assert_eq!(gset1.len(), 2);
    }

    #[test]
    fn test_share_ledger_checkpoint() {
        let mut ledger = ShareLedger::new(10);

        for i in 0..10 {
            let entry = ShareLedgerEntry::new(
                format!("worker{}", i),
                1000,
                [0u8; 32],
                100 + i,
                0,
                [0u8; 32],
                i,
                [0u8; 32],
            );
            ledger.add_share(entry);
        }

        assert!(ledger.needs_checkpoint());

        let checkpoint = ledger.create_checkpoint(110);
        assert_eq!(checkpoint.total_shares, 10);
        assert_eq!(checkpoint.window_id, 0);
    }
}
