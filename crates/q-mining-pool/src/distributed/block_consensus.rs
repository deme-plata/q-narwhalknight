//! Block Found Consensus via Multi-Node Attestation
//!
//! When a pool finds a valid block, multiple pool nodes must attest
//! to its validity before it's considered confirmed. This prevents:
//! - Single node spoofing block finds
//! - Selfish mining attacks
//! - Block withholding from the network

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use super::{DistributedError, DistributedResult, PeerIdBytes, ShareId};

/// Block found announcement from the discovering pool node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockFoundAnnouncement {
    /// Block hash
    pub block_hash: [u8; 32],

    /// Block height
    pub height: u64,

    /// Previous block hash
    pub prev_hash: [u8; 32],

    /// Merkle root
    pub merkle_root: [u8; 32],

    /// Block timestamp
    pub block_timestamp: u64,

    /// Nonce that solved the block
    pub nonce: u64,

    /// Share that found the block
    pub finding_share_id: ShareId,

    /// Worker who found the block
    pub finder_wallet: String,

    /// Pool node that received the finding share
    pub finding_node: PeerIdBytes,

    /// Block difficulty met
    pub difficulty: f64,

    /// Total block reward (in satoshis/base units)
    pub block_reward: u64,

    /// PPLNS state hash at time of find
    pub pplns_state_hash: [u8; 32],

    /// Announcement timestamp
    pub timestamp: u64,

    /// Signature from finding node
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],
}

impl BlockFoundAnnouncement {
    /// Create new block found announcement
    pub fn new(
        block_hash: [u8; 32],
        height: u64,
        prev_hash: [u8; 32],
        merkle_root: [u8; 32],
        block_timestamp: u64,
        nonce: u64,
        finding_share_id: ShareId,
        finder_wallet: String,
        finding_node: PeerIdBytes,
        difficulty: f64,
        block_reward: u64,
        pplns_state_hash: [u8; 32],
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            block_hash,
            height,
            prev_hash,
            merkle_root,
            block_timestamp,
            nonce,
            finding_share_id,
            finder_wallet,
            finding_node,
            difficulty,
            block_reward,
            pplns_state_hash,
            timestamp,
            signature: [0u8; 64], // Filled by signing
        }
    }

    /// Compute announcement hash for signing
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&self.block_hash);
        hasher.update(&self.height.to_le_bytes());
        hasher.update(&self.prev_hash);
        hasher.update(&self.merkle_root);
        hasher.update(&self.block_timestamp.to_le_bytes());
        hasher.update(&self.nonce.to_le_bytes());
        hasher.update(&self.finding_share_id);
        hasher.update(self.finder_wallet.as_bytes());
        hasher.update(&self.finding_node);
        hasher.update(&self.difficulty.to_le_bytes());
        hasher.update(&self.block_reward.to_le_bytes());
        hasher.update(&self.pplns_state_hash);
        hasher.update(&self.timestamp.to_le_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Check if announcement is recent (within 5 minutes)
    pub fn is_recent(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        now - self.timestamp < 300_000 // 5 minutes
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

/// Node attestation for a block find
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAttestation {
    /// Block hash being attested
    pub block_hash: [u8; 32],

    /// Block height
    pub height: u64,

    /// Attesting node's peer ID
    pub attester: PeerIdBytes,

    /// Attestation result
    pub valid: bool,

    /// Reason if invalid
    pub reason: Option<String>,

    /// Attester's view of PPLNS state hash
    pub pplns_state_hash: [u8; 32],

    /// Timestamp
    pub timestamp: u64,

    /// Signature over attestation
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],
}

impl NodeAttestation {
    /// Create new attestation
    pub fn new(
        block_hash: [u8; 32],
        height: u64,
        attester: PeerIdBytes,
        valid: bool,
        reason: Option<String>,
        pplns_state_hash: [u8; 32],
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            block_hash,
            height,
            attester,
            valid,
            reason,
            pplns_state_hash,
            timestamp,
            signature: [0u8; 64],
        }
    }

    /// Compute attestation hash for signing
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&self.block_hash);
        hasher.update(&self.height.to_le_bytes());
        hasher.update(&self.attester);
        hasher.update(&[self.valid as u8]);
        if let Some(reason) = &self.reason {
            hasher.update(reason.as_bytes());
        }
        hasher.update(&self.pplns_state_hash);
        hasher.update(&self.timestamp.to_le_bytes());
        *hasher.finalize().as_bytes()
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

/// Block consensus state for tracking attestations
#[derive(Debug, Clone)]
pub struct BlockConsensusState {
    /// The block announcement
    pub announcement: BlockFoundAnnouncement,

    /// Collected attestations (peer_id -> attestation)
    pub attestations: HashMap<PeerIdBytes, NodeAttestation>,

    /// Set of nodes that attested valid
    pub valid_attesters: HashSet<PeerIdBytes>,

    /// Set of nodes that attested invalid
    pub invalid_attesters: HashSet<PeerIdBytes>,

    /// Consensus status
    pub status: ConsensusStatus,

    /// When consensus was first requested
    pub started_at: u64,

    /// When consensus was reached (if any)
    pub finalized_at: Option<u64>,
}

/// Consensus status for a block
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsensusStatus {
    /// Waiting for attestations
    Pending,

    /// Enough attestations for validity
    Confirmed,

    /// Enough attestations for invalidity
    Rejected,

    /// Timeout without reaching quorum
    TimedOut,
}

impl BlockConsensusState {
    /// Create new consensus state for a block announcement
    pub fn new(announcement: BlockFoundAnnouncement) -> Self {
        let started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            announcement,
            attestations: HashMap::new(),
            valid_attesters: HashSet::new(),
            invalid_attesters: HashSet::new(),
            status: ConsensusStatus::Pending,
            started_at,
            finalized_at: None,
        }
    }

    /// Add an attestation
    pub fn add_attestation(&mut self, attestation: NodeAttestation) {
        // Skip if already have attestation from this node
        if self.attestations.contains_key(&attestation.attester) {
            return;
        }

        // Track validity vote
        if attestation.valid {
            self.valid_attesters.insert(attestation.attester);
        } else {
            self.invalid_attesters.insert(attestation.attester);
        }

        self.attestations.insert(attestation.attester, attestation);
    }

    /// Check if consensus is reached
    ///
    /// Uses BFT quorum: 2f+1 where f = (n-1)/3
    /// For n nodes, need (2n+1)/3 attestations
    pub fn check_consensus(&mut self, total_nodes: usize) -> ConsensusStatus {
        if self.status != ConsensusStatus::Pending {
            return self.status.clone();
        }

        // Calculate quorum threshold (2f+1)
        let quorum = (2 * total_nodes + 2) / 3;

        // Check for valid consensus
        if self.valid_attesters.len() >= quorum {
            self.status = ConsensusStatus::Confirmed;
            self.finalized_at = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            );
            return ConsensusStatus::Confirmed;
        }

        // Check for invalid consensus
        if self.invalid_attesters.len() >= quorum {
            self.status = ConsensusStatus::Rejected;
            self.finalized_at = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            );
            return ConsensusStatus::Rejected;
        }

        // Check for timeout (30 seconds)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        if now - self.started_at > 30_000 {
            self.status = ConsensusStatus::TimedOut;
            self.finalized_at = Some(now);
            return ConsensusStatus::TimedOut;
        }

        ConsensusStatus::Pending
    }

    /// Get attestation count
    pub fn attestation_count(&self) -> usize {
        self.attestations.len()
    }

    /// Get valid attestation count
    pub fn valid_count(&self) -> usize {
        self.valid_attesters.len()
    }

    /// Get invalid attestation count
    pub fn invalid_count(&self) -> usize {
        self.invalid_attesters.len()
    }
}

/// Block consensus manager for tracking multiple pending blocks
pub struct BlockConsensusManager {
    /// Pending block consensus states (block_hash -> state)
    pending: HashMap<[u8; 32], BlockConsensusState>,

    /// Recently finalized blocks (for duplicate detection)
    finalized: HashMap<[u8; 32], ConsensusStatus>,

    /// Known pool nodes (for quorum calculation)
    known_nodes: HashSet<PeerIdBytes>,

    /// Our node's peer ID
    our_peer_id: PeerIdBytes,

    /// Maximum pending blocks to track
    max_pending: usize,
}

impl BlockConsensusManager {
    /// Create new consensus manager
    pub fn new(our_peer_id: PeerIdBytes) -> Self {
        Self {
            pending: HashMap::new(),
            finalized: HashMap::new(),
            known_nodes: HashSet::new(),
            our_peer_id,
            max_pending: 100,
        }
    }

    /// Register a known pool node
    pub fn add_known_node(&mut self, peer_id: PeerIdBytes) {
        self.known_nodes.insert(peer_id);
    }

    /// Remove a known pool node
    pub fn remove_known_node(&mut self, peer_id: &PeerIdBytes) {
        self.known_nodes.remove(peer_id);
    }

    /// Get number of known nodes
    pub fn node_count(&self) -> usize {
        self.known_nodes.len()
    }

    /// Handle new block found announcement
    pub fn handle_announcement(
        &mut self,
        announcement: BlockFoundAnnouncement,
    ) -> DistributedResult<()> {
        // Check if already finalized
        if self.finalized.contains_key(&announcement.block_hash) {
            return Err(DistributedError::DuplicateShare(
                "Block already finalized".to_string(),
            ));
        }

        // Check if already pending
        if self.pending.contains_key(&announcement.block_hash) {
            return Ok(()); // Already tracking
        }

        // Limit pending blocks
        if self.pending.len() >= self.max_pending {
            self.cleanup_old_pending();
        }

        // Create new consensus state
        let state = BlockConsensusState::new(announcement);
        self.pending.insert(state.announcement.block_hash, state);

        Ok(())
    }

    /// Handle attestation from another node
    pub fn handle_attestation(&mut self, attestation: NodeAttestation) -> DistributedResult<ConsensusStatus> {
        // Find pending block
        let state = self
            .pending
            .get_mut(&attestation.block_hash)
            .ok_or_else(|| {
                DistributedError::ConsensusNotReached(format!(
                    "Unknown block: {}",
                    hex::encode(attestation.block_hash)
                ))
            })?;

        // Add attestation
        state.add_attestation(attestation);

        // Check consensus
        let status = state.check_consensus(self.known_nodes.len().max(1));

        // If finalized, move to finalized map
        if status != ConsensusStatus::Pending {
            let block_hash = state.announcement.block_hash;
            self.finalized.insert(block_hash, status.clone());
            self.pending.remove(&block_hash);
        }

        Ok(status)
    }

    /// Create our attestation for a block
    pub fn create_attestation(
        &self,
        block_hash: [u8; 32],
        valid: bool,
        reason: Option<String>,
        pplns_state_hash: [u8; 32],
    ) -> DistributedResult<NodeAttestation> {
        let state = self.pending.get(&block_hash).ok_or_else(|| {
            DistributedError::ConsensusNotReached(format!(
                "Unknown block: {}",
                hex::encode(block_hash)
            ))
        })?;

        Ok(NodeAttestation::new(
            block_hash,
            state.announcement.height,
            self.our_peer_id,
            valid,
            reason,
            pplns_state_hash,
        ))
    }

    /// Get pending block status
    pub fn get_status(&self, block_hash: &[u8; 32]) -> Option<&BlockConsensusState> {
        self.pending.get(block_hash)
    }

    /// Check if block is finalized
    pub fn is_finalized(&self, block_hash: &[u8; 32]) -> Option<&ConsensusStatus> {
        self.finalized.get(block_hash)
    }

    /// Get all pending blocks
    pub fn pending_blocks(&self) -> Vec<&BlockConsensusState> {
        self.pending.values().collect()
    }

    /// Clean up old pending blocks (timed out)
    fn cleanup_old_pending(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let expired: Vec<[u8; 32]> = self
            .pending
            .iter()
            .filter(|(_, state)| now - state.started_at > 60_000) // 1 minute
            .map(|(hash, _)| *hash)
            .collect();

        for hash in expired {
            self.finalized.insert(hash, ConsensusStatus::TimedOut);
            self.pending.remove(&hash);
        }
    }

    /// Clean up old finalized entries
    pub fn cleanup_finalized(&mut self, max_entries: usize) {
        if self.finalized.len() > max_entries {
            // Just clear oldest half
            let to_remove = self.finalized.len() - max_entries / 2;
            let keys: Vec<_> = self.finalized.keys().take(to_remove).cloned().collect();
            for key in keys {
                self.finalized.remove(&key);
            }
        }
    }
}

/// Aggregated block confirmation for network broadcast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedBlockConfirmation {
    /// The original announcement
    pub announcement: BlockFoundAnnouncement,

    /// All valid attestations
    pub attestations: Vec<NodeAttestation>,

    /// Aggregated signature (if using threshold signatures) - 64 bytes when present
    #[serde(default)]
    pub aggregated_signature: Vec<u8>,

    /// Number of attesters
    pub attester_count: u32,

    /// Block was confirmed
    pub confirmed: bool,
}

impl AggregatedBlockConfirmation {
    /// Create from consensus state
    pub fn from_consensus(state: &BlockConsensusState) -> Option<Self> {
        if state.status != ConsensusStatus::Confirmed {
            return None;
        }

        let valid_attestations: Vec<_> = state
            .attestations
            .values()
            .filter(|a| a.valid)
            .cloned()
            .collect();

        Some(Self {
            announcement: state.announcement.clone(),
            attestations: valid_attestations.clone(),
            aggregated_signature: Vec::new(), // Would be filled by threshold sig
            attester_count: valid_attestations.len() as u32,
            confirmed: true,
        })
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

    fn create_test_announcement() -> BlockFoundAnnouncement {
        BlockFoundAnnouncement::new(
            [1u8; 32],
            100,
            [0u8; 32],
            [2u8; 32],
            1700000000,
            12345,
            [3u8; 32],
            "qnk1test".to_string(),
            [4u8; 32],
            1000.0,
            1_000_000_000,
            [5u8; 32],
        )
    }

    #[test]
    fn test_block_consensus_quorum() {
        let announcement = create_test_announcement();
        let mut state = BlockConsensusState::new(announcement);

        // Simulate 5 nodes, need 4 for quorum (2*5+2)/3 = 4
        let total_nodes = 5;

        // Add 3 valid attestations (not enough)
        for i in 0..3 {
            let mut attester = [0u8; 32];
            attester[0] = i;
            let attestation =
                NodeAttestation::new([1u8; 32], 100, attester, true, None, [5u8; 32]);
            state.add_attestation(attestation);
        }

        assert_eq!(
            state.check_consensus(total_nodes),
            ConsensusStatus::Pending
        );

        // Add 4th valid attestation (reaches quorum)
        let mut attester = [0u8; 32];
        attester[0] = 3;
        let attestation = NodeAttestation::new([1u8; 32], 100, attester, true, None, [5u8; 32]);
        state.add_attestation(attestation);

        assert_eq!(
            state.check_consensus(total_nodes),
            ConsensusStatus::Confirmed
        );
    }

    #[test]
    fn test_block_consensus_rejection() {
        let announcement = create_test_announcement();
        let mut state = BlockConsensusState::new(announcement);

        let total_nodes = 5;

        // Add 4 invalid attestations (reaches rejection quorum)
        for i in 0..4 {
            let mut attester = [0u8; 32];
            attester[0] = i;
            let attestation = NodeAttestation::new(
                [1u8; 32],
                100,
                attester,
                false,
                Some("Invalid block".to_string()),
                [5u8; 32],
            );
            state.add_attestation(attestation);
        }

        assert_eq!(
            state.check_consensus(total_nodes),
            ConsensusStatus::Rejected
        );
    }

    #[test]
    fn test_consensus_manager() {
        let our_peer_id = [0u8; 32];
        let mut manager = BlockConsensusManager::new(our_peer_id);

        // Add known nodes
        for i in 0..5 {
            let mut peer = [0u8; 32];
            peer[0] = i;
            manager.add_known_node(peer);
        }

        // Handle announcement
        let announcement = create_test_announcement();
        manager.handle_announcement(announcement).unwrap();

        // Verify pending
        assert!(manager.get_status(&[1u8; 32]).is_some());
        assert!(!manager.pending_blocks().is_empty());
    }

    #[test]
    fn test_attestation_hashing() {
        let attestation =
            NodeAttestation::new([1u8; 32], 100, [2u8; 32], true, None, [3u8; 32]);

        let hash1 = attestation.hash();
        let hash2 = attestation.hash();

        assert_eq!(hash1, hash2); // Deterministic
    }
}
