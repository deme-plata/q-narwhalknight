/// Distributed VM Protocol for Q-NarwhalKnight
/// Enables horizontal scaling of smart contract execution across all nodes
///
/// Architecture:
/// - Contract state synchronized via libp2p gossipsub
/// - Compute jobs distributed using request-response protocol
/// - State verification via merkle proofs
/// - Load balancing across validator nodes

use anyhow::{anyhow, Result};
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Topics for VM gossip
pub const TOPIC_CONTRACT_STATE: &str = "qnk/vm/contract-state/v1";
pub const TOPIC_EXECUTION_RESULT: &str = "qnk/vm/execution-result/v1";
pub const TOPIC_STATE_UPDATE: &str = "qnk/vm/state-update/v1";

/// Contract state synchronization message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractStateMessage {
    pub contract_address: [u8; 32],
    pub state_root: [u8; 32],
    pub updated_at: u64,
    pub updates: Vec<StateUpdate>,
}

/// Individual state update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateUpdate {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub proof: MerkleProof,
}

/// Merkle proof for state verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    pub siblings: Vec<[u8; 32]>,
    pub path: Vec<bool>, // true = right, false = left
}

/// Contract execution request (sent via request-response)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRequest {
    pub request_id: String,
    pub contract_address: [u8; 32],
    pub function_name: String,
    pub parameters: Vec<u8>, // Serialized parameters
    pub caller: [u8; 32],
    pub gas_limit: u64,
    pub state_root: [u8; 32],
}

/// Contract execution response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResponse {
    pub request_id: String,
    pub success: bool,
    pub result: Vec<u8>,
    pub gas_used: u64,
    pub state_changes: Vec<StateUpdate>,
    pub new_state_root: [u8; 32],
    pub error: Option<String>,
}

/// Execution result broadcast (sent via gossipsub)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResultMessage {
    pub request_id: String,
    pub executor_peer: String,
    pub contract_address: [u8; 32],
    pub execution_time_ms: u64,
    pub success: bool,
    pub state_root: [u8; 32],
}

/// Distributed VM coordinator
pub struct DistributedVMCoordinator {
    /// Local node's peer ID
    pub local_peer_id: PeerId,
    /// Contract states indexed by address
    pub contract_states: Arc<RwLock<HashMap<[u8; 32], ContractState>>>,
    /// Pending execution requests
    pub pending_requests: Arc<RwLock<HashMap<String, ExecutionRequest>>>,
    /// Known validator peers that can execute contracts
    pub validator_peers: Arc<RwLock<Vec<PeerId>>>,
    /// Compute load per peer (for load balancing)
    pub peer_loads: Arc<RwLock<HashMap<PeerId, u64>>>,
}

/// Local contract state cache
#[derive(Debug, Clone)]
pub struct ContractState {
    pub address: [u8; 32],
    pub state_root: [u8; 32],
    pub last_updated: u64,
    pub state_data: HashMap<Vec<u8>, Vec<u8>>,
}

impl DistributedVMCoordinator {
    pub fn new(local_peer_id: PeerId) -> Self {
        info!("🔧 Initializing Distributed VM Coordinator");
        Self {
            local_peer_id,
            contract_states: Arc::new(RwLock::new(HashMap::new())),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            validator_peers: Arc::new(RwLock::new(Vec::new())),
            peer_loads: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Handle incoming contract state message from gossipsub
    pub async fn handle_contract_state(&self, msg: ContractStateMessage) -> Result<()> {
        debug!(
            "📥 Received contract state update for {}",
            hex::encode(&msg.contract_address)
        );

        let mut states = self.contract_states.write().await;

        // Update local state cache
        if let Some(state) = states.get_mut(&msg.contract_address) {
            // Verify this is a newer update
            if msg.updated_at > state.last_updated {
                state.state_root = msg.state_root;
                state.last_updated = msg.updated_at;

                // Apply state updates
                for update in msg.updates {
                    // TODO: Verify merkle proof
                    state.state_data.insert(update.key, update.value);
                }

                debug!("✅ Updated contract state to root {}", hex::encode(&msg.state_root));
            }
        } else {
            // Create new state entry
            let mut state_data = HashMap::new();
            for update in msg.updates {
                state_data.insert(update.key, update.value);
            }

            states.insert(
                msg.contract_address,
                ContractState {
                    address: msg.contract_address,
                    state_root: msg.state_root,
                    last_updated: msg.updated_at,
                    state_data,
                },
            );

            info!("🆕 Added new contract state for {}", hex::encode(&msg.contract_address));
        }

        Ok(())
    }

    /// Handle execution result broadcast
    pub async fn handle_execution_result(&self, msg: ExecutionResultMessage) -> Result<()> {
        debug!(
            "📊 Execution result: contract={}, success={}, time={}ms",
            hex::encode(&msg.contract_address),
            msg.success,
            msg.execution_time_ms
        );

        // Update peer load metrics
        if let Ok(peer_id) = msg.executor_peer.parse::<PeerId>() {
            let mut loads = self.peer_loads.write().await;
            *loads.entry(peer_id).or_insert(0) += 1;
        }

        Ok(())
    }

    /// Select best peer for contract execution (load balancing)
    pub async fn select_executor_peer(&self) -> Option<PeerId> {
        let peers = self.validator_peers.read().await;
        let loads = self.peer_loads.read().await;

        if peers.is_empty() {
            return None;
        }

        // Find peer with lowest load
        let mut best_peer = peers[0];
        let mut min_load = loads.get(&best_peer).copied().unwrap_or(0);

        for peer in peers.iter().skip(1) {
            let load = loads.get(peer).copied().unwrap_or(0);
            if load < min_load {
                min_load = load;
                best_peer = *peer;
            }
        }

        Some(best_peer)
    }

    /// Register a new validator peer
    pub async fn register_validator(&self, peer_id: PeerId) {
        let mut peers = self.validator_peers.write().await;
        if !peers.contains(&peer_id) {
            peers.push(peer_id);
            info!("✅ Registered validator peer: {}", peer_id);
        }
    }

    /// Broadcast contract state update to network
    pub async fn broadcast_state_update(
        &self,
        contract_address: [u8; 32],
        updates: Vec<StateUpdate>,
    ) -> Result<ContractStateMessage> {
        let states = self.contract_states.read().await;

        let state_root = if let Some(state) = states.get(&contract_address) {
            state.state_root
        } else {
            [0u8; 32] // Initial state
        };

        let msg = ContractStateMessage {
            contract_address,
            state_root,
            updated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            updates,
        };

        info!(
            "📡 Broadcasting state update for contract {}",
            hex::encode(&contract_address)
        );

        Ok(msg)
    }

    /// Get current state for a contract
    pub async fn get_contract_state(&self, address: &[u8; 32]) -> Option<ContractState> {
        let states = self.contract_states.read().await;
        states.get(address).cloned()
    }

    /// Get network statistics
    pub async fn get_stats(&self) -> VMNetworkStats {
        let states = self.contract_states.read().await;
        let peers = self.validator_peers.read().await;
        let loads = self.peer_loads.read().await;

        VMNetworkStats {
            total_contracts: states.len() as u64,
            validator_count: peers.len() as u64,
            total_executions: loads.values().sum(),
            average_load: if peers.is_empty() {
                0.0
            } else {
                loads.values().sum::<u64>() as f64 / peers.len() as f64
            },
        }
    }
}

/// VM network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VMNetworkStats {
    pub total_contracts: u64,
    pub validator_count: u64,
    pub total_executions: u64,
    pub average_load: f64,
}

/// Verify a merkle proof
pub fn verify_merkle_proof(
    leaf: &[u8; 32],
    proof: &MerkleProof,
    root: &[u8; 32],
) -> bool {
    let mut current = *leaf;

    for (i, sibling) in proof.siblings.iter().enumerate() {
        let is_right = proof.path.get(i).copied().unwrap_or(false);

        current = if is_right {
            // Current is on right, sibling on left
            blake3::hash(&[sibling.as_slice(), current.as_slice()].concat()).into()
        } else {
            // Current is on left, sibling on right
            blake3::hash(&[current.as_slice(), sibling.as_slice()].concat()).into()
        };
    }

    &current == root
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let peer_id = PeerId::random();
        let coordinator = DistributedVMCoordinator::new(peer_id);

        let stats = coordinator.get_stats().await;
        assert_eq!(stats.total_contracts, 0);
        assert_eq!(stats.validator_count, 0);
    }

    #[tokio::test]
    async fn test_validator_registration() {
        let peer_id = PeerId::random();
        let coordinator = DistributedVMCoordinator::new(peer_id);

        let validator = PeerId::random();
        coordinator.register_validator(validator).await;

        let stats = coordinator.get_stats().await;
        assert_eq!(stats.validator_count, 1);
    }

    #[test]
    fn test_merkle_proof_verification() {
        let leaf = blake3::hash(b"test_data").into();
        let sibling = blake3::hash(b"sibling").into();

        // Create a simple proof
        let root = blake3::hash(&[leaf.as_slice(), sibling.as_slice()].concat()).into();

        let proof = MerkleProof {
            siblings: vec![sibling],
            path: vec![false], // leaf on left
        };

        assert!(verify_merkle_proof(&leaf, &proof, &root));
    }
}
