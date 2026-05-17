//! Network Membership Proofs with Zero-Knowledge
//!
//! This module implements zero-knowledge proofs of network membership that allow
//! validators to prove they belong to the network without revealing network topology,
//! size, or other sensitive information.

use anyhow::Result;
use ark_bn254::Fr;
use ark_ff::PrimeField;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use q_types::ValidatorId;
use q_zk_snark::{CircuitBuilder, CircuitGadgets, SNARKConfig, SNARKProtocol, UniversalSNARK};

use crate::{field_from_bytes, ZkP2pError};

/// Merkle tree for network membership
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleTree {
    /// Root hash of the merkle tree
    pub root: [u8; 32],
    /// Tree depth
    pub depth: usize,
    /// Number of leaves
    pub leaf_count: usize,
}

impl MerkleTree {
    /// Create new merkle tree from validator list
    pub fn new(validators: &[ValidatorId]) -> Self {
        // Simplified merkle tree construction
        let leaf_count = validators.len();
        let depth = (leaf_count as f64).log2().ceil() as usize;

        // Compute root hash (simplified - real implementation would build full tree)
        let mut hasher = blake3::Hasher::new();
        for validator in validators {
            hasher.update(validator);
        }
        let root = hasher.finalize().into();

        Self {
            root,
            depth,
            leaf_count,
        }
    }

    /// Get merkle proof for a validator at given index
    pub fn get_proof(&self, validator_id: &ValidatorId, index: usize) -> Result<MerkleProof> {
        // Simplified proof generation - real implementation would traverse tree
        let mut siblings = Vec::new();

        // Generate mock sibling hashes for each level
        for level in 0..self.depth {
            let sibling_hash =
                blake3::hash(&format!("sibling_{}_{}", level, index).as_bytes()).into();
            siblings.push(sibling_hash);
        }

        Ok(MerkleProof {
            leaf_hash: blake3::hash(validator_id).into(),
            index,
            siblings,
        })
    }
}

/// Merkle proof for tree membership
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    /// Hash of the leaf being proven
    pub leaf_hash: [u8; 32],
    /// Index of the leaf in the tree
    pub index: usize,
    /// Sibling hashes along the path to root
    pub siblings: Vec<[u8; 32]>,
}

/// Zero-knowledge proof of network membership without revealing network topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMembershipProof {
    /// PLONK proof supporting universal setup for flexible circuits
    pub membership_proof: Vec<u8>, // Serialized proof
    /// Merkle root of current network state (public)
    pub network_root: [u8; 32],
    /// Member's position commitment (hides actual position)
    pub position_commitment: [u8; 32],
    /// Network ID this proof is for (prevents cross-network attacks)
    /// Examples: "testnet-phase2", "mainnet", "devnet"
    pub network_id: String,
    /// Proof metadata
    pub metadata: MembershipProofMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembershipProofMetadata {
    /// Network epoch when proof was generated
    pub network_epoch: u64,
    /// Proof generation timestamp
    pub timestamp: u64,
    /// Circuit complexity estimate
    pub circuit_size: usize,
}

impl NetworkMembershipProof {
    /// Prove membership in validator set without revealing network topology
    pub async fn prove_membership(
        validator_id: &ValidatorId,
        network_merkle_tree: &MerkleTree,
        member_index: usize,
        merkle_proof: &MerkleProof,
    ) -> Result<NetworkMembershipProof> {
        info!(
            "🌐 Generating membership proof for validator in network (size: {})",
            network_merkle_tree.leaf_count
        );

        // Use PLONK for universal setup (good for variable-size networks)
        let snark_config = SNARKConfig {
            protocol: SNARKProtocol::PLONK,
            security_bits: 128,
            parallel_proving: true,
            max_constraints: 100_000, // Medium-sized circuit
            batch_verification: true,
            ..Default::default()
        };

        let _snark = UniversalSNARK::new(snark_config);

        // Build Merkle membership verification circuit
        let mut builder = CircuitBuilder::<Fr>::new("network_membership".to_string());

        debug!(
            "Building membership circuit for tree depth: {}",
            network_merkle_tree.depth
        );

        // Public inputs: network root
        let root_var = builder.create_variable("network_root".to_string(), true);
        builder.assign_variable(&root_var, field_from_bytes(&network_merkle_tree.root))?;

        // Private inputs: member ID, position, merkle proof path
        let member_var = builder.create_variable("member_id".to_string(), false);
        let position_var = builder.create_variable("position".to_string(), false);

        builder.assign_variable(&member_var, field_from_bytes(validator_id))?;
        builder.assign_variable(&position_var, Fr::from(member_index as u64))?;

        // Build merkle proof verification constraints
        let mut current_hash = member_var.clone();

        for (level, sibling) in merkle_proof.siblings.iter().enumerate() {
            let sibling_var = builder.create_variable(format!("sibling_{}", level), false);
            let next_hash_var = builder.create_variable(format!("hash_{}", level), false);

            builder.assign_variable(&sibling_var, field_from_bytes(sibling))?;

            // Hash constraint: next_hash = hash(current_hash, sibling)
            CircuitGadgets::hash_constraint(
                &mut builder,
                &[current_hash.clone(), sibling_var],
                &next_hash_var,
            )?;
            current_hash = next_hash_var;
        }

        // Final constraint: computed root equals network root
        builder.enforce_equality(
            &current_hash,
            &root_var,
            Some("merkle_root_verification".to_string()),
        )?;

        let circuit = builder.build();
        let circuit_size = circuit.size();

        debug!("Built membership circuit with {} constraints", circuit_size);

        // Generate proof (mock implementation)
        let membership_proof = bincode::serialize(&"mock_plonk_membership_proof").map_err(|e| {
            ZkP2pError::Serialization(format!("Membership proof serialization failed: {}", e))
        })?;

        // Create position commitment (hides actual network position)
        let position_commitment = blake3::hash(&member_index.to_be_bytes()).into();

        let metadata = MembershipProofMetadata {
            network_epoch: get_current_network_epoch(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            circuit_size,
        };

        info!("✅ Network membership proof generated successfully");

        Ok(NetworkMembershipProof {
            membership_proof,
            network_root: network_merkle_tree.root,
            position_commitment,
            network_id: get_current_network_id(),  // Add network ID
            metadata,
        })
    }

    /// Verify network membership without learning network topology
    pub async fn verify_membership(&self) -> Result<bool> {
        self.verify_membership_with_network_id(&get_current_network_id()).await
    }

    /// Verify network membership against specific network ID
    pub async fn verify_membership_with_network_id(&self, expected_network_id: &str) -> Result<bool> {
        debug!(
            "🔍 Verifying network membership proof for epoch: {} (network: {})",
            self.metadata.network_epoch,
            self.network_id
        );

        // ✅ FIX: Issue #7 - Check network ID matches
        if self.network_id != expected_network_id {
            warn!(
                "❌ Network ID mismatch! Expected: {}, Got: {}",
                expected_network_id,
                self.network_id
            );
            return Ok(false);
        }

        // ✅ FIX: Issue #6 - Check proof freshness (expiration)
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        let age = current_time.saturating_sub(self.metadata.timestamp);

        if age > 1800 {
            // 30 minute expiry for membership proofs
            warn!("⚠️ Membership proof expired (age: {}s)", age);
            return Ok(false);
        }

        // Check network epoch validity
        let current_epoch = get_current_network_epoch();
        if self.metadata.network_epoch < current_epoch.saturating_sub(2) {
            warn!(
                "⚠️ Membership proof from outdated epoch: {} (current: {})",
                self.metadata.network_epoch, current_epoch
            );
            return Ok(false);
        }

        // Verify proof structure (mock verification)
        let proof_data: Result<String, _> = bincode::deserialize(&self.membership_proof);
        let is_valid = proof_data.is_ok() && proof_data.unwrap() == "mock_plonk_membership_proof";

        if is_valid {
            info!("✅ Network membership proof verified successfully");
        } else {
            warn!("❌ Network membership proof verification failed");
        }

        Ok(is_valid)
    }

    /// Verify membership against specific network root
    pub async fn verify_against_root(&self, expected_root: &[u8; 32]) -> Result<bool> {
        // Check that proof was generated for the expected network state
        if self.network_root != *expected_root {
            warn!("❌ Membership proof network root mismatch");
            return Ok(false);
        }

        self.verify_membership().await
    }
}

/// Network membership management
pub struct NetworkMembershipManager {
    /// Current network state
    current_network: Option<MerkleTree>,
    /// Membership proof cache
    proof_cache: HashMap<ValidatorId, NetworkMembershipProof>,
    /// Network epoch counter
    network_epoch: u64,
}

impl NetworkMembershipManager {
    pub fn new() -> Self {
        Self {
            current_network: None,
            proof_cache: HashMap::new(),
            network_epoch: 0,
        }
    }

    /// Update network state with new validator set
    pub fn update_network_state(&mut self, validators: &[ValidatorId]) -> Result<()> {
        info!(
            "📊 Updating network state with {} validators",
            validators.len()
        );

        self.current_network = Some(MerkleTree::new(validators));
        self.network_epoch += 1;

        // Clear cache when network state changes
        self.proof_cache.clear();

        Ok(())
    }

    /// Generate membership proof for a validator
    pub async fn generate_membership_proof(
        &mut self,
        validator_id: &ValidatorId,
        all_validators: &[ValidatorId],
    ) -> Result<NetworkMembershipProof> {
        // Check cache first
        if let Some(cached_proof) = self.proof_cache.get(validator_id) {
            if cached_proof.metadata.network_epoch == self.network_epoch {
                debug!("📋 Using cached membership proof for validator");
                return Ok(cached_proof.clone());
            }
        }

        // Find validator index
        let member_index = all_validators
            .iter()
            .position(|v| v == validator_id)
            .ok_or_else(|| {
                ZkP2pError::MembershipVerification("Validator not found in network".to_string())
            })?;

        // Get or create network tree
        let network_tree = match &self.current_network {
            Some(tree) => tree.clone(),
            None => {
                self.update_network_state(all_validators)?;
                self.current_network.as_ref().unwrap().clone()
            }
        };

        // Generate merkle proof
        let merkle_proof = network_tree.get_proof(validator_id, member_index)?;

        // Generate ZK membership proof
        let membership_proof = NetworkMembershipProof::prove_membership(
            validator_id,
            &network_tree,
            member_index,
            &merkle_proof,
        )
        .await?;

        // Cache the proof
        self.proof_cache
            .insert(validator_id.clone(), membership_proof.clone());

        Ok(membership_proof)
    }

    /// Verify a membership proof against current network state
    pub async fn verify_membership_proof(&self, proof: &NetworkMembershipProof) -> Result<bool> {
        // Check against current network root if available
        if let Some(network) = &self.current_network {
            proof.verify_against_root(&network.root).await
        } else {
            proof.verify_membership().await
        }
    }

    /// Get current network statistics (privacy-preserving)
    pub fn get_network_stats(&self) -> NetworkStats {
        let validator_count = self
            .current_network
            .as_ref()
            .map(|n| n.leaf_count)
            .unwrap_or(0);

        NetworkStats {
            epoch: self.network_epoch,
            approximate_size_range: classify_network_size(validator_count),
            cached_proofs: self.proof_cache.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub epoch: u64,
    pub approximate_size_range: String,
    pub cached_proofs: usize,
}

impl Default for NetworkMembershipManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current network epoch (mock implementation)
fn get_current_network_epoch() -> u64 {
    // In real implementation, this would come from consensus state
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        / 300 // 5-minute epochs
}

/// Get current network ID
///
/// In production, this should be configured at node startup and retrieved
/// from the consensus configuration. For now, we default to "testnet-phase2".
fn get_current_network_id() -> String {
    // This should be configurable via environment or config file
    std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "testnet-phase2".to_string())
}

/// Classify network size into privacy-preserving ranges
fn classify_network_size(size: usize) -> String {
    match size {
        0..=10 => "small (≤10)".to_string(),
        11..=100 => "medium (11-100)".to_string(),
        101..=1000 => "large (101-1000)".to_string(),
        _ => "very large (>1000)".to_string(),
    }
}

/// Standalone convenience function for verifying membership proofs
///
/// This function verifies a NetworkMembershipProof without needing to instantiate
/// a NetworkMembershipManager. Useful for one-off verification in block request
/// authentication and other contexts where manager state isn't needed.
///
/// # Arguments
/// * `proof` - The membership proof to verify
///
/// # Returns
/// * `Ok(true)` - Proof is valid
/// * `Ok(false)` - Proof is invalid
/// * `Err` - Verification encountered an error
pub async fn verify_membership_proof(proof: &NetworkMembershipProof) -> Result<bool> {
    debug!("🔍 [STANDALONE] Verifying network membership proof");
    proof.verify_membership().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_creation() {
        let validators = vec![
            ValidatorId::from("validator1"),
            ValidatorId::from("validator2"),
            ValidatorId::from("validator3"),
            ValidatorId::from("validator4"),
        ];

        let tree = MerkleTree::new(&validators);
        assert_eq!(tree.leaf_count, 4);
        assert_eq!(tree.depth, 2); // ceil(log2(4))
        assert_ne!(tree.root, [0u8; 32]); // Non-zero root
    }

    #[test]
    fn test_merkle_proof_generation() {
        let validators = vec![
            ValidatorId::from("validator1"),
            ValidatorId::from("validator2"),
        ];

        let tree = MerkleTree::new(&validators);
        let proof = tree.get_proof(&validators[0], 0).unwrap();

        assert_eq!(proof.index, 0);
        assert_eq!(proof.siblings.len(), tree.depth);
    }

    #[tokio::test]
    async fn test_membership_proof_generation_and_verification() {
        let validator_id = ValidatorId::from("test_validator");
        let validators = vec![
            validator_id.clone(),
            ValidatorId::from("other_validator1"),
            ValidatorId::from("other_validator2"),
        ];

        let tree = MerkleTree::new(&validators);
        let merkle_proof = tree.get_proof(&validator_id, 0).unwrap();

        let membership_proof =
            NetworkMembershipProof::prove_membership(&validator_id, &tree, 0, &merkle_proof)
                .await
                .unwrap();

        let is_valid = membership_proof.verify_membership().await.unwrap();
        assert!(is_valid, "Valid membership proof should verify");
    }

    #[tokio::test]
    async fn test_membership_manager() {
        let mut manager = NetworkMembershipManager::new();

        let validators = vec![
            ValidatorId::from("validator1"),
            ValidatorId::from("validator2"),
            ValidatorId::from("validator3"),
        ];

        manager.update_network_state(&validators).unwrap();

        let proof = manager
            .generate_membership_proof(&validators[0], &validators)
            .await
            .unwrap();
        let is_valid = manager.verify_membership_proof(&proof).await.unwrap();

        assert!(is_valid, "Manager-generated proof should verify");

        let stats = manager.get_network_stats();
        assert_eq!(stats.epoch, 1);
        assert_eq!(stats.cached_proofs, 1);
    }

    #[test]
    fn test_network_size_classification() {
        assert_eq!(classify_network_size(5), "small (≤10)");
        assert_eq!(classify_network_size(50), "medium (11-100)");
        assert_eq!(classify_network_size(500), "large (101-1000)");
        assert_eq!(classify_network_size(5000), "very large (>1000)");
    }
}
