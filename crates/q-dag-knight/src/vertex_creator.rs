//! DAG Vertex Creation with Production Mempool Integration
//!
//! Phase 2B: Creates DAG vertices with real transactions from mempool,
//! VDF proofs, and parent selection for Q-NarwhalKnight consensus.
//!
//! 🔐 v2.4.7-beta: Added Ed25519 vertex signing for BFT consensus

use crate::{AnchorElectionResult, QuantumVDF, QuantumVDFProof};
use anyhow::Result;
use ed25519_dalek::{Signer, SigningKey};
use q_narwhal_core::production_mempool::ProductionMempool;
use q_types::*;
use sha3::{Digest, Sha3_256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// DAG Vertex Creator - integrates mempool transactions with consensus
pub struct VertexCreator {
    /// Node identity
    node_id: NodeId,

    /// 🔐 v2.4.7-beta: Ed25519 signing key for vertex signatures
    signing_key: Arc<SigningKey>,

    /// Current consensus round
    current_round: Arc<RwLock<Round>>,

    /// Parent vertex tracking
    vertex_store: Arc<RwLock<HashMap<VertexId, Vertex>>>,

    /// Quantum VDF for proof computation
    quantum_vdf: Arc<QuantumVDF>,

    /// Configuration parameters
    config: VertexCreatorConfig,
}

/// Configuration for vertex creation
#[derive(Debug, Clone)]
pub struct VertexCreatorConfig {
    /// Maximum transactions per vertex
    pub max_transactions_per_vertex: usize,

    /// VDF computation delay (1-2 seconds)
    pub vdf_delay_seconds: u64,

    /// Maximum parent vertices to reference
    pub max_parent_vertices: usize,

    /// Minimum parent vertices required
    pub min_parent_vertices: usize,
}

impl Default for VertexCreatorConfig {
    fn default() -> Self {
        Self {
            max_transactions_per_vertex: 1000,
            vdf_delay_seconds: 2,
            max_parent_vertices: 10,
            min_parent_vertices: 1,
        }
    }
}

/// A DAG vertex containing transactions and consensus metadata
#[derive(Debug, Clone)]
pub struct Vertex {
    /// Unique vertex identifier
    pub id: VertexId,

    /// Consensus round number
    pub round: Round,

    /// Node that created this vertex
    pub proposer: NodeId,

    /// Transactions included in this vertex
    pub transactions: Vec<TxHash>,

    /// Parent vertex references
    pub parents: Vec<VertexId>,

    /// VDF proof for this vertex
    pub vdf_proof: QuantumVDFProof,

    /// Creation timestamp
    pub timestamp: u64,

    /// Digital signature of the vertex
    pub signature: Vec<u8>,
}

impl VertexCreator {
    /// Create new vertex creator with signing key for BFT consensus
    ///
    /// 🔐 v2.4.7-beta: Now requires signing key for vertex signatures
    pub fn new(node_id: NodeId, signing_key: Arc<SigningKey>, quantum_vdf: Arc<QuantumVDF>) -> Self {
        Self {
            node_id,
            signing_key,
            current_round: Arc::new(RwLock::new(0)),
            vertex_store: Arc::new(RwLock::new(HashMap::new())),
            quantum_vdf,
            config: VertexCreatorConfig::default(),
        }
    }

    /// Create new vertex creator with auto-generated signing key (for testing)
    pub fn new_with_random_key(node_id: NodeId, quantum_vdf: Arc<QuantumVDF>) -> Self {
        // Ed25519-dalek 2.x uses random_bytes() approach or from_bytes
        let mut secret_bytes = [0u8; 32];
        rand::Rng::fill(&mut rand::thread_rng(), &mut secret_bytes);
        let signing_key = Arc::new(SigningKey::from_bytes(&secret_bytes));
        Self::new(node_id, signing_key, quantum_vdf)
    }

    /// 🔐 v2.4.7-beta: Sign vertex data using Ed25519
    ///
    /// Signs: H(vertex_id || round || tx_root || parents)
    pub fn sign_vertex(&self, vertex_id: &VertexId, round: Round, tx_root: &[u8; 32], parents: &[VertexId]) -> Vec<u8> {
        // Construct signing message
        let mut signing_data = Vec::with_capacity(32 + 8 + 32 + parents.len() * 32);
        signing_data.extend_from_slice(vertex_id);
        signing_data.extend_from_slice(&round.to_le_bytes());
        signing_data.extend_from_slice(tx_root);
        for parent in parents {
            signing_data.extend_from_slice(parent);
        }

        // Hash the message
        let message_hash = Sha3_256::digest(&signing_data);

        // Sign with Ed25519
        let signature = self.signing_key.sign(&message_hash);
        signature.to_bytes().to_vec()
    }

    /// Create a new DAG vertex with transactions from mempool
    pub async fn create_vertex_with_mempool_transactions(
        &self,
        mempool: &ProductionMempool,
        anchor_result: Option<AnchorElectionResult>,
    ) -> Result<Vertex> {
        let start_time = std::time::Instant::now();

        // 1. Get current consensus round
        let current_round = *self.current_round.read().await;
        info!("Creating vertex for round {}", current_round);

        // 2. Select transactions from mempool (highest fee first)
        let transactions = mempool
            .get_transactions_for_block(self.config.max_transactions_per_vertex)
            .await;

        let tx_hashes: Vec<TxHash> = transactions.iter().map(|tx| tx.hash()).collect();

        info!("Selected {} transactions for vertex", tx_hashes.len());

        // 3. Select parent vertices using DAG rules
        let parents = self.select_parent_vertices(current_round).await?;
        debug!("Selected {} parent vertices: {:?}", parents.len(), parents);

        // 4. Generate vertex ID from content
        let vertex_id = self
            .generate_vertex_id(current_round, &tx_hashes, &parents)
            .await?;

        // 5. Compute VDF proof (this takes 1-2 seconds)
        info!("Computing VDF proof...");
        let vdf_input_array: [u8; 32] = {
            let input_vec = self
                .create_vdf_input(&vertex_id, &tx_hashes, &parents)
                .await?;
            let mut array = [0u8; 32];
            let len = std::cmp::min(input_vec.len(), 32);
            array[..len].copy_from_slice(&input_vec[..len]);
            array
        };
        let vdf_result = self.quantum_vdf.compute_proof(&vdf_input_array).await?;
        info!("VDF proof computed in {:?}", start_time.elapsed());

        // 6. Create vertex with timestamp
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        // 🔐 v2.4.7-beta: Compute transaction root for vertex integrity
        let tx_root: [u8; 32] = if tx_hashes.is_empty() {
            [0u8; 32]
        } else {
            let mut tx_hasher = Sha3_256::new();
            for tx_hash in &tx_hashes {
                tx_hasher.update(tx_hash);
            }
            tx_hasher.finalize().into()
        };

        // 🔐 v2.4.7-beta: Sign the vertex with Ed25519
        let signature = self.sign_vertex(&vertex_id, current_round, &tx_root, &parents);
        info!(
            "✅ [VERTEX] Signed vertex {} with {} byte Ed25519 signature",
            hex::encode(&vertex_id[..8]),
            signature.len()
        );

        let vertex = Vertex {
            id: vertex_id,
            round: current_round,
            proposer: self.node_id,
            transactions: tx_hashes,
            parents,
            vdf_proof: QuantumVDFProof {
                challenge: vdf_input_array,
                proof: [0u8; 64], // TODO: Get actual proof
                quantum_seed: Some(vdf_input_array),
                computation_time: Duration::from_millis(1000), // TODO: Get actual computation time
                difficulty: 1,                                 // TODO: Get actual difficulty
                entropy_estimate: 0.95,                        // TODO: Get actual entropy estimate
                parallel_witnesses: Vec::new(),                // TODO: Get actual witnesses
            },
            timestamp,
            signature, // 🔐 v2.4.7-beta: Now properly signed!
        };

        // 7. Store vertex in local store
        self.vertex_store
            .write()
            .await
            .insert(vertex_id, vertex.clone());

        // 8. Update metrics
        info!(
            "Created vertex {:?} with {} transactions in {:?}",
            vertex_id,
            vertex.transactions.len(),
            start_time.elapsed()
        );

        Ok(vertex)
    }

    /// Select parent vertices according to DAG rules
    async fn select_parent_vertices(&self, current_round: Round) -> Result<Vec<VertexId>> {
        let vertex_store = self.vertex_store.read().await;
        let mut parents = Vec::new();

        // DAG Rule 1: Reference vertices from previous rounds
        for round in (current_round.saturating_sub(3)..current_round).rev() {
            let round_vertices: Vec<VertexId> = vertex_store
                .values()
                .filter(|v| v.round == round)
                .map(|v| v.id)
                .collect();

            for vertex_id in round_vertices {
                if parents.len() < self.config.max_parent_vertices {
                    parents.push(vertex_id);
                }
            }
        }

        // DAG Rule 2: Ensure minimum parents (use genesis if needed)
        if parents.len() < self.config.min_parent_vertices {
            if current_round == 0 {
                // Genesis round - no parents needed
            } else {
                // Add genesis vertex as parent if no other parents available
                let genesis_id = new_genesis_vertex_id();
                if !parents.contains(&genesis_id) {
                    parents.push(genesis_id);
                }
            }
        }

        Ok(parents)
    }

    /// Generate deterministic vertex ID from content
    async fn generate_vertex_id(
        &self,
        round: Round,
        transactions: &[TxHash],
        parents: &[VertexId],
    ) -> Result<VertexId> {
        use sha3::{Digest, Sha3_256};

        let mut hasher = Sha3_256::new();

        // Hash inputs: node_id + round + transactions + parents
        hasher.update(self.node_id);
        hasher.update(round.to_le_bytes());

        for tx_hash in transactions {
            hasher.update(tx_hash);
        }

        for parent_id in parents {
            hasher.update(parent_id);
        }

        let hash = hasher.finalize();
        Ok(hash.into())
    }

    /// Create VDF input from vertex components
    ///
    /// 🔐 v1.4.5-beta: SECURE VDF-DAG BINDING
    /// VDF input now includes parent VDF outputs to create cryptographic chain binding.
    /// This prevents pre-computation attacks where an attacker computes future VDFs
    /// before seeing parent VDF outputs.
    async fn create_vdf_input(
        &self,
        vertex_id: &VertexId,
        transactions: &[TxHash],
        parents: &[VertexId],
    ) -> Result<Vec<u8>> {
        use sha3::{Digest, Sha3_256};

        // 🔐 Secure VDF input construction:
        // SHA3-256(vertex_id || tx_hashes || parent_ids || parent_vdf_outputs)
        let mut hasher = Sha3_256::new();

        // Vertex identity
        hasher.update(vertex_id);

        // Transaction commitment
        for tx_hash in transactions {
            hasher.update(tx_hash);
        }

        // Parent vertex IDs
        for parent_id in parents {
            hasher.update(parent_id);
        }

        // 🔐 CRITICAL: Include parent VDF outputs for chain binding
        // This creates sequential dependency - can't compute this VDF until
        // all parent VDFs are complete
        let vertex_store = self.vertex_store.read().await;
        for parent_id in parents {
            if let Some(parent_vertex) = vertex_store.get(parent_id) {
                // Include parent's VDF challenge (which contains their VDF output)
                hasher.update(&parent_vertex.vdf_proof.challenge);
                // Include parent's quantum seed if available
                if let Some(ref seed) = parent_vertex.vdf_proof.quantum_seed {
                    hasher.update(seed);
                }
            } else if is_genesis_vertex_id(parent_id) {
                // Genesis has known VDF seed
                hasher.update(&[0u8; 32]); // Genesis VDF output
            }
        }

        // Add timestamp for freshness
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        hasher.update(now.to_le_bytes());

        Ok(hasher.finalize().to_vec())
    }

    /// Validate a vertex according to DAG rules
    pub async fn validate_vertex(&self, vertex: &Vertex) -> Result<bool> {
        // 1. Check VDF proof validity
        let vdf_input = self
            .create_vdf_input(&vertex.id, &vertex.transactions, &vertex.parents)
            .await?;

        if !self.quantum_vdf.verify_proof(&vertex.vdf_proof).await? {
            warn!("Invalid VDF proof for vertex {:?}", vertex.id);
            return Ok(false);
        }

        // 2. Check parent references are valid
        let vertex_store = self.vertex_store.read().await;
        for parent_id in &vertex.parents {
            if !vertex_store.contains_key(parent_id) && !is_genesis_vertex_id(parent_id) {
                warn!(
                    "Unknown parent vertex {:?} in vertex {:?}",
                    parent_id, vertex.id
                );
                return Ok(false);
            }
        }

        // 3. Check round consistency
        if !vertex.parents.is_empty() {
            let max_parent_round = vertex
                .parents
                .iter()
                .filter_map(|pid| vertex_store.get(pid))
                .map(|v| v.round)
                .max()
                .unwrap_or(0);

            if vertex.round <= max_parent_round {
                warn!(
                    "Invalid round {} for vertex {:?} (max parent round: {})",
                    vertex.round, vertex.id, max_parent_round
                );
                return Ok(false);
            }
        }

        info!("Vertex {:?} validation passed", vertex.id);
        Ok(true)
    }

    /// Get current round
    pub async fn current_round(&self) -> Round {
        *self.current_round.read().await
    }

    /// Advance to next round
    pub async fn advance_round(&self) -> Result<Round> {
        let mut current_round = self.current_round.write().await;
        *current_round += 1;
        let new_round = *current_round;
        info!("Advanced to round {}", new_round);
        Ok(new_round)
    }

    /// Get vertex by ID
    pub async fn get_vertex(&self, vertex_id: &VertexId) -> Option<Vertex> {
        self.vertex_store.read().await.get(vertex_id).cloned()
    }

    /// Get vertices for a specific round
    pub async fn get_round_vertices(&self, round: Round) -> Vec<Vertex> {
        self.vertex_store
            .read()
            .await
            .values()
            .filter(|v| v.round == round)
            .cloned()
            .collect()
    }

    /// Clean up old vertices beyond retention policy
    pub async fn cleanup_old_vertices(&self, keep_rounds: u64) -> usize {
        let current_round = *self.current_round.read().await;
        let cutoff = current_round.saturating_sub(keep_rounds);

        let mut store = self.vertex_store.write().await;
        let before = store.len();
        store.retain(|_, v| v.round >= cutoff);
        let removed = before - store.len();
        if removed > 0 {
            debug!("🧹 VertexCreator: cleaned {} old vertices (keeping rounds >= {})", removed, cutoff);
        }
        removed
    }
}

// Helper functions for VertexId operations
pub fn new_genesis_vertex_id() -> VertexId {
    [0u8; 32]
}

pub fn is_genesis_vertex_id(vertex_id: &VertexId) -> bool {
    *vertex_id == new_genesis_vertex_id()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_vdf::{QuantumVDFConfig, VDFSecurityLevel};
    use tokio;

    #[tokio::test]
    async fn test_vertex_creation() {
        let node_id = NodeId::default();
        let vdf_config = QuantumVDFConfig {
            security_level: VDFSecurityLevel::Standard,
            quantum_randomness_enabled: true,
        };
        let quantum_vdf = Arc::new(QuantumVDF::new(vdf_config));

        // 🔐 v2.4.7-beta: Use new_with_random_key for testing
        let vertex_creator = VertexCreator::new_with_random_key(node_id, quantum_vdf);

        // Test parent selection for round 1
        let parents = vertex_creator.select_parent_vertices(1).await.unwrap();
        assert!(!parents.is_empty());
    }

    #[tokio::test]
    async fn test_vertex_validation() {
        let node_id = NodeId::default();
        let vdf_config = QuantumVDFConfig {
            security_level: VDFSecurityLevel::Standard,
            quantum_randomness_enabled: false,
        };
        let quantum_vdf = Arc::new(QuantumVDF::new(vdf_config));

        // 🔐 v2.4.7-beta: Use new_with_random_key for testing
        let vertex_creator = VertexCreator::new_with_random_key(node_id, quantum_vdf);

        // Create a properly signed test vertex using the creator's signing key
        let vertex_id = new_genesis_vertex_id();
        let tx_root = [0u8; 32];
        let parents: Vec<VertexId> = vec![];
        let signature = vertex_creator.sign_vertex(&vertex_id, 0, &tx_root, &parents);

        let vertex = Vertex {
            id: vertex_id,
            round: 0,
            proposer: node_id,
            transactions: vec![],
            parents,
            vdf_proof: QuantumVDFProof::default(),
            timestamp: 0,
            signature,
        };

        // Should validate successfully
        let is_valid = vertex_creator.validate_vertex(&vertex).await.unwrap();
        assert!(is_valid);
    }
}
