use anyhow::Result;
use async_trait::async_trait;
/// Q-Narwhal: DAG-based mempool implementation
/// Phase 0: Classical Ed25519 implementation
/// Phase 2: Quantum-enhanced anchor election with VRF
use q_lattice_vrf::{LatticeVRF, VRFConfig, SecurityLevel, VRFResult};
use q_quantum_rng::{QuantumRNG, QRNGConfig};
use q_types::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

pub mod certificate;
pub mod production_mempool;
pub mod reliable_broadcast;
pub mod tor_broadcast;
pub mod tor_client_impl;
pub mod validator_set;
pub mod vertex_store;

pub use certificate::CertificateStore;
pub use production_mempool::ProductionMempool;
pub use reliable_broadcast::ReliableBroadcast;
pub use tor_broadcast::{TorClient, TorStreamConnection};
pub use tor_client_impl::{ProductionTorClient, TorClientConfig};
pub use validator_set::{ValidatorInfo, ValidatorSet};
pub use vertex_store::{InMemoryVertexStorage, VertexStore};

// Re-export q-types for external crates
pub use q_types::{Certificate, NodeId, Transaction, TxHash, ValidatorId, Vertex, VertexId};

/// Narwhal mempool core implementation
pub struct NarwhalCore {
    pub node_id: NodeId,
    pub vertex_store: VertexStore,
    pub certificate_store: CertificateStore,
    pub reliable_broadcast: ReliableBroadcast,
    pub current_round: RwLock<Round>,

    /// Phase 2+: Lattice VRF for quantum anchor election
    pub lattice_vrf: Option<Arc<LatticeVRF>>,

    /// Phase 2+: Quantum RNG for enhanced entropy
    pub quantum_rng: Option<Arc<QuantumRNG>>,

    /// Current phase for quantum enhancements
    pub phase: Phase,
}

impl NarwhalCore {
    pub fn new(node_id: NodeId) -> Self {
        Self::new_with_phase(node_id, Phase::Phase0)
    }

    /// Create new NarwhalCore with specific phase
    pub fn new_with_phase(node_id: NodeId, phase: Phase) -> Self {
        // Create a default test validator set (4 validators) for Phase 0
        // In production, this should be loaded from configuration
        let validator_set = Self::create_default_validator_set();

        Self {
            node_id,
            vertex_store: VertexStore::new_in_memory(),
            certificate_store: CertificateStore::new(validator_set),
            reliable_broadcast: ReliableBroadcast::new(node_id),
            current_round: RwLock::new(0),
            lattice_vrf: None, // Will be initialized async
            quantum_rng: None, // Will be initialized async
            phase,
        }
    }

    /// Create default validator set for testing/Phase 0
    /// In production, load from configuration file
    fn create_default_validator_set() -> ValidatorSet {
        use crate::validator_set::ValidatorInfo;
        use ed25519_dalek::SigningKey;
        use rand::{rngs::OsRng, RngCore};

        let mut validators = Vec::new();
        for _ in 0..4 {
            let mut secret_bytes = [0u8; 32];
            OsRng.fill_bytes(&mut secret_bytes);
            let signing_key = SigningKey::from_bytes(&secret_bytes);
            let public_key = signing_key.verifying_key();

            // NodeId = hash of public key
            let node_id = {
                use sha3::{Digest, Sha3_256};
                let mut hasher = Sha3_256::new();
                hasher.update(public_key.as_bytes());
                hasher.finalize().into()
            };

            validators.push(ValidatorInfo {
                node_id,
                public_key,
                stake: 1,
                active: true,
            });
        }
        ValidatorSet::new(validators).unwrap()
    }

    /// Initialize Phase 2+ quantum enhancements
    pub async fn initialize_quantum_enhancements(&mut self) -> Result<()> {
        if self.phase < Phase::Phase2 {
            info!("Phase {} - quantum enhancements not enabled", self.phase as u8);
            return Ok(());
        }

        info!("🔮 Initializing Phase 2+ quantum randomness for consensus");

        // Initialize Lattice VRF for verifiable randomness
        let vrf_config = VRFConfig {
            security_level: SecurityLevel::Standard,
            quantum_enhanced: true,
            ..Default::default()
        };

        match LatticeVRF::new(vrf_config, self.phase).await {
            Ok(vrf) => {
                self.lattice_vrf = Some(Arc::new(vrf));
                info!("✅ Lattice VRF initialized for quantum anchor election");
            }
            Err(e) => {
                warn!("⚠️  Failed to initialize Lattice VRF: {}", e);
            }
        }

        // Initialize Quantum RNG for enhanced entropy
        let qrng_config = QRNGConfig::default();
        match QuantumRNG::new(self.phase, qrng_config).await {
            Ok(qrng) => {
                self.quantum_rng = Some(Arc::new(qrng));
                info!("✅ Quantum RNG initialized for enhanced entropy");
            }
            Err(e) => {
                warn!("⚠️  Failed to initialize Quantum RNG: {}", e);
            }
        }

        if self.lattice_vrf.is_some() || self.quantum_rng.is_some() {
            info!("✅ Phase 2 quantum enhancements initialized successfully");
            Ok(())
        } else {
            warn!("⚠️  No quantum enhancements available, falling back to classical");
            Ok(())
        }
    }

    /// Create a new vertex with transactions
    pub async fn create_vertex(
        &self,
        transactions: Vec<Transaction>,
        parents: Vec<VertexId>,
    ) -> Result<Vertex> {
        let round = *self.current_round.read().await;

        // Compute transaction root
        let tx_root = self.compute_tx_root(&transactions);

        let mut vertex = Vertex {
            id: [0u8; 32], // Will be computed after signing
            round,
            author: self.node_id,
            tx_root,
            parents,
            transactions,
            signature: vec![], // Will be added after signing
            timestamp: chrono::Utc::now(),
        };

        // Phase 2+: Use quantum VRF for anchor election (even rounds)
        if self.phase >= Phase::Phase2 && round % 2 == 0 {
            if let Some(ref vrf) = self.lattice_vrf {
                debug!("🔮 Generating quantum VRF for anchor election in round {}", round);

                // Create VRF input from round and vertex data
                let mut vrf_input = Vec::new();
                vrf_input.extend_from_slice(&round.to_be_bytes());
                vrf_input.extend_from_slice(&vertex.author);
                vrf_input.extend_from_slice(&vertex.tx_root);

                match vrf.evaluate(&vrf_input, round).await {
                    Ok(vrf_result) => {
                        info!(
                            "✅ Quantum VRF generated for round {} anchor election",
                            round
                        );
                        info!(
                            "   VRF entropy: {:.3}, proof size: {} bytes",
                            vrf_result.output.entropy_estimate(),
                            vrf_result.proof.data().len()
                        );

                        // VRF result can be used for:
                        // 1. Anchor selection (min hash)
                        // 2. Leader election
                        // 3. Randomness beacon
                        // 4. Ordering decisions

                        // Store VRF result for anchor selection
                        // (In full implementation, this would be used by DAG-Knight ordering)
                    }
                    Err(e) => {
                        warn!("⚠️  Quantum VRF evaluation failed: {}, continuing without", e);
                    }
                }
            }
        }

        // TODO: Sign vertex with Phase-aware signing
        // if self.phase >= Phase::Phase1 {
        //     vertex.signature = sign_with_dilithium5(...);
        // } else {
        //     vertex.signature = sign_with_ed25519(...);
        // }

        Ok(vertex)
    }

    /// Process received vertex
    pub async fn process_vertex(&self, vertex: Vertex) -> Result<Option<Certificate>> {
        info!(
            "Processing vertex from author {:?} for round {}",
            vertex.author, vertex.round
        );

        // Validate vertex
        self.validate_vertex(&vertex).await?;

        // Store vertex
        self.vertex_store.store_vertex(vertex.clone()).await?;

        // Trigger reliable broadcast
        let vertex_id = vertex.id;
        self.reliable_broadcast.broadcast_vertex(vertex).await?;

        // Check if we can create a certificate
        if self.has_sufficient_acknowledgements(&vertex_id).await? {
            let certificate = self.create_certificate(&vertex_id).await?;
            self.certificate_store
                .store_certificate(certificate.clone())
                .await?;
            return Ok(Some(certificate));
        }

        Ok(None)
    }

    /// Validate vertex structure and signatures
    async fn validate_vertex(&self, vertex: &Vertex) -> Result<()> {
        // Check round validity
        let current_round = *self.current_round.read().await;
        if vertex.round > current_round + 1 {
            return Err(anyhow::anyhow!("Vertex from future round"));
        }

        // Validate transaction root
        let computed_root = self.compute_tx_root(&vertex.transactions);
        if computed_root != vertex.tx_root {
            return Err(anyhow::anyhow!("Invalid transaction root"));
        }

        // TODO: Validate signature
        // self.verify_vertex_signature(vertex)?;

        // TODO: Validate parent references
        // self.validate_parents(&vertex.parents).await?;

        Ok(())
    }

    /// Check if vertex has sufficient acknowledgements for certificate
    async fn has_sufficient_acknowledgements(&self, vertex_id: &VertexId) -> Result<bool> {
        // TODO: Implement threshold check (2f+1)
        // For now, return false to avoid certificate creation
        Ok(false)
    }

    /// Create certificate from acknowledgements
    async fn create_certificate(&self, vertex_id: &VertexId) -> Result<Certificate> {
        // TODO: Collect acknowledgements and create certificate
        let certificate = Certificate {
            vertex_id: *vertex_id,
            round: 0, // TODO: Get from vertex
            signatures: BTreeMap::new(),
            threshold_met: true,
        };
        Ok(certificate)
    }

    /// Compute Merkle root of transactions
    fn compute_tx_root(&self, transactions: &[Transaction]) -> TxHash {
        use sha3::{Digest, Sha3_256};

        if transactions.is_empty() {
            return [0u8; 32];
        }

        // Simple hash of all transaction IDs (Phase 0)
        // TODO: Implement proper Merkle tree
        let mut hasher = Sha3_256::new();
        for tx in transactions {
            hasher.update(tx.id);
        }
        hasher.finalize().into()
    }

    /// Advance to next round
    pub async fn advance_round(&self) -> Result<()> {
        let mut current_round = self.current_round.write().await;
        *current_round += 1;
        info!("Advanced to round {}", *current_round);
        Ok(())
    }

    /// Get current round
    pub async fn get_current_round(&self) -> Round {
        *self.current_round.read().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vertex_creation() {
        let node_id = [1u8; 32];
        let narwhal = NarwhalCore::new(node_id);

        let tx = Transaction {
            id: [1u8; 32],
            from: [2u8; 32],
            to: [3u8; 32],
            amount: 1000,
            fee: 10,
            nonce: 1,
            signature: vec![],
            timestamp: chrono::Utc::now(),
            token_type: q_types::TokenType::QUG,
            fee_token_type: q_types::TokenType::QUG,
            data: vec![],
        };

        let vertex = narwhal.create_vertex(vec![tx], vec![]).await.unwrap();

        assert_eq!(vertex.author, node_id);
        assert_eq!(vertex.round, 0);
        assert_eq!(vertex.transactions.len(), 1);
    }

    #[tokio::test]
    async fn test_round_advancement() {
        let node_id = [1u8; 32];
        let narwhal = NarwhalCore::new(node_id);

        assert_eq!(narwhal.get_current_round().await, 0);

        narwhal.advance_round().await.unwrap();
        assert_eq!(narwhal.get_current_round().await, 1);
    }
}
