//! # Q-NarwhalKnight Quantum Mixing Protocol
//!
//! Production-grade privacy mixing system with:
//! - Chaumian mixing protocol for transaction unlinkability
//! - Ring signatures with quantum resistance
//! - Stealth addresses for recipient privacy
//! - Zero-knowledge proofs (ZK-STARK) for mixing validity
//! - Quantum entropy integration for enhanced randomness
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │  Mixing Engine  │◄──►│ Ring Signatures │◄──►│  ZK Proof Gen   │
//! │                 │    │                 │    │                 │
//! │ • Chaumian Mix  │    │ • Linkable Sigs │    │ • STARK Proofs  │
//! │ • Pool Manager  │    │ • Key Images    │    │ • Batch Verify  │
//! │ • Participant   │    │ • Quantum Safe  │    │ • Range Proofs  │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!           │                        │                        │
//!           ▼                        ▼                        ▼
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │ Stealth Address │    │ Quantum Entropy │    │ Network Layer   │
//! │                 │    │                 │    │                 │
//! │ • Address Gen   │    │ • True Random   │    │ • Tor Support   │
//! │ • Payment Scan  │    │ • Noise Inject  │    │ • Pool Sync     │
//! │ • Key Derivation│    │ • Entropy Pool  │    │ • Byzantine Tol │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//! ```

pub mod error;
pub mod mixing_engine;
pub mod ring_signatures;
pub mod clsag;
pub mod aggregated_ring_sig;
pub mod uc_traceable_ring_sig;
pub mod stealth_addresses;
pub mod zkp_prover;
pub mod mixing_pool;
pub mod quantum_entropy;
pub mod compliance;
pub mod network;
pub mod decoy_transactions;
pub mod performance_profiler;
pub mod optimization_engine;
pub mod advanced_zk;
pub mod recursive_stark;

// Bulletproofs++ Range Proofs (v3.9.0: EUROCRYPT 2024)
// 39% smaller proofs (416 bytes for 64-bit), 5x faster proving, 9.5x batch verification speedup
pub mod bulletproofs_pp;

// Threshold Mixing Pool using Multi-Party Computation (v3.9.0: NIST IR 8214C)
// FROST threshold signatures for trustless mixing coordination
pub mod threshold_pool;

// Post-quantum lattice-based ring signatures (v3.9.0)
// Based on IACR ePrint 2025/2170: Module-LWE linkable ring signatures
#[cfg(feature = "lattice-ring-sigs")]
pub mod lattice_ring_sig;

// Re-export main types (only existing ones)
pub use error::{MixingError, Result};
pub use mixing_engine::QuantumMixingEngine;
pub use ring_signatures::{QuantumRingSigner, RingSignature, KeyImage};
pub use clsag::{
    CLSAGSignature, CLSAGSigner,
    batch_verify_clsag, batch_verify_clsag_detailed,
    create_pedersen_commitment, generate_commitment_mask,
    scalar_from_bytes_wide, derive_stealth_address,
};
pub use uc_traceable_ring_sig::{
    UCTraceableRingSigner, UCTraceableRingSignature, UCTRSConfig,
    VRFOutput, VRFProof, TracingTag, TracingResult,
};
pub use stealth_addresses::{StealthAddressGenerator, StealthAddress, DetectedPayment};
pub use zkp_prover::{QuantumZKPProver, ZKProof, ProofType, BalanceCommitment, RangeProof, MixingProof};
pub use quantum_entropy::{QuantumEntropyPool, EntropySource, NoiseInjector};
pub use decoy_transactions::{QuantumDecoyEngine, DecoyStrategy, DecoyType, DecoyTransaction, DecoyCampaign, DecoyMetrics};

// Re-export mixing pool types
pub use mixing_pool::{MixingPool, PoolParticipant, PoolState, MixingInput, MixingOutput, MixingParameters};

// Re-export compliance types
pub use compliance::{ComplianceEngine, ComplianceStatus, RiskFactors, ComplianceConfig};

// Re-export network types
pub use network::{MixingNetworkManager, NetworkPeer, NetworkConfig, NetworkMessage, ConsensusVote};

// Re-export performance profiler types
pub use performance_profiler::{QuantumMixingProfiler, PerformanceBottleneck, CriticalPathAnalysis, OptimizationReport};

// Re-export optimization engine types
pub use optimization_engine::{QuantumMixingOptimizer, OptimizationResult, ProductionReadinessAssessment, Phase3Targets, OptimizationStrategy};

// Re-export advanced ZK types
pub use advanced_zk::{AdvancedZKSystem, AdvancedZKConfig, RecursiveProofTree, UCEnvironment, UCProtocol, AdvancedZKMetrics};

// Re-export recursive STARK types for compressed mixing proofs
pub use recursive_stark::{
    RecursiveStarkProof, RecursiveStarkComposer, RecursiveConfig,
    StarkProofData, RecursiveProofMetadata, VerificationAir,
    VerificationConstraint, ConstraintType, ComposerMetrics,
};

// Re-export aggregated ring signature types (EURASIP 2025 O(log mn) space efficiency)
pub use aggregated_ring_sig::{
    AggregatedRingSignature, RingSignatureAggregator,
    MerkleTree, MerkleProof, MerkleResponseTree,
    BatchHints, SpaceAnalysis,
};

// Re-export Bulletproofs++ types (v3.9.0: EUROCRYPT 2024)
// 39% smaller proofs, 5x faster proving, 9.5x batch verification speedup
pub use bulletproofs_pp::{
    BPPlusConfig, BPPlusRangeProof, BPPlusError, PedersenCommitment,
    GeneratorSet, InnerProductProof, AggregatedBPPlusProof,
    PROOF_SIZE_64BIT,
};

// Re-export threshold mixing pool types (v3.9.0: NIST IR 8214C)
// MPC-based trustless mixing with FROST threshold signatures
pub use threshold_pool::{
    ThresholdPoolConfig, ThresholdMixingPool, MixingState as ThresholdMixingState,
    ParticipantId, ThresholdPedersenCommitment, ParticipantInput,
    SubmitReceipt, MixingOutput as ThresholdMixingOutput,
    ShuffledOutput, ThresholdSignature, ShuffleProof,
    ThresholdKeyShare, DKGRound1Package, DKGRound2Package,
};

// Re-export post-quantum lattice-based ring signature types (v3.9.0)
#[cfg(feature = "lattice-ring-sigs")]
pub use lattice_ring_sig::{
    LatticeRingParams, LatticeRingSignature, LatticeKeyImage,
    LatticeRingKeypair, LatticeRingSigner, SecurityLevel,
    verify as verify_lattice_ring_sig,
    is_linked as lattice_sigs_linked,
    estimate_signature_size as estimate_lattice_sig_size,
    batch_verify as batch_verify_lattice_sigs,
};

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Main configuration for the quantum mixing system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMixingConfig {
    /// Number of participants required for a mixing round
    pub min_participants: usize,
    /// Maximum participants allowed in a single round
    pub max_participants: usize,
    /// Mixing fee in atomic units
    pub mixing_fee: u64,
    /// Ring size for signatures (larger = more privacy)
    pub ring_size: usize,
    /// Enable quantum-resistant cryptography
    pub quantum_resistant: bool,
    /// Enable compliance mode
    pub compliance_enabled: bool,
    /// Stealth address scanning window
    pub scan_window_blocks: u64,
    /// ZK proof system to use
    pub proof_system: ProofType,
    /// Enable decoy transaction generation
    pub decoy_enabled: bool,
    /// Decoy strategy configuration
    pub decoy_strategy: DecoyStrategy,
}

impl Default for QuantumMixingConfig {
    fn default() -> Self {
        Self {
            min_participants: 3,
            max_participants: 100,
            mixing_fee: 1_000_000, // 0.001 QNK
            ring_size: 11,
            quantum_resistant: true,
            compliance_enabled: false,
            scan_window_blocks: 1000,
            proof_system: ProofType::Stark,
            decoy_enabled: true, // Enable by default for max privacy
            decoy_strategy: DecoyStrategy::default(), // 15x decoys by default
        }
    }
}

/// High-level quantum mixing service with advanced ZK integration
pub struct QuantumMixingService {
    engine: mixing_engine::QuantumMixingEngine,
    pool: mixing_pool::MixingPool,
    entropy: quantum_entropy::QuantumEntropyPool,
    compliance: Option<compliance::ComplianceEngine>,
    network: network::MixingNetworkManager,
    decoy_engine: Option<decoy_transactions::QuantumDecoyEngine>,
    advanced_zk: Option<advanced_zk::AdvancedZKSystem>,
    performance_profiler: Option<Arc<performance_profiler::QuantumMixingProfiler>>,
    optimizer: Option<optimization_engine::QuantumMixingOptimizer>,
    config: QuantumMixingConfig,
}

impl QuantumMixingService {
    /// Create a new quantum mixing service with advanced ZK integration
    pub async fn new(config: QuantumMixingConfig) -> Result<Self> {
        let entropy = Arc::new(quantum_entropy::QuantumEntropyPool::new().await?);
        let engine = mixing_engine::QuantumMixingEngine::new(entropy.clone()).await?;
        let pool = mixing_pool::MixingPool::new(config.clone()).await?;
        let compliance = if config.compliance_enabled {
            Some(compliance::ComplianceEngine::new(compliance::ComplianceConfig::default()).await?)
        } else {
            None
        };
        let network = network::MixingNetworkManager::new(network::NetworkConfig::default()).await?;
        
        // Initialize decoy engine if enabled
        let decoy_engine = if config.decoy_enabled {
            // Initialize required components for decoy engine
            let stealth_gen = Arc::new(RwLock::new(
                stealth_addresses::StealthAddressGenerator::new(entropy.clone()).await?
            ));
            let ring_signer = Arc::new(RwLock::new(
                ring_signatures::QuantumRingSigner::new(entropy.clone()).await?
            ));
            let zk_prover = Arc::new(
                zkp_prover::QuantumZKPProver::new(entropy.clone(), zkp_prover::ZKProofConfig::default()).await?
            );

            Some(decoy_transactions::QuantumDecoyEngine::new(
                entropy.clone(),
                stealth_gen,
                ring_signer,
                zk_prover,
                config.decoy_strategy.clone(),
            ).await?)
        } else {
            None
        };

        // Initialize advanced ZK system
        let advanced_zk = if true { // Always enable advanced ZK for enhanced privacy
            let zk_prover = Arc::new(
                zkp_prover::QuantumZKPProver::new(entropy.clone(), zkp_prover::ZKProofConfig::default()).await?
            );
            Some(advanced_zk::AdvancedZKSystem::new(
                advanced_zk::AdvancedZKConfig::default(),
                zk_prover,
                entropy.clone(),
            ).await?)
        } else {
            None
        };

        // Initialize performance profiler
        let profiler = Arc::new(
            performance_profiler::QuantumMixingProfiler::new(performance_profiler::ProfilerConfig::default()).await?
        );

        // Initialize optimizer
        let optimizer = optimization_engine::QuantumMixingOptimizer::new(profiler.clone()).await?;

        Ok(Self {
            engine,
            pool,
            entropy: (*entropy).clone(), // Clone the inner value
            compliance,
            network,
            decoy_engine,
            advanced_zk,
            performance_profiler: Some(profiler),
            optimizer: Some(optimizer),
            config,
        })
    }

    /// Submit a transaction for mixing
    pub async fn submit_for_mixing(
        &mut self,
        input: MixingInput,
    ) -> Result<Uuid> {
        tracing::info!("Submitting transaction for quantum mixing");

        // 1. Compliance check (if enabled)
        if let Some(compliance) = &self.compliance {
            // Create a temporary participant for compliance assessment
            let temp_participant = mixing_pool::PoolParticipant {
                participant_id: uuid::Uuid::new_v4(),
                input_commitment: zkp_prover::BalanceCommitment {
                    commitment: [0u8; 32], // Temporary
                    blinding_factor: [0u8; 32], // Temporary
                    amount: input.amount,
                },
                output_address: input.recipient_address,
                ownership_proof: zkp_prover::ZKProof {
                    proof_data: vec![0u8; 256],
                    proof_type: zkp_prover::ProofType::Stark,
                    public_inputs: vec![[0u8; 32]],
                    timestamp: chrono::Utc::now(),
                    circuit_id: "temp_ownership".to_string(),
                    vk_hash: [0u8; 32],
                },
                joined_at: chrono::Utc::now(),
                mixing_fee: self.config.mixing_fee,
            };

            let status = compliance.assess_participant(&temp_participant, &input).await?;
            match status {
                compliance::ComplianceStatus::Rejected(reason) => {
                    return Err(MixingError::ComplianceRejection(reason));
                }
                compliance::ComplianceStatus::Flagged(reason) => {
                    tracing::warn!("Transaction flagged for manual review: {}", reason);
                    // For now, we'll proceed but in production would queue for review
                }
                compliance::ComplianceStatus::Approved => {
                    tracing::info!("Transaction approved by compliance engine");
                }
            }
        }

        // 2. Add to mixing pool
        let session_id = self.pool.add_participant(input).await?;

        // 3. Check if pool is ready for mixing
        if self.pool.is_ready().await? {
            self.execute_mixing_round().await?;
        }

        Ok(session_id)
    }

    /// Execute a complete mixing round
    async fn execute_mixing_round(&mut self) -> Result<()> {
        tracing::info!("Executing quantum mixing round");

        // 1. Generate decoy transactions if enabled (before real mixing for maximum privacy)
        if let Some(decoy_engine) = &mut self.decoy_engine {
            let decoy_count = (self.config.decoy_strategy.decoy_ratio * self.pool.get_current_size().await? as f64) as usize;
            tracing::info!("Generating {} decoy transactions", decoy_count);
            
            let _decoy_campaign = decoy_engine.start_decoy_campaign(
                decoy_count,
                std::time::Duration::from_secs(300), // 5 minute campaign
            ).await?;
            
            tracing::info!("Decoy campaign started - providing cover traffic");
        }

        // 2. Get participants from pool
        let participants = self.pool.get_participants().await?;

        // 3. Execute mixing with quantum entropy
        let mixing_result = self.engine.execute_mixing_round(participants).await?;

        // 4. Broadcast results to network
        self.network.broadcast_mixing_result(&mixing_result).await?;

        // 5. Clear pool for next round
        self.pool.reset().await?;

        Ok(())
    }

    /// Get mixing statistics
    pub async fn get_statistics(&self) -> Result<MixingStatistics> {
        let decoy_metrics = if let Some(decoy_engine) = &self.decoy_engine {
            Some(decoy_engine.get_metrics().await?)
        } else {
            None
        };
        
        Ok(MixingStatistics {
            total_mixed_transactions: self.pool.get_total_processed().await?,
            current_pool_size: self.pool.get_current_size().await?,
            average_mixing_time: self.pool.get_average_mixing_time().await?,
            privacy_score: self.calculate_privacy_score().await?,
            quantum_entropy_quality: self.entropy.get_quality_score().await?,
            decoy_metrics,
        })
    }

    /// Start a decoy transaction campaign manually
    pub async fn start_decoy_campaign(
        &mut self,
        decoy_count: usize,
        duration: std::time::Duration,
    ) -> Result<Option<String>> {
        if let Some(decoy_engine) = &mut self.decoy_engine {
            let campaign_id = decoy_engine.start_decoy_campaign(decoy_count, duration).await?;
            let campaign_id_hex = hex::encode(&campaign_id[..8]);
            tracing::info!("Started manual decoy campaign {} with {} decoys", campaign_id_hex, decoy_count);
            Ok(Some(campaign_id_hex))
        } else {
            tracing::warn!("Decoy campaigns not enabled in configuration");
            Ok(None)
        }
    }

    /// Get decoy campaign metrics
    pub async fn get_decoy_metrics(&self) -> Result<Option<DecoyMetrics>> {
        if let Some(decoy_engine) = &self.decoy_engine {
            Ok(Some(decoy_engine.get_metrics().await?))
        } else {
            Ok(None)
        }
    }

    /// Generate recursive proof composition for enhanced privacy
    pub async fn compose_privacy_proofs(&self, proofs: Vec<ZKProof>) -> Result<Option<RecursiveProofTree>> {
        if let Some(advanced_zk) = &self.advanced_zk {
            let composition_circuit = "mixing_composition";
            let tree = advanced_zk.compose_proofs(proofs, composition_circuit).await?;
            Ok(Some(tree))
        } else {
            Ok(None)
        }
    }

    /// Verify recursive proof tree
    pub async fn verify_recursive_proof(&self, tree: &RecursiveProofTree) -> Result<Option<bool>> {
        if let Some(advanced_zk) = &self.advanced_zk {
            let result = advanced_zk.verify_recursive_proof(tree).await?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Execute mixing with UC security guarantees
    pub async fn execute_uc_mixing(&self, protocol_id: &str, inputs: Vec<[u8; 32]>) -> Result<Option<Vec<[u8; 32]>>> {
        if let Some(advanced_zk) = &self.advanced_zk {
            let outputs = advanced_zk.execute_uc_protocol(protocol_id, inputs).await?;
            Ok(Some(outputs))
        } else {
            Ok(None)
        }
    }

    /// Get advanced ZK system metrics
    pub async fn get_advanced_zk_metrics(&self) -> Result<Option<AdvancedZKMetrics>> {
        if let Some(advanced_zk) = &self.advanced_zk {
            Ok(Some(advanced_zk.get_metrics().await))
        } else {
            Ok(None)
        }
    }

    /// Get performance optimization report
    pub async fn get_optimization_report(&self) -> Result<Option<OptimizationReport>> {
        if let Some(profiler) = &self.performance_profiler {
            Ok(Some(profiler.generate_optimization_report().await?))
        } else {
            Ok(None)
        }
    }

    /// Run system optimization
    pub async fn optimize_system(&mut self) -> Result<Option<OptimizationResult>> {
        if let Some(optimizer) = &self.optimizer {
            let result = optimizer.optimize_mixing_system(&mut self.engine).await?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Register a UC protocol for advanced security
    pub async fn register_uc_protocol(&self, protocol: UCProtocol) -> Result<bool> {
        if let Some(advanced_zk) = &self.advanced_zk {
            advanced_zk.register_uc_protocol(protocol).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn calculate_privacy_score(&self) -> Result<f64> {
        // Privacy score based on:
        // - Ring size (larger = better)
        // - Pool size (larger = better)  
        // - Mixing frequency (higher = better)
        // - Quantum entropy quality (higher = better)
        // - Decoy ratio (higher = better)
        
        let ring_factor = (self.config.ring_size as f64).log2() / 10.0; // Max ~1.6 for ring size 2^10
        let pool_factor = (self.config.max_participants as f64).sqrt() / 10.0; // Max 1.0 for 100 participants
        let entropy_factor = self.entropy.get_quality_score().await?;
        let decoy_factor = if self.config.decoy_enabled {
            (self.config.decoy_strategy.decoy_ratio / 20.0).min(1.0) // Max 1.0 for 20x decoys
        } else {
            0.0
        };
        
        Ok((ring_factor + pool_factor + entropy_factor + decoy_factor) / 4.0)
    }
}

/// Statistics about the mixing service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixingStatistics {
    pub total_mixed_transactions: u64,
    pub current_pool_size: usize,
    pub average_mixing_time: std::time::Duration,
    pub privacy_score: f64,
    pub quantum_entropy_quality: f64,
    pub decoy_metrics: Option<DecoyMetrics>,
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    // use tokio_test; // Unused import

    #[tokio::test]
    async fn test_mixing_service_creation() {
        let config = QuantumMixingConfig::default();
        let service = QuantumMixingService::new(config).await;
        assert!(service.is_ok(), "Failed to create mixing service: {:?}", service.err());
    }

    #[tokio::test] 
    async fn test_basic_mixing_flow() {
        let config = QuantumMixingConfig {
            min_participants: 2,
            max_participants: 5,
            decoy_enabled: false, // Disable for simpler test
            ..Default::default()
        };
        
        let mut service = QuantumMixingService::new(config).await.unwrap();
        
        // Create test input
        let input = MixingInput {
            amount: 1_000_000_000, // 1 QNK
            sender_key: [1u8; 32],
            recipient_address: [2u8; 32],
            commitment: [3u8; 32],
        };
        
        let session_id = service.submit_for_mixing(input).await;
        assert!(session_id.is_ok(), "Failed to submit for mixing: {:?}", session_id.err());
    }

    #[tokio::test]
    async fn test_decoy_enabled_mixing() {
        let config = QuantumMixingConfig {
            min_participants: 2,
            max_participants: 5,
            decoy_enabled: true, // Enable decoys
            decoy_strategy: DecoyStrategy {
                decoy_ratio: 5.0, // 5x decoys for testing
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut service = QuantumMixingService::new(config).await.unwrap();
        
        // Test manual decoy campaign
        let campaign_result = service.start_decoy_campaign(
            10, 
            std::time::Duration::from_secs(60)
        ).await;
        assert!(campaign_result.is_ok(), "Failed to start decoy campaign: {:?}", campaign_result.err());
        assert!(campaign_result.unwrap().is_some(), "Should return campaign ID");
        
        // Test decoy metrics
        let metrics = service.get_decoy_metrics().await.unwrap();
        assert!(metrics.is_some(), "Should have decoy metrics");
    }
}