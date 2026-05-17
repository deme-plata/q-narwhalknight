//! Quantum Neural Oracle (QNO)
//!
//! A decentralized, verifiable, self-funding prediction infrastructure for Q-NarwhalKnight.
//!
//! # Core Components
//!
//! - **Quantum Simulation**: 128-qubit simulated quantum computing for feature extraction
//! - **Mixture of Experts**: Domain-specific neural networks with quantum attention
//! - **Committee Consensus**: Decentralized prediction verification with reputation
//! - **Prediction Markets**: Trade predictions as financial instruments
//! - **zkML Proofs**: Zero-knowledge proofs for verifiable neural computation
//! - **Neural Evolution**: On-chain architecture search for autonomous improvement
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     QUANTUM NEURAL ORACLE (QNO)                      │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  Layer 5: Prediction Markets (AMM, Staking, Shares)                 │
//! │  Layer 4: zkML Proof System (Plonk, GPU Prover)                     │
//! │  Layer 3: Committee Consensus (BFT, Reputation, Slashing)           │
//! │  Layer 2: Mixture of Quantum Experts (8 domains)                    │
//! │  Layer 1: Quantum Feature Encoder (Variational Circuits)            │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use q_neural_oracle::QuantumNeuralOracle;
//!
//! let oracle = QuantumNeuralOracle::new(OracleConfig::default());
//!
//! // Get prediction with automatic fallback
//! let prediction = oracle.predict(PredictionType::FeeForecasting, &context).await?;
//!
//! // Stake on prediction
//! oracle.stake_prediction(prediction_id, amount, confidence).await?;
//! ```

pub mod quantum;
pub mod experts;
pub mod consensus;
pub mod markets;
pub mod fallback;
pub mod zkml;
pub mod evolution;
pub mod federated;
pub mod governance;
pub mod autonomous;
pub mod quantum_random_forest;

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

// Phase 1: Core quantum and ML components
pub use quantum::{QuantumState, QuantumGate, VariationalQuantumCircuit, QuantumAnnealer};
pub use experts::{MixtureOfQuantumExperts, PredictionDomain, QuantumExpert};
pub use consensus::{PredictionCommittee, CommitteeMember, ReputationSystem};
pub use markets::{PredictionAMM, PredictionStakingPool};
pub use fallback::OracleFallbackSystem;

// Phase 2: GPU-accelerated quantum simulation (128 qubits)
pub use quantum::{GpuQuantumState, GpuQuantumSimulator, Observable};

// Phase 3: Zero-knowledge ML proofs
pub use zkml::{ZkMLConfig, ZkMLProof, ZkMLProver, ZkMLVerifier, ZkNeuralNetwork, ZkSecurityLevel};

// Phase 4: Neural architecture search
pub use evolution::{NasEngine, NasConfig, Architecture, FitnessScore, LayerType};

// Phase 5: Federated learning with zkML gradient proofs
pub use federated::{
    FederatedCoordinator, FederatedConfig, GradientUpdate, AggregatedGradient,
    SecureAggregator, GradientShareValue,
};

// Phase 6: On-chain architecture evolution via NAS governance
pub use governance::{
    NasGovernance, GovernanceConfig, ArchitectureProposal, ProposalStatus,
    Vote, VoteDirection, OnChainMetrics, GovernanceStats,
};

// Phase 7: Full autonomous prediction infrastructure
pub use autonomous::{
    AutonomousOracle, AutonomousConfig, DataFeed, FeedType, DeployedModel,
    PredictionRequest, PredictionResponse, TreasuryState, OracleEvent,
    SelfImprover, SelfImproverConfig, PerformanceTrend,
};

/// Main Quantum Neural Oracle interface
pub struct QuantumNeuralOracle {
    /// Quantum simulation layer
    quantum_layer: Arc<RwLock<quantum::QuantumLayer>>,

    /// Mixture of Quantum Experts
    experts: Arc<RwLock<MixtureOfQuantumExperts>>,

    /// Prediction committee
    committee: Arc<RwLock<PredictionCommittee>>,

    /// Prediction markets
    markets: Arc<RwLock<markets::PredictionMarkets>>,

    /// Fallback system
    fallback: Arc<OracleFallbackSystem>,

    /// Configuration
    config: OracleConfig,

    /// Health metrics
    health: Arc<RwLock<OracleHealth>>,
}

/// Oracle configuration
#[derive(Clone, Debug)]
pub struct OracleConfig {
    /// Number of qubits for quantum simulation
    pub num_qubits: usize,

    /// Number of variational circuit layers
    pub circuit_layers: usize,

    /// Committee size
    pub committee_size: usize,

    /// Minimum stake for committee membership
    pub min_committee_stake: u64,

    /// Prediction timeout (ms)
    pub prediction_timeout_ms: u64,

    /// Enable fallback system
    pub enable_fallback: bool,

    /// Learning rate for online updates
    pub learning_rate: f64,
}

impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            num_qubits: 32,  // Start with 32 qubits (Phase 2 will go to 128)
            circuit_layers: 4,
            committee_size: 21,
            min_committee_stake: 10_000_000_000, // 100 QUG
            prediction_timeout_ms: 5000,
            enable_fallback: true,
            learning_rate: 0.01,
        }
    }
}

/// Prediction result
#[derive(Clone, Debug)]
pub struct Prediction {
    /// Primary prediction value
    pub value: f64,

    /// Confidence score (0-1)
    pub confidence: f64,

    /// Prediction domain
    pub domain: PredictionDomain,

    /// Source layer (quantum, classical, committee, twma)
    pub source: PredictionSource,

    /// Expert contributions
    pub expert_weights: Vec<(PredictionDomain, f64)>,

    /// Quantum fidelity (quality of quantum computation)
    pub quantum_fidelity: f64,

    /// Timestamp
    pub timestamp: u64,

    /// Optional zkML proof (Phase 3+)
    pub proof: Option<Vec<u8>>,
}

/// Source of prediction
#[derive(Clone, Debug, PartialEq)]
pub enum PredictionSource {
    Quantum,
    Classical,
    Committee,
    TWMA,
    Fallback,
}

/// Prediction context (input features)
#[derive(Clone, Debug)]
pub struct PredictionContext {
    /// Blockchain height
    pub block_height: u64,

    /// Current fee rate
    pub current_fee_rate: f64,

    /// Network hashrate
    pub hashrate: f64,

    /// Staking pool size
    pub staking_total: u64,

    /// Active validators
    pub validator_count: u32,

    /// Recent transaction volume
    pub tx_volume: f64,

    /// Historical data (for LSTM)
    pub historical: Vec<HistoricalDataPoint>,

    /// Additional domain-specific features
    pub domain_features: std::collections::HashMap<String, f64>,
}

#[derive(Clone, Debug)]
pub struct HistoricalDataPoint {
    pub timestamp: u64,
    pub value: f64,
    pub metadata: std::collections::HashMap<String, f64>,
}

/// Oracle health metrics
#[derive(Clone, Debug, Default)]
pub struct OracleHealth {
    /// Total predictions made
    pub total_predictions: u64,

    /// Successful predictions
    pub successful_predictions: u64,

    /// Average prediction latency (ms)
    pub avg_latency_ms: f64,

    /// Fallback usage rate
    pub fallback_rate: f64,

    /// Quantum fidelity average
    pub avg_quantum_fidelity: f64,

    /// Committee consensus rate
    pub consensus_rate: f64,

    /// Last health check
    pub last_check: u64,
}

impl QuantumNeuralOracle {
    /// Create new Quantum Neural Oracle
    pub fn new(config: OracleConfig) -> Self {
        info!("🔮 Initializing Quantum Neural Oracle");
        info!("   Qubits: {}", config.num_qubits);
        info!("   Circuit layers: {}", config.circuit_layers);
        info!("   Committee size: {}", config.committee_size);

        let quantum_layer = Arc::new(RwLock::new(
            quantum::QuantumLayer::new(config.num_qubits, config.circuit_layers)
        ));

        let experts = Arc::new(RwLock::new(
            MixtureOfQuantumExperts::new(config.num_qubits)
        ));

        let committee = Arc::new(RwLock::new(
            PredictionCommittee::new(config.committee_size, config.min_committee_stake)
        ));

        let markets = Arc::new(RwLock::new(
            markets::PredictionMarkets::new()
        ));

        let fallback = Arc::new(OracleFallbackSystem::new());

        info!("✅ Quantum Neural Oracle initialized");

        Self {
            quantum_layer,
            experts,
            committee,
            markets,
            fallback,
            config,
            health: Arc::new(RwLock::new(OracleHealth::default())),
        }
    }

    /// Make a prediction
    pub async fn predict(
        &self,
        domain: PredictionDomain,
        context: &PredictionContext,
    ) -> anyhow::Result<Prediction> {
        let start = std::time::Instant::now();

        // Try quantum prediction
        let quantum_result = tokio::time::timeout(
            std::time::Duration::from_millis(self.config.prediction_timeout_ms),
            self.quantum_predict(domain, context),
        ).await;

        match quantum_result {
            Ok(Ok(prediction)) => {
                self.update_health_success(start.elapsed()).await;
                Ok(prediction)
            }
            Ok(Err(e)) => {
                warn!("Quantum prediction failed: {:?}", e);
                if self.config.enable_fallback {
                    self.fallback.get_prediction(domain, context).await
                } else {
                    Err(e)
                }
            }
            Err(_) => {
                warn!("Quantum prediction timeout");
                if self.config.enable_fallback {
                    self.fallback.get_prediction(domain, context).await
                } else {
                    Err(anyhow::anyhow!("Prediction timeout"))
                }
            }
        }
    }

    /// Internal quantum prediction
    async fn quantum_predict(
        &self,
        domain: PredictionDomain,
        context: &PredictionContext,
    ) -> anyhow::Result<Prediction> {
        // 1. Encode context as quantum state
        let features = self.extract_features(context);
        let quantum_state = {
            let ql = self.quantum_layer.read().await;
            ql.encode_features(&features)
        };

        // 2. Route through Mixture of Experts
        let expert_prediction = {
            let experts = self.experts.read().await;
            experts.predict(&quantum_state, domain, context).await
        };

        // 3. Verify through committee (simplified for Phase 1)
        let verified = {
            let committee = self.committee.read().await;
            committee.quick_verify(&expert_prediction).await
        };

        if !verified {
            warn!("Committee verification failed for {:?}", domain);
        }

        Ok(Prediction {
            value: expert_prediction.primary_value,
            confidence: expert_prediction.confidence,
            domain,
            source: PredictionSource::Quantum,
            expert_weights: expert_prediction.expert_weights,
            quantum_fidelity: quantum_state.fidelity(),
            timestamp: chrono::Utc::now().timestamp() as u64,
            proof: None,
        })
    }

    /// Extract features from context
    fn extract_features(&self, context: &PredictionContext) -> Vec<f64> {
        let mut features = vec![
            context.block_height as f64 / 1_000_000.0,
            context.current_fee_rate,
            context.hashrate.ln().max(0.0) / 20.0,
            context.staking_total as f64 / 1e15,
            context.validator_count as f64 / 1000.0,
            context.tx_volume.ln().max(0.0) / 15.0,
        ];

        // Add historical features (last 10 data points)
        for point in context.historical.iter().take(10) {
            features.push(point.value);
        }

        // Pad to power of 2 for quantum encoding
        let target_len = 2usize.pow((features.len() as f64).log2().ceil() as u32);
        features.resize(target_len, 0.0);

        features
    }

    /// Stake on a prediction
    pub async fn stake_prediction(
        &self,
        prediction_id: u64,
        predicted_value: f64,
        amount: u64,
        confidence: f64,
    ) -> anyhow::Result<markets::StakeReceipt> {
        let markets = self.markets.read().await;
        markets.staking.stake_prediction(
            prediction_id,
            predicted_value,
            amount,
            confidence,
        ).await
    }

    /// Update health metrics on success
    async fn update_health_success(&self, latency: std::time::Duration) {
        let mut health = self.health.write().await;
        health.total_predictions += 1;
        health.successful_predictions += 1;

        let latency_ms = latency.as_millis() as f64;
        health.avg_latency_ms = health.avg_latency_ms * 0.9 + latency_ms * 0.1;
        health.last_check = chrono::Utc::now().timestamp() as u64;
    }

    /// Get oracle health
    pub async fn health(&self) -> OracleHealth {
        self.health.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_oracle_creation() {
        let oracle = QuantumNeuralOracle::new(OracleConfig::default());
        let health = oracle.health().await;
        assert_eq!(health.total_predictions, 0);
    }

    #[tokio::test]
    async fn test_feature_extraction() {
        let oracle = QuantumNeuralOracle::new(OracleConfig::default());

        let context = PredictionContext {
            block_height: 100_000,
            current_fee_rate: 1.5,
            hashrate: 1e18,
            staking_total: 1_000_000_000_000,
            validator_count: 100,
            tx_volume: 50_000.0,
            historical: vec![],
            domain_features: std::collections::HashMap::new(),
        };

        let features = oracle.extract_features(&context);
        assert!(features.len().is_power_of_two());
    }
}
