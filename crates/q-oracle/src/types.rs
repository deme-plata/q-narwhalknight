//! Quantum Oracle Types
//!
//! Type definitions for quantum-enhanced oracle system with physics-inspired AI

use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantum oracle submission with physics-enhanced validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumOracleSubmission {
    pub submission_id: String,
    pub oracle_id: String,
    pub feed_id: String,
    pub value: BigDecimal,
    pub timestamp: DateTime<Utc>,
    pub round_id: u64,
    pub quantum_signature: Vec<u8>,
    pub wave_function_data: Option<WaveFunctionData>,
    pub entangled_sources: Vec<String>,
    pub ai_confidence: f64,
    pub uncertainty_bounds: (BigDecimal, BigDecimal),
    pub privacy_level: QuantumPrivacyLevel,
}

/// Wave function data for quantum price evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveFunctionData {
    pub amplitude: f64,
    pub phase: f64,
    pub frequency: f64,
    pub quantum_state: QuantumState,
    pub coherence_time: u64, // microseconds
}

/// Quantum states for price data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantumState {
    /// Superposition state - price uncertain
    Superposition,
    /// Entangled state - correlated with other feeds
    Entangled,
    /// Collapsed state - definite price determined
    Collapsed,
    /// Decoherent state - quantum effects lost
    Decoherent,
}

/// Quantum privacy levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantumPrivacyLevel {
    Basic = 0,
    Enhanced = 1,
    Quantum = 2,
    PostQuantum = 3,
}

impl Default for QuantumPrivacyLevel {
    fn default() -> Self {
        QuantumPrivacyLevel::Enhanced
    }
}

/// Quantum oracle node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumOracleNode {
    pub node_id: String,
    pub address: String,
    pub public_key: Vec<u8>,
    pub quantum_public_key: Option<Vec<u8>>,
    pub reputation_score: f64,
    pub quantum_coherence: f64,
    pub specializations: Vec<QuantumFeedType>,
    pub stake_amount: BigDecimal,
    pub ai_model_version: String,
    pub quantum_capabilities: QuantumCapabilities,
    pub performance_metrics: NodePerformanceMetrics,
    pub entangled_nodes: Vec<String>,
    pub registered_at: DateTime<Utc>,
    pub last_submission: Option<DateTime<Utc>>,
    pub active: bool,
}

/// Quantum capabilities of oracle nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCapabilities {
    pub supports_superposition: bool,
    pub supports_entanglement: bool,
    pub supports_tunneling_detection: bool,
    pub supports_wave_function_collapse: bool,
    pub supports_ai_aggregation: bool,
    pub quantum_processor_type: Option<String>,
    pub max_qubits: Option<u32>,
}

/// Node performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePerformanceMetrics {
    pub average_latency_ms: f64,
    pub accuracy_percentage: f64,
    pub uptime_percentage: f64,
    pub quantum_coherence_stability: f64,
    pub ai_prediction_accuracy: f64,
    pub submissions_count: u64,
    pub successful_submissions: u64,
}

/// Quantum feed type definitions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantumFeedType {
    /// Price feeds with quantum uncertainty
    Price,
    /// Volume feeds with quantum fluctuations
    Volume,
    /// Volatility feeds with wave analysis
    Volatility,
    /// Sentiment analysis with quantum NLP
    Sentiment,
    /// Custom quantum feed type
    Custom(String),
}

/// Quantum feed definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumFeed {
    pub id: String,
    pub symbol: String,
    pub description: String,
    pub feed_type: QuantumFeedType,
    pub quantum_enabled: bool,
    pub ai_enhanced: bool,
    pub uncertainty_tracking: bool,
    pub entanglement_pairs: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub active: bool,
}

/// Quantum price data with physics-inspired properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPriceData {
    pub feed_id: String,
    pub price: BigDecimal,
    pub quantum_confidence: f64,
    pub wave_function_amplitude: f64,
    pub entangled_feeds: Vec<String>,
    pub ai_prediction_score: f64,
    pub uncertainty_range: (BigDecimal, BigDecimal),
    pub quantum_signature: Vec<u8>,
    pub timestamp: DateTime<Utc>,
    pub block_height: Option<u64>,
}

/// Quantum submission result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSubmissionResult {
    pub submission_id: String,
    pub oracle_id: String,
    pub feed_id: String,
    pub accepted: bool,
    pub quantum_score: f64,
    pub uncertainty_adjustment: BigDecimal,
    pub anomaly_probability: f64,
    pub entanglement_proof: Option<QuantumEntanglementProof>,
    pub aggregated_price: BigDecimal,
    pub wave_function_state: QuantumState,
    pub timestamp: DateTime<Utc>,
}

/// Quantum entanglement proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumEntanglementProof {
    pub proof_id: String,
    pub entangled_oracles: Vec<String>,
    pub correlation_matrix: Vec<Vec<f64>>,
    pub bell_inequality_violation: f64,
    pub quantum_signature: Vec<u8>,
    pub generated_at: DateTime<Utc>,
}

/// Quantum aggregation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAggregationResult {
    pub feed_id: String,
    pub quantum_price: BigDecimal,
    pub wave_amplitude: f64,
    pub wave_state: QuantumState,
    pub ai_score: f64,
    pub uncertainty_bounds: (BigDecimal, BigDecimal),
    pub entangled_feeds: Vec<String>,
    pub final_price: BigDecimal,
    pub participants: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

/// Quantum verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumVerificationResult {
    pub is_valid: bool,
    pub quantum_score: f64,
    pub signature_valid: bool,
    pub entropy_check: bool,
    pub coherence_verified: bool,
    pub error_reason: Option<String>,
    pub verification_proof: Option<Vec<u8>>,
}

/// Quantum oracle metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumOracleMetrics {
    pub total_submissions: u64,
    pub successful_submissions: u64,
    pub average_latency_ms: f64,
    pub quantum_confidence_avg: f64,
    pub wave_function_coherence: f64,
    pub ai_accuracy_percentage: f64,
    pub entanglement_success_rate: f64,
    pub throughput_tps: u64,
    pub active_nodes: u64,
    pub active_feeds: u64,
    pub last_updated: DateTime<Utc>,
}

impl Default for QuantumOracleMetrics {
    fn default() -> Self {
        Self {
            total_submissions: 0,
            successful_submissions: 0,
            average_latency_ms: 0.0,
            quantum_confidence_avg: 0.0,
            wave_function_coherence: 0.0,
            ai_accuracy_percentage: 0.0,
            entanglement_success_rate: 0.0,
            throughput_tps: 0,
            active_nodes: 0,
            active_feeds: 0,
            last_updated: Utc::now(),
        }
    }
}

/// Quantum Oracle configuration with physics-inspired parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumOracleConfig {
    /// Maximum number of oracle nodes in quantum superposition
    pub max_oracle_nodes: u64,
    /// Quantum coherence threshold for data validation
    pub coherence_threshold: f64,
    /// Wave function collapse timeout (milliseconds)
    pub wave_collapse_timeout_ms: u64,
    /// Heisenberg uncertainty factor for price aggregation
    pub uncertainty_factor: f64,
    /// Quantum entanglement correlation strength
    pub entanglement_strength: f64,
    /// AI neural network depth (quantum layers)
    pub quantum_neural_depth: u32,
    /// Schrödinger equation time step for price evolution
    pub schrodinger_time_step: f64,
    /// Planck constant scaling for micro-fluctuations
    pub planck_scaling: f64,
    /// Speed of light constraint for data propagation (m/s)
    pub light_speed_constraint: f64,
    /// Quantum tunneling probability for outlier detection
    pub tunneling_probability: f64,
    /// Post-quantum security level
    pub security_level: u8,
    /// Privacy settings
    pub privacy_config: QuantumPrivacyConfig,
    /// Network performance targets
    pub performance_targets: PerformanceTargets,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Target throughput (TPS)
    pub target_tps: u64,
    /// Maximum latency (milliseconds)
    pub max_latency_ms: u64,
    /// Minimum accuracy percentage
    pub min_accuracy_pct: f64,
    /// Maximum cost per query (ORB)
    pub max_cost_per_query: BigDecimal,
}

impl Default for QuantumOracleConfig {
    fn default() -> Self {
        Self {
            max_oracle_nodes: 1000,
            coherence_threshold: 0.95,
            wave_collapse_timeout_ms: 500,
            uncertainty_factor: 0.01618,    // Golden ratio uncertainty
            entanglement_strength: 0.707,   // √2/2 quantum correlation
            quantum_neural_depth: 12,       // Deep quantum network
            schrodinger_time_step: 0.001,   // 1ms quantum evolution
            planck_scaling: 6.62607015e-34, // Planck's constant
            light_speed_constraint: 299792458.0, // c in m/s
            tunneling_probability: 0.001,   // 0.1% quantum tunneling
            security_level: 5,              // Maximum post-quantum security
            privacy_config: QuantumPrivacyConfig::default(),
            performance_targets: PerformanceTargets {
                target_tps: 927000, // 927k TPS target
                max_latency_ms: 1,  // Sub-millisecond latency
                min_accuracy_pct: 99.99,
                max_cost_per_query: "0.1".parse().unwrap(), // 0.1 ORB per query
            },
        }
    }
}

/// Quantum privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPrivacyConfig {
    pub tor_enabled: bool,
    pub circuit_rotation_interval: u64, // seconds
    pub zk_proofs_enabled: bool,
    pub post_quantum_encryption: bool,
    pub privacy_level: QuantumPrivacyLevel,
    pub anonymity_set_size: u32,
}

impl Default for QuantumPrivacyConfig {
    fn default() -> Self {
        Self {
            tor_enabled: true,
            circuit_rotation_interval: 300, // 5 minutes
            zk_proofs_enabled: true,
            post_quantum_encryption: true,
            privacy_level: QuantumPrivacyLevel::Enhanced,
            anonymity_set_size: 100,
        }
    }
}

/// AI model configuration for quantum neural networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAIConfig {
    pub neural_network_depth: u32,
    pub quantum_layers: u32,
    pub classical_layers: u32,
    pub learning_rate: f64,
    pub quantum_learning_rate: f64,
    pub entanglement_strength: f64,
    pub decoherence_rate: f64,
    pub model_type: AIModelType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AIModelType {
    QuantumNeuralNetwork,
    VariationalQuantumEigensolver,
    QuantumApproximateOptimization,
    HybridClassicalQuantum,
}

/// Network statistics for quantum oracle network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumNetworkStats {
    pub total_nodes: u64,
    pub active_nodes: u64,
    pub entangled_pairs: u64,
    pub average_quantum_coherence: f64,
    pub network_throughput_tps: u64,
    pub average_consensus_time_ms: f64,
    pub quantum_advantage_factor: f64, // Performance improvement vs classical
    pub uptime_percentage: f64,
}

/// Oracle reputation data with quantum scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumReputationData {
    pub oracle_id: String,
    pub base_reputation: f64,
    pub quantum_coherence_score: f64,
    pub ai_accuracy_score: f64,
    pub consistency_score: f64,
    pub response_time_score: f64,
    pub entanglement_reliability: f64,
    pub overall_quantum_score: f64,
    pub reputation_history: Vec<ReputationEvent>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationEvent {
    pub event_type: ReputationEventType,
    pub score_change: f64,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReputationEventType {
    SubmissionAccepted,
    SubmissionRejected,
    QuantumCoherenceLoss,
    AIAccuracyImprovement,
    NetworkContribution,
    Penalty,
}
