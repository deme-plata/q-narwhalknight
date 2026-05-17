//! Comprehensive VM Integration System for Q-NarwhalKnight
//! 
//! This module provides complete integration between:
//! - DAG-Knight consensus for transaction ordering
//! - Narwhal mempool for reliable broadcast
//! - Quantum VDF for deterministic randomness
//! - Post-Quantum cryptography for Phase 1 security

pub mod dag_consensus;
pub mod narwhal_broadcast;
pub mod quantum_vdf;
pub mod post_quantum;
pub mod coordinator;

pub use dag_consensus::DAGConsensusIntegration;
pub use narwhal_broadcast::NarwhalBroadcastIntegration;
pub use quantum_vdf::QuantumVDFIntegration;
pub use post_quantum::PostQuantumIntegration;
pub use coordinator::VMIntegrationCoordinator;

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Integration configuration for the VM system
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    pub node_id: String,
    pub phase: CryptographicPhase,
    pub enable_quantum_vdf: bool,
    pub enable_tor_anonymity: bool,
    pub consensus_timeout_ms: u64,
    pub mempool_batch_size: usize,
    pub vdf_difficulty: u64,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            node_id: "vm_node_1".to_string(),
            phase: CryptographicPhase::Phase1,
            enable_quantum_vdf: true,
            enable_tor_anonymity: true,
            consensus_timeout_ms: 3000,
            mempool_batch_size: 1000,
            vdf_difficulty: 1000000,
        }
    }
}

/// Current cryptographic phase
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CryptographicPhase {
    Phase0,  // Classical Ed25519
    Phase1,  // Hybrid Dilithium5 + Kyber1024
    Phase2,  // Post-Quantum only
    Phase3,  // Quantum-resistant lattice
    Phase4,  // Full quantum cryptography
}

/// Integration result with comprehensive metrics
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    pub success: bool,
    pub transaction_hash: String,
    pub execution_result: crate::vm::ExecutionResult,
    pub consensus_round: u64,
    pub vdf_output: Option<Vec<u8>>,
    pub crypto_phase: CryptographicPhase,
    pub processing_time_ms: u64,
    pub integration_metrics: IntegrationMetrics,
}

/// Comprehensive integration metrics
#[derive(Debug, Clone, Default)]
pub struct IntegrationMetrics {
    pub total_transactions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub consensus_rounds: u64,
    pub vdf_computations: u64,
    pub mempool_broadcasts: u64,
    pub average_execution_time_ms: f64,
    pub current_tps: f64,
    pub peak_tps: f64,
}

/// Main integration trait for all subsystem integrations
#[async_trait::async_trait]
pub trait VMIntegration: Send + Sync {
    async fn initialize(&self, config: &IntegrationConfig) -> Result<()>;
    async fn process_transaction(&self, tx: &crate::types::VMTransaction) -> Result<IntegrationResult>;
    async fn get_status(&self) -> Result<IntegrationStatus>;
    async fn shutdown(&self) -> Result<()>;
}

/// Status of the integration system
#[derive(Debug, Clone)]
pub struct IntegrationStatus {
    pub is_healthy: bool,
    pub consensus_status: String,
    pub mempool_status: String,
    pub vdf_status: String,
    pub crypto_status: String,
    pub current_phase: CryptographicPhase,
    pub active_connections: u32,
    pub pending_transactions: u64,
}