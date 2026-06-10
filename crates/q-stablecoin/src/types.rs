//! Q-Stablecoin Types
//!
//! Type definitions for quantum-enhanced stablecoin system

use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Quantum mint request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMintRequest {
    pub user_id: String,
    pub collateral_amount: BigDecimal,
    pub collateral_type: String,
    pub privacy_level: QuantumPrivacyLevel,
}

/// Quantum burn request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumBurnRequest {
    pub user_id: String,
    pub orbusd_amount: BigDecimal,
    pub collateral_type: Option<String>,
    pub privacy_level: QuantumPrivacyLevel,
}

/// Quantum mint result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMintResult {
    pub transaction_id: String,
    pub user_id: String,
    pub orbusd_minted: BigDecimal,
    pub collateral_deposited: BigDecimal,
    pub collateral_type: String,
    pub quantum_signature: Vec<u8>,
    pub privacy_proof: Option<QuantumZkProof>,
    pub wave_function_state: WaveFunctionState,
    pub entanglement_id: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Quantum burn result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumBurnResult {
    pub transaction_id: String,
    pub user_id: String,
    pub orbusd_burned: BigDecimal,
    pub collateral_released: BigDecimal,
    pub collateral_type: String,
    pub quantum_signature: Vec<u8>,
    pub privacy_proof: Option<QuantumZkProof>,
    pub wave_function_state: WaveFunctionState,
    pub timestamp: DateTime<Utc>,
}

/// Quantum privacy levels
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum QuantumPrivacyLevel {
    Basic = 0,
    Enhanced = 1,
    Maximum = 2,
    Quantum = 3,
}

/// Zero-knowledge proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumZkProof {
    pub proof_data: Vec<u8>,
    pub circuit_type: String,
    pub generated_at: DateTime<Utc>,
}

/// Wave function state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveFunctionState {
    pub amplitude: f64,
    pub phase: f64,
    pub coherence: f64,
}

/// System metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub total_supply: BigDecimal,
    pub quantum_coherence_score: f64,
    pub last_updated: DateTime<Utc>,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            total_supply: BigDecimal::from(0),
            quantum_coherence_score: 1.0,
            last_updated: Utc::now(),
        }
    }
}

/// Emergency states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyState {
    Normal,
    WaveCollapse,
    Shutdown,
}

/// Collateral position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollateralPosition {
    pub user_id: String,
    pub collateral_amount: BigDecimal,
    pub collateral_ratio: BigDecimal,
    pub last_updated: DateTime<Utc>,
}
