//! Legacy struct definitions for backwards-compatible deserialization
//!
//! v1.0.80-beta: Support reading blocks stored before v1.0.60-beta
//! v1.0.86-beta: Support reading blocks stored before SQIsign migration
//!
//! The Transaction struct changed in v1.0.60-beta to add the `tx_type` field.
//! The SpectralSignature struct changed in v1.0.86-beta to add the `sqisign_sig` field.
//! Bincode is a positional format that doesn't support #[serde(default)],
//! so we need explicit legacy structs for old block format compatibility.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{
    Address, Amount, TxHash, TokenType, TransactionType, Transaction,
    VertexId, Height, NetworkId,
};
use crate::block::{
    QBlock, BlockHeader, QuantumMetadata, MiningSolution, BalanceUpdate,
    SpectralSignature, SignaturePhase, HypergraphCoordinates, EnergyComponents,
};

/// Legacy Transaction without tx_type field (pre-v1.0.60-beta)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyTransaction {
    pub id: TxHash,
    pub from: Address,
    pub to: Address,
    pub amount: Amount,
    pub fee: Amount,
    pub nonce: u64,
    pub signature: Vec<u8>,
    pub timestamp: DateTime<Utc>,
    pub data: Vec<u8>,
    pub token_type: TokenType,
    pub fee_token_type: TokenType,
    // NOTE: No tx_type field - this is what changed in v1.0.60-beta
}

impl From<LegacyTransaction> for Transaction {
    fn from(legacy: LegacyTransaction) -> Self {
        Transaction {
            id: legacy.id,
            from: legacy.from,
            to: legacy.to,
            amount: legacy.amount,
            fee: legacy.fee,
            nonce: legacy.nonce,
            signature: legacy.signature,
            timestamp: legacy.timestamp,
            data: legacy.data,
            token_type: legacy.token_type,
            fee_token_type: legacy.fee_token_type,
            tx_type: TransactionType::Transfer, // Default for legacy transactions
        }
    }
}

/// Legacy QBlock with LegacyTransaction vec (pre-v1.0.60-beta)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyQBlock {
    pub header: BlockHeader,
    pub mining_solutions: Vec<MiningSolution>,
    pub dag_parents: Vec<VertexId>,
    pub quantum_metadata: QuantumMetadata,
    pub transactions: Vec<LegacyTransaction>,
    #[serde(default)]
    pub balance_updates: Vec<BalanceUpdate>,
    pub size_bytes: usize,
}

impl From<LegacyQBlock> for QBlock {
    fn from(legacy: LegacyQBlock) -> Self {
        QBlock {
            header: legacy.header,
            mining_solutions: legacy.mining_solutions,
            dag_parents: legacy.dag_parents,
            quantum_metadata: legacy.quantum_metadata,
            transactions: legacy.transactions.into_iter().map(Into::into).collect(),
            balance_updates: legacy.balance_updates,
            size_bytes: legacy.size_bytes,
        }
    }
}

// =============================================================================
// v1.0.86-beta LEGACY STRUCTS: Pre-SQIsign migration compatibility
// =============================================================================

/// Legacy SpectralSignature without sqisign_sig field (pre-v1.0.86-beta)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacySpectralSignature {
    pub validator: [u8; 32],
    #[serde(default)]
    pub crypto_phase: SignaturePhase,
    pub classical_sig: Vec<u8>,
    #[serde(default)]
    pub pqc_sig: Option<Vec<u8>>,
    // NOTE: No sqisign_sig field - this is what changed in v1.0.86-beta
    pub spectral_coefficient: f64,
    pub phase_deviation: f64,
    pub timestamp: u64,
}

impl From<LegacySpectralSignature> for SpectralSignature {
    fn from(legacy: LegacySpectralSignature) -> Self {
        SpectralSignature {
            validator: legacy.validator,
            crypto_phase: legacy.crypto_phase,
            classical_sig: legacy.classical_sig,
            pqc_sig: legacy.pqc_sig,
            sqisign_sig: None, // Default for legacy blocks
            spectral_coefficient: legacy.spectral_coefficient,
            phase_deviation: legacy.phase_deviation,
            timestamp: legacy.timestamp,
        }
    }
}

/// Legacy QuantumMetadata with LegacySpectralSignature (pre-v1.0.86-beta)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyQuantumMetadata {
    pub vertex_coordinates: HypergraphCoordinates,
    pub k_parameter: f64,
    pub energy: f64,
    pub energy_components: EnergyComponents,
    pub spectral_signatures: Vec<LegacySpectralSignature>,
    pub wavefunction_phase: f64,
    pub entropy_variance: f64,
    pub byzantine_scores: HashMap<String, f64>,
}

impl From<LegacyQuantumMetadata> for QuantumMetadata {
    fn from(legacy: LegacyQuantumMetadata) -> Self {
        QuantumMetadata {
            vertex_coordinates: legacy.vertex_coordinates,
            k_parameter: legacy.k_parameter,
            energy: legacy.energy,
            energy_components: legacy.energy_components,
            spectral_signatures: legacy.spectral_signatures.into_iter().map(Into::into).collect(),
            wavefunction_phase: legacy.wavefunction_phase,
            entropy_variance: legacy.entropy_variance,
            byzantine_scores: legacy.byzantine_scores,
        }
    }
}

/// Legacy QBlock V2 - with modern Transaction but pre-SQIsign SpectralSignature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyQBlockV2 {
    pub header: BlockHeader,
    pub mining_solutions: Vec<MiningSolution>,
    pub dag_parents: Vec<VertexId>,
    pub quantum_metadata: LegacyQuantumMetadata,
    pub transactions: Vec<Transaction>, // Modern transactions with tx_type
    #[serde(default)]
    pub balance_updates: Vec<BalanceUpdate>,
    pub size_bytes: usize,
}

impl From<LegacyQBlockV2> for QBlock {
    fn from(legacy: LegacyQBlockV2) -> Self {
        QBlock {
            header: legacy.header,
            mining_solutions: legacy.mining_solutions,
            dag_parents: legacy.dag_parents,
            quantum_metadata: legacy.quantum_metadata.into(),
            transactions: legacy.transactions,
            balance_updates: legacy.balance_updates,
            size_bytes: legacy.size_bytes,
        }
    }
}

/// Legacy QBlock V3 - with LegacyTransaction AND pre-SQIsign SpectralSignature
/// (This handles the oldest blocks from before v1.0.60-beta)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyQBlockV3 {
    pub header: BlockHeader,
    pub mining_solutions: Vec<MiningSolution>,
    pub dag_parents: Vec<VertexId>,
    pub quantum_metadata: LegacyQuantumMetadata,
    pub transactions: Vec<LegacyTransaction>, // Old transactions without tx_type
    #[serde(default)]
    pub balance_updates: Vec<BalanceUpdate>,
    pub size_bytes: usize,
}

impl From<LegacyQBlockV3> for QBlock {
    fn from(legacy: LegacyQBlockV3) -> Self {
        QBlock {
            header: legacy.header,
            mining_solutions: legacy.mining_solutions,
            dag_parents: legacy.dag_parents,
            quantum_metadata: legacy.quantum_metadata.into(),
            transactions: legacy.transactions.into_iter().map(Into::into).collect(),
            balance_updates: legacy.balance_updates,
            size_bytes: legacy.size_bytes,
        }
    }
}

/// Deserialize a QBlock with automatic fallback to legacy formats
///
/// Tries formats in order of most recent to oldest:
/// 1. Current format (v1.0.86-beta+) - with sqisign_sig
/// 2. LegacyQBlockV2 (v1.0.60-beta to v1.0.85-beta) - modern tx_type, no sqisign_sig
/// 3. LegacyQBlockV3 (v1.0.60-beta to v1.0.85-beta with legacy tx) - no tx_type, no sqisign_sig
/// 4. LegacyQBlock (pre-v1.0.60-beta) - no tx_type, modern quantum_metadata
///
/// This allows seamless reading of blocks stored with any version.
pub fn deserialize_qblock_with_fallback(data: &[u8]) -> Result<QBlock, bincode::Error> {
    // First try current format (v1.0.86-beta+)
    if let Ok(block) = bincode::deserialize::<QBlock>(data) {
        return Ok(block);
    }

    // Try LegacyQBlockV2: modern Transaction, pre-SQIsign SpectralSignature
    // This is most likely for blocks between v1.0.60-beta and v1.0.85-beta
    if let Ok(legacy_block) = bincode::deserialize::<LegacyQBlockV2>(data) {
        tracing::debug!(
            "📦 Deserialized legacy V2 block at height {} (pre-SQIsign format)",
            legacy_block.header.height
        );
        return Ok(legacy_block.into());
    }

    // Try LegacyQBlockV3: LegacyTransaction + pre-SQIsign SpectralSignature
    // This handles very old blocks
    if let Ok(legacy_block) = bincode::deserialize::<LegacyQBlockV3>(data) {
        tracing::debug!(
            "📦 Deserialized legacy V3 block at height {} (pre-v1.0.60 + pre-SQIsign format)",
            legacy_block.header.height
        );
        return Ok(legacy_block.into());
    }

    // Try original LegacyQBlock (pre-v1.0.60-beta with modern quantum_metadata)
    match bincode::deserialize::<LegacyQBlock>(data) {
        Ok(legacy_block) => {
            tracing::debug!(
                "📦 Deserialized legacy block at height {} (pre-v1.0.60 format)",
                legacy_block.header.height
            );
            Ok(legacy_block.into())
        }
        Err(legacy_err) => {
            // All formats failed - return the last error
            Err(legacy_err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legacy_transaction_conversion() {
        let legacy = LegacyTransaction {
            id: [0u8; 32],
            from: [1u8; 32],
            to: [2u8; 32],
            amount: 1000,
            fee: 10,
            nonce: 1,
            signature: vec![0u8; 64],
            timestamp: Utc::now(),
            data: vec![],
            token_type: TokenType::QUG,
            fee_token_type: TokenType::QUGUSD,
        };

        let modern: Transaction = legacy.into();
        assert_eq!(modern.tx_type, TransactionType::Transfer);
        assert_eq!(modern.amount, 1000);
    }
}
