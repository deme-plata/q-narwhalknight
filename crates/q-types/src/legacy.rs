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

/// v3.1.4: Legacy Amount type (u64) for pre-u128 blocks
/// Old blocks stored amount/fee as u64 (8 bytes), new blocks use u128 (16 bytes)
pub type LegacyAmount = u64;

/// Legacy Transaction without tx_type field (pre-v1.0.60-beta)
/// v3.1.4: Uses LegacyAmount (u64) for backwards compatibility with old blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyTransaction {
    pub id: TxHash,
    pub from: Address,
    pub to: Address,
    pub amount: LegacyAmount, // v3.1.4: u64 for old blocks
    pub fee: LegacyAmount,    // v3.1.4: u64 for old blocks
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
            amount: legacy.amount as u128, // v3.1.4: Convert u64 -> u128
            fee: legacy.fee as u128,       // v3.1.4: Convert u64 -> u128
            nonce: legacy.nonce,
            signature: legacy.signature,
            timestamp: legacy.timestamp,
            data: legacy.data,
            token_type: legacy.token_type,
            fee_token_type: legacy.fee_token_type,
            tx_type: TransactionType::Transfer, // Default for legacy transactions
            // v2.3.0-beta: Default PQC fields for legacy transactions (Phase 0 Ed25519)
            pqc_signature: None,
            signature_phase: crate::TxSignaturePhase::Phase0Ed25519,
            pqc_public_key: None,
            // v3.4.2-beta: Default ZK privacy fields for legacy transactions (transparent)
            zk_proof_bundle: None,
            privacy_level: crate::TransactionPrivacyLevel::Transparent,
            bulletproof: None,
            nullifier: None,
            memo: None,
        }
    }
}

/// v3.1.4: Legacy BalanceUpdate with u64 fields (pre-v2.5.0)
/// Old blocks stored old_balance/new_balance as u64 (8 bytes each)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LegacyBalanceUpdate {
    pub address: Address,
    pub old_balance: u64, // v3.1.4: u64 for old blocks
    pub new_balance: u64, // v3.1.4: u64 for old blocks
    pub reason: String,
    pub timestamp: u64,
}

impl From<LegacyBalanceUpdate> for BalanceUpdate {
    fn from(legacy: LegacyBalanceUpdate) -> Self {
        BalanceUpdate {
            address: legacy.address,
            old_balance: legacy.old_balance as u128, // v3.1.4: Convert u64 -> u128
            new_balance: legacy.new_balance as u128, // v3.1.4: Convert u64 -> u128
            reason: legacy.reason,
            timestamp: legacy.timestamp,
        }
    }
}

/// Legacy QBlock with LegacyTransaction vec (pre-v1.0.60-beta)
/// v3.1.4: Uses LegacyBalanceUpdate for u64 backwards compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyQBlock {
    pub header: BlockHeader,
    pub mining_solutions: Vec<MiningSolution>,
    pub dag_parents: Vec<VertexId>,
    pub quantum_metadata: QuantumMetadata,
    pub transactions: Vec<LegacyTransaction>,
    #[serde(default)]
    pub balance_updates: Vec<LegacyBalanceUpdate>, // v3.1.4: u64 balance updates
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
            balance_updates: legacy.balance_updates.into_iter().map(Into::into).collect(), // v3.1.4: Convert
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

/// v3.1.4: Legacy Transaction V2 - has tx_type field but uses u64 for amounts
/// This is for blocks between v1.0.60-beta (added tx_type) and v2.5.0 (changed to u128)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyTransactionV2 {
    pub id: TxHash,
    pub from: Address,
    pub to: Address,
    pub amount: LegacyAmount, // v3.1.4: u64 for old blocks
    pub fee: LegacyAmount,    // v3.1.4: u64 for old blocks
    pub nonce: u64,
    pub signature: Vec<u8>,
    pub timestamp: DateTime<Utc>,
    pub data: Vec<u8>,
    pub token_type: TokenType,
    pub fee_token_type: TokenType,
    pub tx_type: TransactionType, // Has tx_type (unlike LegacyTransaction)
    // v2.3.0-beta PQC fields - defaulted for old blocks
    #[serde(default)]
    pub pqc_signature: Option<Vec<u8>>,
    #[serde(default)]
    pub signature_phase: crate::TxSignaturePhase,
    #[serde(default)]
    pub pqc_public_key: Option<Vec<u8>>,
}

impl From<LegacyTransactionV2> for Transaction {
    fn from(legacy: LegacyTransactionV2) -> Self {
        Transaction {
            id: legacy.id,
            from: legacy.from,
            to: legacy.to,
            amount: legacy.amount as u128, // v3.1.4: Convert u64 -> u128
            fee: legacy.fee as u128,       // v3.1.4: Convert u64 -> u128
            nonce: legacy.nonce,
            signature: legacy.signature,
            timestamp: legacy.timestamp,
            data: legacy.data,
            token_type: legacy.token_type,
            fee_token_type: legacy.fee_token_type,
            tx_type: legacy.tx_type,
            pqc_signature: legacy.pqc_signature,
            signature_phase: legacy.signature_phase,
            pqc_public_key: legacy.pqc_public_key,
            // v3.4.2-beta: Default ZK privacy fields for legacy transactions (transparent)
            zk_proof_bundle: None,
            privacy_level: crate::TransactionPrivacyLevel::Transparent,
            bulletproof: None,
            nullifier: None,
            memo: None,
        }
    }
}

/// Legacy QBlock V2 - with tx_type field but pre-u128 and pre-SQIsign
/// v3.1.4: Uses LegacyTransactionV2 and LegacyBalanceUpdate for u64 backwards compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyQBlockV2 {
    pub header: BlockHeader,
    pub mining_solutions: Vec<MiningSolution>,
    pub dag_parents: Vec<VertexId>,
    pub quantum_metadata: LegacyQuantumMetadata,
    pub transactions: Vec<LegacyTransactionV2>, // v3.1.4: tx_type + u64 amounts
    #[serde(default)]
    pub balance_updates: Vec<LegacyBalanceUpdate>, // v3.1.4: u64 balance updates
    pub size_bytes: usize,
}

impl From<LegacyQBlockV2> for QBlock {
    fn from(legacy: LegacyQBlockV2) -> Self {
        QBlock {
            header: legacy.header,
            mining_solutions: legacy.mining_solutions,
            dag_parents: legacy.dag_parents,
            quantum_metadata: legacy.quantum_metadata.into(),
            transactions: legacy.transactions.into_iter().map(Into::into).collect(), // v3.1.4: Convert
            balance_updates: legacy.balance_updates.into_iter().map(Into::into).collect(), // v3.1.4: Convert
            size_bytes: legacy.size_bytes,
        }
    }
}

/// Legacy QBlock V3 - with LegacyTransaction AND pre-SQIsign SpectralSignature
/// (This handles the oldest blocks from before v1.0.60-beta)
/// v3.1.4: Uses LegacyBalanceUpdate for u64 backwards compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyQBlockV3 {
    pub header: BlockHeader,
    pub mining_solutions: Vec<MiningSolution>,
    pub dag_parents: Vec<VertexId>,
    pub quantum_metadata: LegacyQuantumMetadata,
    pub transactions: Vec<LegacyTransaction>, // Old transactions without tx_type
    #[serde(default)]
    pub balance_updates: Vec<LegacyBalanceUpdate>, // v3.1.4: u64 balance updates
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
            balance_updates: legacy.balance_updates.into_iter().map(Into::into).collect(), // v3.1.4: Convert
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
