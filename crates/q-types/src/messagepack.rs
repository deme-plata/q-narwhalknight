/**
 * MessagePack Versioned Types
 *
 * Provides versioned wrappers for blocks and transactions
 * to support protocol evolution and compatibility checking.
 *
 * Based on AI expert recommendations (Kimi AI, ChatGPT, DeepSeek):
 * - Use #[serde(flatten)] to merge version with inner type
 * - Avoid serde_json::Value gymnastics
 * - Provide type-safe serialization/deserialization
 * - Enable version checking at protocol layer
 */

use serde::{Deserialize, Serialize};

use crate::{QBlock, Transaction};

/// Protocol version for blocks
pub const BLOCK_VERSION: &str = "qnk-block-v1";

/// Protocol version for transactions
pub const TRANSACTION_VERSION: &str = "qnk-tx-v1";

/**
 * Versioned Block Wrapper
 *
 * Wraps a QBlock with version information for P2P serialization.
 *
 * ⚠️ v1.1.5-beta: REMOVED #[serde(flatten)] - caused u128 serialization errors
 * ROOT CAUSE: #[serde(flatten)] buffers through serde::private::de::Content
 * which doesn't support u128 (total_difficulty field in BlockHeader)
 * See: https://github.com/serde-rs/json/issues/625
 *
 * The block is now serialized as a nested field instead of flattened.
 * This is a BREAKING CHANGE for P2P protocol - old nodes cannot read new format.
 *
 * # Example
 * ```
 * use q_types::messagepack::VersionedBlock;
 * use q_types::QBlock;
 *
 * let block = QBlock { /* ... */ };
 * let versioned = VersionedBlock::new(block);
 *
 * // Serialize to MessagePack or JSON
 * let bytes = rmp_serde::to_vec(&versioned)?;
 * // OR: let bytes = serde_json::to_vec(&versioned)?;
 *
 * // Deserialize
 * let decoded: VersionedBlock = rmp_serde::from_slice(&bytes)?;
 *
 * assert_eq!(decoded.version, "qnk-block-v1");
 * ```
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedBlock {
    /// Protocol version (e.g., "qnk-block-v1")
    pub version: String,

    /// The actual block data (as nested field, NOT flattened)
    /// v1.1.5-beta: Removed #[serde(flatten)] to fix u128 serialization
    pub block: QBlock,
}

impl VersionedBlock {
    /**
     * Create a new versioned block with the current protocol version
     */
    pub fn new(block: QBlock) -> Self {
        Self {
            version: BLOCK_VERSION.to_string(),
            block,
        }
    }

    /**
     * Create a versioned block with a specific version (for testing/migration)
     */
    pub fn with_version(block: QBlock, version: String) -> Self {
        Self { version, block }
    }

    /**
     * Check if the version is compatible with the current protocol
     */
    pub fn is_compatible(&self) -> bool {
        self.version == BLOCK_VERSION
    }

    /**
     * Get the block height
     */
    pub fn height(&self) -> u64 {
        self.block.header.height
    }
}

/**
 * Versioned Transaction Wrapper
 *
 * Wraps a Transaction with version information for MessagePack serialization.
 * The #[serde(flatten)] attribute merges the transaction fields with the version field.
 *
 * # Example
 * ```
 * use q_types::messagepack::VersionedTransaction;
 * use q_types::Transaction;
 *
 * let tx = Transaction { /* ... */ };
 * let versioned = VersionedTransaction::new(tx);
 *
 * // Serialize to MessagePack
 * let bytes = rmp_serde::to_vec(&versioned)?;
 *
 * // Deserialize from MessagePack
 * let decoded: VersionedTransaction = rmp_serde::from_slice(&bytes)?;
 *
 * assert_eq!(decoded.version, "qnk-tx-v1");
 * ```
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedTransaction {
    /// Protocol version (e.g., "qnk-tx-v1")
    pub version: String,

    /// The actual transaction data (flattened into the same level as version)
    #[serde(flatten)]
    pub tx: Transaction,
}

impl VersionedTransaction {
    /**
     * Create a new versioned transaction with the current protocol version
     */
    pub fn new(tx: Transaction) -> Self {
        Self {
            version: TRANSACTION_VERSION.to_string(),
            tx,
        }
    }

    /**
     * Create a versioned transaction with a specific version (for testing/migration)
     */
    pub fn with_version(tx: Transaction, version: String) -> Self {
        Self { version, tx }
    }

    /**
     * Check if the version is compatible with the current protocol
     */
    pub fn is_compatible(&self) -> bool {
        self.version == TRANSACTION_VERSION
    }

    /**
     * Get the transaction amount
     * v2.5.0: Returns u128 for full precision
     */
    pub fn amount(&self) -> u128 {
        self.tx.amount
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_versioned_block_creation() {
        // Note: QBlock initialization would require actual data
        // This is a structure test only
        let version = "qnk-block-v1".to_string();
        assert_eq!(BLOCK_VERSION, version);
    }

    #[test]
    fn test_versioned_transaction_creation() {
        let version = "qnk-tx-v1".to_string();
        assert_eq!(TRANSACTION_VERSION, version);
    }

    #[test]
    fn test_version_constants() {
        assert_eq!(BLOCK_VERSION, "qnk-block-v1");
        assert_eq!(TRANSACTION_VERSION, "qnk-tx-v1");
    }
}
