/// v2.4.0-beta: Swap Indexer for Consensus-Verified Transaction History
///
/// This module indexes swap transactions from verified blocks to provide
/// a queryable history that all nodes agree on.
///
/// Key features:
/// - Processes blocks to extract swap transactions
/// - Records swap history in CF_SWAP_HISTORY
/// - Provides APIs for querying swap history by token
/// - Ensures all nodes have consistent transaction history

use chrono::{DateTime, Utc};
use q_types::{Transaction, TransactionType};
use q_storage::StorageEngine;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Swap history record stored in RocksDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusSwapRecord {
    /// Transaction ID (from block)
    pub tx_id: [u8; 32],
    /// Block height where this swap was confirmed
    pub block_height: u64,
    /// Block hash for verification
    pub block_hash: [u8; 32],
    /// Timestamp of the block
    pub timestamp: i64,
    /// Input token address (for native QUG: [0u8; 32])
    pub token_in: [u8; 32],
    /// Output token address
    pub token_out: [u8; 32],
    /// Input amount in base units
    pub amount_in: u64,
    /// Output amount in base units (calculated from state changes)
    pub amount_out: u64,
    /// Wallet that performed the swap
    pub wallet: [u8; 32],
    /// Pool ID used for the swap
    pub pool_id: [u8; 32],
    /// Swap direction (0 = token_a -> token_b, 1 = token_b -> token_a)
    pub direction: u8,
}

impl ConsensusSwapRecord {
    /// Convert to JSON for API response
    pub fn to_api_json(&self) -> serde_json::Value {
        let amount_in_display = self.amount_in as f64 / 1e24;
        let amount_out_display = self.amount_out as f64 / 1e24;
        let exchange_rate = if self.amount_in > 0 {
            self.amount_out as f64 / self.amount_in as f64
        } else {
            0.0
        };

        serde_json::json!({
            "id": format!("0x{}", hex::encode(self.tx_id)),
            "timestamp": self.timestamp,
            "blockHeight": self.block_height,
            "blockHash": format!("0x{}", hex::encode(self.block_hash)),
            "type": if self.direction == 0 { "sell" } else { "buy" },
            "tokenIn": format!("0x{}", hex::encode(self.token_in)),
            "tokenOut": format!("0x{}", hex::encode(self.token_out)),
            "amountIn": amount_in_display,
            "amountOut": amount_out_display,
            "wallet": format!("qnk{}", hex::encode(self.wallet)),
            "poolId": format!("0x{}", hex::encode(self.pool_id)),
            "exchangeRate": exchange_rate,
            "txHash": format!("0x{}", hex::encode(self.tx_id)),
        })
    }
}

/// Swap Indexer for processing blocks and extracting swap transactions
pub struct SwapIndexer {
    storage: Arc<StorageEngine>,
}

impl SwapIndexer {
    /// Create a new swap indexer
    pub fn new(storage: Arc<StorageEngine>) -> Self {
        Self { storage }
    }

    /// Index swap transactions from a block
    ///
    /// This should be called after a block is verified and applied.
    /// It extracts all swap transactions and records them in history.
    pub async fn index_block(
        &self,
        block_height: u64,
        block_hash: [u8; 32],
        block_timestamp: i64,
        transactions: &[Transaction],
    ) -> Result<usize, String> {
        let mut indexed_count = 0;

        for tx in transactions {
            // Only process swap transactions
            if tx.tx_type != TransactionType::Swap {
                continue;
            }

            // Parse swap transaction data
            // Format: [pool_id:32][direction:1][min_amount_out:8]
            if tx.data.len() < 41 {
                warn!(
                    "⚠️ [SWAP INDEXER] Invalid swap data length {} for tx {}",
                    tx.data.len(),
                    hex::encode(&tx.id[..8])
                );
                continue;
            }

            let mut pool_id = [0u8; 32];
            pool_id.copy_from_slice(&tx.data[0..32]);
            let direction = tx.data[32];

            // Get token addresses from pool_id
            // For now, we use the token_type to determine input token
            let token_in = tx.token_type.address();

            // Output token is the "other" token in the pair
            // We derive this from the pool_id and direction
            // For simplicity, we store the pool's token addresses
            // In production, we would look up the pool to get exact tokens
            let token_out = if direction == 0 {
                // a -> b: output is token_b
                derive_output_token(&pool_id, &token_in)
            } else {
                // b -> a: output is token_a
                derive_output_token(&pool_id, &token_in)
            };

            // The amount_out would ideally come from StateChange::BalanceCredit
            // For now, we estimate it from the transaction data
            // In production, this should be extracted from the actual state changes
            let min_amount_out = u64::from_be_bytes([
                tx.data[33], tx.data[34], tx.data[35], tx.data[36],
                tx.data[37], tx.data[38], tx.data[39], tx.data[40],
            ]);

            // Create the consensus swap record
            let record = ConsensusSwapRecord {
                tx_id: tx.id,
                block_height,
                block_hash,
                timestamp: block_timestamp,
                token_in,
                token_out,
                amount_in: tx.amount as u64, // Cast u128 to u64 for swap record
                amount_out: min_amount_out, // Will be updated with actual from StateChange
                wallet: tx.from,
                pool_id,
                direction,
            };

            // Store in swap history
            if let Err(e) = self.store_swap_record(&record).await {
                warn!(
                    "⚠️ [SWAP INDEXER] Failed to store swap record {}: {}",
                    hex::encode(&tx.id[..8]),
                    e
                );
                continue;
            }

            indexed_count += 1;
            debug!(
                "📝 [SWAP INDEXER] Indexed swap {} at block {}",
                hex::encode(&tx.id[..8]),
                block_height
            );
        }

        if indexed_count > 0 {
            info!(
                "📊 [SWAP INDEXER] Block {}: indexed {} swap(s)",
                block_height, indexed_count
            );
        }

        Ok(indexed_count)
    }

    /// Store a swap record in CF_SWAP_HISTORY
    async fn store_swap_record(&self, record: &ConsensusSwapRecord) -> Result<(), String> {
        // Key format: [token_address:32][timestamp:8][tx_id:8]
        // This allows efficient prefix scans for a specific token

        // Store for input token (sell)
        let key_in = build_swap_key(&record.token_in, record.timestamp, &record.tx_id);
        let value = bincode::serialize(record)
            .map_err(|e| format!("Failed to serialize swap record: {}", e))?;

        self.storage.save_consensus_swap(&key_in, &value).await
            .map_err(|e| format!("Failed to save swap history (in): {}", e))?;

        // Store for output token (buy)
        let key_out = build_swap_key(&record.token_out, record.timestamp, &record.tx_id);
        self.storage.save_consensus_swap(&key_out, &value).await
            .map_err(|e| format!("Failed to save swap history (out): {}", e))?;

        Ok(())
    }

    /// Query swap history for a specific token
    pub async fn get_token_history(
        &self,
        token_address: &[u8; 32],
        limit: usize,
    ) -> Result<Vec<ConsensusSwapRecord>, String> {
        self.storage.load_swap_history_for_token::<ConsensusSwapRecord>(token_address, limit).await
            .map_err(|e| format!("Failed to load swap history: {}", e))
    }
}

/// Build a swap history key
fn build_swap_key(token: &[u8; 32], timestamp: i64, tx_id: &[u8; 32]) -> Vec<u8> {
    let mut key = Vec::with_capacity(48);
    key.extend_from_slice(token);
    // Use inverted timestamp for reverse chronological order
    let inverted_ts = i64::MAX - timestamp;
    key.extend_from_slice(&inverted_ts.to_be_bytes());
    key.extend_from_slice(&tx_id[..8]);
    key
}

/// Derive output token from pool_id and input token
/// This is a placeholder - in production, we would look up the pool state
fn derive_output_token(pool_id: &[u8; 32], _token_in: &[u8; 32]) -> [u8; 32] {
    // For now, return a hash-derived placeholder
    // The actual implementation should look up the pool's token pair
    let mut hasher = Sha3_256::new();
    hasher.update(pool_id);
    hasher.update(b"output_token");
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_key_ordering() {
        let token = [1u8; 32];
        let tx1 = [1u8; 32];
        let tx2 = [2u8; 32];

        let key1 = build_swap_key(&token, 1000, &tx1);
        let key2 = build_swap_key(&token, 2000, &tx2);

        // Key2 (newer) should come before Key1 (older) due to inverted timestamp
        assert!(key2 < key1);
    }

    #[test]
    fn test_consensus_swap_record_to_json() {
        let record = ConsensusSwapRecord {
            tx_id: [1u8; 32],
            block_height: 100,
            block_hash: [2u8; 32],
            timestamp: 1234567890,
            token_in: [0u8; 32], // QUG
            token_out: [3u8; 32],
            amount_in: 100_000_000, // 1 QUG
            amount_out: 42_000_000, // 0.42 token
            wallet: [4u8; 32],
            pool_id: [5u8; 32],
            direction: 0,
        };

        let json = record.to_api_json();
        assert_eq!(json["amountIn"], 1.0);
        assert_eq!(json["amountOut"], 0.42);
        assert_eq!(json["exchangeRate"], 0.42);
        assert_eq!(json["type"], "sell");
    }
}
