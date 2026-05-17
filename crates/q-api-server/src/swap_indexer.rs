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
/// v3.4.19-beta: Uses u128 for amounts to support 24-decimal precision without overflow
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
    /// Input amount in base units (u128 for 24-decimal precision)
    pub amount_in: u128,
    /// Output amount in base units (u128 for 24-decimal precision)
    pub amount_out: u128,
    /// Wallet that performed the swap
    pub wallet: [u8; 32],
    /// Pool ID used for the swap
    pub pool_id: [u8; 32],
    /// Swap direction (0 = token_a -> token_b, 1 = token_b -> token_a)
    pub direction: u8,
}

impl ConsensusSwapRecord {
    /// Convert to JSON for API response
    /// v3.4.19-beta: Uses string representation to preserve full 24-decimal precision
    pub fn to_api_json(&self) -> serde_json::Value {
        // Use string representation to avoid f64 precision loss
        // Frontend can parse these with BigInt/BigNumber libraries
        let amount_in_raw = self.amount_in.to_string();
        let amount_out_raw = self.amount_out.to_string();

        // For display purposes, calculate a human-readable version
        // Split into whole and fractional parts to preserve precision
        let (amount_in_whole, amount_in_frac) = split_amount_for_display(self.amount_in);
        let (amount_out_whole, amount_out_frac) = split_amount_for_display(self.amount_out);

        // Exchange rate with high precision (use u128 math, then convert)
        // rate = amount_out * 1e18 / amount_in (18 decimals for rate precision)
        let exchange_rate_scaled = if self.amount_in > 0 {
            // Scale by 1e18 for precision, then divide
            let scaled_out = self.amount_out.saturating_mul(1_000_000_000_000_000_000u128);
            scaled_out / self.amount_in
        } else {
            0u128
        };
        let exchange_rate = exchange_rate_scaled as f64 / 1e18;

        serde_json::json!({
            "id": format!("0x{}", hex::encode(self.tx_id)),
            "timestamp": self.timestamp,
            "blockHeight": self.block_height,
            "blockHash": format!("0x{}", hex::encode(self.block_hash)),
            "type": if self.direction == 0 { "sell" } else { "buy" },
            "tokenIn": format!("0x{}", hex::encode(self.token_in)),
            "tokenOut": format!("0x{}", hex::encode(self.token_out)),
            // Raw amounts as strings for precision (frontend uses BigInt)
            "amountInRaw": amount_in_raw,
            "amountOutRaw": amount_out_raw,
            // Human-readable display format
            "amountIn": format!("{}.{}", amount_in_whole, amount_in_frac),
            "amountOut": format!("{}.{}", amount_out_whole, amount_out_frac),
            "wallet": format!("qnk{}", hex::encode(self.wallet)),
            "poolId": format!("0x{}", hex::encode(self.pool_id)),
            "exchangeRate": exchange_rate,
            "exchangeRateRaw": exchange_rate_scaled.to_string(),
            "txHash": format!("0x{}", hex::encode(self.tx_id)),
            "decimals": 24,  // Inform frontend of decimal precision
        })
    }
}

/// Split a u128 amount into whole and fractional parts for display (24 decimals)
/// Returns (whole_part_string, fractional_part_string with leading zeros)
fn split_amount_for_display(amount: u128) -> (String, String) {
    const DECIMALS: u128 = 1_000_000_000_000_000_000_000_000u128; // 10^24
    let whole = amount / DECIMALS;
    let frac = amount % DECIMALS;
    // Format fractional part with leading zeros, then trim trailing zeros
    let frac_str = format!("{:024}", frac);
    let frac_trimmed = frac_str.trim_end_matches('0');
    let frac_final = if frac_trimmed.is_empty() { "0" } else { frac_trimmed };
    (whole.to_string(), frac_final.to_string())
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
            // v3.4.19-beta: Updated format to support u128 amounts (16 bytes instead of 8)
            // Format: [pool_id:32][direction:1][min_amount_out:16]
            // Legacy format (41 bytes) still supported for backward compatibility
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
            // v3.4.19-beta: Support both legacy u64 (8 bytes) and new u128 (16 bytes) format
            let min_amount_out: u128 = if tx.data.len() >= 49 {
                // New format: 16 bytes for u128
                u128::from_be_bytes([
                    tx.data[33], tx.data[34], tx.data[35], tx.data[36],
                    tx.data[37], tx.data[38], tx.data[39], tx.data[40],
                    tx.data[41], tx.data[42], tx.data[43], tx.data[44],
                    tx.data[45], tx.data[46], tx.data[47], tx.data[48],
                ])
            } else {
                // Legacy format: 8 bytes for u64, extend to u128
                u64::from_be_bytes([
                    tx.data[33], tx.data[34], tx.data[35], tx.data[36],
                    tx.data[37], tx.data[38], tx.data[39], tx.data[40],
                ]) as u128
            };

            // Create the consensus swap record
            // v3.4.19-beta: No more u64 cast - tx.amount is u128, record uses u128
            let record = ConsensusSwapRecord {
                tx_id: tx.id,
                block_height,
                block_hash,
                timestamp: block_timestamp,
                token_in,
                token_out,
                amount_in: tx.amount, // u128 - full 24-decimal precision preserved
                amount_out: min_amount_out, // u128 - will be updated with actual from StateChange
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

    /// Store a swap record in CF_SWAP_HISTORY and CF_WALLET_SWAP_INDEX
    /// v3.5.8-beta: Also indexes by wallet address for decentralized transaction history
    async fn store_swap_record(&self, record: &ConsensusSwapRecord) -> Result<(), String> {
        // Key format: [token_address:32][timestamp:8][tx_id:8]
        // This allows efficient prefix scans for a specific token

        let value = bincode::serialize(record)
            .map_err(|e| format!("Failed to serialize swap record: {}", e))?;

        // Store for input token (sell)
        let key_in = build_swap_key(&record.token_in, record.timestamp, &record.tx_id);
        self.storage.save_consensus_swap(&key_in, &value).await
            .map_err(|e| format!("Failed to save swap history (in): {}", e))?;

        // Store for output token (buy)
        let key_out = build_swap_key(&record.token_out, record.timestamp, &record.tx_id);
        self.storage.save_consensus_swap(&key_out, &value).await
            .map_err(|e| format!("Failed to save swap history (out): {}", e))?;

        // v3.5.8-beta: Index by wallet address for wallet transaction history
        // This enables efficient lookup of all swaps for a specific wallet
        self.storage.save_wallet_swap_index(
            &record.wallet,
            record.timestamp,
            &record.tx_id,
            &value,
        ).await.map_err(|e| format!("Failed to save wallet swap index: {}", e))?;

        debug!(
            "📝 [SWAP INDEXER v3.5.8] Indexed swap {} for wallet {} (token {} -> {})",
            hex::encode(&record.tx_id[..8]),
            hex::encode(&record.wallet[..8]),
            hex::encode(&record.token_in[..8]),
            hex::encode(&record.token_out[..8]),
        );

        Ok(())
    }

    /// Query swap history for a specific wallet
    /// v3.5.8-beta: Returns all DEX swaps for a wallet address
    pub async fn get_wallet_swap_history(
        &self,
        wallet: &[u8; 32],
        limit: usize,
    ) -> Result<Vec<ConsensusSwapRecord>, String> {
        let swap_data = self.storage.load_swaps_for_wallet(wallet, limit).await
            .map_err(|e| format!("Failed to load wallet swap history: {}", e))?;

        let mut swaps = Vec::new();
        for data in swap_data {
            if let Ok(record) = bincode::deserialize::<ConsensusSwapRecord>(&data) {
                swaps.push(record);
            }
        }

        info!(
            "🔄 [SWAP INDEXER v3.5.8] Loaded {} swaps for wallet {}",
            swaps.len(),
            hex::encode(&wallet[..8])
        );

        Ok(swaps)
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
        // v3.4.19-beta: Use u128 with 24 decimals
        // 1 QUG = 1_000_000_000_000_000_000_000_000 (10^24)
        const ONE_TOKEN: u128 = 1_000_000_000_000_000_000_000_000u128;

        let record = ConsensusSwapRecord {
            tx_id: [1u8; 32],
            block_height: 100,
            block_hash: [2u8; 32],
            timestamp: 1234567890,
            token_in: [0u8; 32], // QUG
            token_out: [3u8; 32],
            amount_in: ONE_TOKEN,                // 1.0 QUG
            amount_out: ONE_TOKEN * 42 / 100,    // 0.42 token
            wallet: [4u8; 32],
            pool_id: [5u8; 32],
            direction: 0,
        };

        let json = record.to_api_json();

        // Check raw amounts are preserved as strings
        assert_eq!(json["amountInRaw"], ONE_TOKEN.to_string());
        assert_eq!(json["amountOutRaw"], (ONE_TOKEN * 42 / 100).to_string());

        // Check display format
        assert_eq!(json["amountIn"], "1.0");
        assert_eq!(json["amountOut"], "0.42");

        // Check exchange rate (0.42)
        let rate: f64 = json["exchangeRate"].as_f64().unwrap();
        assert!((rate - 0.42).abs() < 0.001, "Exchange rate {} should be ~0.42", rate);

        // Check type
        assert_eq!(json["type"], "sell");

        // Check decimals field
        assert_eq!(json["decimals"], 24);
    }

    #[test]
    fn test_split_amount_for_display() {
        const ONE_TOKEN: u128 = 1_000_000_000_000_000_000_000_000u128;

        // Test 1.0
        let (whole, frac) = split_amount_for_display(ONE_TOKEN);
        assert_eq!(whole, "1");
        assert_eq!(frac, "0");

        // Test 0.5
        let (whole, frac) = split_amount_for_display(ONE_TOKEN / 2);
        assert_eq!(whole, "0");
        assert_eq!(frac, "5");

        // Test 123.456789
        let amount = 123 * ONE_TOKEN + 456789 * ONE_TOKEN / 1_000_000;
        let (whole, frac) = split_amount_for_display(amount);
        assert_eq!(whole, "123");
        assert!(frac.starts_with("456789"), "frac {} should start with 456789", frac);

        // Test 0.000001 (very small)
        let (whole, frac) = split_amount_for_display(ONE_TOKEN / 1_000_000);
        assert_eq!(whole, "0");
        assert!(frac.len() > 0, "fractional part should exist");
    }

    #[test]
    fn test_u128_max_amount() {
        // Verify we can handle very large amounts without overflow
        let large_amount: u128 = 340_000_000_000_000_000_000_000_000_000_000_000_000u128; // ~340 trillion QUG
        let (whole, _frac) = split_amount_for_display(large_amount);
        assert!(whole.parse::<u128>().is_ok(), "Should parse whole part");
    }
}
