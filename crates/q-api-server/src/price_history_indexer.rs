/// v3.7.1-beta: Consensus-Verified Price History Indexer
///
/// Computes prices from on-chain swap transactions and stores them
/// persistently in RocksDB. All nodes derive the same prices from
/// the same block data - achieving decentralized price consensus.
///
/// Key features:
/// - Prices derived from swap exchange rates (verifiable on-chain)
/// - Stored persistently in CF_PRICE_HISTORY
/// - All nodes compute identical prices from identical blocks
/// - Enables accurate 1h, 24h, 7d price change calculations

use q_storage::StorageEngine;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Price snapshot record (for API responses)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceSnapshot {
    pub token_address: String,
    pub timestamp_ms: i64,
    pub price_usd: f64,
    pub block_height: u64,
}

/// Price change data for a token
#[derive(Debug, Clone, Default)]
pub struct PriceChangeData {
    pub current_price: f64,
    pub price_1h_ago: Option<f64>,
    pub price_24h_ago: Option<f64>,
    pub price_7d_ago: Option<f64>,
}

impl PriceChangeData {
    /// Calculate percentage changes
    pub fn calculate_changes(&self) -> (f64, f64, f64) {
        let change_1h = self.price_1h_ago.map_or(0.0, |p| {
            if p > 0.0 { ((self.current_price - p) / p) * 100.0 } else { 0.0 }
        });
        let change_24h = self.price_24h_ago.map_or(0.0, |p| {
            if p > 0.0 { ((self.current_price - p) / p) * 100.0 } else { 0.0 }
        });
        let change_7d = self.price_7d_ago.map_or(0.0, |p| {
            if p > 0.0 { ((self.current_price - p) / p) * 100.0 } else { 0.0 }
        });
        (change_1h, change_24h, change_7d)
    }
}

/// Consensus-verified price history indexer
pub struct PriceHistoryIndexer {
    storage: Arc<StorageEngine>,
}

impl PriceHistoryIndexer {
    /// Create a new price history indexer
    pub fn new(storage: Arc<StorageEngine>) -> Self {
        info!("📈 [PRICE HISTORY v3.7.1] Initializing consensus-verified price history indexer");
        Self { storage }
    }

    /// Record a price snapshot derived from a swap transaction
    ///
    /// This computes the price from the swap's exchange rate and stores it.
    /// All nodes processing the same swap will compute the same price.
    pub async fn record_price_from_swap(
        &self,
        token_address: &[u8; 32],
        timestamp_ms: i64,
        amount_in: u128,
        amount_out: u128,
        token_in_decimals: u8,
        token_out_decimals: u8,
        pair_price_usd: f64,
        block_height: u64,
    ) -> Result<(), String> {
        // Convert amounts to display units
        let amount_in_display = amount_in as f64 / 10f64.powi(token_in_decimals as i32);
        let amount_out_display = amount_out as f64 / 10f64.powi(token_out_decimals as i32);

        // Skip invalid swaps
        if amount_in_display <= 0.0 || amount_out_display <= 0.0 {
            return Ok(());
        }

        // Compute price: (amount_out / amount_in) * pair_price
        // This gives us the price of token_in in USD
        let price_usd = (amount_out_display / amount_in_display) * pair_price_usd;

        // Sanity check - skip absurd prices
        if price_usd <= 0.0 || price_usd > 1e18 {
            warn!(
                "⚠️ [PRICE HISTORY] Skipping invalid price {} for token {}",
                price_usd,
                hex::encode(&token_address[..8])
            );
            return Ok(());
        }

        // Store the price snapshot
        self.storage
            .save_price_snapshot(token_address, timestamp_ms, price_usd, block_height)
            .await
            .map_err(|e| format!("Failed to save price snapshot: {}", e))?;

        debug!(
            "📈 [PRICE HISTORY] Recorded price ${:.6} for token {} at block {}",
            price_usd,
            hex::encode(&token_address[..8]),
            block_height
        );

        Ok(())
    }

    /// Get price change data for a token
    ///
    /// v6.1.0: When no snapshot exists at the exact time window, use the earliest
    /// available snapshot as fallback. This prevents showing 0% when there simply
    /// haven't been trades within the window - the last known price before the
    /// window is the correct reference point.
    pub async fn get_price_changes(
        &self,
        token_address: &[u8; 32],
        current_price: f64,
    ) -> (f64, f64, f64) {
        let now_ms = chrono::Utc::now().timestamp_millis();

        // Get prices at different time points
        let price_1h_ago = self.get_price_at_time(token_address, now_ms - 3_600_000).await;
        let price_24h_ago = self.get_price_at_time(token_address, now_ms - 86_400_000).await;
        let price_7d_ago = self.get_price_at_time(token_address, now_ms - 604_800_000).await;

        // v6.1.0: Fallback chain - if no data at requested time, use next available
        // This prevents showing 0% when the token simply hasn't traded in that window.
        // The percentage should reflect change from the last known price before that time.
        let price_1h = price_1h_ago.or(price_24h_ago).or(price_7d_ago);
        let price_24h = price_24h_ago.or(price_7d_ago);
        let price_7d = price_7d_ago;

        let data = PriceChangeData {
            current_price,
            price_1h_ago: price_1h,
            price_24h_ago: price_24h,
            price_7d_ago: price_7d,
        };

        data.calculate_changes()
    }

    /// Get price at a specific time (closest snapshot before that time)
    /// v3.7.3-beta: Made pub for use in oracle price endpoint
    pub async fn get_price_at_time(&self, token_address: &[u8; 32], timestamp_ms: i64) -> Option<f64> {
        match self.storage.get_price_at_time(token_address, timestamp_ms).await {
            Ok(Some((_ts, price))) => Some(price),
            Ok(None) => None,
            Err(e) => {
                debug!(
                    "⚠️ [PRICE HISTORY] Failed to get price at time: {}",
                    e
                );
                None
            }
        }
    }

    /// Get recent price history for a token (for charts)
    pub async fn get_price_history(
        &self,
        token_address: &[u8; 32],
        since_ms: i64,
        limit: usize,
    ) -> Vec<(i64, f64)> {
        match self.storage.load_price_history(token_address, since_ms, limit).await {
            Ok(history) => history.into_iter().map(|(ts, price, _)| (ts, price)).collect(),
            Err(e) => {
                warn!(
                    "⚠️ [PRICE HISTORY] Failed to load history: {}",
                    e
                );
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_change_calculation() {
        let data = PriceChangeData {
            current_price: 110.0,
            price_1h_ago: Some(100.0),
            price_24h_ago: Some(90.0),
            price_7d_ago: Some(80.0),
        };

        let (c1h, c24h, c7d) = data.calculate_changes();

        // 110 vs 100 = +10%
        assert!((c1h - 10.0).abs() < 0.01);
        // 110 vs 90 = +22.22%
        assert!((c24h - 22.22).abs() < 0.1);
        // 110 vs 80 = +37.5%
        assert!((c7d - 37.5).abs() < 0.01);
    }

    #[test]
    fn test_price_change_with_missing_data() {
        let data = PriceChangeData {
            current_price: 100.0,
            price_1h_ago: None,
            price_24h_ago: Some(100.0),
            price_7d_ago: None,
        };

        let (c1h, c24h, c7d) = data.calculate_changes();

        assert_eq!(c1h, 0.0); // No data = 0%
        assert_eq!(c24h, 0.0); // Same price = 0%
        assert_eq!(c7d, 0.0); // No data = 0%
    }

    #[test]
    fn test_negative_price_change() {
        let data = PriceChangeData {
            current_price: 80.0,
            price_1h_ago: Some(100.0),
            price_24h_ago: None,
            price_7d_ago: None,
        };

        let (c1h, _, _) = data.calculate_changes();

        // 80 vs 100 = -20%
        assert!((c1h - (-20.0)).abs() < 0.01);
    }
}
