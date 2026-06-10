//! Oracle integration for QNK-INDEX
//!
//! Provides price feeds with multi-source aggregation, TWAP calculation,
//! and staleness detection.

use crate::types::*;
use anyhow::Result;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Price oracle manager
pub struct PriceOracle {
    /// Cached price feeds
    feeds: DashMap<[u8; 32], PriceFeed>,

    /// Historical prices for TWAP: token -> Vec<(block, price)>
    price_history: DashMap<[u8; 32], Vec<(u64, u64)>>,

    /// Maximum price age in blocks before considered stale
    max_staleness_blocks: u64,

    /// TWAP window in blocks (e.g., 14400 = ~24h at 6s blocks)
    twap_window_blocks: u64,

    /// Current block height
    current_block: Arc<RwLock<u64>>,

    /// Minimum oracle sources required for confidence
    min_sources: u8,
}

impl PriceOracle {
    /// Create new oracle
    pub fn new() -> Self {
        Self {
            feeds: DashMap::new(),
            price_history: DashMap::new(),
            max_staleness_blocks: 100, // ~10 minutes
            twap_window_blocks: 14400, // ~24 hours
            current_block: Arc::new(RwLock::new(0)),
            min_sources: 2,
        }
    }

    /// Set current block height
    pub async fn set_block_height(&self, height: u64) {
        let mut block = self.current_block.write().await;
        *block = height;
    }

    /// Get current block height
    pub async fn get_block_height(&self) -> u64 {
        *self.current_block.read().await
    }

    /// Update price from a source
    pub async fn update_price(
        &self,
        token_address: [u8; 32],
        source_id: String,
        price: u64,
        weight: u8,
    ) -> Result<(), IndexError> {
        let current_block = self.get_block_height().await;

        let mut feed = self.feeds.entry(token_address).or_insert_with(|| {
            PriceFeed {
                token_address,
                current_price: 0,
                twap_24h: 0,
                last_update_block: 0,
                oracle_sources: Vec::new(),
                confidence: 0,
            }
        });

        // Update or add source
        let source = OracleSource {
            source_id: source_id.clone(),
            price,
            weight,
            last_update: current_block,
        };

        if let Some(existing) = feed.oracle_sources.iter_mut().find(|s| s.source_id == source_id) {
            *existing = source;
        } else {
            feed.oracle_sources.push(source);
        }

        // Recalculate aggregated price
        self.recalculate_price(&mut feed, current_block);

        // Update history for TWAP
        self.add_to_history(token_address, current_block, feed.current_price);

        // Calculate TWAP
        feed.twap_24h = self.calculate_twap(token_address, current_block);

        debug!(
            "Price updated for {}: {} QUG (TWAP: {} QUG, confidence: {}%)",
            hex::encode(&token_address[..8]),
            feed.current_price as f64 / 100_000_000.0,
            feed.twap_24h as f64 / 100_000_000.0,
            feed.confidence
        );

        Ok(())
    }

    /// Recalculate aggregated price from sources
    fn recalculate_price(&self, feed: &mut PriceFeed, current_block: u64) {
        // Filter out stale sources
        let active_sources: Vec<&OracleSource> = feed.oracle_sources.iter()
            .filter(|s| current_block.saturating_sub(s.last_update) < self.max_staleness_blocks)
            .collect();

        if active_sources.is_empty() {
            warn!("No active price sources for {}", hex::encode(&feed.token_address[..8]));
            feed.confidence = 0;
            return;
        }

        // Weighted average
        let total_weight: u64 = active_sources.iter().map(|s| s.weight as u64).sum();
        let weighted_sum: u128 = active_sources.iter()
            .map(|s| s.price as u128 * s.weight as u128)
            .sum();

        feed.current_price = (weighted_sum / total_weight as u128) as u64;
        feed.last_update_block = current_block;

        // Calculate confidence based on source count and consistency
        let source_count = active_sources.len();
        let avg_deviation = self.calculate_deviation(&active_sources, feed.current_price);

        // Confidence = base (sources) - deviation penalty
        let base_confidence = std::cmp::min(source_count as u8 * 25, 75);
        let deviation_penalty = (avg_deviation / 100) as u8; // 1% = 1 point
        feed.confidence = base_confidence.saturating_sub(deviation_penalty).saturating_add(25);
    }

    /// Calculate average deviation from mean
    fn calculate_deviation(&self, sources: &[&OracleSource], mean: u64) -> u64 {
        if sources.is_empty() || mean == 0 {
            return 0;
        }

        let total_deviation: u64 = sources.iter()
            .map(|s| {
                if s.price > mean {
                    (s.price - mean) * 10000 / mean
                } else {
                    (mean - s.price) * 10000 / mean
                }
            })
            .sum();

        total_deviation / sources.len() as u64
    }

    /// Add price to history
    fn add_to_history(&self, token: [u8; 32], block: u64, price: u64) {
        let mut history = self.price_history.entry(token).or_insert(Vec::new());

        // Remove old entries outside TWAP window
        let cutoff = block.saturating_sub(self.twap_window_blocks);
        history.retain(|(b, _)| *b > cutoff);

        // Add new entry
        history.push((block, price));
    }

    /// Calculate Time-Weighted Average Price
    fn calculate_twap(&self, token: [u8; 32], current_block: u64) -> u64 {
        let history = match self.price_history.get(&token) {
            Some(h) => h,
            None => return 0,
        };

        if history.is_empty() {
            return 0;
        }

        let cutoff = current_block.saturating_sub(self.twap_window_blocks);

        // Filter to TWAP window
        let relevant: Vec<(u64, u64)> = history.iter()
            .filter(|(b, _)| *b > cutoff)
            .cloned()
            .collect();

        if relevant.is_empty() {
            return history.last().map(|(_, p)| *p).unwrap_or(0);
        }

        // Time-weighted average
        let mut weighted_sum: u128 = 0;
        let mut total_weight: u128 = 0;

        for i in 0..relevant.len() {
            let (block, price) = relevant[i];
            let next_block = if i + 1 < relevant.len() {
                relevant[i + 1].0
            } else {
                current_block
            };

            let weight = next_block.saturating_sub(block) as u128;
            weighted_sum += price as u128 * weight;
            total_weight += weight;
        }

        if total_weight == 0 {
            relevant.last().map(|(_, p)| *p).unwrap_or(0)
        } else {
            (weighted_sum / total_weight) as u64
        }
    }

    /// Get current price feed
    pub fn get_price(&self, token: [u8; 32]) -> Option<PriceFeed> {
        self.feeds.get(&token).map(|f| f.clone())
    }

    /// Get price with staleness check
    pub async fn get_price_checked(&self, token: [u8; 32]) -> Result<PriceFeed, IndexError> {
        let current_block = self.get_block_height().await;

        let feed = self.feeds.get(&token)
            .ok_or(IndexError::OracleError("No price feed".into()))?;

        if current_block.saturating_sub(feed.last_update_block) > self.max_staleness_blocks {
            return Err(IndexError::StalePriceData);
        }

        if feed.confidence < 50 {
            warn!(
                "Low confidence price for {}: {}%",
                hex::encode(&token[..8]),
                feed.confidence
            );
        }

        Ok(feed.clone())
    }

    /// Batch update prices for multiple tokens
    pub async fn batch_update_prices(
        &self,
        updates: Vec<(
            [u8; 32],  // token
            String,    // source_id
            u64,       // price
            u8,        // weight
        )>,
    ) -> Result<usize, IndexError> {
        let mut success_count = 0;

        for (token, source_id, price, weight) in updates {
            if self.update_price(token, source_id, price, weight).await.is_ok() {
                success_count += 1;
            }
        }

        Ok(success_count)
    }

    /// Get all token prices
    pub fn get_all_prices(&self) -> Vec<PriceFeed> {
        self.feeds.iter().map(|f| f.value().clone()).collect()
    }

    /// Check if a token has fresh price data
    pub async fn is_price_fresh(&self, token: [u8; 32]) -> bool {
        let current_block = self.get_block_height().await;

        self.feeds.get(&token)
            .map(|f| current_block.saturating_sub(f.last_update_block) < self.max_staleness_blocks)
            .unwrap_or(false)
    }

    /// Get price volatility (in basis points)
    pub fn get_volatility(&self, token: [u8; 32], window_blocks: u64) -> Option<u64> {
        let history = self.price_history.get(&token)?;

        if history.len() < 2 {
            return Some(0);
        }

        // Get prices in window
        let current = history.last()?.0;
        let cutoff = current.saturating_sub(window_blocks);
        let prices: Vec<u64> = history.iter()
            .filter(|(b, _)| *b > cutoff)
            .map(|(_, p)| *p)
            .collect();

        if prices.len() < 2 {
            return Some(0);
        }

        // Calculate standard deviation as volatility
        let mean = prices.iter().sum::<u64>() / prices.len() as u64;
        if mean == 0 {
            return Some(0);
        }

        let variance: u64 = prices.iter()
            .map(|p| {
                let diff = if *p > mean { *p - mean } else { mean - *p };
                (diff as u128 * diff as u128 / mean as u128) as u64
            })
            .sum::<u64>() / prices.len() as u64;

        // Return as basis points
        Some((variance as f64).sqrt() as u64 * 10000 / mean)
    }
}

impl Default for PriceOracle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_price_update() {
        let oracle = PriceOracle::new();
        oracle.set_block_height(1000).await;

        let token = [1u8; 32];

        // Add first source
        oracle.update_price(token, "dex".into(), 150_000_000, 50).await.unwrap();

        let price = oracle.get_price(token).unwrap();
        assert_eq!(price.current_price, 150_000_000);
        assert!(price.confidence >= 25);

        // Add second source
        oracle.update_price(token, "oracle".into(), 151_000_000, 50).await.unwrap();

        let price = oracle.get_price(token).unwrap();
        // Should be average: (150 + 151) / 2 = 150.5
        assert!(price.current_price > 150_000_000);
        assert!(price.current_price < 151_000_000);
    }

    #[tokio::test]
    async fn test_twap_calculation() {
        let oracle = PriceOracle::new();

        let token = [2u8; 32];

        // Simulate price changes over time
        for i in 0..100 {
            oracle.set_block_height(i * 100).await;
            let price = 100_000_000 + (i as u64 * 1_000_000); // 1.00 -> 1.99
            oracle.update_price(token, "dex".into(), price, 100).await.unwrap();
        }

        let feed = oracle.get_price(token).unwrap();
        // TWAP should be somewhere in the middle
        assert!(feed.twap_24h > 100_000_000);
        assert!(feed.twap_24h < 200_000_000);
    }
}
