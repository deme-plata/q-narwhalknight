//! Oracle-DEX Price Synchronization Bridge
//!
//! Bidirectional price synchronization between on-chain DEX trading
//! and off-chain oracle price feeds. This creates a unified price discovery
//! mechanism that combines market activity with external data sources.

use anyhow::Result;
use bigdecimal::BigDecimal;
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{QuantumDexManager, QuantumTradeResult};
use q_oracle::{QuantumOracle, QuantumOracleSubmission, QuantumFeedType};

/// Bridge between Oracle and DEX for price synchronization
pub struct OraclePriceBridge {
    dex_manager: Arc<QuantumDexManager>,
    oracle: Arc<QuantumOracle>,

    // Track which pairs have oracle feeds
    oracle_enabled_pairs: Arc<RwLock<Vec<String>>>,

    // Weighting for price aggregation
    on_chain_weight: f64,  // Weight for DEX prices
    oracle_weight: f64,     // Weight for Oracle prices
}

impl OraclePriceBridge {
    /// Create a new oracle-DEX price bridge
    pub fn new(
        dex_manager: Arc<QuantumDexManager>,
        oracle: Arc<QuantumOracle>,
        on_chain_weight: f64,
        oracle_weight: f64,
    ) -> Self {
        Self {
            dex_manager,
            oracle,
            oracle_enabled_pairs: Arc::new(RwLock::new(Vec::new())),
            on_chain_weight,
            oracle_weight,
        }
    }

    /// Initialize the price bridge
    pub async fn initialize(&self) -> Result<()> {
        info!("🌉 Initializing Oracle-DEX Price Bridge");
        info!("⚖️  On-chain weight: {}%, Oracle weight: {}%",
            self.on_chain_weight * 100.0,
            self.oracle_weight * 100.0
        );

        // Register default oracle-enabled pairs
        let mut enabled_pairs = self.oracle_enabled_pairs.write().await;
        enabled_pairs.push("ORB/USD".to_string());
        enabled_pairs.push("ORB/ORBUSD".to_string());

        // Start background synchronization tasks
        self.start_dex_to_oracle_sync().await?;
        self.start_oracle_to_dex_sync().await?;

        info!("✅ Oracle-DEX Price Bridge initialized");
        Ok(())
    }

    /// Handle a trade executed on the DEX - submit price to oracle
    pub async fn on_trade_executed(&self, trade: &QuantumTradeResult) -> Result<()> {
        debug!("💱 Processing trade for oracle submission: {} @ {}",
            trade.pair_id, trade.price
        );

        // Check if this pair has oracle integration
        if !self.is_oracle_enabled(&trade.pair_id).await {
            return Ok(()); // Skip oracle submission
        }

        // Calculate AMM price from the trade
        let amm_price = self.calculate_amm_price_from_trade(trade).await?;

        // Submit to oracle as on-chain data source
        let submission = QuantumOracleSubmission {
            submission_id: uuid::Uuid::new_v4().to_string(),
            oracle_id: "dex_on_chain".to_string(),
            feed_id: trade.pair_id.clone(),
            value: amm_price.clone(),
            timestamp: Utc::now(),
            round_id: 0, // TODO: Track actual round IDs
            quantum_signature: vec![0u8; 64], // Post-quantum signature placeholder
            wave_function_data: None,
            entangled_sources: vec!["on_chain_amm".to_string()],
            ai_confidence: 0.95, // High confidence for on-chain data
            uncertainty_bounds: {
                use std::str::FromStr;
                let uncertainty = BigDecimal::from_str("0.01").unwrap_or_else(|_| BigDecimal::from(0));
                let lower = &amm_price - &uncertainty;
                let upper = &amm_price + uncertainty;
                (lower, upper)
            },
            privacy_level: q_oracle::QuantumPrivacyLevel::Basic,
        };

        match self.oracle.submit_quantum_data(submission).await {
            Ok(result) => {
                debug!("✅ DEX price submitted to oracle: {} = {}",
                    trade.pair_id, amm_price
                );
                Ok(())
            }
            Err(e) => {
                warn!("⚠️  Failed to submit DEX price to oracle: {}", e);
                Ok(()) // Don't fail the trade if oracle submission fails
            }
        }
    }

    /// Handle oracle price update - update DEX display prices
    pub async fn on_oracle_price_update(&self, pair_id: &str, oracle_price: BigDecimal) -> Result<()> {
        debug!("📊 Processing oracle price update: {} = {}", pair_id, oracle_price);

        // Get current DEX price (from AMM reserves)
        let dex_price = self.get_current_dex_price(pair_id).await?;

        // Calculate weighted average price
        let combined_price = self.calculate_weighted_price(&dex_price, &oracle_price)?;

        // Update DEX display price (NOT the AMM formula, just for display)
        self.update_dex_display_price(pair_id, combined_price).await?;

        debug!("✅ Combined price updated for {}: DEX={}, Oracle={}", pair_id, dex_price, oracle_price);
        Ok(())
    }

    /// Calculate AMM price from a trade (using reserves if available)
    async fn calculate_amm_price_from_trade(&self, trade: &QuantumTradeResult) -> Result<BigDecimal> {
        // For now, use the trade execution price
        // In production, this would calculate from actual AMM reserves
        Ok(trade.price.clone())
    }

    /// Get current price from DEX AMM
    async fn get_current_dex_price(&self, pair_id: &str) -> Result<BigDecimal> {
        // Get from DEX token data (cached prices)
        let token_data = self.dex_manager.token_data.read().await;

        // Parse pair_id (e.g., "ORB/ORBUSD")
        let tokens: Vec<&str> = pair_id.split('/').collect();
        if tokens.len() == 2 {
            if let Some(token_info) = token_data.get(tokens[0]) {
                if let Some(price) = &token_info.price_usd {
                    return Ok(price.clone());
                }
            }
        }

        // Fallback: try to get from pair data
        let pair_data = self.dex_manager.pair_data.read().await;
        if let Some(pair_info) = pair_data.get(pair_id) {
            return Ok(pair_info.price.clone());
        }

        Ok(BigDecimal::from(0))
    }

    /// Calculate weighted average of DEX and Oracle prices
    fn calculate_weighted_price(&self, dex_price: &BigDecimal, oracle_price: &BigDecimal) -> Result<BigDecimal> {
        use std::str::FromStr;
        // Weighted average: (dex_price * weight_dex + oracle_price * weight_oracle)
        let weight_dex = BigDecimal::from_str(&self.on_chain_weight.to_string())?;
        let weight_oracle = BigDecimal::from_str(&self.oracle_weight.to_string())?;
        let weighted_dex = dex_price * weight_dex;
        let weighted_oracle = oracle_price * weight_oracle;

        Ok(&weighted_dex + &weighted_oracle)
    }

    /// Update DEX display price (for frontend display, not AMM calculations)
    async fn update_dex_display_price(&self, pair_id: &str, price: BigDecimal) -> Result<()> {
        // Parse pair to get base token symbol
        let tokens: Vec<&str> = pair_id.split('/').collect();
        if tokens.len() >= 1 {
            let base_symbol = tokens[0];

            // Update the cached token price
            self.dex_manager.update_quantum_price(
                base_symbol,
                price,
                true, // Collapsed state (observed price)
            ).await?;
        }

        Ok(())
    }

    /// Check if a pair has oracle integration enabled
    async fn is_oracle_enabled(&self, pair_id: &str) -> bool {
        let enabled_pairs = self.oracle_enabled_pairs.read().await;
        enabled_pairs.contains(&pair_id.to_string())
    }

    /// Enable oracle integration for a trading pair
    pub async fn enable_oracle_for_pair(&self, pair_id: String) -> Result<()> {
        let mut enabled_pairs = self.oracle_enabled_pairs.write().await;
        if !enabled_pairs.contains(&pair_id) {
            enabled_pairs.push(pair_id.clone());
            info!("🔌 Oracle integration enabled for pair: {}", pair_id);
        }
        Ok(())
    }

    /// Start background task: DEX → Oracle price submission
    async fn start_dex_to_oracle_sync(&self) -> Result<()> {
        // This is handled by on_trade_executed() which should be called after each trade
        // No background task needed
        Ok(())
    }

    /// Start background task: Oracle → DEX price updates
    async fn start_oracle_to_dex_sync(&self) -> Result<()> {
        let bridge = self.clone_arc_self();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));

            loop {
                interval.tick().await;

                // Get oracle-enabled pairs
                let pairs = bridge.oracle_enabled_pairs.read().await.clone();

                for pair_id in pairs {
                    // Get oracle price for this pair
                    if let Ok(oracle_price_data) = bridge.oracle.get_quantum_price(&pair_id).await {
                        // Update DEX with oracle price
                        if let Err(e) = bridge.on_oracle_price_update(&pair_id, oracle_price_data.price).await {
                            warn!("⚠️  Failed to update DEX with oracle price for {}: {}", pair_id, e);
                        }
                    }
                }
            }
        });

        info!("🔄 Oracle → DEX price sync task started");
        Ok(())
    }

    /// Clone self as Arc for spawning tasks
    fn clone_arc_self(&self) -> Arc<Self> {
        Arc::new(Self {
            dex_manager: self.dex_manager.clone(),
            oracle: self.oracle.clone(),
            oracle_enabled_pairs: self.oracle_enabled_pairs.clone(),
            on_chain_weight: self.on_chain_weight,
            oracle_weight: self.oracle_weight,
        })
    }
}

/// Helper function to create a default price bridge
pub fn create_default_bridge(
    dex_manager: Arc<QuantumDexManager>,
    oracle: Arc<QuantumOracle>,
) -> OraclePriceBridge {
    // 70% weight to on-chain DEX prices (actual market activity)
    // 30% weight to oracle prices (external data sources)
    OraclePriceBridge::new(dex_manager, oracle, 0.7, 0.3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use rocksdb::DB;
    use q_storage::price_history::PriceHistoryManager;
    use q_storage::token_registry::TokenRegistry;

    async fn create_test_dex_manager() -> Arc<QuantumDexManager> {
        let temp_dir = tempdir().unwrap();
        let db = Arc::new(DB::open_default(temp_dir.path()).unwrap());

        let token_registry = Arc::new(TokenRegistry::new(db.clone()));
        token_registry.initialize().await.unwrap();

        let price_history = Arc::new(PriceHistoryManager::new(db));
        price_history.initialize().await.unwrap();

        // Keep temp_dir alive
        std::mem::forget(temp_dir);

        Arc::new(QuantumDexManager::new(token_registry, price_history).unwrap())
    }

    #[tokio::test]
    async fn test_weighted_price_calculation() {
        let dex_price = BigDecimal::from(100);
        let oracle_price = BigDecimal::from(110);

        // Create test DEX manager
        let dex_manager = create_test_dex_manager().await;

        // Create test oracle
        let oracle = Arc::new(
            QuantumOracle::new([0u8; 32], q_types::Phase::Phase1, Default::default())
                .await
                .unwrap()
        );

        // 70% DEX, 30% Oracle
        let bridge = create_default_bridge(dex_manager, oracle);

        let weighted = bridge.calculate_weighted_price(&dex_price, &oracle_price).unwrap();

        // Expected: 100 * 0.7 + 110 * 0.3 = 70 + 33 = 103
        assert_eq!(weighted, BigDecimal::from(103));
    }
}
