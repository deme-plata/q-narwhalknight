//! Rebalancing engine for QNK-INDEX
//!
//! Handles weight calculations, trade generation, and DEX execution
//! for maintaining target allocations.

use crate::oracle::PriceOracle;
use crate::types::*;
use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Maximum slippage allowed during rebalance (basis points)
pub const MAX_REBALANCE_SLIPPAGE_BPS: u16 = 200; // 2%

/// Minimum trade size to execute (in QUG, 8 decimals)
// v8.6.0: lowered from 10_000_000 (0.1 QUG) for finer rebalancing granularity
pub const MIN_TRADE_SIZE: u64 = 1_000_000; // 0.01 QUG

/// Rebalancing engine
pub struct Rebalancer {
    /// Price oracle reference
    oracle: Arc<PriceOracle>,

    /// Maximum trades per rebalance
    max_trades_per_rebalance: usize,

    /// Trade dust threshold (skip if below this %)
    dust_threshold_bps: u16,
}

impl Rebalancer {
    /// Create new rebalancer
    pub fn new(oracle: Arc<PriceOracle>) -> Self {
        Self {
            oracle,
            max_trades_per_rebalance: 50, // v8.6.0: raised from 20 for finer rebalancing
            dust_threshold_bps: 50, // 0.5%
        }
    }

    /// Check if rebalance is needed
    pub fn needs_rebalance(&self, index: &QnkIndex, current_block: u64) -> bool {
        if index.paused_rebalance {
            return false;
        }

        // Check time-based trigger
        let blocks_since_last = current_block.saturating_sub(index.last_rebalance_height);
        if blocks_since_last >= index.rebalance_interval {
            return true;
        }

        // Check weight drift trigger (>5% deviation)
        for component in &index.components {
            let drift = if component.actual_weight_bps > component.target_weight_bps {
                component.actual_weight_bps - component.target_weight_bps
            } else {
                component.target_weight_bps - component.actual_weight_bps
            };

            if drift > 500 { // 5% deviation triggers rebalance
                return true;
            }
        }

        false
    }

    /// Calculate target weights based on methodology
    pub async fn calculate_target_weights(
        &self,
        index: &QnkIndex,
    ) -> Result<Vec<([u8; 32], u16)>, IndexError> {
        match &index.methodology {
            IndexMethodology::MarketCapWeighted { max_component_weight, min_component_weight } => {
                self.calculate_market_cap_weights(
                    &index.components,
                    *max_component_weight,
                    *min_component_weight,
                ).await
            }
            IndexMethodology::EqualWeight { components_count } => {
                self.calculate_equal_weights(&index.components, *components_count)
            }
            IndexMethodology::CustomWeights { weights } => {
                self.apply_custom_weights(&index.components, weights)
            }
            IndexMethodology::RiskAdjusted { volatility_window, max_volatility_bps } => {
                self.calculate_risk_adjusted_weights(
                    &index.components,
                    *volatility_window,
                    *max_volatility_bps,
                ).await
            }
        }
    }

    /// Calculate market-cap weighted allocations
    async fn calculate_market_cap_weights(
        &self,
        components: &[IndexComponent],
        max_weight: u16,
        min_weight: u16,
    ) -> Result<Vec<([u8; 32], u16)>, IndexError> {
        if components.is_empty() {
            return Ok(Vec::new());
        }

        // Get market caps (price * estimated supply)
        // For simplicity, use price as proxy for market cap ranking
        let mut market_caps: Vec<([u8; 32], u64)> = Vec::new();

        for comp in components {
            let price = self.oracle.get_price(comp.token_address)
                .map(|f| f.current_price)
                .unwrap_or(comp.price_qug);

            // Estimate market cap based on price tier
            let estimated_mcap = price * 1_000_000; // Rough estimate
            market_caps.push((comp.token_address, estimated_mcap));
        }

        let total_mcap: u128 = market_caps.iter().map(|(_, m)| *m as u128).sum();

        if total_mcap == 0 {
            // Fall back to equal weight
            let equal_weight = 10000 / components.len() as u16;
            return Ok(components.iter()
                .map(|c| (c.token_address, equal_weight))
                .collect());
        }

        // Calculate raw weights
        let mut weights: Vec<([u8; 32], u16)> = market_caps.iter()
            .map(|(addr, mcap)| {
                let raw_weight = (*mcap as u128 * 10000 / total_mcap) as u16;
                (*addr, raw_weight)
            })
            .collect();

        // Apply caps and floors
        let mut total_capped: u16 = 0;
        let mut uncapped_count: usize = 0;

        for (_, weight) in &mut weights {
            if *weight > max_weight {
                total_capped += *weight - max_weight;
                *weight = max_weight;
            } else if *weight < min_weight {
                total_capped += min_weight - *weight;
                *weight = min_weight;
            } else {
                uncapped_count += 1;
            }
        }

        // Redistribute capped excess
        if uncapped_count > 0 && total_capped > 0 {
            let redistribution = total_capped / uncapped_count as u16;
            for (_, weight) in &mut weights {
                if *weight < max_weight && *weight > min_weight {
                    *weight += redistribution;
                    if *weight > max_weight {
                        *weight = max_weight;
                    }
                }
            }
        }

        // Normalize to ensure sum = 10000
        let sum: u16 = weights.iter().map(|(_, w)| *w).sum();
        if sum != 10000 && sum > 0 {
            let adjustment = (10000 as i32 - sum as i32) / weights.len() as i32;
            for (_, weight) in &mut weights {
                *weight = (*weight as i32 + adjustment).max(0) as u16;
            }
        }

        Ok(weights)
    }

    /// Calculate equal weights
    fn calculate_equal_weights(
        &self,
        components: &[IndexComponent],
        _components_count: u8,
    ) -> Result<Vec<([u8; 32], u16)>, IndexError> {
        if components.is_empty() {
            return Ok(Vec::new());
        }

        let weight_each = 10000 / components.len() as u16;
        let remainder = 10000 % components.len() as u16;

        Ok(components.iter().enumerate()
            .map(|(i, c)| {
                let extra = if i < remainder as usize { 1 } else { 0 };
                (c.token_address, weight_each + extra)
            })
            .collect())
    }

    /// Apply custom weights
    fn apply_custom_weights(
        &self,
        components: &[IndexComponent],
        weights: &[u16],
    ) -> Result<Vec<([u8; 32], u16)>, IndexError> {
        if components.len() != weights.len() {
            return Err(IndexError::InvalidWeight);
        }

        let total: u16 = weights.iter().sum();
        if total != 10000 {
            return Err(IndexError::WeightsNot100Percent);
        }

        Ok(components.iter()
            .zip(weights.iter())
            .map(|(c, w)| (c.token_address, *w))
            .collect())
    }

    /// Calculate risk-adjusted (inverse volatility) weights
    async fn calculate_risk_adjusted_weights(
        &self,
        components: &[IndexComponent],
        volatility_window: u64,
        max_vol_bps: u16,
    ) -> Result<Vec<([u8; 32], u16)>, IndexError> {
        if components.is_empty() {
            return Ok(Vec::new());
        }

        // Get inverse volatility for each component
        let mut inverse_vols: Vec<([u8; 32], u64)> = Vec::new();

        for comp in components {
            let vol = self.oracle.get_volatility(comp.token_address, volatility_window)
                .unwrap_or(1000); // Default 10% vol

            // Skip if volatility too high
            if vol > max_vol_bps as u64 {
                continue;
            }

            // Inverse volatility (higher = less volatile = higher weight)
            let inverse_vol = if vol > 0 { 10000 / vol } else { 100 };
            inverse_vols.push((comp.token_address, inverse_vol));
        }

        if inverse_vols.is_empty() {
            // All too volatile, fall back to equal
            return self.calculate_equal_weights(components, components.len() as u8);
        }

        let total_inv_vol: u64 = inverse_vols.iter().map(|(_, v)| *v).sum();

        Ok(inverse_vols.iter()
            .map(|(addr, inv_vol)| {
                let weight = (*inv_vol * 10000 / total_inv_vol) as u16;
                (*addr, weight)
            })
            .collect())
    }

    /// Generate rebalance trades
    pub async fn generate_trades(
        &self,
        index: &QnkIndex,
        target_weights: &[([u8; 32], u16)],
    ) -> Result<Vec<RebalanceTrade>, IndexError> {
        let mut trades = Vec::new();

        // Calculate total portfolio value
        let total_value = self.calculate_portfolio_value(index).await?;

        if total_value == 0 {
            return Ok(trades);
        }

        // For each component, calculate trade needed
        for (token_addr, target_weight) in target_weights {
            let component = match index.components.iter().find(|c| c.token_address == *token_addr) {
                Some(c) => c,
                None => continue,
            };

            let price = self.oracle.get_price(*token_addr)
                .map(|f| f.current_price)
                .unwrap_or(component.price_qug);

            if price == 0 {
                continue;
            }

            // Current value of this component
            let current_value = component.holdings as u128 * price as u128 / 100_000_000;

            // Target value based on weight
            let target_value = total_value as u128 * *target_weight as u128 / 10000;

            // Calculate trade
            let trade_value = if target_value > current_value {
                // Need to buy
                let buy_value = (target_value - current_value) as u64;

                if buy_value < MIN_TRADE_SIZE {
                    continue; // Skip dust trades
                }

                let buy_amount = buy_value * 100_000_000 / price;

                RebalanceTrade {
                    token_address: *token_addr,
                    side: TradeSide::Buy,
                    amount: buy_amount,
                    price,
                    slippage_bps: 0, // Set during execution
                }
            } else {
                // Need to sell
                let sell_value = (current_value - target_value) as u64;

                if sell_value < MIN_TRADE_SIZE {
                    continue; // Skip dust trades
                }

                let sell_amount = sell_value * 100_000_000 / price;

                RebalanceTrade {
                    token_address: *token_addr,
                    side: TradeSide::Sell,
                    amount: sell_amount,
                    price,
                    slippage_bps: 0,
                }
            };

            trades.push(trade_value);

            if trades.len() >= self.max_trades_per_rebalance {
                break;
            }
        }

        // Sort by value (largest trades first for efficiency)
        trades.sort_by(|a, b| {
            let a_val = a.amount * a.price / 100_000_000;
            let b_val = b.amount * b.price / 100_000_000;
            b_val.cmp(&a_val)
        });

        info!(
            "Generated {} rebalance trades for index {}",
            trades.len(),
            hex::encode(&index.index_id[..8])
        );

        Ok(trades)
    }

    /// Calculate total portfolio value in QUG
    async fn calculate_portfolio_value(&self, index: &QnkIndex) -> Result<u64, IndexError> {
        let mut total: u128 = 0;

        for comp in &index.components {
            let price = self.oracle.get_price(comp.token_address)
                .map(|f| f.current_price)
                .unwrap_or(comp.price_qug);

            total += comp.holdings as u128 * price as u128 / 100_000_000;
        }

        Ok(total as u64)
    }

    /// Execute rebalance (returns updated components)
    pub async fn execute_rebalance(
        &self,
        index: &mut QnkIndex,
        trades: Vec<RebalanceTrade>,
        current_block: u64,
    ) -> Result<RebalanceResult, IndexError> {
        let mut executed_trades = Vec::new();
        let mut total_slippage: u64 = 0;

        for trade in trades {
            // Find component
            let component = match index.components.iter_mut()
                .find(|c| c.token_address == trade.token_address)
            {
                Some(c) => c,
                None => continue,
            };

            // Simulate execution with slippage
            let slippage_bps = self.estimate_slippage(trade.amount, &trade.side);

            if slippage_bps > MAX_REBALANCE_SLIPPAGE_BPS {
                warn!(
                    "Skipping trade for {} due to high slippage: {}bps",
                    hex::encode(&trade.token_address[..8]),
                    slippage_bps
                );
                continue;
            }

            // Update holdings
            match trade.side {
                TradeSide::Buy => {
                    component.holdings = component.holdings
                        .checked_add(trade.amount)
                        .ok_or(IndexError::ArithmeticOverflow)?;
                }
                TradeSide::Sell => {
                    component.holdings = component.holdings
                        .checked_sub(trade.amount)
                        .ok_or(IndexError::ArithmeticUnderflow)?;
                }
            }

            // Update price
            let latest_price = self.oracle.get_price(trade.token_address)
                .map(|f| f.current_price)
                .unwrap_or(trade.price);
            component.price_qug = latest_price;

            total_slippage += slippage_bps as u64;

            executed_trades.push(RebalanceTrade {
                slippage_bps,
                ..trade
            });
        }

        // Update actual weights
        self.update_actual_weights(index).await?;

        // Update rebalance timestamp
        index.last_rebalance_height = current_block;

        // Calculate new NAV
        let new_nav = self.calculate_nav_per_share(index).await?;
        index.nav_per_share = new_nav;

        // Update high water mark if needed
        if new_nav > index.high_water_mark {
            index.high_water_mark = new_nav;
        }

        let avg_slippage = if !executed_trades.is_empty() {
            (total_slippage / executed_trades.len() as u64) as u16
        } else {
            0
        };

        info!(
            "Executed rebalance for {}: {} trades, avg slippage {}bps, new NAV {}",
            hex::encode(&index.index_id[..8]),
            executed_trades.len(),
            avg_slippage,
            new_nav as f64 / 100_000_000.0
        );

        Ok(RebalanceResult {
            trades: executed_trades,
            total_slippage_bps: avg_slippage,
            execution_block: current_block,
            new_nav_per_share: new_nav,
        })
    }

    /// Estimate slippage based on trade size
    fn estimate_slippage(&self, amount: u64, side: &TradeSide) -> u16 {
        // Simple model: larger trades = more slippage
        let base_slippage = 10; // 0.1% base

        // Add size-based slippage
        let size_factor = (amount / 1_000_000_000) as u16; // Per 10 tokens
        let size_slippage = size_factor * 5; // 0.05% per 10 tokens

        // Buys typically have slightly more slippage
        let direction_factor = match side {
            TradeSide::Buy => 5,
            TradeSide::Sell => 0,
        };

        base_slippage + size_slippage + direction_factor
    }

    /// Update actual weights based on current holdings
    async fn update_actual_weights(&self, index: &mut QnkIndex) -> Result<(), IndexError> {
        let total_value = self.calculate_portfolio_value(index).await?;

        if total_value == 0 {
            for comp in &mut index.components {
                comp.actual_weight_bps = 0;
            }
            return Ok(());
        }

        for comp in &mut index.components {
            let price = self.oracle.get_price(comp.token_address)
                .map(|f| f.current_price)
                .unwrap_or(comp.price_qug);

            let value = comp.holdings as u128 * price as u128 / 100_000_000;
            comp.actual_weight_bps = (value * 10000 / total_value as u128) as u16;
        }

        Ok(())
    }

    /// Calculate NAV per share
    async fn calculate_nav_per_share(&self, index: &QnkIndex) -> Result<u64, IndexError> {
        if index.total_supply == 0 {
            return Ok(100_000_000); // 1.0 QUG
        }

        let total_value = self.calculate_portfolio_value(index).await?;
        Ok((total_value as u128 * 100_000_000 / index.total_supply as u128) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_equal_weights() {
        let oracle = Arc::new(PriceOracle::new());
        let rebalancer = Rebalancer::new(oracle);

        let components = vec![
            IndexComponent {
                token_address: [1u8; 32],
                symbol: "TOK1".into(),
                target_weight_bps: 0,
                actual_weight_bps: 0,
                holdings: 0,
                price_qug: 100_000_000,
                rank: 1,
            },
            IndexComponent {
                token_address: [2u8; 32],
                symbol: "TOK2".into(),
                target_weight_bps: 0,
                actual_weight_bps: 0,
                holdings: 0,
                price_qug: 200_000_000,
                rank: 2,
            },
        ];

        let weights = rebalancer.calculate_equal_weights(&components, 2).unwrap();
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].1, 5000); // 50%
        assert_eq!(weights[1].1, 5000); // 50%
    }

    #[tokio::test]
    async fn test_needs_rebalance() {
        let oracle = Arc::new(PriceOracle::new());
        let rebalancer = Rebalancer::new(oracle);

        let mut index = QnkIndex {
            index_id: [0u8; 32],
            name: "Test".into(),
            symbol: "TST".into(),
            total_supply: 1000,
            components: vec![],
            nav_per_share: 100_000_000,
            last_rebalance_height: 0,
            rebalance_interval: 1000,
            management_fee_bps: 100,
            performance_fee_bps: 0,
            min_market_cap: 0,
            max_components: 10,
            manager: [0u8; 32],
            governance_enabled: false,
            methodology: IndexMethodology::EqualWeight { components_count: 10 },
            creation_block: 0,
            total_fees_accrued: 0,
            high_water_mark: 100_000_000,
            paused_mint: false,
            paused_redeem: false,
            paused_rebalance: false,
            emergency_paused_at: None,
        };

        // Should need rebalance after interval
        assert!(rebalancer.needs_rebalance(&index, 1001));
        assert!(!rebalancer.needs_rebalance(&index, 500));

        // Paused should never need rebalance
        index.paused_rebalance = true;
        assert!(!rebalancer.needs_rebalance(&index, 10000));
    }
}
