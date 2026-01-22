//! QNK-INDEX: Decentralized Top Token Index Fund
//!
//! A smart contract implementing S&P 500-style index tokens for the Q-NarwhalKnight DEX.
//! Supports market-cap weighted, equal-weight, and custom weighting methodologies.

pub mod types;
pub mod oracle;
pub mod rebalancer;
pub mod governance;
pub mod fees;
pub mod security;

use crate::types::*;
use anyhow::{Context, Result};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Main Index Fund Contract
pub struct QnkIndexFund {
    /// All active index funds
    pub indices: DashMap<[u8; 32], QnkIndex>,

    /// Shareholder records: (index_id, wallet) -> shares
    pub shareholders: DashMap<([u8; 32], [u8; 32]), IndexShareHolder>,

    /// Price feeds cache
    pub price_feeds: DashMap<[u8; 32], PriceFeed>,

    /// Governance proposals
    pub proposals: DashMap<([u8; 32], u64), GovernanceProposal>,

    /// Rate limiting: (wallet, operation) -> last_block
    pub rate_limits: DashMap<([u8; 32], OperationType), u64>,

    /// Protocol fee treasury address
    pub fee_treasury: [u8; 32],

    /// Protocol fee in basis points (default 10 = 0.1%)
    pub protocol_fee_bps: u16,

    /// Minimum QUG deposit
    pub min_deposit_qug: u64,

    /// Current block height (updated externally)
    current_block: Arc<RwLock<u64>>,
}

impl QnkIndexFund {
    /// Create a new index fund contract
    pub fn new(fee_treasury: [u8; 32]) -> Self {
        Self {
            indices: DashMap::new(),
            shareholders: DashMap::new(),
            price_feeds: DashMap::new(),
            proposals: DashMap::new(),
            rate_limits: DashMap::new(),
            fee_treasury,
            protocol_fee_bps: 10, // 0.1%
            min_deposit_qug: 100_000_000, // 1 QUG minimum
            current_block: Arc::new(RwLock::new(0)),
        }
    }

    /// Update current block height
    pub async fn set_block_height(&self, height: u64) {
        let mut block = self.current_block.write().await;
        *block = height;
    }

    /// Get current block height
    pub async fn get_block_height(&self) -> u64 {
        *self.current_block.read().await
    }

    // ============================================
    // INDEX MANAGEMENT
    // ============================================

    /// Create a new index fund
    pub async fn create_index(
        &self,
        name: String,
        symbol: String,
        manager: [u8; 32],
        methodology: IndexMethodology,
        management_fee_bps: u16,
        min_market_cap: u64,
        max_components: u8,
        rebalance_interval: u64,
        governance_enabled: bool,
    ) -> Result<[u8; 32], IndexError> {
        // Validate inputs
        if name.is_empty() || name.len() > 64 {
            return Err(IndexError::InvalidInput("Name must be 1-64 characters".into()));
        }
        if symbol.is_empty() || symbol.len() > 10 {
            return Err(IndexError::InvalidInput("Symbol must be 1-10 characters".into()));
        }
        if management_fee_bps > 500 {
            return Err(IndexError::FeeTooHigh);
        }
        if max_components == 0 || max_components > 50 {
            return Err(IndexError::InvalidComponentCount);
        }

        Self::validate_methodology(&methodology)?;

        let current_height = self.get_block_height().await;

        // Generate deterministic index ID
        let mut hasher = blake3::Hasher::new();
        hasher.update(name.as_bytes());
        hasher.update(symbol.as_bytes());
        hasher.update(&manager);
        hasher.update(&current_height.to_be_bytes());
        let hash = hasher.finalize();
        let mut index_id = [0u8; 32];
        index_id.copy_from_slice(hash.as_bytes());

        // Create the index
        let index = QnkIndex {
            index_id,
            name: name.clone(),
            symbol: symbol.clone(),
            total_supply: 0,
            components: Vec::new(),
            nav_per_share: 100_000_000, // 1.0 QUG initial
            last_rebalance_height: current_height,
            rebalance_interval,
            management_fee_bps,
            performance_fee_bps: 0,
            min_market_cap,
            max_components,
            manager,
            governance_enabled,
            methodology,
            creation_block: current_height,
            total_fees_accrued: 0,
            high_water_mark: 100_000_000,
            paused_mint: false,
            paused_redeem: false,
            paused_rebalance: false,
            emergency_paused_at: None,
        };

        // Store index
        self.indices.insert(index_id, index);

        info!(
            "Created index fund: {} ({}) - ID: {}",
            name,
            symbol,
            hex::encode(&index_id[..8])
        );

        Ok(index_id)
    }

    /// Validate methodology parameters
    fn validate_methodology(methodology: &IndexMethodology) -> Result<(), IndexError> {
        match methodology {
            IndexMethodology::MarketCapWeighted { max_component_weight, min_component_weight } => {
                if *max_component_weight > 5000 {
                    return Err(IndexError::InvalidWeight);
                }
                if *min_component_weight > *max_component_weight {
                    return Err(IndexError::InvalidWeight);
                }
            }
            IndexMethodology::EqualWeight { components_count } => {
                if *components_count == 0 || *components_count > 50 {
                    return Err(IndexError::InvalidComponentCount);
                }
            }
            IndexMethodology::CustomWeights { weights } => {
                let total: u16 = weights.iter().sum();
                if total != 10000 {
                    return Err(IndexError::WeightsNot100Percent);
                }
            }
            IndexMethodology::RiskAdjusted { volatility_window, max_volatility_bps } => {
                if *volatility_window > 100_000 {
                    return Err(IndexError::InvalidParameter);
                }
                if *max_volatility_bps > 10_000 {
                    return Err(IndexError::InvalidParameter);
                }
            }
        }
        Ok(())
    }

    // ============================================
    // MINT & REDEEM
    // ============================================

    /// Mint index shares by depositing QUG
    pub async fn mint_shares(
        &self,
        index_id: [u8; 32],
        depositor: [u8; 32],
        qug_amount: u64,
        min_shares_out: u64,
    ) -> Result<MintResult, IndexError> {
        // Get index
        let mut index = self.indices.get_mut(&index_id)
            .ok_or(IndexError::IndexNotFound)?;

        // Check not paused
        if index.paused_mint {
            return Err(IndexError::ContractPaused);
        }

        // Check minimum deposit
        if qug_amount < self.min_deposit_qug {
            return Err(IndexError::InsufficientBalance);
        }

        // Rate limiting
        self.check_rate_limit(depositor, OperationType::Mint).await?;

        // Apply mint fee
        let (net_amount, fee) = fees::apply_mint_fee(qug_amount)?;

        // Calculate shares to mint based on NAV
        let shares_to_mint = if index.total_supply == 0 {
            // First deposit - 1:1 ratio
            net_amount
        } else {
            // Calculate based on current NAV
            (net_amount as u128 * index.total_supply as u128 /
             (index.nav_per_share as u128 * index.total_supply as u128 / 100_000_000)) as u64
        };

        // Slippage protection
        if shares_to_mint < min_shares_out {
            return Err(IndexError::SlippageExceeded);
        }

        // Update total supply
        index.total_supply = index.total_supply
            .checked_add(shares_to_mint)
            .ok_or(IndexError::ArithmeticOverflow)?;

        // Update shareholder record
        let key = (index_id, depositor);
        let mut holder = self.shareholders.entry(key).or_insert(IndexShareHolder {
            wallet: depositor,
            shares: 0,
            entry_nav: index.nav_per_share,
            entry_block: self.get_block_height().await,
        });
        holder.shares = holder.shares
            .checked_add(shares_to_mint)
            .ok_or(IndexError::ArithmeticOverflow)?;

        info!(
            "Minted {} shares for {} in index {} (deposited {} QUG)",
            shares_to_mint,
            hex::encode(&depositor[..8]),
            hex::encode(&index_id[..8]),
            qug_amount
        );

        Ok(MintResult {
            shares_minted: shares_to_mint,
            qug_deposited: qug_amount,
            fee_paid: fee,
            nav_per_share: index.nav_per_share,
        })
    }

    /// Redeem index shares for QUG
    pub async fn redeem_shares(
        &self,
        index_id: [u8; 32],
        redeemer: [u8; 32],
        shares_amount: u64,
        min_qug_out: u64,
    ) -> Result<RedeemResult, IndexError> {
        // Get index
        let mut index = self.indices.get_mut(&index_id)
            .ok_or(IndexError::IndexNotFound)?;

        // Check not paused
        if index.paused_redeem {
            return Err(IndexError::ContractPaused);
        }

        // Rate limiting
        self.check_rate_limit(redeemer, OperationType::Redeem).await?;

        // Check shareholder has enough shares
        let key = (index_id, redeemer);
        let mut holder = self.shareholders.get_mut(&key)
            .ok_or(IndexError::InsufficientShares)?;

        if holder.shares < shares_amount {
            return Err(IndexError::InsufficientShares);
        }

        // Calculate QUG to return based on NAV
        let qug_value = (shares_amount as u128 * index.nav_per_share as u128 / 100_000_000) as u64;

        // Apply redeem fee
        let (net_qug, fee) = fees::apply_redeem_fee(qug_value)?;

        // Slippage protection
        if net_qug < min_qug_out {
            return Err(IndexError::SlippageExceeded);
        }

        // Update shares
        holder.shares = holder.shares
            .checked_sub(shares_amount)
            .ok_or(IndexError::ArithmeticUnderflow)?;

        index.total_supply = index.total_supply
            .checked_sub(shares_amount)
            .ok_or(IndexError::ArithmeticUnderflow)?;

        info!(
            "Redeemed {} shares for {} QUG from index {}",
            shares_amount,
            net_qug,
            hex::encode(&index_id[..8])
        );

        Ok(RedeemResult {
            shares_redeemed: shares_amount,
            qug_returned: net_qug,
            fee_paid: fee,
            nav_per_share: index.nav_per_share,
        })
    }

    // ============================================
    // COMPONENT MANAGEMENT
    // ============================================

    /// Add a component to the index
    pub async fn add_component(
        &self,
        index_id: [u8; 32],
        caller: [u8; 32],
        token_address: [u8; 32],
        symbol: String,
        initial_weight_bps: u16,
    ) -> Result<(), IndexError> {
        let mut index = self.indices.get_mut(&index_id)
            .ok_or(IndexError::IndexNotFound)?;

        // Only manager can add components (unless via governance)
        if caller != index.manager {
            return Err(IndexError::Unauthorized);
        }

        // Check capacity
        if index.components.len() >= index.max_components as usize {
            return Err(IndexError::IndexFull);
        }

        // Check not already in index
        if index.components.iter().any(|c| c.token_address == token_address) {
            return Err(IndexError::ComponentAlreadyExists);
        }

        // Get current price
        let price = self.price_feeds.get(&token_address)
            .map(|p| p.current_price)
            .unwrap_or(100_000_000); // Default 1 QUG

        let component = IndexComponent {
            token_address,
            symbol,
            target_weight_bps: initial_weight_bps,
            actual_weight_bps: 0,
            holdings: 0,
            price_qug: price,
            rank: (index.components.len() + 1) as u8,
        };

        index.components.push(component);

        info!(
            "Added component {} to index {}",
            hex::encode(&token_address[..8]),
            hex::encode(&index_id[..8])
        );

        Ok(())
    }

    /// Remove a component from the index
    pub async fn remove_component(
        &self,
        index_id: [u8; 32],
        caller: [u8; 32],
        token_address: [u8; 32],
    ) -> Result<(), IndexError> {
        let mut index = self.indices.get_mut(&index_id)
            .ok_or(IndexError::IndexNotFound)?;

        // Only manager can remove components
        if caller != index.manager {
            return Err(IndexError::Unauthorized);
        }

        let initial_len = index.components.len();
        index.components.retain(|c| c.token_address != token_address);

        if index.components.len() == initial_len {
            return Err(IndexError::ComponentNotFound);
        }

        // Update ranks
        for (i, component) in index.components.iter_mut().enumerate() {
            component.rank = (i + 1) as u8;
        }

        info!(
            "Removed component {} from index {}",
            hex::encode(&token_address[..8]),
            hex::encode(&index_id[..8])
        );

        Ok(())
    }

    // ============================================
    // NAV CALCULATION
    // ============================================

    /// Calculate Net Asset Value of the index
    pub async fn calculate_nav(&self, index: &QnkIndex) -> Result<u64, IndexError> {
        let mut total_value: u128 = 0;

        for component in &index.components {
            let price = self.price_feeds.get(&component.token_address)
                .map(|p| p.current_price)
                .unwrap_or(component.price_qug);

            total_value += component.holdings as u128 * price as u128 / 100_000_000;
        }

        // Deduct accrued fees
        let (mgmt_fee, perf_fee) = self.calculate_accrued_fees(index).await?;
        let total_fees = mgmt_fee.saturating_add(perf_fee);

        let nav = total_value.saturating_sub(total_fees as u128) as u64;

        Ok(nav)
    }

    /// Calculate NAV per share
    pub async fn get_nav_per_share(&self, index_id: [u8; 32]) -> Result<u64, IndexError> {
        let index = self.indices.get(&index_id)
            .ok_or(IndexError::IndexNotFound)?;

        if index.total_supply == 0 {
            return Ok(100_000_000); // 1.0 QUG
        }

        let total_nav = self.calculate_nav(&index).await?;
        Ok((total_nav as u128 * 100_000_000 / index.total_supply as u128) as u64)
    }

    /// Calculate accrued fees
    async fn calculate_accrued_fees(&self, index: &QnkIndex) -> Result<(u64, u64), IndexError> {
        let current_height = self.get_block_height().await;
        let blocks_elapsed = current_height.saturating_sub(index.last_rebalance_height);

        if blocks_elapsed == 0 {
            return Ok((0, 0));
        }

        // Annual blocks (assuming 6-second blocks)
        let annual_blocks: u128 = 365 * 24 * 60 * 60 / 6;

        // Get current NAV
        let mut nav: u128 = 0;
        for component in &index.components {
            let price = self.price_feeds.get(&component.token_address)
                .map(|p| p.current_price)
                .unwrap_or(component.price_qug);
            nav += component.holdings as u128 * price as u128 / 100_000_000;
        }

        // Management fee
        let mgmt_fee = (nav * index.management_fee_bps as u128 * blocks_elapsed as u128 /
            (annual_blocks * 10000)) as u64;

        // Performance fee (if above high water mark)
        let current_nav_per_share = if index.total_supply > 0 {
            (nav * 100_000_000 / index.total_supply as u128) as u64
        } else {
            100_000_000
        };

        let perf_fee = if current_nav_per_share > index.high_water_mark && index.performance_fee_bps > 0 {
            let gain = current_nav_per_share.saturating_sub(index.high_water_mark);
            (gain as u128 * index.performance_fee_bps as u128 * index.total_supply as u128 /
                (10000 * 100_000_000)) as u64
        } else {
            0
        };

        Ok((mgmt_fee, perf_fee))
    }

    // ============================================
    // RATE LIMITING
    // ============================================

    async fn check_rate_limit(&self, caller: [u8; 32], operation: OperationType) -> Result<(), IndexError> {
        let current_height = self.get_block_height().await;
        let key = (caller, operation);

        if let Some(last_op) = self.rate_limits.get(&key) {
            let min_interval = match operation {
                OperationType::Mint => 5,
                OperationType::Redeem => 5,
                OperationType::Rebalance => 1,
            };

            if current_height < *last_op + min_interval {
                return Err(IndexError::RateLimited);
            }
        }

        self.rate_limits.insert(key, current_height);
        Ok(())
    }

    // ============================================
    // GETTERS
    // ============================================

    /// Get index info
    pub fn get_index(&self, index_id: [u8; 32]) -> Option<QnkIndex> {
        self.indices.get(&index_id).map(|i| i.clone())
    }

    /// Get all indices
    pub fn get_all_indices(&self) -> Vec<QnkIndex> {
        self.indices.iter().map(|i| i.value().clone()).collect()
    }

    /// Get shareholder info
    pub fn get_shareholder(&self, index_id: [u8; 32], wallet: [u8; 32]) -> Option<IndexShareHolder> {
        self.shareholders.get(&(index_id, wallet)).map(|h| h.clone())
    }

    /// Get all shareholders for an index
    pub fn get_index_shareholders(&self, index_id: [u8; 32]) -> Vec<IndexShareHolder> {
        self.shareholders.iter()
            .filter(|e| e.key().0 == index_id)
            .map(|e| e.value().clone())
            .collect()
    }
}

// Re-export key types
pub use types::*;
