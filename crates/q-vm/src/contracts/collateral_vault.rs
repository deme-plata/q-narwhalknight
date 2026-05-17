/// CollateralVault: QUGUSD Stablecoin Collateral Management
///
/// This smart contract implements an over-collateralized algorithmic stablecoin (QUGUSD)
/// backed by QUG tokens at a 150% collateralization ratio.
///
/// Key Features:
/// - Mint QUGUSD by locking QUG as collateral (135% ratio)
/// - Redeem QUG by burning QUGUSD
/// - Liquidate undercollateralized positions (< 115%)
/// - Oracle-based QUG/USD price feeds

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Collateralization ratio constants
pub const MIN_COLLATERAL_RATIO: f64 = 1.35; // v8.6.0: 135% minimum (was 150%, more capital efficient)
pub const WARNING_RATIO: f64 = 1.18; // v8.6.0: 118% warning threshold (was 120%, tighter warning)
pub const LIQUIDATION_RATIO: f64 = 1.15; // v8.6.0: 115% liquidation threshold (was 110%, liquidate earlier for safety)
pub const LIQUIDATION_BONUS: f64 = 0.08; // v8.6.0: 8% bonus for liquidators (was 5%, faster liquidation incentive)

/// Base units divisor for 24-decimal precision (v3.0.4: migrated from 1e8)
pub const BASE_UNITS_DIVISOR: f64 = 1e24;

/// Collateral vault for QUGUSD stablecoin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollateralVault {
    /// User address -> Locked QUG amount (in base units, 24 decimals)
    pub locked_qug: HashMap<[u8; 32], u128>,

    /// User address -> Minted QUGUSD amount (in base units, 24 decimals)
    pub minted_qugusd: HashMap<[u8; 32], u128>,

    /// Current QUG price in USD (from oracle)
    pub qug_price_usd: f64,

    /// Total QUG locked in vault
    pub total_qug_locked: u128,

    /// Total QUGUSD minted
    pub total_qugusd_minted: u128,

    /// Last oracle update timestamp
    pub last_price_update: i64,
}

/// Result of a mint operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MintResult {
    pub qug_locked: u128,
    pub qugusd_minted: u128,
    pub collateral_ratio: f64,
    pub liquidation_price: f64,
}

/// Result of a redeem operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedeemResult {
    pub qugusd_burned: u128,
    pub qug_unlocked: u128,
    pub remaining_collateral_ratio: f64,
}

/// Result of a liquidation operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidationResult {
    pub liquidator: [u8; 32],
    pub liquidated_user: [u8; 32],
    pub qug_seized: u128,
    pub qugusd_burned: u128,
    pub liquidator_bonus: u128,
}

/// Position health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionHealth {
    Healthy,    // > 150%
    Warning,    // 120% - 150%
    Danger,     // 110% - 120%
    Liquidatable, // < 110%
}

impl CollateralVault {
    /// Create a new collateral vault
    pub fn new() -> Self {
        Self {
            locked_qug: HashMap::new(),
            minted_qugusd: HashMap::new(),
            qug_price_usd: 3000.00, // Default price $3000.00 (will be updated by oracle)
            total_qug_locked: 0,
            total_qugusd_minted: 0,
            last_price_update: chrono::Utc::now().timestamp(),
        }
    }

    /// Update QUG price from oracle
    pub fn update_price(&mut self, new_price: f64) -> Result<()> {
        if new_price <= 0.0 {
            return Err(anyhow!("Invalid price: must be positive"));
        }

        let price_change_pct = ((new_price - self.qug_price_usd) / self.qug_price_usd * 100.0).abs();

        // Circuit breaker: prevent extreme price changes > 20% in single update
        if price_change_pct > 20.0 {
            warn!(
                "⚠️ Large price change detected: {:.2}% - potential oracle manipulation",
                price_change_pct
            );
            return Err(anyhow!("Price change too large: {:.2}%", price_change_pct));
        }

        self.qug_price_usd = new_price;
        self.last_price_update = chrono::Utc::now().timestamp();

        debug!("💱 QUG price updated: ${:.2}", new_price);
        Ok(())
    }

    /// Mint QUGUSD by locking QUG as collateral
    pub fn mint_qugusd(
        &mut self,
        user: [u8; 32],
        qug_amount: u128,
    ) -> Result<MintResult> {
        if qug_amount == 0 {
            return Err(anyhow!("Cannot mint with zero QUG"));
        }

        // Calculate QUG value in USD (convert from base units with 24 decimals)
        let qug_value_usd = (qug_amount as f64 / BASE_UNITS_DIVISOR) * self.qug_price_usd;

        // Calculate maximum QUGUSD that can be minted (150% collateral ratio)
        let max_qugusd_usd = qug_value_usd / MIN_COLLATERAL_RATIO;
        let qugusd_minted = (max_qugusd_usd * BASE_UNITS_DIVISOR) as u128; // Convert to base units

        if qugusd_minted == 0 {
            return Err(anyhow!("QUG amount too small to mint QUGUSD"));
        }

        // Update user's position
        let current_qug = self.locked_qug.get(&user).copied().unwrap_or(0);
        let current_qugusd = self.minted_qugusd.get(&user).copied().unwrap_or(0);

        self.locked_qug.insert(user, current_qug + qug_amount);
        self.minted_qugusd.insert(user, current_qugusd + qugusd_minted);

        // Update totals
        self.total_qug_locked += qug_amount;
        self.total_qugusd_minted += qugusd_minted;

        // Calculate liquidation price (price at which position becomes liquidatable)
        let total_qugusd_value = (current_qugusd + qugusd_minted) as f64 / BASE_UNITS_DIVISOR;
        let total_qug_locked = (current_qug + qug_amount) as f64 / BASE_UNITS_DIVISOR;
        let liquidation_price = (total_qugusd_value * LIQUIDATION_RATIO) / total_qug_locked;

        info!(
            "🏦 Minted {} QUGUSD for user {} (locked {} QUG)",
            qugusd_minted as f64 / BASE_UNITS_DIVISOR,
            hex::encode(&user[..4]),
            qug_amount as f64 / BASE_UNITS_DIVISOR
        );

        Ok(MintResult {
            qug_locked: qug_amount,
            qugusd_minted,
            collateral_ratio: MIN_COLLATERAL_RATIO,
            liquidation_price,
        })
    }

    /// Redeem QUG by burning QUGUSD
    pub fn redeem_qug(
        &mut self,
        user: [u8; 32],
        qugusd_amount: u128,
    ) -> Result<RedeemResult> {
        if qugusd_amount == 0 {
            return Err(anyhow!("Cannot redeem zero QUGUSD"));
        }

        // Check user has enough minted QUGUSD
        let current_qugusd = self.minted_qugusd.get(&user).copied().unwrap_or(0);
        if current_qugusd < qugusd_amount {
            return Err(anyhow!(
                "Insufficient QUGUSD balance: {} < {}",
                current_qugusd,
                qugusd_amount
            ));
        }

        // Calculate QUG to unlock (based on current price)
        let qugusd_value_usd = qugusd_amount as f64 / BASE_UNITS_DIVISOR;
        let qug_to_unlock = ((qugusd_value_usd / self.qug_price_usd) * BASE_UNITS_DIVISOR) as u128;

        // Check user has enough locked QUG
        let current_qug = self.locked_qug.get(&user).copied().unwrap_or(0);
        if current_qug < qug_to_unlock {
            return Err(anyhow!(
                "Insufficient locked QUG: {} < {}",
                current_qug,
                qug_to_unlock
            ));
        }

        // Update user's position
        let remaining_qug = current_qug - qug_to_unlock;
        let remaining_qugusd = current_qugusd - qugusd_amount;

        if remaining_qug > 0 {
            self.locked_qug.insert(user, remaining_qug);
        } else {
            self.locked_qug.remove(&user);
        }

        if remaining_qugusd > 0 {
            self.minted_qugusd.insert(user, remaining_qugusd);
        } else {
            self.minted_qugusd.remove(&user);
        }

        // Update totals (use saturating_sub to prevent underflow)
        self.total_qug_locked = self.total_qug_locked.saturating_sub(qug_to_unlock);
        self.total_qugusd_minted = self.total_qugusd_minted.saturating_sub(qugusd_amount);

        // Calculate remaining collateral ratio
        let remaining_collateral_ratio = if remaining_qugusd > 0 {
            let remaining_qug_value = (remaining_qug as f64 / BASE_UNITS_DIVISOR) * self.qug_price_usd;
            let remaining_qugusd_value = remaining_qugusd as f64 / BASE_UNITS_DIVISOR;
            remaining_qug_value / remaining_qugusd_value
        } else {
            0.0
        };

        info!(
            "🔓 Redeemed {} QUG for user {} (burned {} QUGUSD)",
            qug_to_unlock as f64 / BASE_UNITS_DIVISOR,
            hex::encode(&user[..4]),
            qugusd_amount as f64 / BASE_UNITS_DIVISOR
        );

        Ok(RedeemResult {
            qugusd_burned: qugusd_amount,
            qug_unlocked: qug_to_unlock,
            remaining_collateral_ratio,
        })
    }

    /// Liquidate an undercollateralized position
    pub fn liquidate(
        &mut self,
        liquidator: [u8; 32],
        liquidated_user: [u8; 32],
    ) -> Result<LiquidationResult> {
        // Get user's position
        let locked_qug = self.locked_qug.get(&liquidated_user).copied().unwrap_or(0);
        let minted_qugusd = self.minted_qugusd.get(&liquidated_user).copied().unwrap_or(0);

        if locked_qug == 0 || minted_qugusd == 0 {
            return Err(anyhow!("No position to liquidate"));
        }

        // Calculate current collateral ratio
        let qug_value_usd = (locked_qug as f64 / BASE_UNITS_DIVISOR) * self.qug_price_usd;
        let qugusd_value = minted_qugusd as f64 / BASE_UNITS_DIVISOR;
        let collateral_ratio = qug_value_usd / qugusd_value;

        // Check if position is liquidatable
        if collateral_ratio >= LIQUIDATION_RATIO {
            return Err(anyhow!(
                "Position is healthy ({:.2}% collateral ratio)",
                collateral_ratio * 100.0
            ));
        }

        // Calculate liquidation amounts
        let liquidator_bonus = (locked_qug as f64 * LIQUIDATION_BONUS) as u128;
        let qug_seized = locked_qug; // Seize all collateral

        // Remove user's position
        self.locked_qug.remove(&liquidated_user);
        self.minted_qugusd.remove(&liquidated_user);

        // Update totals (use saturating_sub to prevent underflow)
        self.total_qug_locked = self.total_qug_locked.saturating_sub(locked_qug);
        self.total_qugusd_minted = self.total_qugusd_minted.saturating_sub(minted_qugusd);

        warn!(
            "⚡ Liquidated position: user={}, ratio={:.2}%, seized={} QUG",
            hex::encode(&liquidated_user[..4]),
            collateral_ratio * 100.0,
            qug_seized as f64 / BASE_UNITS_DIVISOR
        );

        Ok(LiquidationResult {
            liquidator,
            liquidated_user,
            qug_seized,
            qugusd_burned: minted_qugusd,
            liquidator_bonus,
        })
    }

    /// Get collateral ratio for a user
    pub fn get_collateral_ratio(&self, user: &[u8; 32]) -> Result<f64> {
        let locked_qug = self.locked_qug.get(user).copied().unwrap_or(0);
        let minted_qugusd = self.minted_qugusd.get(user).copied().unwrap_or(0);

        if minted_qugusd == 0 {
            return Ok(0.0); // No debt = no ratio
        }

        let qug_value_usd = (locked_qug as f64 / BASE_UNITS_DIVISOR) * self.qug_price_usd;
        let qugusd_value = minted_qugusd as f64 / BASE_UNITS_DIVISOR;

        Ok(qug_value_usd / qugusd_value)
    }

    /// Get position health status
    pub fn get_position_health(&self, user: &[u8; 32]) -> Result<PositionHealth> {
        let ratio = self.get_collateral_ratio(user)?;

        if ratio == 0.0 {
            return Ok(PositionHealth::Healthy); // No position
        }

        Ok(if ratio >= MIN_COLLATERAL_RATIO {
            PositionHealth::Healthy
        } else if ratio >= WARNING_RATIO {
            PositionHealth::Warning
        } else if ratio >= LIQUIDATION_RATIO {
            PositionHealth::Danger
        } else {
            PositionHealth::Liquidatable
        })
    }

    /// Get all users with liquidatable positions
    pub fn get_liquidatable_positions(&self) -> Vec<[u8; 32]> {
        let mut liquidatable = Vec::new();

        for (user, locked_qug) in &self.locked_qug {
            if let Some(&minted_qugusd) = self.minted_qugusd.get(user) {
                let qug_value_usd = (*locked_qug as f64 / BASE_UNITS_DIVISOR) * self.qug_price_usd;
                let qugusd_value = minted_qugusd as f64 / BASE_UNITS_DIVISOR;
                let ratio = qug_value_usd / qugusd_value;

                if ratio < LIQUIDATION_RATIO {
                    liquidatable.push(*user);
                }
            }
        }

        liquidatable
    }

    /// Get vault statistics
    pub fn get_vault_stats(&self) -> VaultStats {
        let total_qug_value_usd = (self.total_qug_locked as f64 / BASE_UNITS_DIVISOR) * self.qug_price_usd;
        let total_qugusd_value = self.total_qugusd_minted as f64 / BASE_UNITS_DIVISOR;

        let global_collateral_ratio = if total_qugusd_value > 0.0 {
            total_qug_value_usd / total_qugusd_value
        } else {
            0.0
        };

        VaultStats {
            total_qug_locked: self.total_qug_locked,
            total_qugusd_minted: self.total_qugusd_minted,
            qug_price_usd: self.qug_price_usd,
            global_collateral_ratio,
            num_positions: self.locked_qug.len(),
            last_price_update: self.last_price_update,
        }
    }

    /// Get QUGUSD balance for a user (helper for DEX integration)
    pub fn get_balance(&self, user: &[u8; 32]) -> u128 {
        self.minted_qugusd.get(user).copied().unwrap_or(0)
    }

    /// Burn QUGUSD that was minted via CDP (removes debt without unlocking collateral)
    ///
    /// ⚠️ IMPORTANT: This function should ONLY be called for QUGUSD that was minted via
    /// `mint_qugusd()` (CDP model). For QUGUSD received via `credit_from_pool()` (DEX swaps),
    /// use `burn_pool_qugusd()` instead.
    ///
    /// The distinction matters because:
    /// - CDP-minted QUGUSD increases `total_qugusd_minted` when created
    /// - Pool-transferred QUGUSD does NOT increase `total_qugusd_minted`
    ///
    /// If you burn pool QUGUSD with this function, you'll cause an underflow!
    pub fn burn(&mut self, user: &[u8; 32], amount: u128) -> Result<()> {
        let current_qugusd = self.minted_qugusd.get(user).copied().unwrap_or(0);

        if current_qugusd < amount {
            return Err(anyhow!(
                "Insufficient QUGUSD balance to burn: {} < {}",
                current_qugusd,
                amount
            ));
        }

        let new_balance = current_qugusd - amount;
        if new_balance > 0 {
            self.minted_qugusd.insert(*user, new_balance);
        } else {
            self.minted_qugusd.remove(user);
        }

        // v2.4.0: Use saturating_sub to prevent underflow
        // If this would underflow, log a warning - it indicates accounting mismatch
        // (likely burning pool QUGUSD with this function instead of burn_pool_qugusd)
        if self.total_qugusd_minted < amount {
            warn!(
                "⚠️ ACCOUNTING MISMATCH: Attempted to burn {} QUGUSD but total is only {}. Using saturating_sub. \
                This may indicate pool QUGUSD being burned via burn() instead of burn_pool_qugusd().",
                amount as f64 / BASE_UNITS_DIVISOR,
                self.total_qugusd_minted as f64 / BASE_UNITS_DIVISOR
            );
        }
        self.total_qugusd_minted = self.total_qugusd_minted.saturating_sub(amount);

        debug!(
            "🔥 Burned {} QUGUSD for user {} (CDP debt)",
            amount as f64 / BASE_UNITS_DIVISOR,
            hex::encode(&user[..4])
        );

        Ok(())
    }

    /// Burn QUGUSD that was received from pool transfers (DEX swaps)
    ///
    /// This function removes QUGUSD from user balance WITHOUT decrementing total_qugusd_minted,
    /// because pool-transferred QUGUSD never incremented the total in the first place.
    ///
    /// Use this when burning QUGUSD that was credited via `credit_from_pool()`.
    pub fn burn_pool_qugusd(&mut self, user: &[u8; 32], amount: u128) -> Result<()> {
        let current_qugusd = self.minted_qugusd.get(user).copied().unwrap_or(0);

        if current_qugusd < amount {
            return Err(anyhow!(
                "Insufficient QUGUSD balance to burn: {} < {}",
                current_qugusd,
                amount
            ));
        }

        let new_balance = current_qugusd - amount;
        if new_balance > 0 {
            self.minted_qugusd.insert(*user, new_balance);
        } else {
            self.minted_qugusd.remove(user);
        }

        // Note: We do NOT decrement total_qugusd_minted because this QUGUSD
        // came from pool reserves (via credit_from_pool) which never incremented it.

        debug!(
            "🔥 Burned {} QUGUSD for user {} (pool transfer)",
            amount as f64 / BASE_UNITS_DIVISOR,
            hex::encode(&user[..4])
        );

        Ok(())
    }

    /// ⚠️ DEPRECATED: DO NOT USE - This function mints QUGUSD without collateral!
    ///
    /// v2.3.6-beta: This function has been deprecated because it violates sound monetary policy.
    /// QUGUSD should ONLY be created via:
    /// 1. mint_qugusd() with locked QUG collateral (CDP model)
    /// 2. Transfers from existing pool reserves (DEX swaps)
    ///
    /// For DEX swaps: Use token_balances map instead of minting new supply.
    /// The pool reserves already contain QUGUSD from liquidity providers who locked collateral.
    #[deprecated(since = "2.3.6", note = "Use mint_qugusd() with collateral or pool reserves for swaps")]
    pub fn mint_unbacked_deprecated(&mut self, _user: &[u8; 32], _amount: u128) -> Result<()> {
        Err(anyhow!(
            "SECURITY: Unbacked QUGUSD minting is disabled. Use mint_qugusd() with collateral."
        ))
    }

    /// Credit QUGUSD from pool reserves to user (for DEX swaps)
    /// This does NOT create new supply - it tracks transfers from liquidity pools.
    /// The pool liquidity providers already locked collateral when adding liquidity.
    pub fn credit_from_pool(&mut self, user: &[u8; 32], amount: u128) -> Result<()> {
        if amount == 0 {
            return Err(anyhow!("Cannot credit zero QUGUSD"));
        }

        // Note: This doesn't increase total_qugusd_minted because the QUGUSD
        // already exists in the pool (from LPs who locked collateral).
        // We're just tracking the transfer to the user's balance.
        let current_qugusd = self.minted_qugusd.get(user).copied().unwrap_or(0);
        self.minted_qugusd.insert(*user, current_qugusd + amount);

        debug!(
            "💰 Credited {} QUGUSD to user {} (from pool reserves)",
            amount as f64 / BASE_UNITS_DIVISOR,
            hex::encode(&user[..4])
        );

        Ok(())
    }

    /// Update QUG price from AMM pool reserves (oracle bridge)
    /// This allows the CollateralVault to use real market prices for CDP calculations.
    pub fn update_price_from_amm(&mut self, qug_reserve: u128, qugusd_reserve: u128) -> Result<()> {
        if qug_reserve == 0 || qugusd_reserve == 0 {
            return Err(anyhow!("Invalid pool reserves: cannot have zero reserves"));
        }

        // Price = QUGUSD reserve / QUG reserve (how many QUGUSD per 1 QUG)
        let new_price = (qugusd_reserve as f64) / (qug_reserve as f64);

        // Sanity check: price should be positive and reasonable
        if new_price <= 0.0 || new_price > 1_000_000.0 {
            return Err(anyhow!("Invalid price from AMM: {}", new_price));
        }

        let price_change_pct = ((new_price - self.qug_price_usd) / self.qug_price_usd * 100.0).abs();

        // Allow gradual price changes without circuit breaker for AMM updates
        // The AMM is the source of truth, so we accept its prices (with logging)
        if price_change_pct > 20.0 {
            warn!(
                "📊 Large AMM price change: ${:.2} → ${:.2} ({:.1}%)",
                self.qug_price_usd, new_price, price_change_pct
            );
        }

        self.qug_price_usd = new_price;
        self.last_price_update = chrono::Utc::now().timestamp();

        debug!(
            "💱 QUG price updated from AMM: ${:.4} (reserves: {} QUG / {} QUGUSD)",
            new_price, qug_reserve, qugusd_reserve
        );

        Ok(())
    }

    /// Get current QUG price in USD
    pub fn get_qug_price(&self) -> f64 {
        self.qug_price_usd
    }
}

impl Default for CollateralVault {
    fn default() -> Self {
        Self::new()
    }
}

/// Vault statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultStats {
    pub total_qug_locked: u128,
    pub total_qugusd_minted: u128,
    pub qug_price_usd: f64,
    pub global_collateral_ratio: f64,
    pub num_positions: usize,
    pub last_price_update: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1 QUG in base units with 24 decimals = 10^24
    const ONE_QUG: u128 = 1_000_000_000_000_000_000_000_000;

    #[test]
    fn test_vault_creation() {
        let vault = CollateralVault::new();
        assert_eq!(vault.total_qug_locked, 0);
        assert_eq!(vault.total_qugusd_minted, 0);
        assert_eq!(vault.qug_price_usd, 3000.00);
    }

    #[test]
    fn test_mint_qugusd() {
        let mut vault = CollateralVault::new();
        let user = [1u8; 32];

        // Lock 1000 QUG ($3,000,000 at $3000.00/QUG)
        let qug_amount = 1000 * ONE_QUG; // 1000 QUG in base units
        let result = vault.mint_qugusd(user, qug_amount).unwrap();

        // Should mint $3,000,000 / 1.5 = $2,000,000 QUGUSD
        assert_eq!(result.qug_locked, qug_amount);
        // Expected: 28333.333... QUGUSD in base units
        let expected_qugusd = (28333.333333333333 * BASE_UNITS_DIVISOR) as u128;
        // Allow 1% tolerance due to floating point
        let tolerance = expected_qugusd / 100;
        assert!((result.qugusd_minted as i128 - expected_qugusd as i128).abs() < tolerance as i128);
        assert_eq!(result.collateral_ratio, 1.5);
    }

    #[test]
    fn test_redeem_qug() {
        let mut vault = CollateralVault::new();
        let user = [1u8; 32];

        // Mint first with 1000 QUG
        let qug_amount = 1000 * ONE_QUG;
        vault.mint_qugusd(user, qug_amount).unwrap();

        // Get the actual minted amount
        let minted = vault.minted_qugusd.get(&user).copied().unwrap();

        // Redeem half
        let redeem_amount = minted / 2;
        let result = vault.redeem_qug(user, redeem_amount).unwrap();

        assert_eq!(result.qugusd_burned, redeem_amount);
        // QUG unlocked should be proportional
        assert!(result.qug_unlocked > 0);
    }

    #[test]
    fn test_liquidation() {
        let mut vault = CollateralVault::new();
        let user = [1u8; 32];
        let liquidator = [2u8; 32];

        // Mint with 1000 QUG
        let qug_amount = 1000 * ONE_QUG;
        vault.mint_qugusd(user, qug_amount).unwrap();

        // Drop QUG price to trigger liquidation ($3000.00 -> $6.50)
        vault.update_price(35.0).unwrap(); // 15% drop first
        vault.update_price(30.0).unwrap(); // Another drop
        vault.update_price(25.0).unwrap(); // Keep dropping
        vault.update_price(21.0).unwrap();
        vault.update_price(18.0).unwrap();
        vault.update_price(15.0).unwrap();
        vault.update_price(12.5).unwrap();
        vault.update_price(10.5).unwrap();
        vault.update_price(9.0).unwrap();
        vault.update_price(7.5).unwrap();
        vault.update_price(6.5).unwrap();

        // Check position is liquidatable
        let ratio = vault.get_collateral_ratio(&user).unwrap();
        assert!(ratio < LIQUIDATION_RATIO);

        // Liquidate
        let result = vault.liquidate(liquidator, user).unwrap();
        assert_eq!(result.qug_seized, qug_amount);
        // Liquidator bonus is 5% of seized QUG
        let expected_bonus = (qug_amount as f64 * LIQUIDATION_BONUS) as u128;
        assert_eq!(result.liquidator_bonus, expected_bonus);
    }

    #[test]
    fn test_price_circuit_breaker() {
        let mut vault = CollateralVault::new();

        // Try to update price by > 20%
        let result = vault.update_price(55.0); // ~29% increase from 3000.00
        assert!(result.is_err());

        // Small change should work (15% increase)
        let result = vault.update_price(48.0); // ~13% increase
        assert!(result.is_ok());
    }
}
