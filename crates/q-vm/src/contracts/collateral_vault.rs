/// CollateralVault: QUGUSD Stablecoin Collateral Management
///
/// This smart contract implements an over-collateralized algorithmic stablecoin (QUGUSD)
/// backed by QUG tokens at a 150% collateralization ratio.
///
/// Key Features:
/// - Mint QUGUSD by locking QUG as collateral (150% ratio)
/// - Redeem QUG by burning QUGUSD
/// - Liquidate undercollateralized positions (< 110%)
/// - Oracle-based QUG/USD price feeds

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Collateralization ratio constants
pub const MIN_COLLATERAL_RATIO: f64 = 1.50; // 150% minimum
pub const WARNING_RATIO: f64 = 1.20; // 120% warning threshold
pub const LIQUIDATION_RATIO: f64 = 1.10; // 110% liquidation threshold
pub const LIQUIDATION_BONUS: f64 = 0.05; // 5% bonus for liquidators

/// Collateral vault for QUGUSD stablecoin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollateralVault {
    /// User address -> Locked QUG amount (in base units)
    pub locked_qug: HashMap<[u8; 32], u64>,

    /// User address -> Minted QUGUSD amount (in base units)
    pub minted_qugusd: HashMap<[u8; 32], u64>,

    /// Current QUG price in USD (from oracle)
    pub qug_price_usd: f64,

    /// Total QUG locked in vault
    pub total_qug_locked: u64,

    /// Total QUGUSD minted
    pub total_qugusd_minted: u64,

    /// Last oracle update timestamp
    pub last_price_update: i64,
}

/// Result of a mint operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MintResult {
    pub qug_locked: u64,
    pub qugusd_minted: u64,
    pub collateral_ratio: f64,
    pub liquidation_price: f64,
}

/// Result of a redeem operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedeemResult {
    pub qugusd_burned: u64,
    pub qug_unlocked: u64,
    pub remaining_collateral_ratio: f64,
}

/// Result of a liquidation operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidationResult {
    pub liquidator: [u8; 32],
    pub liquidated_user: [u8; 32],
    pub qug_seized: u64,
    pub qugusd_burned: u64,
    pub liquidator_bonus: u64,
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
            qug_price_usd: 42.50, // Default price $42.50 (will be updated by oracle)
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
        qug_amount: u64,
    ) -> Result<MintResult> {
        if qug_amount == 0 {
            return Err(anyhow!("Cannot mint with zero QUG"));
        }

        // Calculate QUG value in USD (convert from base units)
        let qug_value_usd = (qug_amount as f64 / 1e8) * self.qug_price_usd;

        // Calculate maximum QUGUSD that can be minted (150% collateral ratio)
        let max_qugusd_usd = qug_value_usd / MIN_COLLATERAL_RATIO;
        let qugusd_minted = (max_qugusd_usd * 1e8) as u64; // Convert to base units

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
        let total_qugusd_value = (current_qugusd + qugusd_minted) as f64 / 1e8;
        let total_qug_locked = (current_qug + qug_amount) as f64 / 1e8;
        let liquidation_price = (total_qugusd_value * LIQUIDATION_RATIO) / total_qug_locked;

        info!(
            "🏦 Minted {} QUGUSD for user {} (locked {} QUG)",
            qugusd_minted as f64 / 1e8,
            hex::encode(&user[..4]),
            qug_amount as f64 / 1e8
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
        qugusd_amount: u64,
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
        let qugusd_value_usd = qugusd_amount as f64 / 1e8;
        let qug_to_unlock = ((qugusd_value_usd / self.qug_price_usd) * 1e8) as u64;

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

        // Update totals
        self.total_qug_locked -= qug_to_unlock;
        self.total_qugusd_minted -= qugusd_amount;

        // Calculate remaining collateral ratio
        let remaining_collateral_ratio = if remaining_qugusd > 0 {
            let remaining_qug_value = (remaining_qug as f64 / 1e8) * self.qug_price_usd;
            let remaining_qugusd_value = remaining_qugusd as f64 / 1e8;
            remaining_qug_value / remaining_qugusd_value
        } else {
            0.0
        };

        info!(
            "🔓 Redeemed {} QUG for user {} (burned {} QUGUSD)",
            qug_to_unlock as f64 / 1e8,
            hex::encode(&user[..4]),
            qugusd_amount as f64 / 1e8
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
        let qug_value_usd = (locked_qug as f64 / 1e8) * self.qug_price_usd;
        let qugusd_value = minted_qugusd as f64 / 1e8;
        let collateral_ratio = qug_value_usd / qugusd_value;

        // Check if position is liquidatable
        if collateral_ratio >= LIQUIDATION_RATIO {
            return Err(anyhow!(
                "Position is healthy ({:.2}% collateral ratio)",
                collateral_ratio * 100.0
            ));
        }

        // Calculate liquidation amounts
        let liquidator_bonus = (locked_qug as f64 * LIQUIDATION_BONUS) as u64;
        let qug_seized = locked_qug; // Seize all collateral

        // Remove user's position
        self.locked_qug.remove(&liquidated_user);
        self.minted_qugusd.remove(&liquidated_user);

        // Update totals
        self.total_qug_locked -= locked_qug;
        self.total_qugusd_minted -= minted_qugusd;

        warn!(
            "⚡ Liquidated position: user={}, ratio={:.2}%, seized={} QUG",
            hex::encode(&liquidated_user[..4]),
            collateral_ratio * 100.0,
            qug_seized as f64 / 1e8
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

        let qug_value_usd = (locked_qug as f64 / 1e8) * self.qug_price_usd;
        let qugusd_value = minted_qugusd as f64 / 1e8;

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
                let qug_value_usd = (*locked_qug as f64 / 1e8) * self.qug_price_usd;
                let qugusd_value = minted_qugusd as f64 / 1e8;
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
        let total_qug_value_usd = (self.total_qug_locked as f64 / 1e8) * self.qug_price_usd;
        let total_qugusd_value = self.total_qugusd_minted as f64 / 1e8;

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
    pub fn get_balance(&self, user: &[u8; 32]) -> u64 {
        self.minted_qugusd.get(user).copied().unwrap_or(0)
    }

    /// Burn QUGUSD (helper for DEX integration - removes debt without unlocking collateral)
    pub fn burn(&mut self, user: &[u8; 32], amount: u64) -> Result<()> {
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

        self.total_qugusd_minted -= amount;

        debug!(
            "🔥 Burned {} QUGUSD for user {} (DEX swap)",
            amount as f64 / 1e8,
            hex::encode(&user[..4])
        );

        Ok(())
    }

    /// Mint QUGUSD directly (helper for DEX integration - adds debt without collateral)
    /// WARNING: This bypasses collateral requirements and should only be used for DEX swaps
    pub fn mint(&mut self, user: &[u8; 32], amount: u64) -> Result<()> {
        if amount == 0 {
            return Err(anyhow!("Cannot mint zero QUGUSD"));
        }

        let current_qugusd = self.minted_qugusd.get(user).copied().unwrap_or(0);
        self.minted_qugusd.insert(*user, current_qugusd + amount);
        self.total_qugusd_minted += amount;

        debug!(
            "💰 Minted {} QUGUSD for user {} (DEX swap)",
            amount as f64 / 1e8,
            hex::encode(&user[..4])
        );

        Ok(())
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
    pub total_qug_locked: u64,
    pub total_qugusd_minted: u64,
    pub qug_price_usd: f64,
    pub global_collateral_ratio: f64,
    pub num_positions: usize,
    pub last_price_update: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vault_creation() {
        let vault = CollateralVault::new();
        assert_eq!(vault.total_qug_locked, 0);
        assert_eq!(vault.total_qugusd_minted, 0);
        assert_eq!(vault.qug_price_usd, 10.0);
    }

    #[test]
    fn test_mint_qugusd() {
        let mut vault = CollateralVault::new();
        let user = [1u8; 32];

        // Lock 1000 QUG ($10,000 at $10/QUG)
        let result = vault.mint_qugusd(user, 100_000_000_000).unwrap(); // 1000 QUG in base units

        // Should mint $10,000 / 1.5 = $6,666.67 QUGUSD
        assert_eq!(result.qug_locked, 100_000_000_000);
        assert_eq!(result.qugusd_minted, 66_666_666_666); // ~666.67 QUGUSD
        assert_eq!(result.collateral_ratio, 1.5);
    }

    #[test]
    fn test_redeem_qug() {
        let mut vault = CollateralVault::new();
        let user = [1u8; 32];

        // Mint first
        vault.mint_qugusd(user, 100_000_000_000).unwrap();

        // Redeem 100 QUGUSD
        let result = vault.redeem_qug(user, 10_000_000_000).unwrap();

        // Should unlock 10 QUG (100 QUGUSD / $10 per QUG)
        assert_eq!(result.qugusd_burned, 10_000_000_000);
        assert_eq!(result.qug_unlocked, 10_000_000_000);
    }

    #[test]
    fn test_liquidation() {
        let mut vault = CollateralVault::new();
        let user = [1u8; 32];
        let liquidator = [2u8; 32];

        // Mint with 1000 QUG
        vault.mint_qugusd(user, 100_000_000_000).unwrap();

        // Drop QUG price to trigger liquidation
        vault.update_price(1.5).unwrap(); // $10 -> $1.50

        // Check position is liquidatable
        let ratio = vault.get_collateral_ratio(&user).unwrap();
        assert!(ratio < LIQUIDATION_RATIO);

        // Liquidate
        let result = vault.liquidate(liquidator, user).unwrap();
        assert_eq!(result.qug_seized, 100_000_000_000);
        assert_eq!(result.liquidator_bonus, 5_000_000_000); // 5% bonus
    }

    #[test]
    fn test_price_circuit_breaker() {
        let mut vault = CollateralVault::new();

        // Try to update price by > 20%
        let result = vault.update_price(13.0); // 30% increase
        assert!(result.is_err());

        // Small change should work
        let result = vault.update_price(11.5); // 15% increase
        assert!(result.is_ok());
    }
}
