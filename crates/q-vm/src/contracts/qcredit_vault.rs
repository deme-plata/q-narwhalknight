//! Quillon Credit (QCREDIT) Yield Vault
//!
//! Lock QUG 1:1 → mint QCREDIT, earn tiered yield.
//! Digital credit layer inspired by Strategy's 3-layer stack:
//!   L1 digital capital (QUG) → L2 digital credit (QCREDIT) → L3 products
//!
//! Tiers:
//!   Bronze   —  7 days lock,  5% APY
//!   Silver   — 30 days lock, 10% APY
//!   Gold     — 90 days lock, 15% APY
//!   Platinum — 180 days lock, 25% APY

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============ TIER PARAMETERS ============

const BRONZE_LOCK_SECONDS: u64 = 7 * 24 * 3600;
const SILVER_LOCK_SECONDS: u64 = 30 * 24 * 3600;
const GOLD_LOCK_SECONDS: u64 = 90 * 24 * 3600;
const PLATINUM_LOCK_SECONDS: u64 = 180 * 24 * 3600;

/// APY in basis points (500 = 5%)
const BRONZE_APY_BPS: u64 = 500;
const SILVER_APY_BPS: u64 = 1000;
const GOLD_APY_BPS: u64 = 1500;
const PLATINUM_APY_BPS: u64 = 2500;

/// Seconds in a 365-day year (for yield calculations)
const SECONDS_PER_YEAR: u64 = 365 * 24 * 3600;

// ============ DATA STRUCTURES ============

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CreditTier {
    Bronze,
    Silver,
    Gold,
    Platinum,
}

impl CreditTier {
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "bronze" => Some(Self::Bronze),
            "silver" => Some(Self::Silver),
            "gold" => Some(Self::Gold),
            "platinum" => Some(Self::Platinum),
            _ => None,
        }
    }

    pub fn lock_duration(&self) -> u64 {
        match self {
            Self::Bronze => BRONZE_LOCK_SECONDS,
            Self::Silver => SILVER_LOCK_SECONDS,
            Self::Gold => GOLD_LOCK_SECONDS,
            Self::Platinum => PLATINUM_LOCK_SECONDS,
        }
    }

    pub fn apy_bps(&self) -> u64 {
        match self {
            Self::Bronze => BRONZE_APY_BPS,
            Self::Silver => SILVER_APY_BPS,
            Self::Gold => GOLD_APY_BPS,
            Self::Platinum => PLATINUM_APY_BPS,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Bronze => "Bronze",
            Self::Silver => "Silver",
            Self::Gold => "Gold",
            Self::Platinum => "Platinum",
        }
    }

    pub fn all_tiers() -> &'static [CreditTier] {
        &[Self::Bronze, Self::Silver, Self::Gold, Self::Platinum]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditPosition {
    pub wallet: String,
    /// QUG locked (24-decimal base units)
    pub amount_locked: u128,
    /// QCREDIT minted 1:1 (24-decimal base units)
    pub qcredit_minted: u128,
    pub tier: CreditTier,
    /// Unix timestamp when position was opened
    pub lock_timestamp: u64,
    /// Unix timestamp when unlock becomes available
    pub unlock_timestamp: u64,
    /// Yield already claimed (24-decimal base units)
    pub claimed_yield: u128,
    /// Last time yield was claimed
    pub last_claim_timestamp: u64,
}

impl CreditPosition {
    /// Calculate pending (unclaimed) yield since last claim
    pub fn pending_yield(&self, now: u64) -> u128 {
        let elapsed = now.saturating_sub(self.last_claim_timestamp);
        if elapsed == 0 || self.amount_locked == 0 {
            return 0;
        }
        // yield = principal * apy_bps / 10000 * elapsed / SECONDS_PER_YEAR
        // Use u128 arithmetic to avoid overflow
        let apy = self.tier.apy_bps() as u128;
        (self.amount_locked)
            .checked_mul(apy)
            .and_then(|v| v.checked_mul(elapsed as u128))
            .map(|v| v / (10_000u128 * SECONDS_PER_YEAR as u128))
            .unwrap_or(0)
    }

    /// Whether the lock period has expired
    pub fn is_unlockable(&self, now: u64) -> bool {
        now >= self.unlock_timestamp
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierInfo {
    pub tier: CreditTier,
    pub lock_days: u64,
    pub apy_percent: f64,
    pub lock_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultStatus {
    pub total_locked: u128,
    pub total_qcredit_supply: u128,
    pub protocol_reserve: u128,
    pub total_yield_paid: u128,
    pub position_count: usize,
    pub tiers: Vec<TierInfo>,
}

// ============ VAULT ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QCreditVault {
    /// wallet → list of positions
    pub positions: HashMap<String, Vec<CreditPosition>>,
    /// Protocol reserve that funds yield payouts (24-decimal base units)
    pub protocol_reserve: u128,
    /// Total QUG locked across all positions
    pub total_locked: u128,
    /// Total QCREDIT minted across all positions
    pub total_qcredit_supply: u128,
    /// Cumulative yield paid out
    pub total_yield_paid: u128,
}

impl Default for QCreditVault {
    fn default() -> Self {
        Self::new()
    }
}

impl QCreditVault {
    pub fn new() -> Self {
        Self {
            positions: HashMap::new(),
            protocol_reserve: 0,
            total_locked: 0,
            total_qcredit_supply: 0,
            total_yield_paid: 0,
        }
    }

    /// Get tier info for display
    pub fn get_tiers() -> Vec<TierInfo> {
        CreditTier::all_tiers()
            .iter()
            .map(|t| TierInfo {
                tier: *t,
                lock_days: t.lock_duration() / 86400,
                apy_percent: t.apy_bps() as f64 / 100.0,
                lock_seconds: t.lock_duration(),
            })
            .collect()
    }

    /// Get vault status summary
    pub fn status(&self) -> VaultStatus {
        VaultStatus {
            total_locked: self.total_locked,
            total_qcredit_supply: self.total_qcredit_supply,
            protocol_reserve: self.protocol_reserve,
            total_yield_paid: self.total_yield_paid,
            position_count: self.positions.values().map(|v| v.len()).sum(),
            tiers: Self::get_tiers(),
        }
    }

    /// Lock QUG and mint QCREDIT 1:1
    ///
    /// Returns the created position on success.
    /// Caller is responsible for:
    ///   1. Deducting `amount` from wallet's QUG balance
    ///   2. Crediting `amount` QCREDIT to wallet's token balance
    pub fn lock(
        &mut self,
        wallet: &str,
        amount: u128,
        tier: CreditTier,
        now: u64,
    ) -> Result<CreditPosition, String> {
        if amount == 0 {
            return Err("Amount must be > 0".into());
        }

        let position = CreditPosition {
            wallet: wallet.to_string(),
            amount_locked: amount,
            qcredit_minted: amount, // 1:1
            tier,
            lock_timestamp: now,
            unlock_timestamp: now + tier.lock_duration(),
            claimed_yield: 0,
            last_claim_timestamp: now,
        };

        self.total_locked = self.total_locked.saturating_add(amount);
        self.total_qcredit_supply = self.total_qcredit_supply.saturating_add(amount);

        self.positions
            .entry(wallet.to_string())
            .or_default()
            .push(position.clone());

        Ok(position)
    }

    /// Unlock a position: burn QCREDIT, return QUG + unclaimed yield.
    ///
    /// Returns `(qug_returned, yield_claimed)` on success.
    /// Caller is responsible for:
    ///   1. Burning `qug_returned` QCREDIT from wallet's token balance
    ///   2. Crediting `qug_returned + yield_claimed` QUG to wallet
    pub fn unlock(
        &mut self,
        wallet: &str,
        position_index: usize,
        now: u64,
    ) -> Result<(u128, u128), String> {
        let positions = self
            .positions
            .get_mut(wallet)
            .ok_or("No positions found")?;

        if position_index >= positions.len() {
            return Err("Invalid position index".into());
        }

        let pos = &positions[position_index];
        if !pos.is_unlockable(now) {
            return Err(format!(
                "Position locked until {} (now={})",
                pos.unlock_timestamp, now
            ));
        }

        let pending = pos.pending_yield(now);
        let qug_returned = pos.amount_locked;

        // Check protocol reserve can cover yield
        let yield_to_pay = if pending > self.protocol_reserve {
            self.protocol_reserve // Pay what we can
        } else {
            pending
        };

        self.total_locked = self.total_locked.saturating_sub(qug_returned);
        self.total_qcredit_supply = self.total_qcredit_supply.saturating_sub(qug_returned);
        self.total_yield_paid = self.total_yield_paid.saturating_add(yield_to_pay);
        self.protocol_reserve = self.protocol_reserve.saturating_sub(yield_to_pay);

        positions.remove(position_index);
        if positions.is_empty() {
            self.positions.remove(wallet);
        }

        Ok((qug_returned, yield_to_pay))
    }

    /// Claim accrued yield without unlocking the position.
    ///
    /// Returns the yield amount claimed.
    /// Caller credits this as QUG to the wallet.
    pub fn claim_yield(
        &mut self,
        wallet: &str,
        position_index: usize,
        now: u64,
    ) -> Result<u128, String> {
        let positions = self
            .positions
            .get_mut(wallet)
            .ok_or("No positions found")?;

        if position_index >= positions.len() {
            return Err("Invalid position index".into());
        }

        let pos = &mut positions[position_index];
        let pending = pos.pending_yield(now);
        if pending == 0 {
            return Err("No yield to claim".into());
        }

        let yield_to_pay = if pending > self.protocol_reserve {
            self.protocol_reserve
        } else {
            pending
        };

        pos.claimed_yield = pos.claimed_yield.saturating_add(yield_to_pay);
        pos.last_claim_timestamp = now;

        self.total_yield_paid = self.total_yield_paid.saturating_add(yield_to_pay);
        self.protocol_reserve = self.protocol_reserve.saturating_sub(yield_to_pay);

        Ok(yield_to_pay)
    }

    /// Get all positions for a wallet
    pub fn get_positions(&self, wallet: &str) -> Vec<CreditPosition> {
        self.positions.get(wallet).cloned().unwrap_or_default()
    }

    /// Get positions with pending yield calculated
    pub fn get_positions_with_yield(&self, wallet: &str, now: u64) -> Vec<(CreditPosition, u128)> {
        self.get_positions(wallet)
            .into_iter()
            .map(|p| {
                let pending = p.pending_yield(now);
                (p, pending)
            })
            .collect()
    }

    /// Add to the protocol reserve (called when protocol fees are collected)
    pub fn fund_reserve(&mut self, amount: u128) {
        self.protocol_reserve = self.protocol_reserve.saturating_add(amount);
    }
}

// ============ TESTS ============

#[cfg(test)]
mod tests {
    use super::*;

    const ONE_QUG: u128 = 1_000_000_000_000_000_000_000_000; // 10^24

    #[test]
    fn test_lock_and_mint() {
        let mut vault = QCreditVault::new();
        let now = 1_700_000_000u64;
        let result = vault.lock("alice", 100 * ONE_QUG, CreditTier::Silver, now);
        assert!(result.is_ok());
        let pos = result.unwrap();
        assert_eq!(pos.amount_locked, 100 * ONE_QUG);
        assert_eq!(pos.qcredit_minted, 100 * ONE_QUG);
        assert_eq!(pos.unlock_timestamp, now + SILVER_LOCK_SECONDS);
        assert_eq!(vault.total_locked, 100 * ONE_QUG);
        assert_eq!(vault.total_qcredit_supply, 100 * ONE_QUG);
    }

    #[test]
    fn test_zero_amount_rejected() {
        let mut vault = QCreditVault::new();
        assert!(vault.lock("alice", 0, CreditTier::Bronze, 1_700_000_000).is_err());
    }

    #[test]
    fn test_pending_yield_calculation() {
        let pos = CreditPosition {
            wallet: "alice".into(),
            amount_locked: 1000 * ONE_QUG,
            qcredit_minted: 1000 * ONE_QUG,
            tier: CreditTier::Gold, // 15% APY
            lock_timestamp: 0,
            unlock_timestamp: GOLD_LOCK_SECONDS,
            claimed_yield: 0,
            last_claim_timestamp: 0,
        };
        // After 1 year: yield = 1000 * 0.15 = 150 QUG
        let yield_1y = pos.pending_yield(SECONDS_PER_YEAR);
        let expected = 150 * ONE_QUG;
        // Allow small rounding error from integer division
        assert!(
            (yield_1y as i128 - expected as i128).unsigned_abs() < ONE_QUG / 100,
            "Expected ~{} got {}",
            expected,
            yield_1y
        );
    }

    #[test]
    fn test_unlock_before_expiry_fails() {
        let mut vault = QCreditVault::new();
        let now = 1_700_000_000u64;
        vault.lock("bob", 50 * ONE_QUG, CreditTier::Platinum, now).unwrap();
        // Try unlock immediately
        assert!(vault.unlock("bob", 0, now).is_err());
        // Try unlock 1 day later (Platinum = 180 days)
        assert!(vault.unlock("bob", 0, now + 86400).is_err());
    }

    #[test]
    fn test_unlock_after_expiry_succeeds() {
        let mut vault = QCreditVault::new();
        vault.fund_reserve(1000 * ONE_QUG);
        let now = 1_700_000_000u64;
        vault.lock("bob", 50 * ONE_QUG, CreditTier::Bronze, now).unwrap();
        // Bronze = 7 days
        let unlock_time = now + BRONZE_LOCK_SECONDS;
        let (returned, yield_claimed) = vault.unlock("bob", 0, unlock_time).unwrap();
        assert_eq!(returned, 50 * ONE_QUG);
        assert!(yield_claimed > 0);
        assert_eq!(vault.total_locked, 0);
        assert_eq!(vault.total_qcredit_supply, 0);
    }

    #[test]
    fn test_claim_yield_without_unlock() {
        let mut vault = QCreditVault::new();
        vault.fund_reserve(1000 * ONE_QUG);
        let now = 1_700_000_000u64;
        vault.lock("carol", 200 * ONE_QUG, CreditTier::Silver, now).unwrap();
        // Claim after 30 days
        let claim_time = now + 30 * 86400;
        let claimed = vault.claim_yield("carol", 0, claim_time).unwrap();
        assert!(claimed > 0);
        // Position still exists
        assert_eq!(vault.get_positions("carol").len(), 1);
        assert_eq!(vault.get_positions("carol")[0].amount_locked, 200 * ONE_QUG);
    }

    #[test]
    fn test_yield_capped_by_reserve() {
        let mut vault = QCreditVault::new();
        // Only 1 QUG in reserve
        vault.fund_reserve(ONE_QUG);
        let now = 1_700_000_000u64;
        vault.lock("dave", 10_000 * ONE_QUG, CreditTier::Platinum, now).unwrap();
        // After 180 days, yield would be huge but capped by reserve
        let unlock_time = now + PLATINUM_LOCK_SECONDS;
        let (_, yield_claimed) = vault.unlock("dave", 0, unlock_time).unwrap();
        assert_eq!(yield_claimed, ONE_QUG); // Capped at reserve
    }

    #[test]
    fn test_tier_info() {
        let tiers = QCreditVault::get_tiers();
        assert_eq!(tiers.len(), 4);
        assert_eq!(tiers[0].lock_days, 7);
        assert_eq!(tiers[0].apy_percent, 5.0);
        assert_eq!(tiers[3].lock_days, 180);
        assert_eq!(tiers[3].apy_percent, 25.0);
    }

    #[test]
    fn test_multiple_positions() {
        let mut vault = QCreditVault::new();
        let now = 1_700_000_000u64;
        vault.lock("eve", 10 * ONE_QUG, CreditTier::Bronze, now).unwrap();
        vault.lock("eve", 20 * ONE_QUG, CreditTier::Gold, now).unwrap();
        assert_eq!(vault.get_positions("eve").len(), 2);
        assert_eq!(vault.total_locked, 30 * ONE_QUG);
    }
}
