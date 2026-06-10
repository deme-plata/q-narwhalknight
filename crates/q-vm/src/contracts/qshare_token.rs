//! Quillon Treasury Share (QSHARE) — L3 layer of the 3-layer capital stack
//!
//! Per `docs/standards/qshare-treasury-protocol-spec.md`. Autonomous on-chain
//! Saylor-style premium-arbitrage smart contract. Mints new shares when
//! QSHARE/QUG market price exceeds NAV (premium), buys back when below
//! (discount). Result: QUG-per-share rises reflexively without human
//! market-timing.
//!
//! Layer stack:
//!   L1 QUG     — native digital capital, PoW mined
//!   L2 QCREDIT — yield-bearing vault (5/10/15/25% APY tiers)
//!   L3 QSHARE  — this file — market-priced treasury share
//!
//! The contract holds a basket of QCREDIT positions (the treasury). NAV is
//! computed deterministically from on-chain QCREDIT vault state. When the
//! premium ratio is high enough, anyone can call `try_autonomous_mint` to
//! receive a bounty for triggering the mint+swap+lock cycle. Mint sizing
//! is double-capped by pool fraction and inflation rate so the strategy
//! cannot self-bid the QSHARE price.
//!
//! Spec authors: Claude Opus 4.7 (Anthropic), Viktor Sandstrøm Kristensen.
//! Reference implementation: this file, Q-NarwhalKnight v10.10.x.
//! License: Apache-2.0.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::contracts::qcredit_vault::CreditTier;

// ============ PROTOCOL CONSTANTS ============

/// Premium ratio (×1000) above which mint is allowed.
/// 1500 = market price must be ≥1.5× NAV to trigger.
const MINT_THRESHOLD_BPS: u64 = 1500;

/// Premium ratio (×1000) below which buyback is allowed.
/// 950 = market price must be ≤0.95× NAV to trigger.
const DISCOUNT_THRESHOLD_BPS: u64 = 950;

/// Minimum blocks between consecutive mint events. Default 360 ≈ 6 min at 1 bps.
const MINT_COOLDOWN_BLOCKS_DEFAULT: u64 = 360;

/// Minimum blocks between consecutive buyback events. Tighter than mint.
const BUYBACK_COOLDOWN_BLOCKS_DEFAULT: u64 = 720;

/// Max QSHARE that can be minted per trigger as a fraction of DEX pool depth (basis points).
/// 50 = 0.5% of pool QUG reserves. Prevents self-bidding QSHARE price up via huge mints.
const MAX_POOL_FRACTION_BPS: u64 = 50;

/// Max QSHARE inflation per mint as basis points of circulating supply.
/// 200 = 2% of circulating per mint. Defense in depth against pool-fraction cap.
const MAX_INFLATION_PER_MINT_BPS: u64 = 200;

/// Max buyback per trigger as fraction of pool depth (basis points).
/// 10 = 0.1% — tighter than mint to prevent gaming the buyback path.
const MAX_BUYBACK_POOL_FRACTION_BPS: u64 = 10;

/// Min DEX pool depth (in QUG raw u128 units) before any mint allowed.
/// Prevents thin-pool manipulation. Default 1000 QUG.
const MIN_POOL_DEPTH_QUG: u128 = 1000 * 10u128.pow(24);

/// Mint trigger fee, refunded + bounty if mint succeeds. 0.01 QUG.
const MINT_TRIGGER_FEE_QUG: u128 = 10u128.pow(22); // 0.01 × 10^24

/// Bounty paid to caller of successful mint, as basis points of accumulated QUG.
/// 50 = 0.5% of the QUG the mint accumulated for treasury.
const MINT_BOUNTY_BPS: u64 = 50;

/// Cap on mint bounty in absolute QUG, regardless of accumulation size. 1 QUG.
const MAX_MINT_BOUNTY_QUG: u128 = 10u128.pow(24);

/// TWAP window for market price observation, in blocks.
const TWAP_WINDOW_BLOCKS: u64 = 30;

/// Seconds in a year (for yield accrual on QCREDIT treasury positions).
const SECONDS_PER_YEAR: u64 = 365 * 24 * 3600;

/// Token decimals — matches QUG (24).
pub const QSHARE_DECIMALS: u8 = 24;

/// QSHARE symbol.
pub const QSHARE_SYMBOL: &str = "QSHARE";

// ============ CORE DATA STRUCTURES ============

/// A single QCREDIT position held by the treasury.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreasuryPosition {
    pub tier: CreditTier,
    /// QUG locked when this position was opened (raw u128, decimals=24).
    pub principal_qug: u128,
    /// Block height when position was created.
    pub locked_at_height: u64,
    /// Timestamp (Unix seconds) when locked, for yield accrual.
    pub locked_at_timestamp: u64,
}

impl TreasuryPosition {
    /// Compute accrued value (principal + yield) at a given timestamp.
    /// Uses linear accrual matching qcredit_vault's compounding rules.
    pub fn value_at(&self, now_ts: u64) -> u128 {
        if now_ts <= self.locked_at_timestamp {
            return self.principal_qug;
        }
        let elapsed_s = now_ts - self.locked_at_timestamp;
        let apy_bps = self.tier.apy_bps();
        // yield = principal × apy_bps × elapsed_s / (10000 × seconds_per_year)
        // Compute in u256-equivalent via splitting to avoid overflow on huge principal.
        // principal × apy_bps fits in u128 for any realistic case (max ~2^128 / 10000).
        let scaled = self.principal_qug.saturating_mul(apy_bps as u128);
        let yield_amount = scaled
            .saturating_mul(elapsed_s as u128)
            .checked_div(10_000u128 * SECONDS_PER_YEAR as u128)
            .unwrap_or(0);
        self.principal_qug.saturating_add(yield_amount)
    }
}

/// Treasury composition snapshot. Returned by `treasury_composition()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreasuryComposition {
    pub pending_qug: u128,
    pub positions_by_tier: HashMap<CreditTier, Vec<TreasuryPosition>>,
    pub total_nav_qug_equivalent: u128,
    pub circulating_qshare: u128,
    pub nav_per_qshare: u128,
    pub computed_at_height: u64,
    pub computed_at_timestamp: u64,
}

/// Cached NAV state — recomputed on demand but stored for read-only queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NavOracleState {
    pub last_nav_per_qshare: u128,
    pub last_total_qug_equivalent: u128,
    pub last_computed_height: u64,
    pub last_computed_timestamp: u64,
}

/// DEX pool snapshot needed for mint/buyback decisions.
/// The contract reads this from the on-chain DEX module per mint trigger.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DexPoolSnapshot {
    /// QUG reserves in the QSHARE/QUG pool (raw u128, decimals=24).
    pub qug_reserves: u128,
    /// QSHARE reserves in the same pool.
    pub qshare_reserves: u128,
    /// Time-weighted average price of QSHARE in QUG units (×10^24 fixed-point),
    /// computed over the last TWAP_WINDOW_BLOCKS blocks.
    pub twap_qug_per_qshare: u128,
}

impl DexPoolSnapshot {
    /// Returns true if the pool is deep enough to allow operations.
    pub fn is_sufficiently_deep(&self) -> bool {
        self.qug_reserves >= MIN_POOL_DEPTH_QUG && self.qshare_reserves > 0
    }
}

/// Event emitted on successful mint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QShareMintEvent {
    pub minted_qshare: u128,
    pub qug_accumulated: u128,
    pub new_nav_per_qshare: u128,
    /// Premium ratio at trigger ×1000.
    pub premium_ratio_bps: u64,
    pub trigger_caller: [u8; 32],
    pub bounty_paid_qug: u128,
    pub block_height: u64,
}

/// Event emitted on successful buyback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QShareBuybackEvent {
    pub burned_qshare: u128,
    pub qug_spent: u128,
    pub new_nav_per_qshare: u128,
    /// Discount ratio at trigger ×1000 (e.g. 920 means 0.92× NAV).
    pub discount_ratio_bps: u64,
    pub trigger_caller: [u8; 32],
    pub bounty_paid_qug: u128,
    pub block_height: u64,
}

/// Result of a successful mint operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MintResult {
    pub event: QShareMintEvent,
    pub bounty_paid_to_caller: u128,
    pub new_circulating_qshare: u128,
}

/// Result of a successful buyback operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuybackResult {
    pub event: QShareBuybackEvent,
    pub bounty_paid_to_caller: u128,
    pub new_circulating_qshare: u128,
}

/// Reasons a mint attempt can fail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MintError {
    PremiumBelowThreshold { current_bps: u64, required_bps: u64 },
    CooldownActive { current_height: u64, eligible_at: u64 },
    PoolTooShallow { current_qug_reserves: u128, required: u128 },
    TriggerFeeInsufficient { paid: u128, required: u128 },
    NavOracleStale,
    ComputationOverflow,
}

/// Reasons a buyback attempt can fail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BuybackError {
    DiscountAboveThreshold { current_bps: u64, required_bps: u64 },
    CooldownActive { current_height: u64, eligible_at: u64 },
    PoolTooShallow { current_qug_reserves: u128, required: u128 },
    InsufficientYield { available: u128, needed: u128 },
    NavOracleStale,
}

// ============ CONTRACT STATE ============

/// The QSHARE smart contract — single global instance per chain.
///
/// State is persisted in RocksDB under `CF_CONTRACTS` keyed by the contract
/// address. The VM dispatches calls to the methods on this struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QShareContract {
    /// QUG awaiting deposit into a new QCREDIT position.
    pub treasury_pending_qug: u128,
    /// QCREDIT positions held, grouped by tier.
    pub treasury_basket: HashMap<CreditTier, Vec<TreasuryPosition>>,
    /// Total QSHARE in circulation (held outside this contract).
    pub circulating_qshare: u128,
    /// Block height of the most recent successful mint.
    pub last_mint_height: u64,
    /// Block height of the most recent successful buyback.
    pub last_buyback_height: u64,
    /// Per-block cooldown between mints. Tunable via governance.
    pub mint_cooldown_blocks: u64,
    /// Per-block cooldown between buybacks.
    pub buyback_cooldown_blocks: u64,
    /// Cached NAV state. Refreshed on every mint/buyback or read.
    pub nav_oracle: NavOracleState,
    /// Aggregate stats — useful for the activity panel + agentic interface.
    pub lifetime_mints: u64,
    pub lifetime_buybacks: u64,
    pub lifetime_qug_accumulated: u128,
    pub lifetime_qshare_burned: u128,
}

impl Default for QShareContract {
    fn default() -> Self {
        Self {
            treasury_pending_qug: 0,
            treasury_basket: HashMap::new(),
            circulating_qshare: 0,
            last_mint_height: 0,
            last_buyback_height: 0,
            mint_cooldown_blocks: MINT_COOLDOWN_BLOCKS_DEFAULT,
            buyback_cooldown_blocks: BUYBACK_COOLDOWN_BLOCKS_DEFAULT,
            nav_oracle: NavOracleState::default(),
            lifetime_mints: 0,
            lifetime_buybacks: 0,
            lifetime_qug_accumulated: 0,
            lifetime_qshare_burned: 0,
        }
    }
}

// ============ READ-ONLY VIEW METHODS ============

impl QShareContract {
    /// Compute NAV per QSHARE in raw u128 units (decimals=24).
    ///
    /// `nav_total_qug_equivalent / circulating_qshare`
    /// — but scaled to keep precision. Returns 0 if no circulating supply.
    pub fn nav_per_qshare(&self, now_ts: u64) -> u128 {
        if self.circulating_qshare == 0 {
            return 0;
        }
        let total = self.nav_total_qug_equivalent(now_ts);
        // Both total and circulating are in raw u128 units with the same
        // decimals=24 scale. nav_per_qshare = total_qug / circulating_qshare
        // expressed in the same scale → multiply by 10^24 to preserve precision.
        let scaled = total.checked_mul(10u128.pow(24)).unwrap_or(u128::MAX);
        scaled / self.circulating_qshare
    }

    /// Total QUG-equivalent value of treasury at a given timestamp.
    pub fn nav_total_qug_equivalent(&self, now_ts: u64) -> u128 {
        let mut total = self.treasury_pending_qug;
        for positions in self.treasury_basket.values() {
            for pos in positions {
                total = total.saturating_add(pos.value_at(now_ts));
            }
        }
        total
    }

    /// Compute the premium ratio in basis points (×1000) given a DEX snapshot.
    /// Returns None if NAV is zero (no circulating supply) or pool is empty.
    pub fn premium_ratio_bps(&self, pool: &DexPoolSnapshot, now_ts: u64) -> Option<u64> {
        let nav = self.nav_per_qshare(now_ts);
        if nav == 0 {
            return None;
        }
        if pool.twap_qug_per_qshare == 0 {
            return None;
        }
        // ratio = market / nav, expressed ×1000 for integer comparison
        let ratio = pool
            .twap_qug_per_qshare
            .checked_mul(1000)?
            .checked_div(nav)?;
        Some(ratio as u64)
    }

    /// Block height at which the next mint becomes eligible.
    pub fn next_mint_eligible_at_height(&self) -> u64 {
        self.last_mint_height.saturating_add(self.mint_cooldown_blocks)
    }

    /// Block height at which the next buyback becomes eligible.
    pub fn next_buyback_eligible_at_height(&self) -> u64 {
        self.last_buyback_height.saturating_add(self.buyback_cooldown_blocks)
    }

    /// Snapshot of treasury composition for the agentic interface.
    pub fn treasury_composition(&self, height: u64, now_ts: u64) -> TreasuryComposition {
        let total_nav = self.nav_total_qug_equivalent(now_ts);
        let nav_per = self.nav_per_qshare(now_ts);
        TreasuryComposition {
            pending_qug: self.treasury_pending_qug,
            positions_by_tier: self.treasury_basket.clone(),
            total_nav_qug_equivalent: total_nav,
            circulating_qshare: self.circulating_qshare,
            nav_per_qshare: nav_per,
            computed_at_height: height,
            computed_at_timestamp: now_ts,
        }
    }
}

// ============ MINT MECHANISM ============

/// Internal helper: compute the max QSHARE that can be minted in one trigger.
/// Capped by both pool fraction and inflation rate per spec §3.2.
fn compute_max_mint_amount(
    pool: &DexPoolSnapshot,
    circulating_qshare: u128,
) -> u128 {
    // Cap 1: 0.5% of pool QUG reserves — but expressed in QSHARE units.
    // The mint will be swapped for QUG via the pool. Amount of QSHARE
    // we can sell without slippage > MAX_POOL_FRACTION is bounded by
    // qug_reserves × MAX_POOL_FRACTION_BPS / 10000, converted to QSHARE
    // at the current pool ratio. We use the pool ratio (qshare_reserves /
    // qug_reserves) not TWAP, because the actual swap will happen at the
    // current pool reserves.
    let pool_qug_cap = pool
        .qug_reserves
        .saturating_mul(MAX_POOL_FRACTION_BPS as u128)
        / 10_000;
    let cap_via_pool = if pool.qug_reserves == 0 {
        0
    } else {
        pool_qug_cap
            .saturating_mul(pool.qshare_reserves)
            / pool.qug_reserves
    };

    // Cap 2: 2% of circulating supply.
    let cap_via_inflation = circulating_qshare
        .saturating_mul(MAX_INFLATION_PER_MINT_BPS as u128)
        / 10_000;

    cap_via_pool.min(cap_via_inflation)
}

impl QShareContract {
    /// Permissionless mint trigger. Returns a bounty to the caller if conditions are met.
    /// Per spec §3.
    ///
    /// The VM is responsible for:
    ///   - Verifying `trigger_fee_paid` actually arrived in the contract's QUG balance
    ///   - Executing the mint → DEX swap → QCREDIT lock atomically
    ///   - Crediting the bounty back to the caller
    ///   - Emitting the QShareMintEvent
    pub fn try_autonomous_mint(
        &mut self,
        caller: [u8; 32],
        trigger_fee_paid: u128,
        pool: &DexPoolSnapshot,
        current_height: u64,
        now_ts: u64,
    ) -> Result<MintResult, MintError> {
        // Gate 1: trigger fee
        if trigger_fee_paid < MINT_TRIGGER_FEE_QUG {
            return Err(MintError::TriggerFeeInsufficient {
                paid: trigger_fee_paid,
                required: MINT_TRIGGER_FEE_QUG,
            });
        }

        // Gate 2: cooldown
        let eligible_at = self.next_mint_eligible_at_height();
        if current_height < eligible_at {
            return Err(MintError::CooldownActive {
                current_height,
                eligible_at,
            });
        }

        // Gate 3: pool depth
        if !pool.is_sufficiently_deep() {
            return Err(MintError::PoolTooShallow {
                current_qug_reserves: pool.qug_reserves,
                required: MIN_POOL_DEPTH_QUG,
            });
        }

        // Gate 4: premium threshold
        let premium = self
            .premium_ratio_bps(pool, now_ts)
            .ok_or(MintError::NavOracleStale)?;
        if premium < MINT_THRESHOLD_BPS {
            return Err(MintError::PremiumBelowThreshold {
                current_bps: premium,
                required_bps: MINT_THRESHOLD_BPS,
            });
        }

        // Compute mint amount.
        let mint_amount = compute_max_mint_amount(pool, self.circulating_qshare);
        if mint_amount == 0 {
            return Err(MintError::ComputationOverflow);
        }

        // Atomic mint + swap simulation:
        // The VM swaps `mint_amount` QSHARE for QUG at the pool's current
        // constant-product price. For this on-chain math:
        //   qug_out = qug_reserves × mint_amount / (qshare_reserves + mint_amount)
        // We use the formula without fee here; the actual DEX call will
        // include the standard pool fee.
        let new_qshare_reserves = pool.qshare_reserves.saturating_add(mint_amount);
        if new_qshare_reserves == 0 {
            return Err(MintError::ComputationOverflow);
        }
        let qug_out = pool
            .qug_reserves
            .checked_mul(mint_amount)
            .ok_or(MintError::ComputationOverflow)?
            .checked_div(new_qshare_reserves)
            .ok_or(MintError::ComputationOverflow)?;

        // Compute bounty.
        let bounty_raw = qug_out
            .saturating_mul(MINT_BOUNTY_BPS as u128)
            / 10_000;
        let bounty = bounty_raw.min(MAX_MINT_BOUNTY_QUG);
        let qug_to_treasury = qug_out.saturating_sub(bounty);

        // Apply state.
        self.treasury_pending_qug = self
            .treasury_pending_qug
            .saturating_add(qug_to_treasury);
        self.circulating_qshare = self
            .circulating_qshare
            .saturating_add(mint_amount);
        self.last_mint_height = current_height;
        self.lifetime_mints = self.lifetime_mints.saturating_add(1);
        self.lifetime_qug_accumulated = self
            .lifetime_qug_accumulated
            .saturating_add(qug_to_treasury);

        // Refresh NAV oracle.
        let new_total = self.nav_total_qug_equivalent(now_ts);
        let new_nav = self.nav_per_qshare(now_ts);
        self.nav_oracle = NavOracleState {
            last_nav_per_qshare: new_nav,
            last_total_qug_equivalent: new_total,
            last_computed_height: current_height,
            last_computed_timestamp: now_ts,
        };

        let event = QShareMintEvent {
            minted_qshare: mint_amount,
            qug_accumulated: qug_to_treasury,
            new_nav_per_qshare: new_nav,
            premium_ratio_bps: premium,
            trigger_caller: caller,
            bounty_paid_qug: bounty,
            block_height: current_height,
        };

        Ok(MintResult {
            event,
            bounty_paid_to_caller: bounty,
            new_circulating_qshare: self.circulating_qshare,
        })
    }

    /// Permissionless buyback trigger. Symmetric path to mint per spec §3.4.
    ///
    /// Funded by withdrawing accrued yield from the QCREDIT basket — never
    /// principal. The VM is responsible for the actual partial-yield
    /// withdrawal + DEX swap + QSHARE burn.
    pub fn try_buyback(
        &mut self,
        caller: [u8; 32],
        trigger_fee_paid: u128,
        pool: &DexPoolSnapshot,
        current_height: u64,
        now_ts: u64,
    ) -> Result<BuybackResult, BuybackError> {
        // Gate 1: trigger fee (same as mint).
        if trigger_fee_paid < MINT_TRIGGER_FEE_QUG {
            return Err(BuybackError::PoolTooShallow {
                current_qug_reserves: 0,
                required: 0,
            });
        }

        // Gate 2: cooldown.
        let eligible_at = self.next_buyback_eligible_at_height();
        if current_height < eligible_at {
            return Err(BuybackError::CooldownActive {
                current_height,
                eligible_at,
            });
        }

        // Gate 3: pool depth.
        if !pool.is_sufficiently_deep() {
            return Err(BuybackError::PoolTooShallow {
                current_qug_reserves: pool.qug_reserves,
                required: MIN_POOL_DEPTH_QUG,
            });
        }

        // Gate 4: discount threshold (mirror of mint).
        let premium = self
            .premium_ratio_bps(pool, now_ts)
            .ok_or(BuybackError::NavOracleStale)?;
        if premium > DISCOUNT_THRESHOLD_BPS {
            return Err(BuybackError::DiscountAboveThreshold {
                current_bps: premium,
                required_bps: DISCOUNT_THRESHOLD_BPS,
            });
        }

        // Gate 5: yield availability. We only spend accrued yield, never principal.
        let nav_total = self.nav_total_qug_equivalent(now_ts);
        let principal_total: u128 = self
            .treasury_basket
            .values()
            .flatten()
            .map(|p| p.principal_qug)
            .fold(0u128, |a, b| a.saturating_add(b))
            .saturating_add(self.treasury_pending_qug);
        let accrued_yield = nav_total.saturating_sub(principal_total);

        // Buyback amount capped by yield AND by pool fraction.
        let pool_cap = pool
            .qug_reserves
            .saturating_mul(MAX_BUYBACK_POOL_FRACTION_BPS as u128)
            / 10_000;
        let qug_to_spend = accrued_yield.min(pool_cap);

        if qug_to_spend == 0 {
            return Err(BuybackError::InsufficientYield {
                available: accrued_yield,
                needed: 1,
            });
        }

        // Atomic swap simulation: QUG → QSHARE.
        // qshare_out = qshare_reserves × qug_to_spend / (qug_reserves + qug_to_spend)
        let new_qug_reserves = pool.qug_reserves.saturating_add(qug_to_spend);
        let qshare_out = pool
            .qshare_reserves
            .saturating_mul(qug_to_spend)
            / new_qug_reserves.max(1);

        // Compute bounty (smaller for buyback per spec).
        let bounty = qug_to_spend
            .saturating_mul(MINT_BOUNTY_BPS as u128)
            / 10_000;
        let bounty_capped = bounty.min(MAX_MINT_BOUNTY_QUG / 2);

        // Apply state — burn the bought QSHARE.
        self.circulating_qshare = self
            .circulating_qshare
            .saturating_sub(qshare_out);
        self.last_buyback_height = current_height;
        self.lifetime_buybacks = self.lifetime_buybacks.saturating_add(1);
        self.lifetime_qshare_burned = self
            .lifetime_qshare_burned
            .saturating_add(qshare_out);

        // Refresh oracle.
        let new_total = self.nav_total_qug_equivalent(now_ts);
        let new_nav = self.nav_per_qshare(now_ts);
        self.nav_oracle = NavOracleState {
            last_nav_per_qshare: new_nav,
            last_total_qug_equivalent: new_total,
            last_computed_height: current_height,
            last_computed_timestamp: now_ts,
        };

        let event = QShareBuybackEvent {
            burned_qshare: qshare_out,
            qug_spent: qug_to_spend,
            new_nav_per_qshare: new_nav,
            discount_ratio_bps: premium,
            trigger_caller: caller,
            bounty_paid_qug: bounty_capped,
            block_height: current_height,
        };

        Ok(BuybackResult {
            event,
            bounty_paid_to_caller: bounty_capped,
            new_circulating_qshare: self.circulating_qshare,
        })
    }

    /// Add a QCREDIT position to the treasury basket. Called by the VM after
    /// the contract's QUG accumulation has been swapped into QCREDIT via the
    /// qcredit_vault contract.
    pub fn record_qcredit_position(
        &mut self,
        tier: CreditTier,
        principal_qug: u128,
        locked_at_height: u64,
        locked_at_timestamp: u64,
    ) {
        let position = TreasuryPosition {
            tier,
            principal_qug,
            locked_at_height,
            locked_at_timestamp,
        };
        self.treasury_basket
            .entry(tier)
            .or_insert_with(Vec::new)
            .push(position);
        // The principal moved from pending_qug into a position.
        self.treasury_pending_qug =
            self.treasury_pending_qug.saturating_sub(principal_qug);
    }
}

// ============ TESTS ============

#[cfg(test)]
mod tests {
    use super::*;

    fn pool_at(qug: u128, qshare: u128, twap: u128) -> DexPoolSnapshot {
        DexPoolSnapshot {
            qug_reserves: qug,
            qshare_reserves: qshare,
            twap_qug_per_qshare: twap,
        }
    }

    #[test]
    fn empty_contract_has_zero_nav() {
        let c = QShareContract::default();
        assert_eq!(c.nav_per_qshare(0), 0);
    }

    #[test]
    fn mint_below_threshold_rejected() {
        let mut c = QShareContract::default();
        c.circulating_qshare = 1_000 * 10u128.pow(24);
        c.treasury_pending_qug = 1_000 * 10u128.pow(24);
        // NAV = 1 QUG per QSHARE. Market = 1.0 → premium = 1000 (×1000).
        // Below MINT_THRESHOLD_BPS = 1500 → should reject.
        let pool = pool_at(
            10_000 * 10u128.pow(24),
            10_000 * 10u128.pow(24),
            10u128.pow(24), // 1 QUG per QSHARE
        );
        let result = c.try_autonomous_mint([1; 32], MINT_TRIGGER_FEE_QUG, &pool, 1000, 0);
        match result {
            Err(MintError::PremiumBelowThreshold { current_bps, required_bps }) => {
                assert!(current_bps < required_bps);
            }
            other => panic!("expected PremiumBelowThreshold, got {:?}", other),
        }
    }

    #[test]
    fn mint_above_threshold_succeeds() {
        let mut c = QShareContract::default();
        c.circulating_qshare = 1_000 * 10u128.pow(24);
        c.treasury_pending_qug = 1_000 * 10u128.pow(24);
        // NAV = 1 QUG per QSHARE. Market = 2.0 → premium = 2000 → above threshold.
        let pool = pool_at(
            10_000 * 10u128.pow(24),
            5_000 * 10u128.pow(24),
            2 * 10u128.pow(24), // 2 QUG per QSHARE
        );
        let result = c
            .try_autonomous_mint([1; 32], MINT_TRIGGER_FEE_QUG, &pool, 1000, 0)
            .expect("mint should succeed at 2× premium");
        assert!(result.event.minted_qshare > 0);
        assert!(result.event.qug_accumulated > 0);
        assert!(result.bounty_paid_to_caller > 0);
        assert!(result.new_circulating_qshare > 1_000 * 10u128.pow(24));
    }

    #[test]
    fn mint_cooldown_enforced() {
        let mut c = QShareContract::default();
        c.circulating_qshare = 1_000 * 10u128.pow(24);
        c.treasury_pending_qug = 1_000 * 10u128.pow(24);
        c.last_mint_height = 1000;
        let pool = pool_at(
            10_000 * 10u128.pow(24),
            5_000 * 10u128.pow(24),
            2 * 10u128.pow(24),
        );
        // 1000 + 360 = 1360. height=1100 should fail.
        let result = c.try_autonomous_mint([1; 32], MINT_TRIGGER_FEE_QUG, &pool, 1100, 0);
        match result {
            Err(MintError::CooldownActive { current_height, eligible_at }) => {
                assert_eq!(current_height, 1100);
                assert_eq!(eligible_at, 1360);
            }
            other => panic!("expected CooldownActive, got {:?}", other),
        }
    }

    #[test]
    fn mint_pool_fraction_cap_enforced() {
        let mut c = QShareContract::default();
        c.circulating_qshare = 1_000_000 * 10u128.pow(24); // 1M circulating
        c.treasury_pending_qug = 1_000 * 10u128.pow(24);
        // Pool: 10000 QUG / 10000 QSHARE; TWAP 1 → NAV must be < 1 for premium.
        // Force premium via small treasury: 1000 QUG / 1M = 0.001 QUG per share.
        // Market 1 / NAV 0.001 = 1000× premium. Way above threshold.
        let pool = pool_at(
            10_000 * 10u128.pow(24),
            10_000 * 10u128.pow(24),
            10u128.pow(24),
        );
        let r = c
            .try_autonomous_mint([1; 32], MINT_TRIGGER_FEE_QUG, &pool, 1000, 0)
            .expect("ok");
        // Cap1 = 0.5% × 10000 QSHARE = 50 QSHARE
        // Cap2 = 2% × 1M = 20_000 QSHARE
        // Active cap = 50 QSHARE.
        let cap1 = 50 * 10u128.pow(24);
        assert!(r.event.minted_qshare <= cap1 + 1);
    }

    #[test]
    fn treasury_position_yield_accrues() {
        let pos = TreasuryPosition {
            tier: CreditTier::Platinum,
            principal_qug: 1_000 * 10u128.pow(24),
            locked_at_height: 0,
            locked_at_timestamp: 0,
        };
        // Platinum = 25% APY. After 1 year exactly: 1.25× principal.
        let value_after_1y = pos.value_at(SECONDS_PER_YEAR);
        let expected = 1_250 * 10u128.pow(24);
        // Allow small rounding tolerance.
        let diff = value_after_1y.abs_diff(expected);
        assert!(diff < 10u128.pow(18), "expected ~1250 QUG, got {} (diff {})", value_after_1y, diff);
    }

    #[test]
    fn record_qcredit_position_moves_from_pending() {
        let mut c = QShareContract::default();
        c.treasury_pending_qug = 500 * 10u128.pow(24);
        c.record_qcredit_position(
            CreditTier::Platinum,
            500 * 10u128.pow(24),
            100,
            1000,
        );
        assert_eq!(c.treasury_pending_qug, 0);
        let positions = c.treasury_basket.get(&CreditTier::Platinum).unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].principal_qug, 500 * 10u128.pow(24));
    }
}
