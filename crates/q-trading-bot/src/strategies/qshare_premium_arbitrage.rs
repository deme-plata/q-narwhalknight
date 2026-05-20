//! QSHARE Premium Arbitrage — Phase 2 of QSHARE-1 protocol.
//!
//! Watches the QSHARE/QUG market price vs on-chain NAV. When premium ratio
//! crosses MINT_THRESHOLD (default 1.5×), calls `try_autonomous_mint` on the
//! QShareContract to trigger a mint+swap+lock cycle and collect the bounty.
//!
//! Symmetric path: when premium drops below DISCOUNT_THRESHOLD (default 0.95×),
//! calls `try_buyback` to burn QSHARE using accrued treasury yield.
//!
//! This strategy is COOPERATIVE — the on-chain contract enforces cooldowns,
//! pool-fraction caps, and inflation caps. The bot just polls, decides
//! whether the call is profitable (bounty > gas cost), and submits. Any
//! agent or human can run an identical bot; the first to trigger after the
//! cooldown wins the bounty.
//!
//! Companion: `docs/standards/qshare-treasury-protocol-spec.md` §3 + §5.
//! Phase 1 (QCREDIT-DCA): `qcredit_dca.rs` — builds the treasury pool that
//! QSHARE backs against.
//!
//! Wire-up: this strategy posts one on-chain operation per cycle (when
//! conditions allow):
//!   1. Read on-chain state: nav_per_qshare, dex_pool_snapshot, last_mint_height
//!   2. Compute premium_ratio_bps locally (matches contract's view)
//!   3. If premium > MINT_THRESHOLD && height >= eligible: submit try_autonomous_mint
//!   4. Else if premium < DISCOUNT_THRESHOLD && height >= eligible: submit try_buyback
//!   5. Otherwise sleep until next cycle

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

// ============ STRATEGY PARAMETERS ============

/// Default polling interval — how often to check the premium ratio.
/// 30 seconds is half the smallest cooldown (mint cooldown = 360 blocks = ~6 min).
const DEFAULT_POLL_INTERVAL_SECS: u64 = 30;

/// Minimum expected bounty (in raw QUG units, decimals=24) before
/// we'll attempt a trigger. Below this, gas isn't worth the call.
/// 0.001 QUG = 10^21 raw units.
const MIN_PROFITABLE_BOUNTY_QUG: u128 = 10u128.pow(21);

/// Premium threshold to attempt mint (basis points × 1000).
/// 1500 = 1.5× premium ratio. Matches QShareContract::MINT_THRESHOLD_BPS.
const MINT_THRESHOLD_BPS: u64 = 1500;

/// Discount threshold to attempt buyback (basis points × 1000).
/// 950 = 0.95× discount ratio.
const DISCOUNT_THRESHOLD_BPS: u64 = 950;

// ============ STRATEGY CONFIG ============

/// Configuration for the QSHARE arbitrage strategy.
/// Loaded from CLI flags or env vars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QSharePremiumConfig {
    /// Wallet to use for mint/buyback trigger fee + bounty receipt.
    pub wallet_address: String,
    /// API endpoint to query QSHARE contract state.
    pub api_endpoint: String,
    /// How often to poll (seconds).
    pub poll_interval_secs: u64,
    /// Min profitable bounty to attempt trigger. 0 = always try.
    pub min_profitable_bounty_qug: u128,
    /// Stop after this many successful triggers (0 = run forever).
    pub max_triggers: u64,
    /// Dry-run mode — log what we'd do, don't actually submit.
    pub dry_run: bool,
}

impl Default for QSharePremiumConfig {
    fn default() -> Self {
        Self {
            wallet_address: String::new(),
            api_endpoint: "https://quillon.xyz".to_string(),
            poll_interval_secs: DEFAULT_POLL_INTERVAL_SECS,
            min_profitable_bounty_qug: MIN_PROFITABLE_BOUNTY_QUG,
            max_triggers: 0,
            dry_run: false,
        }
    }
}

// ============ ON-CHAIN STATE SNAPSHOT ============

/// What the bot needs to know each cycle. Fetched from the chain via REST API.
#[derive(Debug, Clone, Deserialize)]
pub struct QShareStateSnapshot {
    pub nav_per_qshare: u128,
    pub market_price_twap: u128,
    pub premium_ratio_bps: u64,
    pub circulating_qshare: u128,
    pub dex_pool_qug_reserves: u128,
    pub dex_pool_qshare_reserves: u128,
    pub last_mint_height: u64,
    pub last_buyback_height: u64,
    pub current_height: u64,
    pub mint_cooldown_blocks: u64,
    pub buyback_cooldown_blocks: u64,
    pub estimated_mint_bounty_qug: u128,
    pub estimated_buyback_bounty_qug: u128,
}

impl QShareStateSnapshot {
    pub fn mint_eligible_now(&self) -> bool {
        self.current_height >= self.last_mint_height + self.mint_cooldown_blocks
    }

    pub fn buyback_eligible_now(&self) -> bool {
        self.current_height >= self.last_buyback_height + self.buyback_cooldown_blocks
    }
}

// ============ STRATEGY LOOP ============

/// Decision a cycle can make.
#[derive(Debug, Clone, PartialEq)]
pub enum CycleDecision {
    /// Conditions met — attempt mint trigger.
    AttemptMint { expected_bounty_qug: u128 },
    /// Conditions met — attempt buyback trigger.
    AttemptBuyback { expected_bounty_qug: u128 },
    /// No action — premium between thresholds.
    Neutral { premium_ratio_bps: u64 },
    /// No action — cooldown active.
    Cooldown { eligible_at_height: u64 },
    /// No action — pool too shallow.
    PoolTooShallow,
    /// No action — bounty below minimum profitable.
    UnprofitableBounty { bounty: u128, minimum: u128 },
}

/// Pure decision function: given on-chain state + config, what should we do?
/// No side effects — testable in isolation.
pub fn decide_cycle(
    state: &QShareStateSnapshot,
    config: &QSharePremiumConfig,
) -> CycleDecision {
    // Pool depth gate — both directions need it.
    const MIN_POOL_DEPTH_QUG: u128 = 1000 * 10u128.pow(24);
    if state.dex_pool_qug_reserves < MIN_POOL_DEPTH_QUG {
        return CycleDecision::PoolTooShallow;
    }

    // Mint path
    if state.premium_ratio_bps >= MINT_THRESHOLD_BPS {
        if !state.mint_eligible_now() {
            return CycleDecision::Cooldown {
                eligible_at_height: state.last_mint_height + state.mint_cooldown_blocks,
            };
        }
        if state.estimated_mint_bounty_qug < config.min_profitable_bounty_qug {
            return CycleDecision::UnprofitableBounty {
                bounty: state.estimated_mint_bounty_qug,
                minimum: config.min_profitable_bounty_qug,
            };
        }
        return CycleDecision::AttemptMint {
            expected_bounty_qug: state.estimated_mint_bounty_qug,
        };
    }

    // Buyback path
    if state.premium_ratio_bps <= DISCOUNT_THRESHOLD_BPS {
        if !state.buyback_eligible_now() {
            return CycleDecision::Cooldown {
                eligible_at_height: state.last_buyback_height + state.buyback_cooldown_blocks,
            };
        }
        if state.estimated_buyback_bounty_qug < config.min_profitable_bounty_qug {
            return CycleDecision::UnprofitableBounty {
                bounty: state.estimated_buyback_bounty_qug,
                minimum: config.min_profitable_bounty_qug,
            };
        }
        return CycleDecision::AttemptBuyback {
            expected_bounty_qug: state.estimated_buyback_bounty_qug,
        };
    }

    // Neutral zone — no action.
    CycleDecision::Neutral {
        premium_ratio_bps: state.premium_ratio_bps,
    }
}

/// Stub for the actual on-chain submission path. Returns Ok(tx_hash) in
/// production. In Phase 2 dev, leaves as TODO so we can wire to the
/// signed-submit endpoint when the REST API exposes try_autonomous_mint.
pub async fn submit_mint_trigger(
    _config: &QSharePremiumConfig,
    _expected_bounty: u128,
) -> Result<String> {
    if _config.dry_run {
        return Ok("dry-run-mint-tx-hash".to_string());
    }
    Err(anyhow!(
        "submit_mint_trigger: not yet wired to /api/v1/qshare/try_mint_signed \
         — see docs/standards/qshare-treasury-protocol-spec.md §5 for the endpoint shape"
    ))
}

pub async fn submit_buyback_trigger(
    _config: &QSharePremiumConfig,
    _expected_bounty: u128,
) -> Result<String> {
    if _config.dry_run {
        return Ok("dry-run-buyback-tx-hash".to_string());
    }
    Err(anyhow!(
        "submit_buyback_trigger: not yet wired to /api/v1/qshare/try_buyback_signed"
    ))
}

/// Fetch the on-chain QSHARE state. Stub — would call REST GET /api/v1/qshare/state.
pub async fn fetch_qshare_state(_config: &QSharePremiumConfig) -> Result<QShareStateSnapshot> {
    Err(anyhow!(
        "fetch_qshare_state: not yet wired to /api/v1/qshare/state \
         — see docs/standards/qshare-treasury-protocol-spec.md §5 for the endpoint"
    ))
}

/// Main strategy loop — Phase 2 scaffold.
/// Returns Ok(total_triggers_attempted) when max_triggers reached or external stop.
pub async fn run_qshare_premium_arbitrage(config: QSharePremiumConfig) -> Result<u64> {
    let mut triggers_attempted = 0u64;
    let start = Instant::now();

    loop {
        // Fetch on-chain state.
        let state = match fetch_qshare_state(&config).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(target: "qshare", "fetch state failed: {}", e);
                tokio::time::sleep(Duration::from_secs(config.poll_interval_secs)).await;
                continue;
            }
        };

        // Decide.
        let decision = decide_cycle(&state, &config);

        match decision {
            CycleDecision::AttemptMint { expected_bounty_qug } => {
                tracing::info!(
                    target: "qshare",
                    "MINT trigger: premium={}.{}× expected bounty={} QUG raw",
                    state.premium_ratio_bps / 1000,
                    state.premium_ratio_bps % 1000,
                    expected_bounty_qug
                );
                match submit_mint_trigger(&config, expected_bounty_qug).await {
                    Ok(tx) => {
                        tracing::info!(target: "qshare", "MINT submitted: {}", tx);
                        triggers_attempted += 1;
                    }
                    Err(e) => tracing::error!(target: "qshare", "MINT submit failed: {}", e),
                }
            }
            CycleDecision::AttemptBuyback { expected_bounty_qug } => {
                tracing::info!(
                    target: "qshare",
                    "BUYBACK trigger: discount={}.{}× expected bounty={} QUG raw",
                    state.premium_ratio_bps / 1000,
                    state.premium_ratio_bps % 1000,
                    expected_bounty_qug
                );
                match submit_buyback_trigger(&config, expected_bounty_qug).await {
                    Ok(tx) => {
                        tracing::info!(target: "qshare", "BUYBACK submitted: {}", tx);
                        triggers_attempted += 1;
                    }
                    Err(e) => tracing::error!(target: "qshare", "BUYBACK submit failed: {}", e),
                }
            }
            CycleDecision::Neutral { premium_ratio_bps } => {
                tracing::debug!(
                    target: "qshare",
                    "neutral zone — premium {} bps, no action",
                    premium_ratio_bps
                );
            }
            CycleDecision::Cooldown { eligible_at_height } => {
                tracing::debug!(
                    target: "qshare",
                    "cooldown — eligible at height {}, current {}",
                    eligible_at_height,
                    state.current_height
                );
            }
            CycleDecision::PoolTooShallow => {
                tracing::warn!(target: "qshare", "DEX pool too shallow — pool needs deposits");
            }
            CycleDecision::UnprofitableBounty { bounty, minimum } => {
                tracing::debug!(
                    target: "qshare",
                    "bounty {} below profitable threshold {} — skipping",
                    bounty,
                    minimum
                );
            }
        }

        // Termination check.
        if config.max_triggers > 0 && triggers_attempted >= config.max_triggers {
            tracing::info!(
                target: "qshare",
                "reached max_triggers={}, exiting after {:?}",
                config.max_triggers,
                start.elapsed()
            );
            break;
        }

        tokio::time::sleep(Duration::from_secs(config.poll_interval_secs)).await;
    }

    Ok(triggers_attempted)
}

// ============ TESTS ============

#[cfg(test)]
mod tests {
    use super::*;

    fn snap_premium(premium_bps: u64) -> QShareStateSnapshot {
        QShareStateSnapshot {
            nav_per_qshare: 10u128.pow(24),
            market_price_twap: 10u128.pow(24),
            premium_ratio_bps: premium_bps,
            circulating_qshare: 1_000 * 10u128.pow(24),
            dex_pool_qug_reserves: 10_000 * 10u128.pow(24),
            dex_pool_qshare_reserves: 10_000 * 10u128.pow(24),
            last_mint_height: 0,
            last_buyback_height: 0,
            current_height: 1000,
            mint_cooldown_blocks: 360,
            buyback_cooldown_blocks: 720,
            estimated_mint_bounty_qug: 10u128.pow(22), // 0.01 QUG — profitable
            estimated_buyback_bounty_qug: 10u128.pow(22),
        }
    }

    #[test]
    fn neutral_when_premium_inside_band() {
        let state = snap_premium(1200); // 1.2× — between discount (0.95) and mint (1.5)
        let decision = decide_cycle(&state, &QSharePremiumConfig::default());
        assert!(matches!(decision, CycleDecision::Neutral { .. }));
    }

    #[test]
    fn mint_when_premium_exceeds_threshold() {
        let state = snap_premium(1600); // 1.6×
        let decision = decide_cycle(&state, &QSharePremiumConfig::default());
        assert!(matches!(decision, CycleDecision::AttemptMint { .. }));
    }

    #[test]
    fn buyback_when_discount_exceeds_threshold() {
        let state = snap_premium(900); // 0.9× discount
        let decision = decide_cycle(&state, &QSharePremiumConfig::default());
        assert!(matches!(decision, CycleDecision::AttemptBuyback { .. }));
    }

    #[test]
    fn cooldown_blocks_mint_even_at_premium() {
        let mut state = snap_premium(1600);
        state.last_mint_height = 800;
        // 800 + 360 = 1160. Current = 1000 < 1160 → cooldown.
        let decision = decide_cycle(&state, &QSharePremiumConfig::default());
        assert!(matches!(decision, CycleDecision::Cooldown { .. }));
    }

    #[test]
    fn pool_too_shallow_blocks_action() {
        let mut state = snap_premium(1600);
        state.dex_pool_qug_reserves = 100 * 10u128.pow(24); // way under 1000 QUG min
        let decision = decide_cycle(&state, &QSharePremiumConfig::default());
        assert_eq!(decision, CycleDecision::PoolTooShallow);
    }

    #[test]
    fn unprofitable_bounty_skips_trigger() {
        let mut state = snap_premium(1600);
        state.estimated_mint_bounty_qug = 10u128.pow(18); // 0.000001 QUG, below profit threshold
        let decision = decide_cycle(&state, &QSharePremiumConfig::default());
        assert!(matches!(decision, CycleDecision::UnprofitableBounty { .. }));
    }
}
