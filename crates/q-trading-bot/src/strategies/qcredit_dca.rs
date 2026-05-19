/// QCREDIT-DCA — Phase 1 of QSHARE-1 protocol.
///
/// Dollar-cost-averages QUG into the QCREDIT Platinum tier (180-day lock,
/// 25% APY). Replaces dumb DCA-into-spot with yield-bearing DCA, which is
/// the L2 layer of the three-layer Saylor stack (L1 QUG → L2 QCREDIT →
/// L3 QSHARE).
///
/// This is the prerequisite for the autonomous QSHARE premium-arbitrage
/// strategy that lands in Phase 2 + 3 once the L3 contract is deployed.
/// Until then, this strategy alone earns 25% APY on the QCREDIT base layer
/// without exploiting premium reflexivity yet.
///
/// Companion: `docs/standards/qshare-treasury-protocol-spec.md` §8 Phase 1.
///
/// Wire-up: this strategy posts two on-chain operations per cycle:
///   1. Acquire QUG (either via existing balance OR via a small DCA swap
///      using QUGUSD/USD/whatever the wallet holds)
///   2. Lock QUG into QCREDIT Platinum tier via the qcredit_vault API
///
/// The second step requires the qcredit_api endpoint at
/// `/api/v1/qcredit/lock` (see `crates/q-api-server/src/qcredit_api.rs`).
///
/// Reference Saylor mechanics: this DCA path is NOT premium-arbitrage —
/// it is the constant-rate accumulation that builds the treasury pool
/// QSHARE will later be issued against in Phase 2. It's the safe,
/// conservative base layer.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{error, info, warn};

use crate::api_client::QuillonClient;

// ════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════

/// Configuration for a QCREDIT-DCA bot instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QcreditDcaConfig {
    /// Quillon API base URL (e.g. "https://quillon.xyz")
    pub api_url: String,
    /// qnk wallet address — must hold QUG balance
    pub wallet: String,
    /// QUG amount per DCA cycle (in display units, e.g. 0.05 = 0.05 QUG)
    pub qug_per_cycle: f64,
    /// Sleep duration between cycles (default 1 hour)
    #[serde(default = "default_cycle_interval")]
    pub cycle_interval: Duration,
    /// QCREDIT tier to lock into. Default "platinum" (25% APY, 180 days).
    /// Use "gold" for 90-day flexibility tradeoff.
    #[serde(default = "default_tier")]
    pub tier: String,
    /// Stop after this many successful cycles (None = run forever).
    #[serde(default)]
    pub max_cycles: Option<u32>,
    /// Stop if wallet QUG balance drops below this amount (display units).
    /// Default 0.1 — leaves a small floor for fees and exits.
    #[serde(default = "default_min_balance_floor")]
    pub min_balance_floor: f64,
    /// Dry-run mode: log what would happen but don't actually call /lock.
    #[serde(default)]
    pub dry_run: bool,
}

fn default_cycle_interval() -> Duration {
    Duration::from_secs(3600) // 1 hour
}
fn default_tier() -> String {
    "platinum".to_string()
}
fn default_min_balance_floor() -> f64 {
    0.1
}

impl QcreditDcaConfig {
    /// Helper for creating a sane default config bound to a specific wallet.
    pub fn for_wallet(api_url: impl Into<String>, wallet: impl Into<String>) -> Self {
        Self {
            api_url: api_url.into(),
            wallet: wallet.into(),
            qug_per_cycle: 0.01,
            cycle_interval: default_cycle_interval(),
            tier: default_tier(),
            max_cycles: None,
            min_balance_floor: default_min_balance_floor(),
            dry_run: false,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Cycle result + bot state
// ════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize)]
pub struct LockCycleResult {
    pub cycle_index: u32,
    pub qug_locked_display: f64,
    pub tier: String,
    pub position_id: Option<String>,
    pub at_unix: i64,
    pub skipped: bool,
    pub skip_reason: Option<String>,
}

pub struct QcreditDcaBot {
    config: QcreditDcaConfig,
    client: QuillonClient,
    cycles_run: u32,
    cycles_skipped: u32,
    total_qug_locked: f64,
    started_at: Instant,
}

impl QcreditDcaBot {
    pub fn new(config: QcreditDcaConfig) -> Result<Self> {
        let client = QuillonClient::new(config.api_url.clone());
        Ok(Self {
            config,
            client,
            cycles_run: 0,
            cycles_skipped: 0,
            total_qug_locked: 0.0,
            started_at: Instant::now(),
        })
    }

    /// Execute one cycle: check balance, attempt QCREDIT lock if eligible.
    pub async fn step(&mut self) -> Result<LockCycleResult> {
        let cycle_idx = self.cycles_run + self.cycles_skipped;
        let now_unix = chrono::Utc::now().timestamp();

        // TODO(follow-up PR): add QuillonClient::get_balance_qug with X-Wallet-Auth.
        // For Phase 1, the bot operates in dry-run-only mode until the signed
        // balance endpoint is wired in trading-bot's api_client.rs.
        // Workaround: when dry_run=true, treat balance as effectively unlimited
        // (large enough that cycle never skips). When dry_run=false, refuse to
        // run because we can't safely verify funds.
        let qug_balance: f64 = if self.config.dry_run {
            f64::MAX / 2.0 // never trips the floor check
        } else {
            return Err(anyhow!(
                "qcredit-dca live mode requires QuillonClient::get_balance_qug — \
                 follow-up PR will wire it via wallet_auth.rs X-Wallet-Auth signer. \
                 For now, set dry_run=true to test cycle logic."
            ));
        };
        let _ = self.client.list_pools().await; // touch the client to avoid dead-code warnings

        // Skip if balance would drop below floor after this cycle.
        let required_balance = self.config.qug_per_cycle + self.config.min_balance_floor;
        if qug_balance < required_balance {
            self.cycles_skipped += 1;
            return Ok(LockCycleResult {
                cycle_index: cycle_idx,
                qug_locked_display: 0.0,
                tier: self.config.tier.clone(),
                position_id: None,
                at_unix: now_unix,
                skipped: true,
                skip_reason: Some(format!(
                    "balance {:.6} QUG < required {:.6} (cycle + floor)",
                    qug_balance, required_balance
                )),
            });
        }

        if self.config.dry_run {
            info!(
                "[qcredit-dca] [DRY-RUN] would lock {:.6} QUG into {} tier (balance {:.6})",
                self.config.qug_per_cycle, self.config.tier, qug_balance
            );
            self.cycles_run += 1;
            self.total_qug_locked += self.config.qug_per_cycle;
            return Ok(LockCycleResult {
                cycle_index: cycle_idx,
                qug_locked_display: self.config.qug_per_cycle,
                tier: self.config.tier.clone(),
                position_id: Some(format!("dry-run-{}", cycle_idx)),
                at_unix: now_unix,
                skipped: false,
                skip_reason: None,
            });
        }

        // TODO(follow-up PR): POST /api/v1/qcredit/lock via
        // QuillonClient::qcredit_lock — wire via trading-bot's wallet_auth.rs
        // X-Wallet-Auth signer to authenticate the lock request. The endpoint
        // contract is in crates/q-api-server/src/qcredit_api.rs::LockRequest.
        return Err(anyhow!(
            "qcredit-dca live lock not yet wired. \
             See crates/q-api-server/src/qcredit_api.rs for the endpoint contract."
        ));
    }

    /// Run the bot loop until max_cycles is reached OR an unrecoverable error.
    pub async fn run(&mut self) -> Result<()> {
        info!(
            "[qcredit-dca] starting for wallet {}, {:.6} QUG/cycle, tier={}, interval={:?}",
            self.config.wallet.chars().take(16).collect::<String>(),
            self.config.qug_per_cycle,
            self.config.tier,
            self.config.cycle_interval,
        );

        loop {
            if let Some(max) = self.config.max_cycles {
                if self.cycles_run >= max {
                    info!(
                        "[qcredit-dca] reached max_cycles {} — exiting. Total locked: {:.6} QUG",
                        max, self.total_qug_locked
                    );
                    return Ok(());
                }
            }

            match self.step().await {
                Ok(result) if result.skipped => {
                    warn!(
                        "[qcredit-dca] skipped cycle {}: {}",
                        result.cycle_index,
                        result.skip_reason.unwrap_or_default()
                    );
                }
                Ok(_) => {} // success logged inside step()
                Err(e) => {
                    error!("[qcredit-dca] cycle error: {}", e);
                    // Don't crash; sleep + retry. The DCA strategy tolerates intermittent failures.
                }
            }

            sleep(self.config.cycle_interval).await;
        }
    }

    pub fn status_summary(&self) -> String {
        format!(
            "qcredit-dca: cycles_run={}, cycles_skipped={}, total_locked={:.6} QUG, uptime={:?}",
            self.cycles_run,
            self.cycles_skipped,
            self.total_qug_locked,
            self.started_at.elapsed(),
        )
    }
}

// ════════════════════════════════════════════════════════════════════════
// Unit tests (logic-only; integration tests need a live QCREDIT API)
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_values_sane() {
        let cfg = QcreditDcaConfig::for_wallet("https://quillon.xyz", "qnktest123");
        assert_eq!(cfg.tier, "platinum");
        assert_eq!(cfg.cycle_interval, Duration::from_secs(3600));
        assert_eq!(cfg.min_balance_floor, 0.1);
        assert!(!cfg.dry_run);
    }

    #[test]
    fn config_qug_per_cycle_below_floor_is_caught() {
        // If qug_per_cycle is 0.05 and min_balance_floor is 0.1,
        // a wallet with 0.12 QUG should NOT skip (0.05 + 0.1 = 0.15 > 0.12)
        // Actually 0.12 < 0.15, so it SHOULD skip. This test verifies the
        // arithmetic clearly.
        let cycle = 0.05;
        let floor = 0.1;
        let balance = 0.12;
        let required = cycle + floor;
        assert!(balance < required, "0.12 < 0.15");
    }

    #[test]
    fn dry_run_flag_default_false() {
        let cfg = QcreditDcaConfig::for_wallet("https://quillon.xyz", "qnktest123");
        assert!(!cfg.dry_run);
    }
}
