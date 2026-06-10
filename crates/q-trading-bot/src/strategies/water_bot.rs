/// Water Robot DCA Bot — Tunneling Octopus strategy.
///
/// Executes DCA swaps on the Quillon DEX using:
///   1. Breit-Wigner resonance gate (skip if pool is imbalanced)
///   2. Proof-of-Biosynthesis swarm consensus (>66% mass YES)
///   3. FCC-ee operating mode (auto-selected by pool depth)
///   4. DNA-encoded trade history with genetic evolution
///   5. Kelly Criterion + Renormalization Group adaptive sizing (v2 design doc)

use crate::api_client::{QuillonClient, to_raw, to_display, DECIMALS_24};
use crate::kelly::{PriceHistory, adaptive_dca_amount};
use crate::resonance::{compute_resonance, gamma_from_tvl, FccMode};
use crate::swarm::WaterRobotSwarm;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{error, info, warn};

/// Configuration for a single Tunneling Octopus DCA instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterBotConfig {
    /// Quillon API base URL (e.g. "https://quillon.xyz")
    pub api_url: String,
    /// Your qnk wallet address — must hold token_in balance on-chain
    pub wallet: String,
    /// Input token symbol, e.g. "QUG"
    pub token_in: String,
    /// Output token symbol, e.g. "QUGUSD"
    pub token_out: String,
    /// Base amount of token_in per DCA execution (display units, e.g. 10.0).
    /// Actual amount per cycle is scaled by Kelly/RG sizing unless kelly_sizing=false.
    pub amount_per_execution: f64,
    /// Total capital available for DCA (display units). Used for Kelly fraction.
    /// If 0, Kelly sizing falls back to amount_per_execution.
    pub total_capital: f64,
    /// Interval between DCA executions in seconds (e.g. 3600 = hourly)
    pub interval_secs: u64,
    /// Max allowed slippage as a fraction (e.g. 0.01 = 1%)
    pub max_slippage: f64,
    /// Override FCC-ee resonance threshold (0–1). None = auto from pool depth.
    pub resonance_threshold: Option<f64>,
    /// Enable Kelly Criterion + RG adaptive sizing (default true).
    pub kelly_sizing: bool,
    /// Dry run: print what would happen but don't call execute_swap
    pub dry_run: bool,
}

impl Default for WaterBotConfig {
    fn default() -> Self {
        WaterBotConfig {
            api_url: "https://quillon.xyz".to_string(),
            wallet: String::new(),
            token_in: "QUG".to_string(),
            token_out: "QUGUSD".to_string(),
            amount_per_execution: 10.0,
            total_capital: 0.0,
            interval_secs: 3600,
            max_slippage: 0.01,
            resonance_threshold: None,
            kelly_sizing: true,
            dry_run: false,
        }
    }
}

/// One cycle result for metrics/logging.
#[derive(Debug, Clone)]
pub struct CycleResult {
    pub executed: bool,
    pub skip_reason: Option<String>,
    pub amount_in_display: f64,
    pub amount_out_display: f64,
    pub resonance_efficiency: f64,
    pub yes_fraction: f64,
    pub tx_hash: Option<String>,
    pub kelly_sizing_note: Option<String>,
}

pub struct TunnelingOctopusBot {
    cfg: WaterBotConfig,
    client: QuillonClient,
    swarm: WaterRobotSwarm,
    cycle_count: u64,
    /// Rolling price history for Kelly/RG sizing
    price_history: PriceHistory,
    /// Exponential moving average of reserve_in for stochastic correction
    mean_reserve_in: f64,
    /// Timestamp of last executed swap (for τ in RG coupling)
    last_executed_at: Option<Instant>,
}

impl TunnelingOctopusBot {
    pub fn new(cfg: WaterBotConfig) -> Self {
        let client = QuillonClient::new(cfg.api_url.clone());
        let swarm = WaterRobotSwarm::new(1.0);
        TunnelingOctopusBot {
            cfg,
            client,
            swarm,
            cycle_count: 0,
            price_history: PriceHistory::new(48), // ~48 price samples
            mean_reserve_in: 0.0,
            last_executed_at: None,
        }
    }

    /// Run indefinitely, executing DCA swaps on the configured interval.
    pub async fn run(&mut self) -> Result<()> {
        self.print_banner();

        if !self.client.dex_status().await {
            warn!("⚠️  DEX status check failed — continuing anyway (node may still serve swaps)");
        }

        loop {
            self.cycle_count += 1;
            info!("🔄 Cycle #{} — {}/{}", self.cycle_count, self.cfg.token_in, self.cfg.token_out);

            match self.execute_cycle().await {
                Ok(result) => self.print_cycle_result(&result),
                Err(e) => error!("❌ Cycle error: {}", e),
            }

            info!("⏳ Sleeping {}s until next DCA cycle…", self.cfg.interval_secs);
            sleep(Duration::from_secs(self.cfg.interval_secs)).await;
        }
    }

    /// Execute one DCA cycle: poll pool → resonance → swarm vote → Kelly sizing → swap.
    pub async fn execute_cycle(&mut self) -> Result<CycleResult> {
        // 1. Fetch pool state
        let pool = self.client
            .find_pool(&self.cfg.token_in, &self.cfg.token_out)
            .await?;

        let pool = match pool {
            Some(p) => p,
            None => {
                return Ok(CycleResult {
                    executed: false,
                    skip_reason: Some(format!(
                        "No pool found for {}/{}",
                        self.cfg.token_in, self.cfg.token_out
                    )),
                    amount_in_display: 0.0,
                    amount_out_display: 0.0,
                    resonance_efficiency: 0.0,
                    yes_fraction: 0.0,
                    tx_hash: None,
                    kelly_sizing_note: None,
                });
            }
        };

        // Determine trade direction
        let (reserve_in, reserve_out) = if pool.token0.to_uppercase() == self.cfg.token_in.to_uppercase() {
            (pool.reserve0_raw(), pool.reserve1_raw())
        } else {
            (pool.reserve1_raw(), pool.reserve0_raw())
        };

        let tvl = pool.tvl_display();

        // Update EMA of reserve_in for stochastic correction
        let reserve_in_display = to_display(reserve_in, DECIMALS_24);
        if self.mean_reserve_in == 0.0 {
            self.mean_reserve_in = reserve_in_display;
        } else {
            self.mean_reserve_in = 0.05 * reserve_in_display + 0.95 * self.mean_reserve_in;
        }

        // 2. Compute Breit-Wigner resonance
        let gamma = gamma_from_tvl(tvl);
        let resonance = compute_resonance(reserve_in, reserve_out, gamma);
        let fcc_mode = FccMode::from_pool_depth(tvl);
        let threshold = self.cfg.resonance_threshold.unwrap_or(fcc_mode.resonance_threshold());

        // 3. Get current price + update history
        let current_price = self.client.get_price(&self.cfg.token_in).await.unwrap_or(0.0);
        if current_price > 0.0 {
            self.price_history.push(current_price);
        }

        // 4. Proof-of-Biosynthesis swarm vote
        let vote = self.swarm.vote(&resonance, self.cfg.amount_per_execution, current_price, threshold);

        info!(
            "🧬 Resonance: R={:.4} η={:.3} (threshold={:.2}) | Mode: {:?}",
            resonance.ratio, resonance.efficiency, threshold, fcc_mode
        );
        info!("🗳️  Swarm: {}", vote.summary_line());

        if !vote.approved {
            let reason = if resonance.efficiency < threshold {
                format!("Pool off-resonance (η={:.3} < {:.2})", resonance.efficiency, threshold)
            } else {
                format!("Swarm rejected ({:.0}% YES < 66%)", vote.yes_fraction * 100.0)
            };
            return Ok(CycleResult {
                executed: false,
                skip_reason: Some(reason),
                amount_in_display: self.cfg.amount_per_execution,
                amount_out_display: 0.0,
                resonance_efficiency: resonance.efficiency,
                yes_fraction: vote.yes_fraction,
                tx_hash: None,
                kelly_sizing_note: None,
            });
        }

        // 5. Kelly/RG adaptive sizing
        let hours_since_last = self.last_executed_at
            .map(|t| t.elapsed().as_secs_f64() / 3600.0)
            .unwrap_or(24.0);

        let capital = if self.cfg.total_capital > 0.0 {
            self.cfg.total_capital
        } else {
            self.cfg.amount_per_execution * 100.0 // fallback: assume 100× configured amount
        };

        let (adaptive_amount, kelly_note) = if self.cfg.kelly_sizing && self.price_history.len() >= 3 {
            adaptive_dca_amount(
                self.cfg.amount_per_execution,
                capital,
                &self.price_history,
                self.cfg.interval_secs,
                tvl,
                reserve_in,
                reserve_out,
                self.mean_reserve_in,
                hours_since_last,
            )
        } else {
            (self.cfg.amount_per_execution, "fixed (insufficient price history)".to_string())
        };

        info!("📐 Kelly/RG amount: {:.6} {} ({})", adaptive_amount, self.cfg.token_in, kelly_note);

        // 6. Get quote using adaptive amount
        let amount_in_raw = to_raw(adaptive_amount, DECIMALS_24);
        let quote = match self.client
            .get_quote(&self.cfg.token_in, &self.cfg.token_out, amount_in_raw, self.cfg.max_slippage)
            .await
        {
            Ok(q) => q,
            Err(e) => {
                return Ok(CycleResult {
                    executed: false,
                    skip_reason: Some(format!("Quote failed: {e}")),
                    amount_in_display: adaptive_amount,
                    amount_out_display: 0.0,
                    resonance_efficiency: resonance.efficiency,
                    yes_fraction: vote.yes_fraction,
                    tx_hash: None,
                    kelly_sizing_note: Some(kelly_note),
                });
            }
        };

        let amount_out_display = to_display(quote.amount_out_raw(), DECIMALS_24);
        let price_impact = quote.price_impact;

        if price_impact > self.cfg.max_slippage {
            return Ok(CycleResult {
                executed: false,
                skip_reason: Some(format!(
                    "Price impact {:.2}% > max slippage {:.2}%",
                    price_impact * 100.0, self.cfg.max_slippage * 100.0
                )),
                amount_in_display: adaptive_amount,
                amount_out_display,
                resonance_efficiency: resonance.efficiency,
                yes_fraction: vote.yes_fraction,
                tx_hash: None,
                kelly_sizing_note: Some(kelly_note),
            });
        }

        info!(
            "💱 Quote: {:.6} {} → {:.6} {} (impact: {:.3}%)",
            adaptive_amount, self.cfg.token_in,
            amount_out_display, self.cfg.token_out,
            price_impact * 100.0
        );

        // 7. Execute (or dry-run)
        if self.cfg.dry_run {
            info!("🧪 DRY RUN — skipping execute_swap call");
            self.swarm.record_outcome(current_price, adaptive_amount, 0.0);
            return Ok(CycleResult {
                executed: false,
                skip_reason: Some("dry_run".to_string()),
                amount_in_display: adaptive_amount,
                amount_out_display,
                resonance_efficiency: resonance.efficiency,
                yes_fraction: vote.yes_fraction,
                tx_hash: Some("DRY-RUN".to_string()),
                kelly_sizing_note: Some(kelly_note),
            });
        }

        let result = self.client.execute_swap(
            &self.cfg.token_in,
            &self.cfg.token_out,
            amount_in_raw,
            quote.minimum_out_raw(),
            &self.cfg.wallet,
        ).await?;

        self.last_executed_at = Some(Instant::now());
        self.swarm.record_outcome(current_price, adaptive_amount, 0.0);

        Ok(CycleResult {
            executed: true,
            skip_reason: None,
            amount_in_display: adaptive_amount,
            amount_out_display,
            resonance_efficiency: resonance.efficiency,
            yes_fraction: vote.yes_fraction,
            tx_hash: Some(result.transaction_hash),
            kelly_sizing_note: Some(kelly_note),
        })
    }

    fn print_banner(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║       🐙 Tunneling Octopus — Water Robot DCA Bot v2           ║");
        println!("╠═══════════════════════════════════════════════════════════════╣");
        println!("║  API:        {:<50} ║", self.cfg.api_url);
        println!("║  Wallet:     {:<50} ║", truncate(&self.cfg.wallet, 50));
        println!("║  Pair:       {} → {:<46} ║", self.cfg.token_in, self.cfg.token_out);
        println!("║  Base amount:{:.4} {} per cycle{:<34} ║", self.cfg.amount_per_execution, self.cfg.token_in, "");
        println!("║  Capital:    {:.2}{:<50} ║", self.cfg.total_capital, "");
        println!("║  Interval:   {}s{:<52} ║", self.cfg.interval_secs, "");
        println!("║  Max slippage:{:.1}%{:<49} ║", self.cfg.max_slippage * 100.0, "");
        println!("║  Kelly sizing:{:<50} ║", if self.cfg.kelly_sizing { "ON (RG adaptive)" } else { "OFF (fixed)" });
        println!("║  Dry run:    {:<50} ║", if self.cfg.dry_run { "YES" } else { "NO (LIVE TRADING)" });
        println!("╠═══════════════════════════════════════════════════════════════╣");
        println!("║  Swarm: 6 droplets — Proof-of-Biosynthesis consensus (>66%)   ║");
        println!("║  Resonance: Breit-Wigner η gate + FCC-ee mode auto-select     ║");
        println!("║  Sizing: Kelly criterion + Renormalization Group DCA (v2)     ║");
        println!("║  DNA: trade history encoded ATGC, evolves over time           ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");
    }

    fn print_cycle_result(&self, r: &CycleResult) {
        if r.executed {
            info!(
                "✅ EXECUTED | {:.6} {} → {:.6} {} | η={:.3} | YES={:.0}% | swarm={:.1}pg | tx={}",
                r.amount_in_display, self.cfg.token_in,
                r.amount_out_display, self.cfg.token_out,
                r.resonance_efficiency,
                r.yes_fraction * 100.0,
                self.swarm.total_mass_pg(),
                r.tx_hash.as_deref().unwrap_or("?"),
            );
            if let Some(note) = &r.kelly_sizing_note {
                info!("   📐 Sizing: {}", note);
            }
        } else {
            info!(
                "⏭  SKIPPED | η={:.3} | YES={:.0}% | reason: {}",
                r.resonance_efficiency,
                r.yes_fraction * 100.0,
                r.skip_reason.as_deref().unwrap_or("unknown"),
            );
        }
    }
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max { s } else { &s[..max] }
}
