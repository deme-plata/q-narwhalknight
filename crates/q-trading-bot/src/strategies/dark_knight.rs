/// Dark Knight Water Robot — Combined Dagknight + Water Robot strategy.
///
/// Architecture:
///   1. Technical analysis layer (Dagknight SIMD indicators):
///      Bollinger Bands, RSI, MACD, ATR, ADX, VWAP, OBV, Ichimoku Cloud
///   2. Resonance gate (FCC-ee Breit-Wigner, from water_bot.rs)
///   3. Proof-of-Biosynthesis swarm vote (DNA-mass >66%)
///   4. Kelly/RG adaptive sizing
///   5. Optional: Bracha Reliable Broadcast for P2P multi-node consensus
///
/// libp2p / Bracha integration:
///   - `BotP2pHandle` wraps q-narwhal-core's ReliableBroadcast
///   - Before each swap, the bot broadcasts a `TradeProposal` vertex via BRB
///   - The proposal must be delivered (2f+1 echoes) before execution proceeds
///   - This turns the single-process swarm vote into a network-wide BFT vote
///   - Each node runs its own IndicatorSet + resonance check; the BRB ensures
///     all honest nodes agree before any money moves

use crate::api_client::{QuillonClient, to_raw, to_display, DECIMALS_24};
use crate::indicators::{IndicatorSet, AnalyticsSnapshot};
use crate::kelly::{PriceHistory, adaptive_dca_amount};
use crate::resonance::{compute_resonance, gamma_from_tvl, FccMode};
use crate::swarm::WaterRobotSwarm;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{error, info, warn};

// ── Config ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DarkKnightConfig {
    pub api_url: String,
    pub wallet: String,
    pub token_in: String,
    pub token_out: String,
    /// Base DCA amount per execution (display units)
    pub base_amount: f64,
    /// Total capital — used for Kelly sizing
    pub total_capital: f64,
    /// DCA interval in seconds
    pub interval_secs: u64,
    /// Max slippage fraction
    pub max_slippage: f64,
    /// Override resonance threshold (None = auto FCC-ee mode)
    pub resonance_threshold: Option<f64>,
    /// Require Ichimoku confluence for entry (long_signal or short_signal must be true)
    pub require_ichimoku: bool,
    /// Require ADX > this value to confirm trend strength
    pub min_adx: f64,
    /// Enable Kelly/RG adaptive sizing
    pub kelly_sizing: bool,
    /// Enable P2P Bracha broadcast before execution (requires node connectivity)
    pub p2p_bracha: bool,
    pub dry_run: bool,
}

impl Default for DarkKnightConfig {
    fn default() -> Self {
        DarkKnightConfig {
            api_url: "https://quillon.xyz".to_string(),
            wallet: String::new(),
            token_in: "QUG".to_string(),
            token_out: "QUGUSD".to_string(),
            base_amount: 10.0,
            total_capital: 0.0,
            interval_secs: 3600,
            max_slippage: 0.01,
            resonance_threshold: None,
            require_ichimoku: true,
            min_adx: 20.0,
            kelly_sizing: true,
            p2p_bracha: false, // enable once node is connected
            dry_run: false,
        }
    }
}

// ── Result ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DarkKnightCycleResult {
    pub executed: bool,
    pub skip_reason: Option<String>,
    pub amount_in: f64,
    pub amount_out: f64,
    pub resonance_eta: f64,
    pub swarm_yes_pct: f64,
    pub rsi: f64,
    pub adx: f64,
    pub ichimoku_long: bool,
    pub tx_hash: Option<String>,
}

// ── Bot ──────────────────────────────────────────────────────────────────────

pub struct DarkKnightBot {
    cfg: DarkKnightConfig,
    client: QuillonClient,
    swarm: WaterRobotSwarm,
    indicators: IndicatorSet,
    price_history: PriceHistory,
    mean_reserve_in: f64,
    last_executed: Option<Instant>,
    cycle_count: u64,
}

impl DarkKnightBot {
    pub fn new(cfg: DarkKnightConfig) -> Self {
        let client = QuillonClient::new(cfg.api_url.clone());
        DarkKnightBot {
            cfg,
            client,
            swarm: WaterRobotSwarm::new(1.0),
            indicators: IndicatorSet::new(),
            price_history: PriceHistory::new(48),
            mean_reserve_in: 0.0,
            last_executed: None,
            cycle_count: 0,
        }
    }

    pub async fn run(&mut self) -> Result<()> {
        self.print_banner();

        if !self.client.dex_status().await {
            warn!("⚠️  DEX health check failed — continuing (node may still serve swaps)");
        }

        loop {
            self.cycle_count += 1;
            info!("🦇 Dark Knight cycle #{} — {}/{}",
                self.cycle_count, self.cfg.token_in, self.cfg.token_out);

            match self.execute_cycle().await {
                Ok(r) => self.log_result(&r),
                Err(e) => error!("❌ Cycle error: {}", e),
            }

            info!("⏳ Next cycle in {}s", self.cfg.interval_secs);
            sleep(Duration::from_secs(self.cfg.interval_secs)).await;
        }
    }

    pub async fn execute_cycle(&mut self) -> Result<DarkKnightCycleResult> {
        // ── 1. Fetch pool ────────────────────────────────────────────────────
        let pool = match self.client.find_pool(&self.cfg.token_in, &self.cfg.token_out).await? {
            Some(p) => p,
            None => return Ok(self.skip("No pool found")),
        };

        let (reserve_in, reserve_out) = if pool.token0.to_uppercase() == self.cfg.token_in.to_uppercase() {
            (pool.reserve0_raw(), pool.reserve1_raw())
        } else {
            (pool.reserve1_raw(), pool.reserve0_raw())
        };

        let tvl = pool.tvl_display();
        let reserve_in_display = to_display(reserve_in, DECIMALS_24);

        // Update reserve EMA
        if self.mean_reserve_in == 0.0 { self.mean_reserve_in = reserve_in_display; }
        else { self.mean_reserve_in = 0.05 * reserve_in_display + 0.95 * self.mean_reserve_in; }

        // ── 2. Oracle price + indicator update ───────────────────────────────
        let price = self.client.get_price(&self.cfg.token_in).await.unwrap_or(0.0);
        if price > 0.0 { self.price_history.push(price); }

        // Use pool ratio as a proxy for OHLCV (real integration: pull from candle API)
        let high = price * 1.001;
        let low = price * 0.999;
        let volume = tvl * 0.01; // estimate 1% of TVL as rolling volume
        let snap: AnalyticsSnapshot = self.indicators.update(high, low, price, volume);

        info!(
            "📊 Indicators: RSI={:.1} ADX={:.1} MACD={:.4} | Ichimoku long={} short={}",
            snap.rsi, snap.adx, snap.macd_hist, snap.long_signal, snap.short_signal
        );

        // ── 3. Ichimoku confluence gate ──────────────────────────────────────
        if self.cfg.require_ichimoku && !snap.long_signal && !snap.short_signal {
            return Ok(self.skip_with(
                format!("Ichimoku: no confluence (RSI={:.1} ADX={:.1} long={} short={})",
                    snap.rsi, snap.adx, snap.long_signal, snap.short_signal),
                &snap,
            ));
        }

        if snap.adx < self.cfg.min_adx {
            return Ok(self.skip_with(
                format!("ADX {:.1} < min {:.1} (trending strength insufficient)", snap.adx, self.cfg.min_adx),
                &snap,
            ));
        }

        // ── 4. Breit-Wigner resonance gate ───────────────────────────────────
        let gamma = gamma_from_tvl(tvl);
        let resonance = compute_resonance(reserve_in, reserve_out, gamma);
        let fcc_mode = FccMode::from_pool_depth(tvl);
        let threshold = self.cfg.resonance_threshold.unwrap_or(fcc_mode.resonance_threshold());

        info!(
            "🧬 Resonance: R={:.4} η={:.3} (threshold={:.2}) | {:?}",
            resonance.ratio, resonance.efficiency, threshold, fcc_mode
        );

        // ── 5. Swarm vote ────────────────────────────────────────────────────
        let vote = self.swarm.vote(&resonance, self.cfg.base_amount, price, threshold);
        info!("🗳️  Swarm: {}", vote.summary_line());

        if !vote.approved {
            let reason = if resonance.efficiency < threshold {
                format!("Off-resonance η={:.3} < {:.2}", resonance.efficiency, threshold)
            } else {
                format!("Swarm rejected {:.0}% YES", vote.yes_fraction * 100.0)
            };
            return Ok(self.skip_with(reason, &snap));
        }

        // ── 6. Kelly/RG sizing ───────────────────────────────────────────────
        let hours_since = self.last_executed.map(|t| t.elapsed().as_secs_f64() / 3600.0).unwrap_or(24.0);
        let capital = if self.cfg.total_capital > 0.0 { self.cfg.total_capital } else { self.cfg.base_amount * 100.0 };

        let (amount, kelly_note) = if self.cfg.kelly_sizing && self.price_history.len() >= 3 {
            adaptive_dca_amount(
                self.cfg.base_amount, capital, &self.price_history,
                self.cfg.interval_secs, tvl, reserve_in, reserve_out,
                self.mean_reserve_in, hours_since,
            )
        } else {
            (self.cfg.base_amount, "fixed".to_string())
        };
        info!("📐 Kelly/RG amount: {:.6} {} ({})", amount, self.cfg.token_in, kelly_note);

        // ── 7. P2P Bracha broadcast (optional) ──────────────────────────────
        // When cfg.p2p_bracha = true, the bot would:
        //   a. Create a TradeProposal vertex with the swap parameters
        //   b. Call reliable_broadcast.broadcast_vertex(proposal).await
        //   c. Wait for BRB delivery confirmation (2f+1 echo votes)
        //   d. Only then proceed to execute_swap
        //
        // This makes the water robot swarm multi-node: each Quillon node running
        // this bot participates as a BFT voter. The swap only executes when >2/3
        // of network nodes have delivered the proposal via Bracha's protocol.
        //
        // For now this is gated behind cfg.p2p_bracha (default=false) since it
        // requires a live q-narwhal-core NarwhalNode instance to be wired in.
        // See `p2p_bridge.rs` for the integration shim.

        if self.cfg.p2p_bracha {
            info!("📡 P2P Bracha broadcast required but not yet wired — set p2p_bracha=false or wire BotP2pBridge");
            return Ok(self.skip_with("p2p_bracha enabled but bridge not wired".into(), &snap));
        }

        // ── 8. Quote ─────────────────────────────────────────────────────────
        let amount_raw = to_raw(amount, DECIMALS_24);
        let quote = match self.client.get_quote(&self.cfg.token_in, &self.cfg.token_out, amount_raw, self.cfg.max_slippage).await {
            Ok(q) => q,
            Err(e) => return Ok(self.skip_with(format!("Quote failed: {e}"), &snap)),
        };

        let amount_out = to_display(quote.amount_out_raw(), DECIMALS_24);
        if quote.price_impact > self.cfg.max_slippage {
            return Ok(self.skip_with(
                format!("Price impact {:.2}% > max {:.2}%",
                    quote.price_impact * 100.0, self.cfg.max_slippage * 100.0),
                &snap,
            ));
        }

        info!("💱 Quote: {:.6} {} → {:.6} {} (impact {:.3}%)",
            amount, self.cfg.token_in, amount_out, self.cfg.token_out, quote.price_impact * 100.0);

        // ── 9. Execute ───────────────────────────────────────────────────────
        if self.cfg.dry_run {
            self.swarm.record_outcome(price, amount, 0.0);
            return Ok(DarkKnightCycleResult {
                executed: false,
                skip_reason: Some("dry_run".into()),
                amount_in: amount,
                amount_out,
                resonance_eta: resonance.efficiency,
                swarm_yes_pct: vote.yes_fraction,
                rsi: snap.rsi,
                adx: snap.adx,
                ichimoku_long: snap.long_signal,
                tx_hash: Some("DRY-RUN".into()),
            });
        }

        let result = self.client.execute_swap(
            &self.cfg.token_in,
            &self.cfg.token_out,
            amount_raw,
            quote.minimum_out_raw(),
            &self.cfg.wallet,
        ).await?;

        self.last_executed = Some(Instant::now());
        self.swarm.record_outcome(price, amount, 0.0);

        Ok(DarkKnightCycleResult {
            executed: true,
            skip_reason: None,
            amount_in: amount,
            amount_out,
            resonance_eta: resonance.efficiency,
            swarm_yes_pct: vote.yes_fraction,
            rsi: snap.rsi,
            adx: snap.adx,
            ichimoku_long: snap.long_signal,
            tx_hash: Some(result.transaction_hash),
        })
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    fn skip(&self, reason: impl Into<String>) -> DarkKnightCycleResult {
        DarkKnightCycleResult {
            executed: false,
            skip_reason: Some(reason.into()),
            amount_in: 0.0, amount_out: 0.0,
            resonance_eta: 0.0, swarm_yes_pct: 0.0,
            rsi: 0.0, adx: 0.0, ichimoku_long: false,
            tx_hash: None,
        }
    }

    fn skip_with(&self, reason: String, snap: &AnalyticsSnapshot) -> DarkKnightCycleResult {
        DarkKnightCycleResult {
            executed: false,
            skip_reason: Some(reason),
            amount_in: 0.0, amount_out: 0.0,
            resonance_eta: 0.0, swarm_yes_pct: 0.0,
            rsi: snap.rsi, adx: snap.adx,
            ichimoku_long: snap.long_signal,
            tx_hash: None,
        }
    }

    fn log_result(&self, r: &DarkKnightCycleResult) {
        if r.executed {
            info!(
                "✅ DARK KNIGHT EXECUTED | {:.6} {} → {:.6} {} | η={:.3} | YES={:.0}% | RSI={:.1} ADX={:.1} | tx={}",
                r.amount_in, self.cfg.token_in,
                r.amount_out, self.cfg.token_out,
                r.resonance_eta, r.swarm_yes_pct * 100.0,
                r.rsi, r.adx,
                r.tx_hash.as_deref().unwrap_or("?"),
            );
        } else {
            info!(
                "⏭  SKIPPED | RSI={:.1} ADX={:.1} | reason: {}",
                r.rsi, r.adx,
                r.skip_reason.as_deref().unwrap_or("unknown"),
            );
        }
    }

    fn print_banner(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║     🦇🐙 Dark Knight Water Robot — Combined Strategy             ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║  Dagknight indicators: BB·RSI·MACD·ATR·ADX·VWAP·OBV·Ichimoku   ║");
        println!("║  Water Robot: Breit-Wigner resonance + Proof-of-Biosynthesis    ║");
        println!("║  Sizing: Kelly Criterion + Renormalization Group DCA            ║");
        println!("║  P2P: Bracha reliable broadcast (BFT multi-node swarm vote)     ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║  API:      {:<56} ║", self.cfg.api_url);
        println!("║  Pair:     {} → {:<50} ║", self.cfg.token_in, self.cfg.token_out);
        println!("║  Ichimoku: {:<56} ║", if self.cfg.require_ichimoku { "REQUIRED (multi-confluence)" } else { "optional" });
        println!("║  Bracha:   {:<56} ║", if self.cfg.p2p_bracha { "ENABLED (BFT multi-node)" } else { "disabled" });
        println!("║  Kelly:    {:<56} ║", if self.cfg.kelly_sizing { "ENABLED (adaptive sizing)" } else { "disabled (fixed)" });
        println!("║  Dry run:  {:<56} ║", if self.cfg.dry_run { "YES" } else { "NO (LIVE TRADING)" });
        println!("╚══════════════════════════════════════════════════════════════════╝\n");
    }
}
