/// BTC Advantage Trading — Polymarket vs Black-Scholes edge.
///
/// Strategy:
///   For each active BTC price-target market on Polymarket
///   (e.g. "Will BTC be above $100k by Dec 31?"):
///
///     1. Fetch the Polymarket YES probability P_poly  (crowd wisdom, 0–1)
///     2. Compute Black-Scholes N(d2) implied probability P_bs using:
///            S  = current Binance BTC mark price
///            K  = strike price parsed from market question
///            T  = days_to_expiry / 365
///            σ  = 30-day realised vol (from Binance daily klines)
///            r  = 0.05 (USD risk-free rate)
///     3. Advantage = P_poly - P_bs
///        Positive advantage → market prices higher than BS → BTC has alpha
///        Negative advantage → market prices lower than BS → short hedge
///     4. If |advantage| > ENTRY_THRESHOLD:
///          Long  BTC futures (Binance) if advantage > 0
///          Short BTC futures (Binance) if advantage < 0
///     5. Size = Kelly fraction of USDT balance × advantage magnitude
///     6. Stop-loss at 2% adverse move from entry
///
/// Why this works:
///   Polymarket is an efficient information aggregator — sophisticated traders
///   with private information bet into the market and push odds toward truth.
///   When Polymarket diverges significantly from the no-arbitrage BS price,
///   it usually means the crowd has information (supply/demand, macro events)
///   not yet priced into spot BTC. We trade in the direction of that information.
///
/// Risk management:
///   - Maximum one position at a time (per symbol)
///   - Hard stop-loss 2% from entry
///   - Maximum 5% of account per trade
///   - Requires testnet = true during development

use crate::binance::{BinanceFutures, BinanceConfig, Side, Position};
use crate::kelly::{kelly_fraction, PriceHistory};
use crate::polymarket::{PolymarketClient, BtcPolymarketOpportunity};

use anyhow::{Context, Result};
use std::f64::consts::PI;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn, error};

const SYMBOL: &str = "BTCUSDT";
const ENTRY_THRESHOLD: f64 = 0.08;    // minimum |advantage| to trade (8% edge)
const STOP_LOSS_PCT: f64  = 0.02;     // 2% stop-loss from entry
const TAKE_PROFIT_PCT: f64 = 0.05;    // 5% take-profit from entry
const RISK_FREE_RATE: f64 = 0.05;     // annualised USD risk-free rate
const MAX_POSITION_FRACTION: f64 = 0.05; // max 5% of balance per trade
const POLL_INTERVAL_SECS: u64 = 300;  // check every 5 minutes

// ── Config ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BtcAdvantageConfig {
    pub binance: BinanceConfig,
    /// Minimum USDC volume on a Polymarket market to consider it liquid.
    pub min_poly_volume: f64,
    /// Maximum days to expiry to consider (very short = erratic; very long = illiquid).
    pub max_dte: f64,
    /// Minimum days to expiry.
    pub min_dte: f64,
    /// Minimum |advantage| to enter a trade.
    pub entry_threshold: f64,
    /// Stop-loss as fraction of entry price.
    pub stop_loss_pct: f64,
    /// Take-profit as fraction of entry price.
    pub take_profit_pct: f64,
    /// Dry run — analyse but don't place orders.
    pub dry_run: bool,
}

impl Default for BtcAdvantageConfig {
    fn default() -> Self {
        BtcAdvantageConfig {
            binance: BinanceConfig::default(),
            min_poly_volume: 10_000.0,
            max_dte: 90.0,
            min_dte: 3.0,
            entry_threshold: ENTRY_THRESHOLD,
            stop_loss_pct: STOP_LOSS_PCT,
            take_profit_pct: TAKE_PROFIT_PCT,
            dry_run: true,
        }
    }
}

// ── Black-Scholes ─────────────────────────────────────────────────────────────

/// Standard normal CDF via Abramowitz & Stegun approximation (error < 7.5e-8).
fn norm_cdf(x: f64) -> f64 {
    if x > 8.0 { return 1.0; }
    if x < -8.0 { return 0.0; }
    let k = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = k * (0.319_381_530
        + k * (-0.356_563_782
        + k * (1.781_477_937
        + k * (-1.821_255_978
        + k * 1.330_274_429))));
    let pdf = (-0.5 * x * x).exp() / (2.0 * PI).sqrt();
    if x >= 0.0 { 1.0 - pdf * poly } else { pdf * poly }
}

/// Black-Scholes N(d2) = risk-neutral probability that S_T > K.
///
///   d1 = (ln(S/K) + (r + σ²/2)×T) / (σ×√T)
///   d2 = d1 - σ×√T
///   P(S_T > K) = N(d2)
pub fn bs_probability(spot: f64, strike: f64, sigma: f64, risk_free: f64, t_years: f64) -> f64 {
    if t_years <= 0.0 || sigma <= 0.0 || spot <= 0.0 || strike <= 0.0 {
        return 0.0;
    }
    let sqrt_t = t_years.sqrt();
    let d1 = ((spot / strike).ln() + (risk_free + sigma * sigma / 2.0) * t_years) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    norm_cdf(d2)
}

// ── Opportunity evaluation ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Evaluation {
    pub opp: BtcPolymarketOpportunity,
    pub spot_price: f64,
    pub sigma: f64,
    /// Black-Scholes implied probability P(BTC > strike at expiry)
    pub bs_prob: f64,
    /// Advantage = P_poly - P_bs
    pub advantage: f64,
    /// Recommended direction: Long (advantage > 0), Short (advantage < 0), None
    pub direction: Option<Side>,
}

impl Evaluation {
    pub fn summary(&self) -> String {
        format!(
            "Q: {} | K=${:.0} DTE={:.1} | P_poly={:.3} P_bs={:.3} ADV={:+.3} σ={:.1}%",
            &self.opp.question[..self.opp.question.len().min(50)],
            self.opp.strike_usd,
            self.opp.days_to_expiry,
            self.opp.poly_prob,
            self.bs_prob,
            self.advantage,
            self.sigma * 100.0,
        )
    }
}

// ── Active trade tracker ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ActiveTrade {
    side: Side,
    entry_price: f64,
    qty_btc: f64,
    stop_loss: f64,
    take_profit: f64,
    condition_id: String,
}

// ── Bot ──────────────────────────────────────────────────────────────────────

pub struct BtcAdvantageBot {
    cfg: BtcAdvantageConfig,
    binance: BinanceFutures,
    polymarket: PolymarketClient,
    price_history: PriceHistory,
    active_trade: Option<ActiveTrade>,
    cycle: u64,
}

impl BtcAdvantageBot {
    pub fn new(cfg: BtcAdvantageConfig) -> Self {
        let binance = BinanceFutures::new(cfg.binance.clone());
        BtcAdvantageBot {
            cfg,
            binance,
            polymarket: PolymarketClient::new(),
            price_history: PriceHistory::new(35),
            active_trade: None,
            cycle: 0,
        }
    }

    pub async fn run(&mut self) -> Result<()> {
        self.print_banner();

        // Set leverage on Binance
        if !self.cfg.dry_run {
            if let Err(e) = self.binance.set_leverage(SYMBOL, self.cfg.binance.leverage).await {
                warn!("Could not set leverage: {} — continuing", e);
            }
        }

        loop {
            self.cycle += 1;
            info!("━━━ Cycle #{} ━━━", self.cycle);

            if let Err(e) = self.tick().await {
                error!("Cycle error: {}", e);
            }

            info!("⏳ Sleeping {}s", POLL_INTERVAL_SECS);
            sleep(Duration::from_secs(POLL_INTERVAL_SECS)).await;
        }
    }

    async fn tick(&mut self) -> Result<()> {
        // ── 1. Fetch BTC price + vol ──────────────────────────────────────────
        let spot = self.binance.mark_price(SYMBOL).await
            .context("Fetching BTC mark price")?;
        self.price_history.push(spot);

        let sigma = self.binance.historical_volatility(SYMBOL).await
            .unwrap_or_else(|_| {
                // Fallback: use price history if Binance klines fail
                self.price_history.volatility_annualized(86400)
            });

        info!("₿  BTC mark=${:.2} | σ_ann={:.1}%", spot, sigma * 100.0);

        // ── 2. Manage existing position ───────────────────────────────────────
        if let Some(trade) = &self.active_trade.clone() {
            self.manage_position(spot, trade).await?;
            // If position was closed, active_trade is now None — continue to next cycle
            if self.active_trade.is_none() {
                info!("🔄 Position closed — scanning for new opportunity next cycle");
                return Ok(());
            }
            info!("📊 Holding {} | entry=${:.2} stop=${:.2} tp=${:.2} | current=${:.2}",
                if trade.side == Side::Buy { "LONG" } else { "SHORT" },
                trade.entry_price, trade.stop_loss, trade.take_profit, spot);
            return Ok(());
        }

        // ── 3. Scan Polymarket for opportunities ──────────────────────────────
        let opps = self.polymarket.find_btc_opportunities().await
            .context("Fetching Polymarket opportunities")?;

        info!("📋 Found {} BTC Polymarket markets", opps.len());

        let mut best: Option<Evaluation> = None;
        let mut best_abs_adv = 0.0_f64;

        for opp in opps {
            // Filter by liquidity and DTE
            if opp.volume_usdc < self.cfg.min_poly_volume { continue; }
            if opp.days_to_expiry < self.cfg.min_dte { continue; }
            if opp.days_to_expiry > self.cfg.max_dte { continue; }

            let t = opp.days_to_expiry / 365.0;
            let bs_prob = bs_probability(spot, opp.strike_usd, sigma, RISK_FREE_RATE, t);
            let advantage = opp.poly_prob - bs_prob;
            let direction = if advantage > self.cfg.entry_threshold {
                Some(Side::Buy)
            } else if advantage < -self.cfg.entry_threshold {
                Some(Side::Sell)
            } else {
                None
            };

            let ev = Evaluation { opp, spot_price: spot, sigma, bs_prob, advantage, direction };
            info!("  {}", ev.summary());

            if ev.direction.is_some() && advantage.abs() > best_abs_adv {
                best_abs_adv = advantage.abs();
                best = Some(ev);
            }
        }

        // ── 4. Enter trade if edge found ──────────────────────────────────────
        if let Some(ev) = best {
            info!("⚡ BEST EDGE: {}", ev.summary());
            self.enter_trade(&ev).await?;
        } else {
            info!("💤 No edge found above {:.0}% threshold", self.cfg.entry_threshold * 100.0);
        }

        Ok(())
    }

    async fn enter_trade(&mut self, ev: &Evaluation) -> Result<()> {
        let side = match ev.direction {
            Some(s) => s,
            None => return Ok(()),
        };

        // Kelly-sized position
        let mu = self.price_history.drift_annualized(86400);
        let sigma = ev.sigma;
        let kf = kelly_fraction(mu, sigma, ev.spot_price * 1000.0); // large pool depth (spot market)
        let adv_scale = ev.advantage.abs().min(0.5); // scale by advantage, cap at 50%
        let fraction = (kf * adv_scale).min(MAX_POSITION_FRACTION);

        let usdt_balance = if self.cfg.dry_run {
            10_000.0 // paper capital
        } else {
            self.binance.usdt_balance().await.unwrap_or(0.0)
        };

        let notional_usdt = usdt_balance * fraction * self.cfg.binance.leverage as f64;
        if notional_usdt < self.cfg.binance.min_notional_usdt {
            warn!("Notional ${:.2} below minimum ${:.2} — skipping",
                notional_usdt, self.cfg.binance.min_notional_usdt);
            return Ok(());
        }

        let qty_btc = notional_usdt / ev.spot_price;
        let stop_loss = if side == Side::Buy {
            ev.spot_price * (1.0 - self.cfg.stop_loss_pct)
        } else {
            ev.spot_price * (1.0 + self.cfg.stop_loss_pct)
        };
        let take_profit = if side == Side::Buy {
            ev.spot_price * (1.0 + self.cfg.take_profit_pct)
        } else {
            ev.spot_price * (1.0 - self.cfg.take_profit_pct)
        };

        info!(
            "🎯 {} {} | qty={:.3} BTC | notional=${:.0} | entry≈${:.0} | stop=${:.0} | tp=${:.0}",
            if side == Side::Buy { "LONG" } else { "SHORT" },
            SYMBOL, qty_btc, notional_usdt, ev.spot_price, stop_loss, take_profit
        );
        info!("   Advantage: {:+.3} | Kelly fraction: {:.4} | ADV scale: {:.3}",
            ev.advantage, kf, adv_scale);
        info!("   Polymarket: {} | Market: {}", ev.opp.poly_prob, &ev.opp.condition_id[..16.min(ev.opp.condition_id.len())]);

        if self.cfg.dry_run {
            info!("🧪 DRY RUN — trade not placed");
            self.active_trade = Some(ActiveTrade {
                side,
                entry_price: ev.spot_price,
                qty_btc,
                stop_loss,
                take_profit,
                condition_id: ev.opp.condition_id.clone(),
            });
            return Ok(());
        }

        let order = self.binance.market_order(SYMBOL, side, qty_btc).await?;
        let fill_price = order.avg_fill_price().max(ev.spot_price);

        self.active_trade = Some(ActiveTrade {
            side,
            entry_price: fill_price,
            qty_btc: order.filled_qty(),
            stop_loss,
            take_profit,
            condition_id: ev.opp.condition_id.clone(),
        });
        Ok(())
    }

    async fn manage_position(&mut self, spot: f64, trade: &ActiveTrade) -> Result<()> {
        let hit_stop = if trade.side == Side::Buy {
            spot <= trade.stop_loss
        } else {
            spot >= trade.stop_loss
        };
        let hit_tp = if trade.side == Side::Buy {
            spot >= trade.take_profit
        } else {
            spot <= trade.take_profit
        };

        if hit_stop || hit_tp {
            let reason = if hit_stop { "STOP-LOSS" } else { "TAKE-PROFIT" };
            let pnl_pct = if trade.side == Side::Buy {
                (spot - trade.entry_price) / trade.entry_price
            } else {
                (trade.entry_price - spot) / trade.entry_price
            };
            info!("🔔 {} triggered at ${:.0} | PnL ≈ {:+.2}%", reason, spot, pnl_pct * 100.0);

            if !self.cfg.dry_run {
                // Build a minimal Position for close_position
                let pos = crate::binance::Position {
                    symbol: SYMBOL.into(),
                    position_amt: if trade.side == Side::Buy {
                        format!("{:.3}", trade.qty_btc)
                    } else {
                        format!("-{:.3}", trade.qty_btc)
                    },
                    entry_price: format!("{:.2}", trade.entry_price),
                    unrealized_profit: "0".into(),
                    leverage: self.cfg.binance.leverage.to_string(),
                    position_side: "BOTH".into(),
                };
                self.binance.close_position(SYMBOL, &pos).await?;
            } else {
                info!("🧪 DRY RUN — position close simulated");
            }

            self.active_trade = None;
        }
        Ok(())
    }

    fn print_banner(&self) {
        let net = if self.cfg.binance.testnet { "TESTNET" } else { "⚠️  MAINNET LIVE" };
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║   ₿ BTC Advantage Bot — Polymarket vs Black-Scholes edge        ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║  Binance:  {} {:<51} ║", net, "");
        println!("║  Leverage: {}× | Max pos: {:.0}% of balance{:<33} ║",
            self.cfg.binance.leverage,
            self.cfg.binance.position_size_fraction * 100.0, "");
        println!("║  Entry threshold: {:+.0}% advantage{:<38} ║",
            self.cfg.entry_threshold * 100.0, "");
        println!("║  Stop-loss: {:.0}% | Take-profit: {:.0}%{:<35} ║",
            self.cfg.stop_loss_pct * 100.0, self.cfg.take_profit_pct * 100.0, "");
        println!("║  Poly min volume: ${:.0} | DTE: {:.0}–{:.0} days{:<24} ║",
            self.cfg.min_poly_volume, self.cfg.min_dte, self.cfg.max_dte, "");
        println!("║  Dry run: {:<58} ║", if self.cfg.dry_run { "YES (safe)" } else { "NO — REAL MONEY" });
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║  Strategy: P_poly > P_bs+threshold → LONG BTC futures           ║");
        println!("║            P_poly < P_bs-threshold → SHORT BTC futures          ║");
        println!("║  Sizing:   Kelly fraction × advantage magnitude                 ║");
        println!("╚══════════════════════════════════════════════════════════════════╝\n");
    }
}

// ── BS formula tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_cdf() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((norm_cdf(1.0) - 0.8413).abs() < 1e-3);
        assert!((norm_cdf(-1.0) - 0.1587).abs() < 1e-3);
    }

    #[test]
    fn test_bs_probability() {
        // ATM call: S=K, σ=80%, T=0.25yr → P ≈ N(-σ√T/2)
        let p = bs_probability(100_000.0, 100_000.0, 0.80, 0.05, 0.25);
        assert!(p > 0.3 && p < 0.6, "ATM prob should be ~0.5 got {}", p);

        // Deep OTM: BTC at 80k, strike 200k, 30 days, σ=80%
        let p2 = bs_probability(80_000.0, 200_000.0, 0.80, 0.05, 30.0 / 365.0);
        assert!(p2 < 0.01, "Deep OTM should be near 0, got {}", p2);

        // Deep ITM: strike 10k, spot 80k
        let p3 = bs_probability(80_000.0, 10_000.0, 0.80, 0.05, 30.0 / 365.0);
        assert!(p3 > 0.99, "Deep ITM should be near 1, got {}", p3);
    }
}
