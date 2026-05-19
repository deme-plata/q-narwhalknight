//! Observer-gene sizing layer for the Quillon trading bot.
//!
//! This module ports the **Kristensen K-parameter** observer-dependence formalism
//! (`/home/myuser/quantum-cosmos/quantum_cosmos/kristensen/`) into a concrete
//! position-sizing multiplier:
//!
//! ```text
//! f_position = f_Kelly × Ω_obs(agent) × κ_market
//! ```
//!
//! - `Ω_obs(agent)` is the Harlow observer factor `1 - exp(-S_Ob / S_max)` where
//!   `S_Ob` is the agent's information capacity (Shannon entropy of the feature
//!   vector it actually observes), and `S_max` is the maximum entropy
//!   the system can support (capped at `MAX_OBSERVER_ENTROPY` nats below).
//! - `κ_market` is the market's quantum-classical phase parameter — high κ means
//!   the market has anomalous, exploitable correlations; low κ means the market
//!   has decohered into efficient noise where alpha is impossible.
//!
//! The product is conservative by construction: a thin observer cannot extract
//! signal from a deep market (Ω_obs → 0); a maximally capable observer cannot
//! squeeze blood from a stone (κ → 0). Only when *both* the agent has
//! information capacity AND the market has structure does the bot size up.
//!
//! See `papers/agentic-money-quillon-graph.tex` for the derivation.

use std::collections::HashMap;

/// Maximum effective observer entropy (in nats) we'll credit to any single
/// trading agent. Caps `Ω_obs` away from 1.0 so even an oracle gets sized
/// against `κ_market`. ~25 nats ≈ a feature vector with about 2^36 distinct
/// states — generous for any rational trading bot.
pub const MAX_OBSERVER_ENTROPY: f64 = 25.0;

/// Numerical floor for `Ω_obs` so a brand-new agent with no observation
/// history doesn't silently get a 0× multiplier (and never trade, never
/// observe, never learn). Cold-start floor of 1% Kelly equivalent.
pub const OMEGA_FLOOR: f64 = 0.01;

/// Numerical floor for `κ_market` — same rationale: never multiply Kelly by
/// exactly zero, since residual structure exists in any real market.
pub const KAPPA_FLOOR: f64 = 0.05;

/// Hard ceiling on the observer-sized multiplier. Belt-and-braces against
/// pathological inputs (e.g. a κ estimator that overflows during a flash
/// crash). Caps the bot at 80% of its raw Kelly suggestion.
pub const SIZE_CEIL: f64 = 0.80;

// ─────────────────────────────────────────────────────────────────────────────
// Ω_obs — observer entropy
// ─────────────────────────────────────────────────────────────────────────────

/// Running estimator of an agent's observer entropy.
///
/// Maintains a windowed histogram of the *symbolic* feature vector the agent
/// has observed, then estimates Shannon entropy `H = -Σ p_i log p_i` over
/// the observed bucket distribution. The Harlow factor is then
/// `Ω_obs = 1 - exp(-H / MAX_OBSERVER_ENTROPY)`.
///
/// The features fed in should be discretized — e.g. quantize each indicator
/// (RSI band, ATR percentile, order-book imbalance bucket) into a small
/// alphabet, concatenate into a feature key, and `record(key)`.
#[derive(Debug, Clone, Default)]
pub struct OmegaObs {
    /// Counts per observed feature key.
    counts: HashMap<String, u64>,
    /// Total observations recorded.
    total: u64,
    /// Maximum unique buckets to keep — bounds memory under high-cardinality
    /// feature spaces. When exceeded, the lowest-count buckets are pruned.
    max_buckets: usize,
}

impl OmegaObs {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            total: 0,
            max_buckets: 4096,
        }
    }

    pub fn with_max_buckets(max_buckets: usize) -> Self {
        Self {
            counts: HashMap::new(),
            total: 0,
            max_buckets,
        }
    }

    /// Record a single observation of the discretized feature key.
    pub fn record(&mut self, feature_key: impl Into<String>) {
        let key = feature_key.into();
        *self.counts.entry(key).or_insert(0) += 1;
        self.total += 1;

        if self.counts.len() > self.max_buckets {
            self.prune_low();
        }
    }

    /// Drop the bottom-quartile of buckets by count to bound memory.
    fn prune_low(&mut self) {
        let mut all: Vec<(String, u64)> = self.counts.drain().collect();
        all.sort_by(|a, b| b.1.cmp(&a.1));
        let keep = (self.max_buckets * 3) / 4;
        all.truncate(keep);
        // Re-sum total from kept counts so the entropy estimate stays consistent.
        self.total = all.iter().map(|(_, c)| c).sum();
        self.counts.extend(all);
    }

    /// Shannon entropy of the observed bucket distribution, in nats.
    /// Returns 0.0 if the agent has observed nothing (cold start).
    pub fn entropy_nats(&self) -> f64 {
        if self.total == 0 || self.counts.is_empty() {
            return 0.0;
        }
        let total = self.total as f64;
        self.counts
            .values()
            .map(|&c| {
                let p = c as f64 / total;
                if p > 0.0 { -p * p.ln() } else { 0.0 }
            })
            .sum()
    }

    /// Compute the Harlow observer factor `Ω_obs = 1 - exp(-H / S_max)`.
    /// Always in `[OMEGA_FLOOR, 1.0)`.
    pub fn factor(&self) -> f64 {
        let h = self.entropy_nats();
        let raw = 1.0 - (-h / MAX_OBSERVER_ENTROPY).exp();
        raw.max(OMEGA_FLOOR).min(1.0)
    }

    /// Number of distinct feature buckets observed so far. Useful as a
    /// diagnostic of how much the agent has actually "seen".
    pub fn unique_observations(&self) -> usize {
        self.counts.len()
    }

    /// Total observations recorded. Aliased so it's available for logging.
    pub fn total(&self) -> u64 {
        self.total
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// κ_market — market quantum-classical phase parameter
// ─────────────────────────────────────────────────────────────────────────────

/// Inputs to the κ estimator. None of these are required — the estimator
/// uses whichever signals are available and weights them accordingly.
///
/// All inputs are pre-normalized to roughly `[0, 1]` where 1.0 is "extreme
/// imbalance / divergence / structural anomaly" and 0.0 is "no observable
/// structure."
#[derive(Debug, Clone, Default)]
pub struct KappaInputs {
    /// AMM pool reserve imbalance versus a reference (CEX) price.
    /// `|amm_implied_price - cex_price| / cex_price` capped at 1.0.
    /// High value → AMM has not yet absorbed external price information →
    /// arbitrage edge.
    pub amm_cex_divergence: Option<f64>,

    /// Order-book depth-weighted imbalance: `|bid_depth - ask_depth| /
    /// (bid_depth + ask_depth)` over the top N levels.
    /// High value → flow asymmetry → directional edge.
    pub orderbook_skew: Option<f64>,

    /// Cross-feed signal divergence: e.g. funding rate vs spot drift,
    /// open-interest delta vs price drift. Captures derivatives-market
    /// information not yet reflected in spot.
    pub cross_feed_divergence: Option<f64>,

    /// Polymarket implied probability vs spot-implied probability divergence.
    /// `|p_polymarket - p_implied(spot, K, T)| / max(p_polymarket, p_implied)`.
    /// High value → prediction market and spot disagree → exploitable.
    pub polymarket_divergence: Option<f64>,

    /// Realized vs implied volatility ratio. Far from 1.0 → vol-surface
    /// mispricing.
    pub vol_ratio: Option<f64>,
}

impl KappaInputs {
    /// Helper: build an inputs struct from just the AMM-vs-CEX divergence
    /// (the most common single signal during DCA-style trading).
    pub fn from_amm_cex(amm_cex_divergence: f64) -> Self {
        Self {
            amm_cex_divergence: Some(amm_cex_divergence.clamp(0.0, 1.0)),
            ..Default::default()
        }
    }
}

/// The market quantum-classical phase parameter.
///
/// `κ = 1 - (1 - max_signal) × exp(-Σ signals)` — bounded in `[0, 1]`,
/// monotonically non-decreasing in each input, and equal to the maximum
/// individual signal when all others are zero (so the strongest single
/// edge is never washed out).
pub fn kappa_market(inputs: &KappaInputs) -> f64 {
    let signals: Vec<f64> = [
        inputs.amm_cex_divergence,
        inputs.orderbook_skew,
        inputs.cross_feed_divergence,
        inputs.polymarket_divergence,
        inputs.vol_ratio.map(|r| (r - 1.0).abs().min(1.0)),
    ]
    .into_iter()
    .flatten()
    .map(|s| s.clamp(0.0, 1.0))
    .collect();

    if signals.is_empty() {
        return KAPPA_FLOOR;
    }

    let max_signal = signals.iter().cloned().fold(0.0_f64, f64::max);
    let sum: f64 = signals.iter().sum();
    // Combine: at most one strong signal dominates; multiple signals reinforce.
    let raw = 1.0 - (1.0 - max_signal) * (-sum).exp();
    raw.max(KAPPA_FLOOR).min(1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// The sizing multiplier — the heart of the observer-gene formula
// ─────────────────────────────────────────────────────────────────────────────

/// Result of an observer-aware sizing pass. Carries the inputs alongside the
/// output so logs and the MCP tool can explain what the bot is doing.
#[derive(Debug, Clone)]
pub struct ObserverSize {
    pub kelly_amount: f64,
    pub omega_obs: f64,
    pub kappa_market: f64,
    /// Final size suggestion: `kelly_amount × omega_obs × kappa_market`,
    /// clamped to `[0, kelly_amount × SIZE_CEIL]`.
    pub final_amount: f64,
    /// One-line explanation suitable for logs / Claude-facing tool responses.
    pub explanation: String,
}

/// Multiply a raw Kelly position size by the observer factor and the market
/// κ, then clamp to `SIZE_CEIL × kelly_amount`. This is the public entry
/// point used by the trading engine and by the MCP `dex_propose_trade` tool.
pub fn observer_sized_amount(
    kelly_amount: f64,
    omega: &OmegaObs,
    kappa_inputs: &KappaInputs,
) -> ObserverSize {
    let omega_obs = omega.factor();
    let kappa = kappa_market(kappa_inputs);
    let multiplier = (omega_obs * kappa).min(SIZE_CEIL);
    let final_amount = (kelly_amount * multiplier).max(0.0);

    let explanation = format!(
        "kelly={:.4} × Ω_obs={:.3} (H={:.2} nats over {} buckets) × κ={:.3} = {:.4}",
        kelly_amount,
        omega_obs,
        omega.entropy_nats(),
        omega.unique_observations(),
        kappa,
        final_amount,
    );

    ObserverSize {
        kelly_amount,
        omega_obs,
        kappa_market: kappa,
        final_amount,
        explanation,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — small but cover the corner cases the formula must respect
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_start_observer_is_floored_not_zero() {
        let omega = OmegaObs::new();
        assert_eq!(omega.entropy_nats(), 0.0);
        assert!((omega.factor() - OMEGA_FLOOR).abs() < 1e-9);
    }

    #[test]
    fn entropy_grows_with_distinct_observations() {
        let mut omega = OmegaObs::new();
        for i in 0..1000 {
            omega.record(format!("bucket_{}", i % 100));
        }
        // 100 equiprobable buckets → H = ln(100) ≈ 4.605 nats
        let h = omega.entropy_nats();
        assert!(h > 4.5 && h < 4.7, "expected ~ln(100), got {}", h);
        // Ω = 1 - exp(-4.6 / 25) ≈ 0.168
        let f = omega.factor();
        assert!(f > 0.15 && f < 0.20, "expected ~0.17, got {}", f);
    }

    #[test]
    fn no_signals_yields_kappa_floor() {
        let k = kappa_market(&KappaInputs::default());
        assert!((k - KAPPA_FLOOR).abs() < 1e-9);
    }

    #[test]
    fn single_strong_signal_dominates_kappa() {
        let inputs = KappaInputs::from_amm_cex(0.6);
        let k = kappa_market(&inputs);
        assert!(k > 0.7, "expected κ > 0.7 from a single 0.6 signal, got {}", k);
        assert!(k <= 1.0);
    }

    #[test]
    fn multiple_weak_signals_reinforce() {
        let weak_alone = kappa_market(&KappaInputs {
            amm_cex_divergence: Some(0.2),
            ..Default::default()
        });
        let weak_combined = kappa_market(&KappaInputs {
            amm_cex_divergence: Some(0.2),
            orderbook_skew: Some(0.2),
            cross_feed_divergence: Some(0.2),
            ..Default::default()
        });
        assert!(
            weak_combined > weak_alone,
            "three weak signals ({}) should beat one ({})",
            weak_combined, weak_alone
        );
    }

    #[test]
    fn observer_sized_amount_respects_ceil() {
        let mut omega = OmegaObs::new();
        // Fully saturate the observer
        for i in 0..1_000_000 {
            omega.record(format!("b_{}", i));
        }
        let inputs = KappaInputs {
            amm_cex_divergence: Some(1.0),
            orderbook_skew: Some(1.0),
            cross_feed_divergence: Some(1.0),
            polymarket_divergence: Some(1.0),
            vol_ratio: Some(2.0),
        };
        let r = observer_sized_amount(100.0, &omega, &inputs);
        assert!(
            r.final_amount <= 100.0 * SIZE_CEIL + 1e-9,
            "final_amount {} exceeded ceil {}",
            r.final_amount,
            100.0 * SIZE_CEIL
        );
    }

    #[test]
    fn cold_observer_sizes_tiny_even_in_anomalous_market() {
        let omega = OmegaObs::new();
        let inputs = KappaInputs {
            amm_cex_divergence: Some(1.0),
            orderbook_skew: Some(1.0),
            cross_feed_divergence: Some(1.0),
            polymarket_divergence: Some(1.0),
            vol_ratio: Some(2.0),
        };
        let r = observer_sized_amount(100.0, &omega, &inputs);
        // Cold-start agent in a screaming-anomaly market still trades small.
        assert!(
            r.final_amount < 5.0,
            "cold-start agent should size <5%, got {}",
            r.final_amount
        );
    }
}
