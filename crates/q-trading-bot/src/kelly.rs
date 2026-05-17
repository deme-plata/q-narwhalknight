/// Kelly Criterion + Renormalization Group DCA sizing.
///
/// Combines three mathematical frameworks from the v2 design doc:
///
///   1. Kelly-optimal capital fraction for each DCA interval
///   2. Running coupling α_DCA(L,σ,τ) — renormalization group adaptation
///      β = 0.618034 (golden ratio conjugate), γ = 0.5, δ = 0.236
///   3. Stochastic optimal control: reserve-corrected swap schedule
///
/// All produce a recommended `amount_display` that replaces the flat
/// `amount_per_execution` configured in WaterBotConfig.

use std::collections::VecDeque;
use tracing::debug;

// ── Constants from v2 design doc ─────────────────────────────────────────────

/// Golden ratio conjugate: 1 - φ⁻¹ ≈ 0.3820
const BETA: f64 = 0.618_034;
/// Square-root volatility scaling
const GAMMA: f64 = 0.5;
/// Derived from Kelly criterion fixed-point
const DELTA: f64 = 0.236;
/// Reference liquidity baseline ($100K)
const L0: f64 = 100_000.0;
/// Reference time constant (hours)
const TAU0: f64 = 24.0;
/// Target volatility (5% / day)
const SIGMA_TARGET: f64 = 0.05;
/// Staking yield as risk-free rate
const RISK_FREE_RATE: f64 = 0.05;
/// Maximum Kelly fraction (never bet >25% of capital per interval)
const MAX_KELLY: f64 = 0.25;
/// Maximum price impact per swap (1%)
const MAX_PRICE_IMPACT: f64 = 0.01;

// ── Price history tracker ─────────────────────────────────────────────────────

/// Rolling price window for drift + volatility estimation.
#[derive(Debug, Clone)]
pub struct PriceHistory {
    prices: VecDeque<f64>,
    capacity: usize,
}

impl PriceHistory {
    pub fn new(capacity: usize) -> Self {
        PriceHistory { prices: VecDeque::with_capacity(capacity), capacity }
    }

    pub fn push(&mut self, price: f64) {
        if price > 0.0 {
            if self.prices.len() == self.capacity {
                self.prices.pop_front();
            }
            self.prices.push_back(price);
        }
    }

    pub fn len(&self) -> usize { self.prices.len() }
    pub fn is_empty(&self) -> bool { self.prices.is_empty() }

    /// Annualized drift estimate from log returns (mean log return × periods/year).
    /// Uses interval_secs to scale from per-sample to per-year.
    pub fn drift_annualized(&self, interval_secs: u64) -> f64 {
        let n = self.prices.len();
        if n < 2 { return 0.05; } // assume 5% if no data
        let periods_per_year = 365.25 * 24.0 * 3600.0 / interval_secs as f64;
        let log_returns: Vec<f64> = self.prices.iter()
            .zip(self.prices.iter().skip(1))
            .filter(|(a, _)| **a > 0.0)
            .map(|(a, b)| (b / a).ln())
            .collect();
        if log_returns.is_empty() { return 0.05; }
        let mean_log_return = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
        mean_log_return * periods_per_year
    }

    /// Annualized volatility from rolling log returns.
    pub fn volatility_annualized(&self, interval_secs: u64) -> f64 {
        let n = self.prices.len();
        if n < 3 { return 0.8; } // assume 80% vol for crypto if no data
        let periods_per_year = 365.25 * 24.0 * 3600.0 / interval_secs as f64;
        let log_returns: Vec<f64> = self.prices.iter()
            .zip(self.prices.iter().skip(1))
            .filter(|(a, _)| **a > 0.0)
            .map(|(a, b)| (b / a).ln())
            .collect();
        if log_returns.len() < 2 { return 0.8; }
        let mean = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
        let variance = log_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (log_returns.len() - 1) as f64;
        (variance * periods_per_year).sqrt().max(0.01)
    }

    /// Current price (latest sample).
    pub fn current(&self) -> Option<f64> { self.prices.back().copied() }
}

// ── Kelly fraction ────────────────────────────────────────────────────────────

/// Kelly-optimal fraction of capital to deploy per DCA interval.
///
/// f* = (μ - r) / σ² × impact_correction
///
/// Capped at MAX_KELLY (25%) for safety.
pub fn kelly_fraction(mu: f64, sigma: f64, pool_depth_display: f64) -> f64 {
    if sigma < 1e-6 { return 0.0; }
    let kelly_raw = (mu - RISK_FREE_RATE) / (sigma * sigma);
    // Pool impact penalty: deeper pools allow larger fractions
    let impact_penalty = 1.0 - (0.001 / pool_depth_display.sqrt().max(1.0));
    let f = kelly_raw * impact_penalty;
    f.clamp(0.0, MAX_KELLY)
}

// ── Renormalization group running coupling ────────────────────────────────────

/// Running coupling α_DCA(L, σ, τ) from the renormalization group analogy.
///
/// α_DCA = α₀ × (L₀/L)^β × (σ_target/σ)^γ × (τ₀/τ)^δ
///
/// - L: pool TVL in display units
/// - sigma: annualized vol
/// - hours_since_last_trade: τ in hours
pub fn running_coupling(alpha0: f64, tvl_display: f64, sigma: f64, hours_since_last_trade: f64) -> f64 {
    let l = tvl_display.max(1.0);
    let tau = hours_since_last_trade.max(0.1);

    let liquidity_factor = (L0 / l).powf(BETA);
    let volatility_factor = (SIGMA_TARGET / sigma.max(0.001)).powf(GAMMA);
    let time_factor = (TAU0 / tau).powf(DELTA);

    let alpha = alpha0 * liquidity_factor * volatility_factor * time_factor;
    // Clamp to sensible bounds: 10% – 300% of base alpha0
    alpha.clamp(alpha0 * 0.1, alpha0 * 3.0)
}

// ── Stochastic optimal control schedule ──────────────────────────────────────

/// Stochastic optimal control swap sizing.
///
/// s* = T/N + α(R - R̄)  (simplified HJB solution for constant-product AMM)
///
/// Adjusts the base amount upward when the pool is imbalanced in our favour
/// and downward when we would cause more impact.
pub fn optimal_schedule_amount(
    base_amount: f64,
    reserve_in: u128,
    reserve_out: u128,
    mean_reserve_in: f64,
) -> f64 {
    if reserve_in == 0 || reserve_out == 0 { return base_amount; }
    let r_in = reserve_in as f64;
    let r_out = reserve_out as f64;
    // AMM curvature term: ∂²impact/∂s² ∝ R_out / (R_in + s)³
    let curvature = r_out / (r_in + base_amount).powi(3);
    // Reserve correction: positive when R_in > mean (more in = cheaper buy)
    let reserve_deviation = r_in - mean_reserve_in;
    let correction = -reserve_deviation / (2.0 * r_out.max(1.0)) * curvature;
    // Clamp to ±50% of base
    let adj = correction.clamp(-base_amount * 0.5, base_amount * 0.5);
    (base_amount + adj).max(base_amount * 0.01)
}

// ── Max impact amount ─────────────────────────────────────────────────────────

/// Maximum amount we can swap without exceeding MAX_PRICE_IMPACT.
///
/// From constant-product AMM: impact = s/(R_x + s)
/// Solving for s: s* = R_x × impact / (1 - impact)
pub fn max_impact_amount_display(reserve_in_display: f64) -> f64 {
    reserve_in_display * MAX_PRICE_IMPACT / (1.0 - MAX_PRICE_IMPACT)
}

// ── Volatility dampener (v1 formula) ─────────────────────────────────────────

/// 1 / (1 + σ/σ_target) — reduces size during high-volatility periods.
pub fn volatility_dampener(sigma: f64) -> f64 {
    1.0 / (1.0 + (sigma / SIGMA_TARGET).max(0.0))
}

// ── Main entry: compute adaptive DCA amount ───────────────────────────────────

/// All-in-one: compute the Kelly + RG + stochastic-control adjusted swap amount.
///
/// Returns `(adaptive_amount, description)` where description explains the
/// adjustment for logging.
pub fn adaptive_dca_amount(
    configured_amount: f64,
    capital_display: f64,
    history: &PriceHistory,
    interval_secs: u64,
    tvl_display: f64,
    reserve_in: u128,
    reserve_out: u128,
    mean_reserve_in: f64,
    hours_since_last_trade: f64,
) -> (f64, String) {
    let mu = history.drift_annualized(interval_secs);
    let sigma = history.volatility_annualized(interval_secs);

    // 1. Kelly fraction → target capital per interval
    let kf = kelly_fraction(mu, sigma, tvl_display);
    let kelly_amount = if capital_display > 0.0 && kf > 0.0 {
        capital_display * kf
    } else {
        configured_amount
    };

    // 2. Renormalization group coupling → scale kelly_amount
    let alpha0 = configured_amount / capital_display.max(1.0);
    let rg_alpha = running_coupling(alpha0, tvl_display, sigma, hours_since_last_trade);
    let rg_amount = capital_display * rg_alpha;

    // 3. Blend Kelly and RG (equal weight — both are evidence of same truth)
    let blended = (kelly_amount + rg_amount) / 2.0;

    // 4. Volatility dampener
    let dampened = blended * volatility_dampener(sigma);

    // 5. Stochastic reserve correction
    let reserve_corrected = optimal_schedule_amount(dampened, reserve_in, reserve_out, mean_reserve_in);

    // 6. Hard cap: never exceed max impact amount
    let reserve_in_display = (reserve_in as f64) / 1e24;
    let max_by_impact = max_impact_amount_display(reserve_in_display);
    let final_amount = reserve_corrected
        .min(max_by_impact)
        .min(configured_amount * 3.0) // safety: never >3× configured
        .max(configured_amount * 0.1); // never <10% of configured

    let desc = format!(
        "μ={:.2}% σ={:.1}% Kelly={:.4} RG_α={:.4} dampener={:.2} max_impact={:.4}",
        mu * 100.0, sigma * 100.0, kf, rg_alpha,
        volatility_dampener(sigma), max_by_impact,
    );
    debug!("📐 Kelly/RG sizing: {} → {:.4} (configured {:.4})", desc, final_amount, configured_amount);

    (final_amount, desc)
}
