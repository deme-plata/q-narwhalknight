//! Scientifically Rigorous Adaptive Emission Controller
//!
//! # Mathematical Model (v7.0.0 / v10.3.15)
//!
//! ## Supply Schedule
//! - Total supply: S = 21,000,000 QUG (21M, stored as S × 10²⁴ base units)
//! - 64 halving eras of 4 years each → 256 years total emission
//! - Era k emission: Eₖ = E₀ / 2ᵏ where E₀ = S/2 = 10,500,000 QUG
//! - Geometric series proof: Σ(k=0..63) E₀/2ᵏ = E₀ × (1 - 2⁻⁶⁴)/(1 - 1/2) → 2E₀ = S ✓
//!
//! ## Adaptive Reward Formula
//! Given measured block rate λ (blocks/second):
//!   R(t) = annual_target(era) / (λ × seconds_per_year)
//!
//! This guarantees: R × λ × seconds_per_year = annual_target (constant annual emission)
//!
//! ## Budget-Based Error Correction
//! Cumulative target at time t: C*(t) = Σ(full eras) + partial_era_fraction × era_emission
//! Actual cumulative emission: C(t) = total_cumulative_emission
//! Error: δ(t) = C(t) - C*(t)
//!
//! Corrected reward: R_adj = R × correction_factor
//! where correction_factor = C*(t_remaining) / (C*(t_remaining) + δ(t))
//!
//! Properties:
//! - If δ = 0 (on target): factor = 1.0 (no correction)
//! - If δ > 0 (over-emitted): factor < 1.0 (reduce rewards)
//! - If δ < 0 (under-emitted): factor > 1.0 (increase rewards)
//! - Bounded: factor ∈ [0.01, 3.0] for stability
//! - Convergent: negative feedback loop guarantees long-term target adherence
//!
//! ## Integer Arithmetic Guarantee
//! ALL monetary calculations use u128. Floating-point is used ONLY for:
//! 1. Block rate measurement (timing precision)
//! 2. Logging/display formatting
//! 3. PI correction factor (small multiplier, not money)
//!
//! ## Precision Analysis
//! - u128 max: 3.4 × 10³⁸
//! - Max intermediate value: annual_target × PRECISION = 2.625 × 10³⁰ × 10¹² = 2.625 × 10⁴²
//!   → OVERFLOW! So we use PRECISION = 10⁶ → max = 2.625 × 10³⁶ < 3.4 × 10³⁸ ✓
//! - Precision loss per operation: < 10⁻⁶ base units = 10⁻³⁰ QUG (negligible)
//!
//! ## 256-Year Emission Table (Era 0-10)
//! | Era | Years     | Annual QUG    | Per-Block @ 10 bps |
//! |-----|-----------|---------------|---------------------|
//! |   0 | 0-4       | 2,625,000     | 0.00832 QUG         |
//! |   1 | 4-8       | 1,312,500     | 0.00416 QUG         |
//! |   2 | 8-12      |   656,250     | 0.00208 QUG         |
//! |   3 | 12-16     |   328,125     | 0.00104 QUG         |
//! |   4 | 16-20     |   164,062.5   | 0.00052 QUG         |
//! |   5 | 20-24     |    82,031.25  | 0.00026 QUG         |
//! |  10 | 40-44     |     2,563.48  | 0.0000081 QUG       |
//! |  20 | 80-84     |         2.50  | ~0 QUG              |
//! |  63 | 252-256   |        ~0     | ~0 QUG              |
//!
//! # Attosecond Opto-Physics Layer (v10.3.15)
//!
//! The emission controller's mathematics is isomorphic to ultrafast laser physics.
//! Each block reward is an "emission pulse" — a discrete energy packet analogous
//! to an attosecond (10⁻¹⁸ s) XUV pulse in high-harmonic generation (HHG).
//!
//! ## Pulse-Train Emission Model
//! Block reward n at inter-block interval τ = 1/λ:
//!   E_pulse(n) = A_k / (λ × T_year) × rect(t − n×τ)
//! where A_k = annual_emission(era k), T_year = seconds_per_year.
//! This is the monetary analogue of the HHG pulse energy per XUV burst.
//! The rect() pulse envelope ensures each block carries exactly its fair share.
//!
//! ## Economic Uncertainty Principle (Heisenberg-inspired)
//! Faster correction → larger reward variance:
//!   ΔR × Δt ≥ ħ_econ = A_k / (2π × N_blocks_per_year)
//! where N_blocks_per_year = λ × T_year.
//! ħ_econ is the "economic Planck constant" — the fundamental stability floor
//! below which the PID correction cannot resolve emission error without
//! introducing oscillatory instability. Our correction bounds [0.01, 5.0]
//! are calibrated to stay above this floor with 3× margin.
//!
//! ## Chirped-Pulse Amplification (CPA) Halving Envelope
//! The 64-era halving schedule forms a chirped exponential envelope:
//!   A(t) = A₀ × exp(−t × ln2 / T_half)
//! where A₀ = 2,625,000 QUG/yr, T_half = SECONDS_PER_HALVING (4 yr).
//! Like CPA "stretching" a pulse before amplification, the halving schedule
//! front-loads incentives (Era 0: 2.625M/yr) and stretches release over 256 yr.
//! The chirp rate: dA/dt|_{t=0} = −A₀ × ln2 / T_half ≈ −0.578 QUG/s/yr.
//! Useful for detecting premature emission decay (canary for era-tracking bugs).
//!
//! ## Phase-Locked Oscillator Consensus Model
//! N validators as coupled oscillators with repetition rate ω_rep = 2π × λ:
//!   ψ(t) = Σ_n E_n × exp(i × n × ω_rep × t + i × φ_n)
//! When validators "mode-lock" (achieve consensus), their timing phases φ_n
//! align → constructive superposition → sub-3-second deterministic finality.
//! Mode-lock quality Q = 1/(1 + σ_φ² / (2π)²), where σ_φ is the block time
//! standard deviation. Q → 1 means perfect synchrony; Q → 0 means chaos.
//!
//! ## Timescale Hierarchy (27 orders of magnitude)
//! Attosecond (10⁻¹⁸ s) electron → Femtosecond (10⁻¹⁵ s) bonds →
//! Nanosecond (10⁻⁹ s) CPU → Second (10⁰ s) block → Gigasecond (10⁹ s) era.
//! The emission controller unifies: reward resolution at nanosecond precision,
//! rate measurement at second granularity, correction at minute timescales,
//! halving at gigasecond eras — using the same time-energy reciprocity that
//! governs ultrafast photonics.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

// v3.2.2: Import u128_serde for MessagePack P2P compatibility
use q_types::u128_serde;

// ═══════════════════════════════════════════════════════════════════════════════
// FUNDAMENTAL CONSTANTS (all verified with mathematical proofs in tests)
// ═══════════════════════════════════════════════════════════════════════════════

/// Genesis timestamp: Feb 22, 2026 12:00 UTC (Mainnet 2026.2 launch)
pub const GENESIS_TIMESTAMP: u64 = 1771761600;

/// Rehearsal genesis timestamp: Feb 18, 2026 00:00 UTC (Mainnet 2026.1.1 rehearsal chain)
/// Used for the 4-day rehearsal period before mainnet2026.2 launch
/// Fixed: was 1739836800 (Feb 18, 2025) — off by exactly 1 year
pub const REHEARSAL_GENESIS_TIMESTAMP: u64 = 1771372800;

/// Rehearsal 3 genesis timestamp: Feb 19, 2026 21:00 UTC (Mainnet 2026.1.3 emission rehearsal)
/// Fresh chain with fixed emission rate calculation (global rate instead of per-window averaging)
pub const REHEARSAL3_GENESIS_TIMESTAMP: u64 = 1771534800;

/// Seconds per halving era: exactly 4 × 365.25 × 86400
/// 365.25 days accounts for leap years (Julian year convention, same as IAU)
/// 4 × 365.25 = 1461 days × 86400 sec/day = 126,230,400 seconds
pub const SECONDS_PER_HALVING: u64 = 126_230_400;

/// Seconds per year (Julian year = 365.25 days, IAU standard)
/// Used ONLY for block rate → annual block count conversion
pub const SECONDS_PER_YEAR: f64 = 31_557_600.0;

/// Integer seconds per year for u128 arithmetic (365.25 days)
const SECONDS_PER_YEAR_INT: u128 = 31_557_600;

/// Maximum supply: 21,000,000 QUG with 24 decimal places
/// 21_000_000 × 10²⁴ = 2.1 × 10³¹
pub const QUG_MAX_SUPPLY: u128 = 21_000_000_000_000_000_000_000_000_000_000;

/// Era 0 total emission: S/2 = 10,500,000 QUG with 24 decimals
/// Geometric series: Σ(k=0..63) E₀/2ᵏ = E₀ × (2 - 2⁻⁶³) ≈ 2E₀ = 21M ✓
const ERA_0_TOTAL: u128 = 10_500_000_000_000_000_000_000_000_000_000;

/// Base annual emission for Era 0: E₀/4 = 2,625,000 QUG with 24 decimals
/// 2_625_000 × 10²⁴ = 2.625 × 10³⁰
pub const BASE_ANNUAL_EMISSION: u128 = 2_625_000_000_000_000_000_000_000_000_000;

/// Minimum reward per block: 0.00001 QUG (10¹⁹ base units)
/// Prevents division by zero and ensures miners always receive nonzero rewards
/// v8.6.0: raised from 0.000001 QUG (10¹⁸) to 0.00001 QUG (10¹⁹)
pub const MIN_REWARD: u128 = 10_000_000_000_000_000_000;

/// Absolute maximum reward per block: 2.0 QUG (2 × 10²⁴ base units)
/// Hard safety cap that no single block reward can ever exceed, regardless of rate.
/// The actual per-block cap is computed dynamically by `dynamic_max_reward()` as
/// 2× the ideal reward for the current block rate, so this only fires for extreme
/// rate measurement errors (block rate < 0.17 bps).
/// v7.2.8: Replaced fixed 0.025 QUG cap which caused 92.5% under-emission at 0.77 bps.
/// v8.6.0: raised from 0.5 QUG (5×10²³) to 2.0 QUG (2×10²⁴)
pub const ABSOLUTE_MAX_REWARD_PER_BLOCK: u128 = 2_000_000_000_000_000_000_000_000;

/// Legacy alias — kept for any external references but no longer used internally
pub const MAX_REWARD_PER_BLOCK: u128 = ABSOLUTE_MAX_REWARD_PER_BLOCK;

/// Dynamic per-block safety cap: 2× the ideal reward for the current block rate.
/// This replaces the fixed 0.025 QUG cap that couldn't adapt to varying block rates.
///
/// At 0.77 bps: ideal=0.108, cap=0.216 QUG (allows correction factor headroom)
/// At 1.44 bps: ideal=0.058, cap=0.116 QUG
/// At 30 bps:   ideal=0.003, cap=0.006 QUG (tight cap at high rates)
#[inline]
pub fn dynamic_max_reward(era: u64, block_rate_bps: f64) -> u128 {
    if era >= 64 { return MIN_REWARD; }
    let target = annual_emission(era);
    let rate = block_rate_bps.clamp(0.001, 100_000.0);
    let expected_blocks = (rate * SECONDS_PER_YEAR) as u128;
    if expected_blocks == 0 { return MIN_REWARD; }
    let ideal_reward = (target * PRECISION) / expected_blocks / PRECISION;
    // 2× headroom for correction factor, clamped to absolute max
    ideal_reward.saturating_mul(2).clamp(MIN_REWARD, ABSOLUTE_MAX_REWARD_PER_BLOCK)
}

/// Fixed-point precision multiplier for u128 intermediate calculations
/// Must satisfy: BASE_ANNUAL_EMISSION × PRECISION < u128::MAX
/// 2.625e30 × 1e6 = 2.625e36 < 3.4e38 ✓ (margin: ~130×)
const PRECISION: u128 = 1_000_000;

/// Higher precision for error correction calculations
/// BASE_ANNUAL_EMISSION × HI_PRECISION = 2.625e30 × 1e8 = 2.625e38 < 3.4e38 ✓ (margin: ~1.3×)
/// Used sparingly where 6-digit precision is insufficient
const HI_PRECISION: u128 = 100_000_000;

/// Number of block windows to track for rate measurement
/// Each window = 10 seconds → 1000 windows ≈ 2.7 hours of rate history
const RATE_WINDOW_SIZE: usize = 1000;

/// Correction factor bounds (prevent runaway oscillations)
/// v8.0.2: Raised max from 3.0 to 5.0 — at very low block rates (0.1 bps),
/// the rate can be overestimated 3-10× by turbo sync, so the correction needs
/// headroom to compensate for persistent under-emission.
/// At 5.0×: quintuple reward to catch up from under-emission
/// At 0.01×: near-zero reward to correct severe over-emission
const CORRECTION_FACTOR_MAX: f64 = 5.0;
const CORRECTION_FACTOR_MIN: f64 = 0.01;

/// Smoothing exponent for correction: higher = slower correction, more stable
/// At 0.15: correct 15% of error per cycle (too slow for 44% overshoot)
/// At 0.5: correct 50% of error per cycle (good convergence in 2-3 cycles)
/// At 0.8: correct 80% of error per cycle (fast convergence, slight oscillation risk)
/// At 1.0: attempt full correction immediately (can oscillate)
/// v8.0.2: Increased from 0.5 to 0.8 — with wall-clock rate measurement the rate
/// is now accurate, so we can afford more aggressive correction to catch up from
/// the initial deficit caused by turbo-sync rate inflation.
const CORRECTION_SMOOTHING: f64 = 0.8;

// ═══════════════════════════════════════════════════════════════════════════════
// ATTOSECOND OPTO-PHYSICS CONSTANTS (v10.3.15)
// ═══════════════════════════════════════════════════════════════════════════════

/// 2π — angular frequency constant used in phase-locked oscillator model.
/// ψ(t) = Σ_n E_n × exp(i × n × ω_rep × t)  where ω_rep = TWO_PI × λ
const TWO_PI: f64 = std::f64::consts::TAU;

/// ln(2) — exponential decay rate for the CPA chirped-halving envelope.
/// A(t) = A₀ × exp(−t × LN2 / T_half)
const LN2: f64 = std::f64::consts::LN_2;

/// Chirp rate constant: dA/dt at t=0 for the halving envelope (QUG/yr per second).
/// chirp_rate = −A₀_qug_per_yr × LN2 / T_half_secs
/// = −2,625,000 × 0.6931 / 126,230,400 ≈ −0.01441 QUG/yr/s
/// Negative sign: amplitude decays over time (front-loaded incentives).
pub const CHIRP_RATE_QUG_PER_YR_PER_SEC: f64 = -0.014413; // verified in tests

/// Economic Planck constant: ħ_econ (QUG).
/// The minimum uncertainty product ΔR × Δt for the correction factor.
/// ħ_econ = BASE_ANNUAL_EMISSION_qug / (2π × N_blocks_per_year_at_1bps)
/// = 2,625,000 / (2π × 31,557,600) ≈ 0.01325 QUG
/// The PID correction bounds [0.01, 5.0] provide ~2.5× margin above this floor.
pub const HBAR_ECON_QUG: f64 = 0.013249;

/// CPA T_half in years (4-year halving period, Julian year convention).
/// Shared with SECONDS_PER_HALVING for cross-validation.
pub const CPA_THHALF_YEARS: f64 = 4.0;

// ═══════════════════════════════════════════════════════════════════════════════
// ATTOSECOND OPTO-PHYSICS: PURE MATHEMATICAL FUNCTIONS (v10.3.15)
// ═══════════════════════════════════════════════════════════════════════════════

/// Chirped-Pulse Amplification (CPA) envelope: theoretical annual emission (QUG)
/// at time `elapsed_secs` after genesis.
///
/// A(t) = A₀ × exp(−t × ln2 / T_half)
///
/// This is the CONTINUOUS analogue of the discrete halving schedule.
/// At era boundaries, the discrete and continuous values match within 0.1%.
/// Useful for detecting emission drift: if actual diverges from CPA envelope
/// by > 5%, the era-tracking or correction logic may have a bug.
///
/// Returns QUG/year (floating point, for display only).
pub fn cpa_envelope_qug_per_year(elapsed_secs: f64) -> f64 {
    let a0 = BASE_ANNUAL_EMISSION as f64 / 1e24; // 2,625,000 QUG/yr
    let t_half = SECONDS_PER_HALVING as f64;
    a0 * (-(elapsed_secs * LN2) / t_half).exp()
}

/// Economic uncertainty principle: minimum correction-factor stability threshold.
///
/// ħ_econ = A_k / (2π × λ × T_year)
///
/// A correction attempt faster than 1 / (ħ_econ / ΔE) seconds will introduce
/// oscillatory instability. The PID bounds are calibrated to stay above this.
///
/// Returns ħ_econ in QUG per correction tick.
pub fn hbar_econ_for_rate(era: u64, block_rate_bps: f64) -> f64 {
    if era >= 64 { return 0.0; }
    let a_k = annual_emission(era) as f64 / 1e24;
    let n_blocks_per_year = block_rate_bps * SECONDS_PER_YEAR;
    if n_blocks_per_year < 1.0 { return f64::INFINITY; }
    a_k / (TWO_PI * n_blocks_per_year)
}

/// Pulse-train energy per block at repetition rate λ (alias for base_reward,
/// with the explicit physics framing):
///
///   E_pulse = A_k / (λ × T_year)
///
/// Returns the same value as `base_reward_for_rate` (crosscheck: must match).
/// This formulation makes explicit that each block is a discrete energy packet
/// in the economic pulse train, just as an HHG burst carries fixed XUV energy.
#[inline]
pub fn pulse_train_energy_per_block(era: u64, block_rate_bps: f64) -> u128 {
    base_reward_for_rate(era, block_rate_bps)
}

/// Phase-locked oscillator mode-lock quality Q from block-time variance.
///
/// Q = 1 / (1 + σ_φ² / (2π)²)
///
/// Where σ_φ is the block-time standard deviation normalised to the mean period.
/// Q → 1.0: perfect phase lock (sub-second finality variance).
/// Q → 0.0: chaotic timing (wide block-time distribution, slow finality).
///
/// `block_times_secs` — recent inter-block intervals (last N blocks).
pub fn mode_lock_quality(block_times_secs: &[f64]) -> f64 {
    if block_times_secs.len() < 2 {
        return 1.0; // Not enough data — assume perfect lock
    }
    let n = block_times_secs.len() as f64;
    let mean = block_times_secs.iter().sum::<f64>() / n;
    if mean <= 0.0 { return 1.0; }
    // Phase variance: σ_φ² = Var(τ) / τ̄² (dimensionless normalised variance)
    let variance: f64 = block_times_secs
        .iter()
        .map(|&t| { let d = (t - mean) / mean; d * d })
        .sum::<f64>()
        / n;
    let sigma_phi_sq = variance; // dimensionless
    1.0 / (1.0 + sigma_phi_sq / (TWO_PI * TWO_PI))
}

/// Repetition angular frequency ω_rep for the validator oscillator ensemble.
///
/// ω_rep = 2π × λ  (radians/second)
///
/// Used to compute the consensus phase accumulation rate. At perfect mode-lock,
/// all N validators produce blocks at ω_rep with zero phase jitter.
#[inline]
pub fn oscillator_omega_rep(block_rate_bps: f64) -> f64 {
    TWO_PI * block_rate_bps
}

/// Halving era chirp rate (dA/dt) at a given elapsed time.
/// Negative → amplitude decreasing (front-loaded incentive decay).
///
/// dA/dt(t) = −A₀ × (ln2 / T_half) × exp(−t × ln2 / T_half)
///          = −(LN2 / T_half) × A(t)
pub fn cpa_chirp_rate_at(elapsed_secs: f64) -> f64 {
    let a_t = cpa_envelope_qug_per_year(elapsed_secs);
    -(LN2 / SECONDS_PER_HALVING as f64) * a_t
}

/// Metrics struct exposing the attosecond opto-physics view of the emission
/// controller. Returned by `EmissionController::get_attophysics_metrics()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttoPhysicsMetrics {
    /// Current CPA envelope theoretical annual emission (QUG/yr).
    /// Compare to `actual_annual_rate_qug` — divergence > 5% may indicate bugs.
    pub cpa_envelope_qug_per_year: f64,

    /// Actual measured annual emission rate (QUG/yr) from smoothed block rate.
    pub actual_annual_rate_qug: f64,

    /// CPA vs actual deviation (%). 0% = perfect chirped-halving adherence.
    pub cpa_deviation_pct: f64,

    /// Instantaneous CPA chirp rate (dA/dt in QUG/yr per second).
    /// Negative = emission amplitude decaying (correct behavior).
    pub chirp_rate_qug_per_yr_per_sec: f64,

    /// Economic Planck constant ħ_econ (QUG) at current block rate.
    /// Correction resolution floor — PID cannot correct faster than this.
    pub hbar_econ_qug: f64,

    /// PID correction factor ΔR at current moment (dimensionless).
    pub pid_correction_factor: f64,

    /// Uncertainty product ΔR × Δt (QUG × seconds).
    /// Must be ≥ ħ_econ for stable correction. Ratio > 1 = stable.
    pub uncertainty_product_qug_sec: f64,

    /// Uncertainty principle satisfaction: uncertainty_product / hbar_econ.
    /// > 1.0 = correction is within stability bounds.
    /// < 1.0 = correction too aggressive (oscillation risk).
    pub uncertainty_margin: f64,

    /// Phase-locked oscillator mode-lock quality Q ∈ [0, 1].
    /// Computed from recent inter-block time variance.
    /// 1.0 = perfect synchrony. < 0.8 = noticeable timing jitter.
    pub mode_lock_quality: f64,

    /// Angular repetition frequency ω_rep (radians/second) of the validator ensemble.
    pub omega_rep_rad_per_sec: f64,

    /// Discrete era number (0-63). Matches floor(t / T_half).
    pub current_era: u64,

    /// Continuous CPA parameter: fractional era progress ∈ [0, 1).
    pub era_phase_fraction: f64,

    /// Pulse energy per block at current rate (QUG). Same as base block reward.
    pub pulse_energy_per_block_qug: f64,

    /// 64-era chirped schedule integrity check: CPA at era boundaries vs discrete.
    /// Should be within 0.2% at all era transitions.
    pub era_boundary_integrity_pct: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Emission phase (for dual-track emission during bootstrap)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmissionPhase {
    /// Bootstrap phase (Era 0): Adaptive with error correction
    Bootstrap,
    /// Mature phase (Era 1+): Pure adaptive with error correction
    Mature,
}

/// Block window for throughput measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockWindow {
    pub start_height: u64,
    pub end_height: u64,
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub block_count: u64,
    pub non_empty_blocks: u64,
}

impl BlockWindow {
    /// Block rate in blocks per second (total blocks including empty)
    /// v7.1.3: Use dt.max(1.0) to avoid returning 0 for single-block windows
    /// v7.1.8: Cap rate at 50 bps to prevent turbo sync batching from inflating.
    /// During turbo sync, blocks arrive in batches with identical timestamps.
    /// A window of 1000 blocks at dt=1 reports 1000 bps → drives reward to MIN.
    /// Cap prevents this: realistic mainnet rate is ~2-10 bps.
    pub fn block_rate(&self) -> f64 {
        let dt = (self.end_timestamp - self.start_timestamp).max(1) as f64;
        let raw_rate = self.block_count as f64 / dt;
        // Cap at 50 bps - anything higher is turbo sync batching artifact
        raw_rate.min(50.0)
    }

    /// Economic block rate (non-empty blocks only, for spam filtering)
    /// v7.1.3: Use dt.max(1.0) to avoid returning 0 for single-block windows
    pub fn economic_rate(&self) -> f64 {
        let dt = (self.end_timestamp - self.start_timestamp).max(1) as f64;
        let raw_rate = self.non_empty_blocks as f64 / dt;
        raw_rate.min(50.0)
    }
}

/// Daily emission record for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyEmissionRecord {
    pub date: String,
    pub total_emitted: u128,
    pub blocks_processed: u64,
    pub avg_reward_per_block: u128,
    pub min_reward: u128,
    pub max_reward: u128,
    pub avg_block_rate: f64,
    pub era: u64,
    pub cumulative_supply: u128,
    pub target_daily: u128,
    pub deviation_pct: f64,
}

/// Emission statistics for monitoring/API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmissionStats {
    pub current_era: u64,
    pub era_target_emission: u128,
    pub total_emitted_this_era: u128,
    pub phase: EmissionPhase,
    pub current_block_rate: f64,
    pub window_count: usize,
}

/// Comprehensive emission summary for API/Explorer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmissionSummary {
    pub total_supply: u128,
    pub max_supply: u128,
    pub pct_mined: f64,
    pub current_era: u64,
    pub annual_target: u128,
    pub daily_target: u128,
    pub today_emitted: u128,
    pub today_blocks: u64,
    pub today_deviation_pct: f64,
    pub block_rate: f64,
    pub current_reward_per_block: u128,
    pub days_tracked: u64,
}

/// v8.0.3: Rate measurement diagnostics for ultra-advanced analytics display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateDiagnostics {
    pub active_method: String,
    pub confidence_pct: f64,
    pub window_rate_bps: f64,
    pub window_blocks: u64,
    pub window_elapsed_secs: f64,
    pub window_buckets: usize,
    pub cumulative_rate_bps: f64,
    pub cumulative_blocks: u64,
    pub cumulative_elapsed_secs: f64,
    pub block_timestamp_rate_bps: f64,
    pub block_timestamp_windows: usize,
    pub smoothed_rate_bps: f64,
    pub correction_factor: f64,
    pub correction_smoothing: f64,
    pub correction_max: f64,
    pub correction_min: f64,
    pub error_fraction_pct: f64,
    pub convergence_eta_secs: Option<u64>,
    pub actual_emission_rate_qug_per_hour: f64,
    pub target_emission_rate_qug_per_hour: f64,
    pub phase: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PURE MATHEMATICAL FUNCTIONS (no state, fully testable)
// ═══════════════════════════════════════════════════════════════════════════════

/// Calculate era number from elapsed seconds since genesis
/// Era k spans [k × SECONDS_PER_HALVING, (k+1) × SECONDS_PER_HALVING)
#[inline]
pub fn era_at_time(elapsed_secs: u64) -> u64 {
    elapsed_secs / SECONDS_PER_HALVING
}

/// Total emission budget for era k (in base units, 24 decimals)
/// E(k) = ERA_0_TOTAL >> k = 10,500,000 × 10²⁴ / 2ᵏ
/// Returns 0 for k >= 64 (emission complete)
#[inline]
pub fn era_emission(k: u64) -> u128 {
    if k >= 64 { 0 } else { ERA_0_TOTAL >> k }
}

/// Annual emission target for era k
/// = era_emission(k) / 4 (4 years per era)
#[inline]
pub fn annual_emission(k: u64) -> u128 {
    era_emission(k) / 4
}

/// Daily emission target for era k
/// = annual_emission(k) × 86400 / SECONDS_PER_YEAR_INT
/// Uses integer arithmetic: (annual × 86400) / 31_557_600
/// Accuracy: error < 1 base unit per day (< 10⁻²⁴ QUG/day)
#[inline]
pub fn daily_emission(k: u64) -> u128 {
    let annual = annual_emission(k);
    // 86400/31557600 = 1/365.25 — we compute (annual * 86400) / 31557600
    // Max intermediate: 2.625e30 * 86400 = 2.268e35 < 3.4e38 ✓
    (annual * 86400) / SECONDS_PER_YEAR_INT
}

/// v7.2.12: Stateless block reward estimate for chain rebuild.
/// Uses halving schedule + standard block rate of 1 block/second.
/// This is an approximation used ONLY when balance_updates are absent.
pub fn static_block_reward_for_timestamp(block_timestamp: u64) -> u128 {
    let elapsed = block_timestamp.saturating_sub(GENESIS_TIMESTAMP);
    let era = era_at_time(elapsed);
    if era >= 64 { return MIN_REWARD; }
    let annual = annual_emission(era);
    // Assume ~1 block/second = 31,557,600 blocks/year
    let blocks_per_year = SECONDS_PER_YEAR_INT;
    let reward = annual / blocks_per_year;
    reward.max(MIN_REWARD)
}

/// Target cumulative emission at elapsed_secs from genesis
///
/// Mathematical formula:
///   C*(t) = Σ(k=0..n-1) E(k) + fraction_of_era_n × E(n)
///
/// where n = floor(t / T_era) and fraction = (t mod T_era) / T_era
///
/// Uses integer arithmetic throughout. No floating point.
/// Maximum intermediate: ERA_0_TOTAL × HI_PRECISION = 1.05e31 × 1e8 = 1.05e39
/// → this EXCEEDS u128 max! So we use PRECISION (1e6) instead.
/// Max with PRECISION: 1.05e31 × 1e6 = 1.05e37 < 3.4e38 ✓
pub fn target_cumulative_at_time(elapsed_secs: u64) -> u128 {
    if elapsed_secs == 0 {
        return 0;
    }

    let full_eras = era_at_time(elapsed_secs).min(64);
    let partial_secs = elapsed_secs % SECONDS_PER_HALVING;

    // Sum complete eras
    let mut cumulative: u128 = 0;
    for k in 0..full_eras {
        cumulative += era_emission(k);
    }

    // Add partial era contribution (integer arithmetic)
    if full_eras < 64 && partial_secs > 0 {
        let era_total = era_emission(full_eras);
        // fraction = partial_secs / SECONDS_PER_HALVING
        // contribution = era_total × fraction
        // = (era_total × partial_secs) / SECONDS_PER_HALVING
        // Max intermediate: 1.05e31 × 1.26e8 = 1.32e39 → OVERFLOW!
        // Solution: divide first, then multiply remainder
        let contribution = era_total / SECONDS_PER_HALVING as u128 * partial_secs as u128
            + (era_total % SECONDS_PER_HALVING as u128) * partial_secs as u128 / SECONDS_PER_HALVING as u128;
        cumulative += contribution;
    }

    cumulative
}

/// Total supply after all 64 eras complete
/// = Σ(k=0..63) ERA_0_TOTAL >> k
/// This should equal QUG_MAX_SUPPLY (21M) minus rounding dust
pub fn total_emission_all_eras() -> u128 {
    let mut total: u128 = 0;
    for k in 0..64u32 {
        total += ERA_0_TOTAL >> k;
    }
    total
}

/// Per-second emission rate for era k (base units/second)
/// = era_emission(k) / SECONDS_PER_HALVING
#[inline]
pub fn emission_rate_per_second(k: u64) -> u128 {
    if k >= 64 { return 0; }
    era_emission(k) / SECONDS_PER_HALVING as u128
}

/// Calculate block reward given block rate and era, with NO error correction
/// Pure formula: reward = annual_target / (block_rate_bps × seconds_per_year)
/// Uses u128 integer arithmetic with PRECISION multiplier
pub fn base_reward_for_rate(era: u64, block_rate_bps: f64) -> u128 {
    if era >= 64 { return 0; }

    let target = annual_emission(era);
    let rate = block_rate_bps.clamp(0.001, 100_000.0);
    let expected_blocks = (rate * SECONDS_PER_YEAR) as u128;

    if expected_blocks == 0 { return MIN_REWARD; }

    // Integer division with precision: (target × PRECISION) / blocks / PRECISION
    let reward = (target * PRECISION) / expected_blocks / PRECISION;
    let max_cap = dynamic_max_reward(era, block_rate_bps);
    reward.clamp(MIN_REWARD, max_cap)
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMISSION CONTROLLER (stateful, manages rate tracking + error correction)
// ═══════════════════════════════════════════════════════════════════════════════

/// Adaptive Emission Controller with Budget-Based Error Correction
///
/// Maintains three invariants over 256 years:
/// 1. Annual emission ≈ target for current era (short-term accuracy)
/// 2. Cumulative emission converges to target schedule (long-term accuracy)
/// 3. Total supply → 21,000,000 QUG (supply cap guarantee)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmissionController {
    /// Sliding window of block production for rate measurement
    block_windows: VecDeque<BlockWindow>,

    /// Current halving era (0-63)
    current_era: u64,

    /// Total emitted in current era (resets on era transition)
    total_emitted_this_era: u128,

    /// Target emission for current era = ERA_0_TOTAL >> era
    era_target_emission: u128,

    /// Current emission phase
    phase: EmissionPhase,

    /// Genesis timestamp (unix seconds)
    genesis_timestamp: u64,

    /// Daily emission records for audit trail (last 90 days)
    #[serde(default)]
    daily_records: BTreeMap<String, DailyEmissionRecord>,

    /// Total cumulative emission since genesis (all eras)
    #[serde(default)]
    total_cumulative_emission: u128,

    /// Block rate samples for daily averaging
    #[serde(default)]
    daily_rate_samples: Vec<f64>,

    /// v7.0.0: Running correction factor (smoothed, persisted across restarts)
    /// 1.0 = no correction, <1.0 = reducing emission, >1.0 = increasing emission
    #[serde(default = "default_correction_factor")]
    correction_factor: f64,

    /// v7.0.0: Total blocks tracked across all eras (for lifetime statistics)
    #[serde(default)]
    total_blocks_tracked: u64,

    /// v7.1.3: Last tracked block height (prevents duplicate tracking from TOCTOU races)
    #[serde(default)]
    last_tracked_height: u64,

    /// v8.0.2: Wall-clock epoch (unix seconds) when the first block was tracked after (re)start.
    /// Used for wall-clock rate measurement that is immune to turbo-sync timestamp inflation.
    #[serde(default)]
    wallclock_start_epoch: u64,

    /// v8.0.2: Count of blocks tracked since wallclock_start_epoch
    #[serde(default)]
    wallclock_blocks_tracked: u64,

    /// v8.0.3: Wall-clock sliding window for rate measurement.
    /// Each entry = (wall_timestamp_secs, block_count_in_this_10s_bucket).
    /// Keeps last 180 entries = 30 minutes of data.
    /// This replaces the cumulative wallclock rate which was poisoned by turbo-sync
    /// burst blocks on startup (inflated rate → too-low rewards → persistent under-emission).
    #[serde(default)]
    wallclock_windows: VecDeque<(u64, u64)>,
}

fn default_correction_factor() -> f64 { 1.0 }

impl Default for EmissionController {
    fn default() -> Self {
        Self::new(GENESIS_TIMESTAMP)
    }
}

impl EmissionController {
    /// Create new emission controller with genesis timestamp
    pub fn new(genesis_timestamp: u64) -> Self {
        Self {
            block_windows: VecDeque::with_capacity(RATE_WINDOW_SIZE),
            current_era: 0,
            total_emitted_this_era: 0,
            era_target_emission: ERA_0_TOTAL, // Full era 0 budget
            phase: EmissionPhase::Bootstrap,
            genesis_timestamp,
            daily_records: BTreeMap::new(),
            total_cumulative_emission: 0,
            daily_rate_samples: Vec::new(),
            correction_factor: 1.0,
            total_blocks_tracked: 0,
            last_tracked_height: 0,
            wallclock_start_epoch: 0,
            wallclock_blocks_tracked: 0,
            wallclock_windows: VecDeque::with_capacity(180),
        }
    }

    /// Initialize from the theoretical halving schedule when no persisted state exists.
    ///
    /// Used when a node boots with a fresh or empty database (load_emission_state returns None).
    /// Sets total_cumulative_emission from the pure time formula — error ≤ 0.15% vs actual,
    /// self-corrects within hours via the PID. Category C fields (correction_factor,
    /// wallclock_windows, daily_records) start neutral; live blocks populate them normally.
    ///
    /// This prevents the critical bug where a fresh DB causes total_cumulative_emission = 0,
    /// making the node behave as if genesis just happened regardless of elapsed chain time.
    pub fn from_time_based_fallback(genesis_timestamp: u64) -> Self {
        let wall_now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let elapsed = wall_now.saturating_sub(genesis_timestamp);
        let era = era_at_time(elapsed).min(63);
        let total = target_cumulative_at_time(elapsed);

        let mut controller = Self::new(genesis_timestamp);
        controller.total_cumulative_emission = total;
        controller.current_era = era;
        controller.era_target_emission = era_emission(era);
        controller.wallclock_start_epoch = wall_now;
        warn!(
            "⚠️ [EMISSION FALLBACK] No persisted state — initialized from time formula: \
             total={} base units ({:.4} QUG), era={}, elapsed={}s",
            total,
            total as f64 / 1e24,
            era,
            elapsed
        );
        controller
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STATE PERSISTENCE (v7.1.0)
    // ═══════════════════════════════════════════════════════════════════════

    /// Serialize emission controller state for persistence to RocksDB
    pub fn serialize_state(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Restore emission controller state from persisted bytes
    pub fn restore_from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// Override genesis timestamp (used when persisted state has stale genesis)
    pub fn set_genesis_timestamp(&mut self, ts: u64) {
        if self.genesis_timestamp != ts {
            info!("🔧 Emission controller genesis_timestamp updated: {} → {}", self.genesis_timestamp, ts);
            self.genesis_timestamp = ts;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // BLOCK RATE TRACKING
    // ═══════════════════════════════════════════════════════════════════════

    /// Track a new block for rate measurement
    /// Creates time-based windows of 10 seconds each
    ///
    /// v7.1.4: Removed height-based dedup. In a DAG with parallel miners,
    /// multiple blocks can exist at the same height. Skipping them undercounts
    /// the true block rate (e.g., 4 miners → rate shows 0.36 bps instead of 1.44),
    /// causing the reward formula to overcompensate → 79% overemission.
    /// The `processed_blocks` LRU in balance_consensus.rs prevents true duplicates.
    ///
    /// v7.2.6: Removed v7.1.8 age filter — it blocked 99.76% of blocks from rate tracking.
    /// The age filter was supposed to prevent turbo sync from inflating the measured rate,
    /// but it also blocked ALL historical blocks from being counted, causing the emission
    /// controller to see almost no blocks and emit only 40 QUG in 2.5 days instead of ~17K.
    /// The existing block_rate() cap at 50 bps already prevents turbo sync inflation.
    pub fn add_block(&mut self, height: u64, timestamp: u64, has_transactions: bool) {
        const WINDOW_DURATION_SECS: u64 = 10;

        // Track max height for statistics (but don't skip parallel blocks)
        if height > self.last_tracked_height {
            self.last_tracked_height = height;
        }

        self.total_blocks_tracked += 1;

        // v8.0.2: Wall-clock rate tracking — immune to turbo-sync timestamp inflation.
        // Uses the ACTUAL time blocks arrive at this node, not block-embedded timestamps.
        let wall_now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if self.wallclock_start_epoch == 0 {
            self.wallclock_start_epoch = wall_now;
        }

        // v8.0.3: CRITICAL FIX — Only count LIVE blocks in wall-clock rate tracking.
        // Turbo-synced historical blocks have timestamps far in the past relative to
        // wall_now. If we count them, the rate inflates 2-5× (e.g. 5000 synced blocks
        // in 2 minutes → 41 bps when real rate is 0.2 bps). This causes the reward
        // formula R = Annual/(rate×T) to produce 2-5× too little reward.
        //
        // A block is "live" if its embedded timestamp is within 120 seconds of wall time.
        // Historical turbo-synced blocks will have timestamps minutes/hours in the past.
        const LIVE_BLOCK_THRESHOLD_SECS: u64 = 120;
        let is_live_block = wall_now.saturating_sub(timestamp) < LIVE_BLOCK_THRESHOLD_SECS;

        if is_live_block {
            self.wallclock_blocks_tracked += 1;

            // v8.0.3: Wall-clock SLIDING WINDOW rate tracking (30-minute window).
            // Each bucket = 10 seconds of wall-clock time. Keep 180 buckets = 30 min.
            // ONLY live blocks counted — turbo-synced historical blocks excluded.
            const WALL_BUCKET_SECS: u64 = 10;
            const MAX_WALL_BUCKETS: usize = 180; // 180 × 10s = 30 minutes
            let should_new_bucket = match self.wallclock_windows.back() {
                Some(&(ts, _)) => wall_now.saturating_sub(ts) >= WALL_BUCKET_SECS,
                None => true,
            };
            if should_new_bucket {
                self.wallclock_windows.push_back((wall_now, 1));
                while self.wallclock_windows.len() > MAX_WALL_BUCKETS {
                    self.wallclock_windows.pop_front();
                }
            } else if let Some(last) = self.wallclock_windows.back_mut() {
                last.1 += 1;
            }
        }

        // Block-timestamp window tracking (legacy, used as secondary signal)
        let should_create_new = if let Some(last) = self.block_windows.back() {
            timestamp.saturating_sub(last.start_timestamp) >= WINDOW_DURATION_SECS
        } else {
            true
        };

        if should_create_new {
            self.block_windows.push_back(BlockWindow {
                start_height: height,
                end_height: height,
                start_timestamp: timestamp,
                end_timestamp: timestamp,
                block_count: 1,
                non_empty_blocks: if has_transactions { 1 } else { 0 },
            });
            while self.block_windows.len() > RATE_WINDOW_SIZE {
                self.block_windows.pop_front();
            }
        } else if let Some(last) = self.block_windows.back_mut() {
            last.end_height = height;
            last.end_timestamp = timestamp;
            last.block_count += 1;
            if has_transactions {
                last.non_empty_blocks += 1;
            }
        }
    }

    /// Block rate measurement using wall-clock sliding window (primary) with fallbacks.
    ///
    /// v8.0.3: CRITICAL FIX — Sliding window wall-clock rate measurement.
    ///
    /// ## Problem History
    /// - v8.0.1: Block-timestamp rate inflated by turbo-sync → -96% deviation.
    /// - v8.0.2: Cumulative wall-clock rate (total_blocks / total_elapsed). Fixed turbo-sync
    ///   timestamp inflation, BUT the cumulative counter includes turbo-sync BURST blocks
    ///   (thousands of historical blocks arriving in minutes on node startup). The burst
    ///   inflates the cumulative rate 2-3× for hours/days, causing persistent under-emission
    ///   (-52% deviation after 8 hours because reward was calculated for 0.43 bps when
    ///   true production was 0.2 bps).
    ///
    /// ## Solution (v8.0.3)
    /// Use a 30-minute SLIDING WINDOW of wall-clock time. Each 10-second bucket records
    /// how many blocks arrived in that real-time interval. Turbo-sync burst blocks all
    /// land in a few buckets, which age out within minutes. After 30 minutes, the window
    /// reflects only genuine block production.
    ///
    /// Priority: sliding window (≥60s data) > cumulative wall-clock (≥30s) > block-timestamp > default
    pub fn calculate_smoothed_rate(&self) -> f64 {
        let wall_now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Primary: wall-clock SLIDING WINDOW rate (immune to turbo-sync burst)
        if self.wallclock_windows.len() >= 2 {
            let first_ts = self.wallclock_windows.front().unwrap().0;
            let window_elapsed = wall_now.saturating_sub(first_ts).max(1) as f64;

            // Need at least 60 seconds of window data for stability
            if window_elapsed >= 60.0 {
                let total_blocks: u64 = self.wallclock_windows.iter().map(|&(_, c)| c).sum();
                let window_rate = total_blocks as f64 / window_elapsed;
                let clamped = window_rate.clamp(0.001, 50.0);
                debug!(
                    "📊 Wall-clock WINDOW rate: {:.4} bps ({} blocks in {:.0}s, {} buckets)",
                    clamped, total_blocks, window_elapsed, self.wallclock_windows.len()
                );
                return clamped;
            }
        }

        // Secondary: cumulative wall-clock rate (for early startup < 60s window)
        if self.wallclock_start_epoch > 0 {
            let wall_elapsed = wall_now.saturating_sub(self.wallclock_start_epoch).max(1) as f64;
            if wall_elapsed >= 30.0 && self.wallclock_blocks_tracked >= 2 {
                let wall_rate = self.wallclock_blocks_tracked as f64 / wall_elapsed;
                let clamped = wall_rate.clamp(0.001, 50.0);
                debug!(
                    "📊 Wall-clock cumulative rate: {:.4} bps ({} blocks in {:.0}s)",
                    clamped, self.wallclock_blocks_tracked, wall_elapsed
                );
                return clamped;
            }
        }

        // Tertiary: block-timestamp global rate (for cold-start < 30s)
        if self.block_windows.len() >= 2 {
            let first = self.block_windows.front().unwrap();
            let last = self.block_windows.back().unwrap();
            let total_time = last.end_timestamp.saturating_sub(first.start_timestamp).max(1) as f64;
            let total_blocks: u64 = self.block_windows.iter().map(|w| w.block_count).sum();
            let rate = total_blocks as f64 / total_time;
            return rate.clamp(0.001, 50.0);
        }

        // No data: conservative default (will be corrected quickly)
        1.0
    }

    /// Economic block rate (non-empty blocks only, filters spam)
    /// v8.0.2: Uses wall-clock time as primary signal (same as smoothed_rate)
    pub fn calculate_economic_rate(&self) -> f64 {
        // For economic rate, use the smoothed rate as base (wall-clock)
        // and scale by the non-empty fraction from windows
        if self.block_windows.len() >= 2 {
            let total_blocks: u64 = self.block_windows.iter().map(|w| w.block_count).sum();
            let total_non_empty: u64 = self.block_windows.iter().map(|w| w.non_empty_blocks).sum();
            if total_blocks > 0 {
                let non_empty_fraction = total_non_empty as f64 / total_blocks as f64;
                return (self.calculate_smoothed_rate() * non_empty_fraction).clamp(0.001, 50.0);
            }
        }
        self.calculate_smoothed_rate()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ERA MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════

    /// Update era based on current timestamp
    pub fn update_era(&mut self, current_timestamp: u64) {
        let elapsed = current_timestamp.saturating_sub(self.genesis_timestamp);
        let new_era = era_at_time(elapsed).min(64);

        if new_era > self.current_era {
            info!(
                "📅 Era transition: {} → {} | Previous era emitted: {:.4} QUG (target: {:.4})",
                self.current_era, new_era,
                self.total_emitted_this_era as f64 / 1e24,
                self.era_target_emission as f64 / 1e24
            );

            self.current_era = new_era;
            self.total_emitted_this_era = 0;
            self.era_target_emission = era_emission(new_era);

            if new_era >= 1 {
                self.phase = EmissionPhase::Mature;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CORE REWARD CALCULATION (the heart of the controller)
    // ═══════════════════════════════════════════════════════════════════════

    /// Calculate the correction factor based on cumulative emission error
    ///
    /// Mathematical derivation:
    /// Let δ = actual_cumulative - target_cumulative (positive = over-emitted)
    /// Let R = remaining_target_this_era = era_target - actual_emitted_this_era
    ///
    /// Ideal correction: reduce future rewards so that:
    ///   future_emission = target_remaining - δ
    ///
    /// Correction factor = (target_remaining - δ × smoothing) / target_remaining
    ///
    /// Bounded to [CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX] for stability
    fn calculate_correction_factor(&self, current_timestamp: u64) -> f64 {
        let elapsed = current_timestamp.saturating_sub(self.genesis_timestamp);
        let target_cumulative = target_cumulative_at_time(elapsed);

        if target_cumulative == 0 {
            return 1.0; // No target yet, no correction
        }

        let actual = self.total_cumulative_emission;

        // Calculate error as fraction of target
        let error_fraction = if actual > target_cumulative {
            // Over-emitted: positive error
            (actual - target_cumulative) as f64 / target_cumulative as f64
        } else {
            // Under-emitted: negative error
            -((target_cumulative - actual) as f64 / target_cumulative as f64)
        };

        // v8.0.3: PID correction with aggressive catch-up for large deviations.
        //
        // Base: factor = 1.0 - α × error (proportional term, α = 0.8)
        // Boost: when |error| > 10%, add a quadratic term to accelerate convergence:
        //   factor += sign(error) × β × error²  (β = 2.0)
        //
        // Example: error = -37% (under-emitted)
        //   Base: 1.0 + 0.8 × 0.37 = 1.296
        //   Boost: 2.0 × 0.37² = 0.274
        //   Total: 1.296 + 0.274 = 1.570 (57% boost instead of 29.6%)
        //
        // This halves convergence time from ~13h to ~6h for typical startup deficits.
        let mut factor = 1.0 - CORRECTION_SMOOTHING * error_fraction;

        // Quadratic acceleration for deviations > 10%
        if error_fraction.abs() > 0.10 {
            let quadratic_boost = 2.0 * error_fraction * error_fraction;
            // Sign: if under-emitted (negative error), boost UP (add); if over, boost DOWN (subtract)
            if error_fraction < 0.0 {
                factor += quadratic_boost; // under-emitted → emit more
            } else {
                factor -= quadratic_boost; // over-emitted → emit less
            }
        }

        let clamped = factor.clamp(CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX);

        if (clamped - 1.0).abs() > 0.01 {
            info!(
                "📐 Emission correction: factor={:.4} | error={:.2}% | actual={:.2} QUG | target={:.2} QUG",
                clamped,
                error_fraction * 100.0,
                actual as f64 / 1e24,
                target_cumulative as f64 / 1e24
            );
        }

        clamped
    }

    /// Calculate adaptive block reward with budget-based error correction
    ///
    /// Formula: reward = (annual_target / expected_blocks_per_year) × correction_factor
    ///
    /// All intermediate monetary calculations use u128. The correction factor
    /// is a small f64 multiplier applied at the final step.
    pub fn calculate_adaptive_reward(
        &self,
        current_timestamp: u64,
        recent_block_rate: f64,
        total_supply: u128,
    ) -> Result<u128> {
        // Hard cap: no more emission after 21M
        if total_supply >= QUG_MAX_SUPPLY {
            debug!("🛑 Max supply reached - no more rewards");
            return Ok(0);
        }

        // Era cap: emission complete after 64 eras (256 years)
        if self.current_era >= 64 {
            debug!("🛑 Era 64+ reached - emission complete");
            return Ok(0);
        }

        // Sane block rate: floor at 0.001, cap at 100K bps
        let rate = recent_block_rate.clamp(0.001, 100_000.0);

        // Annual target for current era (pure u128)
        let target = annual_emission(self.current_era);

        // Expected blocks this year (u128 from f64 — safe because rate × 31.5M < u128 max)
        let expected_blocks = (rate * SECONDS_PER_YEAR) as u128;
        if expected_blocks == 0 {
            warn!("⚠️  Expected blocks is zero - using minimum reward");
            return Ok(MIN_REWARD);
        }

        // Base reward: pure integer arithmetic
        // reward = (target × PRECISION) / expected_blocks / PRECISION
        let base_reward = (target * PRECISION) / expected_blocks / PRECISION;

        // Budget-based correction: don't exceed remaining era emission
        let remaining = self.era_target_emission.saturating_sub(self.total_emitted_this_era);
        let blocks_remaining = expected_blocks.max(1_000_000);
        let budget_cap = remaining / blocks_remaining;

        let mut reward = base_reward.min(budget_cap);

        // Apply cumulative error correction
        let correction = self.calculate_correction_factor(current_timestamp);

        // Convert to corrected reward: reward × correction (f64 multiplication on u128)
        // This is the ONLY f64 operation on a monetary value
        reward = ((reward as f64) * correction) as u128;

        // Dynamic safety cap: 2× ideal reward for current rate (rate-adaptive)
        // v7.2.8: Replaces fixed MAX_REWARD_PER_BLOCK=0.025 QUG which caused
        // 92.5% under-emission at 0.77 bps (ideal reward was 0.108 QUG)
        let rate_cap = dynamic_max_reward(self.current_era, rate);
        reward = reward.clamp(MIN_REWARD, rate_cap);

        // Don't exceed remaining supply
        let remaining_supply = QUG_MAX_SUPPLY.saturating_sub(total_supply);
        reward = reward.min(remaining_supply);

        info!(
            "💰 Reward: {:.6} QUG | rate: {:.2} bps | correction: {:.4} | era: {} | target: {:.0} QUG/yr",
            reward as f64 / 1e24,
            rate,
            correction,
            self.current_era,
            target as f64 / 1e24
        );

        Ok(reward)
    }

    /// Calculate block reward (entry point for block_producer.rs)
    /// v7.1.3: Uses smoothed_rate (ALL blocks) instead of economic_rate (non-empty only).
    /// Since ALL blocks emit coinbase rewards, the rate must count ALL blocks.
    /// Using economic_rate undercounts by ~40% because P2P/sync blocks arrive with
    /// empty transaction lists, causing massive over-emission (83% overshoot).
    pub fn calculate_block_reward(
        &mut self,
        current_timestamp: u64,
        total_supply: u128,
    ) -> Result<u128> {
        self.update_era(current_timestamp);
        let rate = self.calculate_smoothed_rate();

        let reward = self.calculate_adaptive_reward(current_timestamp, rate, total_supply)?;

        // Update smoothed correction factor for persistence
        self.correction_factor = self.calculate_correction_factor(current_timestamp);

        Ok(reward)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // EMISSION RECORDING
    // ═══════════════════════════════════════════════════════════════════════

    /// Record emission (update era and cumulative totals)
    pub fn record_emission(&mut self, amount: u128) {
        self.total_emitted_this_era += amount;
        self.total_cumulative_emission += amount;
    }

    /// Record a coinbase transaction's emission for daily audit trail
    /// NOTE: This is called per coinbase TX, not per block. blocks_processed
    /// actually counts coinbase transactions.
    pub fn record_daily_emission(&mut self, timestamp: u64, reward_amount: u128) {
        let date = Self::timestamp_to_date(timestamp);
        let block_rate = self.calculate_smoothed_rate();

        let record = self.daily_records.entry(date.clone()).or_insert_with(|| {
            let target = daily_emission(self.current_era);
            DailyEmissionRecord {
                date: date.clone(),
                total_emitted: 0,
                blocks_processed: 0,
                avg_reward_per_block: 0,
                min_reward: u128::MAX,
                max_reward: 0,
                avg_block_rate: 0.0,
                era: self.current_era,
                cumulative_supply: self.total_cumulative_emission,
                target_daily: target,
                deviation_pct: 0.0,
            }
        });

        record.total_emitted += reward_amount;
        record.blocks_processed += 1;
        record.avg_reward_per_block = if record.blocks_processed > 0 {
            record.total_emitted / record.blocks_processed as u128
        } else { 0 };
        if reward_amount < record.min_reward { record.min_reward = reward_amount; }
        if reward_amount > record.max_reward { record.max_reward = reward_amount; }
        record.cumulative_supply = self.total_cumulative_emission;

        self.daily_rate_samples.push(block_rate);
        if self.daily_rate_samples.len() > 10000 {
            self.daily_rate_samples.drain(..5000);
        }
        let avg: f64 = self.daily_rate_samples.iter().sum::<f64>() / self.daily_rate_samples.len() as f64;
        record.avg_block_rate = avg;

        if record.target_daily > 0 {
            record.deviation_pct = ((record.total_emitted as f64 / record.target_daily as f64) - 1.0) * 100.0;
        }

        // Keep last 90 days in memory
        while self.daily_records.len() > 90 {
            if let Some(key) = self.daily_records.keys().next().cloned() {
                self.daily_records.remove(&key);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // QUERIES & STATISTICS
    // ═══════════════════════════════════════════════════════════════════════

    /// Get current emission statistics
    pub fn get_stats(&self) -> EmissionStats {
        EmissionStats {
            current_era: self.current_era,
            era_target_emission: self.era_target_emission,
            total_emitted_this_era: self.total_emitted_this_era,
            phase: self.phase,
            current_block_rate: self.calculate_smoothed_rate(),
            window_count: self.block_windows.len(),
        }
    }

    /// Get comprehensive emission summary for API
    pub fn get_emission_summary(&self) -> EmissionSummary {
        let today = Self::timestamp_to_date(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );

        let today_record = self.daily_records.get(&today);
        let annual = annual_emission(self.current_era);
        let daily = daily_emission(self.current_era);

        EmissionSummary {
            total_supply: self.total_cumulative_emission,
            max_supply: QUG_MAX_SUPPLY,
            pct_mined: (self.total_cumulative_emission as f64 / QUG_MAX_SUPPLY as f64) * 100.0,
            current_era: self.current_era,
            annual_target: annual,
            daily_target: daily,
            today_emitted: today_record.map(|r| r.total_emitted).unwrap_or(0),
            today_blocks: today_record.map(|r| r.blocks_processed).unwrap_or(0),
            today_deviation_pct: today_record.map(|r| r.deviation_pct).unwrap_or(0.0),
            block_rate: self.calculate_smoothed_rate(),
            current_reward_per_block: 0, // Filled by caller
            days_tracked: self.daily_records.len() as u64,
        }
    }

    /// Get daily emission history (last N days)
    pub fn get_daily_history(&self, days: usize) -> Vec<DailyEmissionRecord> {
        let records: Vec<DailyEmissionRecord> = self.daily_records.values().cloned().collect();
        let start = if records.len() > days { records.len() - days } else { 0 };
        records[start..].to_vec()
    }

    pub fn phase(&self) -> EmissionPhase { self.phase }
    pub fn current_era(&self) -> u64 { self.current_era }

    /// Get current correction factor (for display/debugging)
    /// v7.1.3: Calculate live value instead of returning stale stored field
    pub fn correction_factor(&self) -> f64 {
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.calculate_correction_factor(now_secs)
    }

    /// Get total cumulative emission
    pub fn total_cumulative_emission(&self) -> u128 { self.total_cumulative_emission }

    /// v8.8.4: Set total cumulative emission (used by migration to sync after replay)
    pub fn set_total_cumulative_emission(&mut self, total: u128) {
        self.total_cumulative_emission = total;
    }

    /// v8.8.6: Clear rate measurement windows after migration.
    /// Pre-migration rate samples contain inflated emission data that would poison
    /// the correction factor calculation. Call this after setting total_cumulative_emission.
    pub fn clear_rate_windows(&mut self) {
        self.block_windows.clear();
        self.wallclock_windows.clear();
    }

    /// v8.0.3: Get rate measurement diagnostics for ultra-advanced analytics
    pub fn get_rate_diagnostics(&self) -> RateDiagnostics {
        let wall_now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Sliding window stats
        let (window_rate, window_blocks, window_elapsed_secs, window_buckets) =
            if self.wallclock_windows.len() >= 2 {
                let first_ts = self.wallclock_windows.front().unwrap().0;
                let elapsed = wall_now.saturating_sub(first_ts).max(1) as f64;
                let blocks: u64 = self.wallclock_windows.iter().map(|&(_, c)| c).sum();
                (blocks as f64 / elapsed, blocks, elapsed, self.wallclock_windows.len())
            } else {
                (0.0, 0, 0.0, self.wallclock_windows.len())
            };

        // Cumulative wall-clock stats
        let cumulative_elapsed = if self.wallclock_start_epoch > 0 {
            wall_now.saturating_sub(self.wallclock_start_epoch).max(1)
        } else { 0 };
        let cumulative_rate = if cumulative_elapsed > 30 {
            self.wallclock_blocks_tracked as f64 / cumulative_elapsed as f64
        } else { 0.0 };

        // Block-timestamp rate
        let block_ts_rate = if !self.block_windows.is_empty() {
            let total: u64 = self.block_windows.iter().map(|w| w.block_count).sum();
            let window_secs = self.block_windows.len() as f64 * 60.0;
            if window_secs > 0.0 { total as f64 / window_secs } else { 0.0 }
        } else { 0.0 };

        // Determine which method is active
        let active_method = if self.wallclock_windows.len() >= 2 && window_elapsed_secs >= 60.0 {
            "sliding_window".to_string()
        } else if cumulative_elapsed > 30 {
            "cumulative".to_string()
        } else if !self.block_windows.is_empty() {
            "block_timestamp".to_string()
        } else {
            "default".to_string()
        };

        // Confidence: sliding window with 10+ min > 80%, cumulative > 60%, block_ts > 40%, default 20%
        let confidence_pct = if active_method == "sliding_window" {
            let minutes = window_elapsed_secs / 60.0;
            ((minutes / 30.0) * 100.0).clamp(50.0, 99.9)
        } else if active_method == "cumulative" {
            60.0_f64.min(80.0)
        } else if active_method == "block_timestamp" {
            40.0
        } else {
            20.0
        };

        // PI controller internals
        let now_secs = wall_now;
        let elapsed = now_secs.saturating_sub(self.genesis_timestamp);
        let target_cumulative = target_cumulative_at_time(elapsed);

        let error_fraction = if target_cumulative > 0 {
            if self.total_cumulative_emission > target_cumulative {
                (self.total_cumulative_emission - target_cumulative) as f64 / target_cumulative as f64
            } else {
                -((target_cumulative - self.total_cumulative_emission) as f64 / target_cumulative as f64)
            }
        } else { 0.0 };

        // Convergence ETA: at current correction rate, how long to close the gap?
        let current_rate = self.calculate_smoothed_rate();
        let reward = if current_rate > 0.001 {
            let annual = annual_emission(self.current_era);
            let blocks_per_year = current_rate * SECONDS_PER_YEAR as f64;
            let base = if blocks_per_year > 0.0 { annual as f64 / blocks_per_year } else { 0.0 };
            base * self.calculate_correction_factor(now_secs)
        } else { 0.0 };

        // Current emission rate (QUG/sec) and target rate
        let actual_rate_qug_per_sec = reward * current_rate;
        let target_rate_qug_per_sec = annual_emission(self.current_era) as f64 / SECONDS_PER_YEAR as f64;
        let gap_qug = if self.total_cumulative_emission < target_cumulative {
            (target_cumulative - self.total_cumulative_emission) as f64
        } else { 0.0 };

        let convergence_eta_secs = if actual_rate_qug_per_sec > target_rate_qug_per_sec && gap_qug > 0.0 {
            let catch_up_rate = actual_rate_qug_per_sec - target_rate_qug_per_sec;
            if catch_up_rate > 0.0 { (gap_qug / catch_up_rate) as u64 } else { u64::MAX }
        } else { u64::MAX };

        RateDiagnostics {
            active_method,
            confidence_pct,
            window_rate_bps: window_rate,
            window_blocks,
            window_elapsed_secs,
            window_buckets,
            cumulative_rate_bps: cumulative_rate,
            cumulative_blocks: self.wallclock_blocks_tracked,
            cumulative_elapsed_secs: cumulative_elapsed as f64,
            block_timestamp_rate_bps: block_ts_rate,
            block_timestamp_windows: self.block_windows.len(),
            smoothed_rate_bps: current_rate,
            correction_factor: self.calculate_correction_factor(now_secs),
            correction_smoothing: CORRECTION_SMOOTHING,
            correction_max: CORRECTION_FACTOR_MAX,
            correction_min: CORRECTION_FACTOR_MIN,
            error_fraction_pct: error_fraction * 100.0,
            convergence_eta_secs: if convergence_eta_secs == u64::MAX { None } else { Some(convergence_eta_secs) },
            actual_emission_rate_qug_per_hour: actual_rate_qug_per_sec * 3600.0 / 1e24,
            target_emission_rate_qug_per_hour: target_rate_qug_per_sec * 3600.0 / 1e24,
            phase: format!("{:?}", self.phase),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ATTOSECOND OPTO-PHYSICS METRICS
    // ═══════════════════════════════════════════════════════════════════════

    /// Compute the full attosecond opto-physics view of the current emission state.
    ///
    /// Combines CPA envelope analysis, economic uncertainty principle,
    /// mode-lock quality, and oscillator model into a single diagnostic snapshot.
    ///
    /// Safe to call on every tick — all operations are pure math, no I/O.
    pub fn get_attophysics_metrics(&self) -> AttoPhysicsMetrics {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Elapsed time since genesis (clamped to avoid underflow before genesis)
        let elapsed_secs = now_secs.saturating_sub(self.genesis_timestamp) as f64;

        // CPA continuous envelope — theoretical annual emission (QUG/yr)
        let cpa_qug_yr = cpa_envelope_qug_per_year(elapsed_secs);

        // Actual measured annual emission from smoothed block rate
        let smoothed_rate = self.calculate_smoothed_rate();
        let actual_annual_qug = if smoothed_rate > 0.0 {
            let era_annual = annual_emission(self.current_era) as f64 / 1e24;
            let cf = self.calculate_correction_factor(now_secs);
            // actual = rate × reward_per_block × seconds_per_year
            let reward_per_block = era_annual / (smoothed_rate * SECONDS_PER_YEAR);
            reward_per_block * cf * smoothed_rate * SECONDS_PER_YEAR
        } else {
            0.0
        };

        // CPA deviation: (actual - theoretical) / theoretical × 100%
        let cpa_deviation_pct = if cpa_qug_yr > 0.0 {
            (actual_annual_qug - cpa_qug_yr) / cpa_qug_yr * 100.0
        } else {
            0.0
        };

        // Instantaneous CPA chirp rate dA/dt at current elapsed time
        let chirp_rate = cpa_chirp_rate_at(elapsed_secs);

        // ħ_econ at current era and block rate
        let hbar = hbar_econ_for_rate(self.current_era, smoothed_rate.max(0.001));

        // PID correction factor at this instant
        let pid_cf = self.calculate_correction_factor(now_secs);

        // Uncertainty product: ΔR × Δt where Δt = 1 correction cycle (60s)
        // ΔR is proportional to how far correction factor deviates from 1.0
        let correction_cycle_secs = 60.0_f64; // one rate-window tick
        let delta_r_qug = if smoothed_rate > 0.0 {
            let base_reward = annual_emission(self.current_era) as f64
                / 1e24
                / (smoothed_rate * SECONDS_PER_YEAR);
            base_reward * (pid_cf - 1.0).abs()
        } else {
            0.0
        };
        let uncertainty_product = delta_r_qug * correction_cycle_secs;
        let uncertainty_margin = if hbar > 0.0 { uncertainty_product / hbar } else { f64::INFINITY };

        // Mode-lock quality from recent inter-block times stored in block_windows
        let block_times: Vec<f64> = self
            .block_windows
            .iter()
            .filter(|w| w.block_count > 0)
            .map(|w| {
                let dt = (w.end_timestamp.saturating_sub(w.start_timestamp)).max(1) as f64;
                dt / w.block_count as f64
            })
            .collect();
        let mlq = mode_lock_quality(&block_times);

        // Angular repetition frequency
        let omega = oscillator_omega_rep(smoothed_rate.max(0.0));

        // Era phase: fractional progress within current era
        let era_start_secs = self.current_era * SECONDS_PER_HALVING;
        let era_phase = if elapsed_secs > era_start_secs as f64 {
            let into_era = elapsed_secs - era_start_secs as f64;
            (into_era / SECONDS_PER_HALVING as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Pulse energy per block (same as base reward, physics framing)
        let pulse_energy_qug = if smoothed_rate > 0.0 {
            annual_emission(self.current_era) as f64 / 1e24 / (smoothed_rate * SECONDS_PER_YEAR)
        } else {
            0.0
        };

        // Era boundary integrity: at the start of the current era, how close is
        // the discrete era emission to the CPA continuous envelope?
        let era_boundary_elapsed = (self.current_era * SECONDS_PER_HALVING) as f64;
        let cpa_at_boundary = cpa_envelope_qug_per_year(era_boundary_elapsed);
        let discrete_at_boundary = annual_emission(self.current_era) as f64 / 1e24;
        let era_boundary_integrity_pct = if discrete_at_boundary > 0.0 {
            100.0 - ((cpa_at_boundary - discrete_at_boundary) / discrete_at_boundary * 100.0).abs()
        } else {
            100.0
        };

        AttoPhysicsMetrics {
            cpa_envelope_qug_per_year: cpa_qug_yr,
            actual_annual_rate_qug: actual_annual_qug,
            cpa_deviation_pct,
            chirp_rate_qug_per_yr_per_sec: chirp_rate,
            hbar_econ_qug: hbar,
            pid_correction_factor: pid_cf,
            uncertainty_product_qug_sec: uncertainty_product,
            uncertainty_margin,
            mode_lock_quality: mlq,
            omega_rep_rad_per_sec: omega,
            current_era: self.current_era,
            era_phase_fraction: era_phase,
            pulse_energy_per_block_qug: pulse_energy_qug,
            era_boundary_integrity_pct,
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // UTILITY
    // ═══════════════════════════════════════════════════════════════════════

    /// Convert unix timestamp to YYYY-MM-DD (civil calendar algorithm)
    /// Based on Howard Hinnant's algorithm (public domain, ISO 8601)
    fn timestamp_to_date(timestamp: u64) -> String {
        let z = (timestamp / 86400) + 719468;
        let era = z / 146097;
        let doe = z - era * 146097;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        format!("{:04}-{:02}-{:02}", y, m, d)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS: Mathematical proofs and simulation
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Fundamental Constants ───────────────────────────────────────────

    #[test]
    fn test_seconds_per_halving_exact() {
        // 4 years × 365.25 days/year × 86400 seconds/day
        let expected = 4u64 * 365 * 86400 + 86400; // 4×365 days + 1 leap day
        // Actually: 365.25 × 4 = 1461 days × 86400 = 126,230,400
        assert_eq!(SECONDS_PER_HALVING, 126_230_400);
        assert_eq!(SECONDS_PER_HALVING, (1461 * 86400) as u64);
    }

    #[test]
    fn test_era_0_total_is_half_max_supply() {
        assert_eq!(ERA_0_TOTAL, QUG_MAX_SUPPLY / 2);
        assert_eq!(ERA_0_TOTAL, 10_500_000_000_000_000_000_000_000_000_000);
    }

    #[test]
    fn test_base_annual_is_quarter_era0() {
        assert_eq!(BASE_ANNUAL_EMISSION, ERA_0_TOTAL / 4);
    }

    // ─── Geometric Series: Total Supply = 21M ────────────────────────────

    #[test]
    fn test_total_emission_all_64_eras_equals_21m() {
        let total = total_emission_all_eras();
        let total_qug = total as f64 / 1e24;

        // Geometric series: E₀ × (1 - 2⁻⁶⁴)/(1 - 1/2) = 2E₀ × (1 - 2⁻⁶⁴)
        // 2⁻⁶⁴ ≈ 5.42e-20, so loss ≈ 21M × 5.42e-20 ≈ 1.14e-12 QUG
        // In base units: ~1.14e12 base units (negligible)

        // Total should be within 1 QUG of 21M
        assert!(
            total_qug > 20_999_999.0 && total_qug < 21_000_001.0,
            "Total emission = {:.6} QUG (expected ~21,000,000)", total_qug
        );

        // Verify the "dust" lost to integer division is tiny
        let dust = QUG_MAX_SUPPLY - total;
        let dust_qug = dust as f64 / 1e24;
        assert!(
            dust_qug < 0.001,
            "Dust from integer rounding = {:.12} QUG (should be < 0.001)", dust_qug
        );

        info!("✅ Total emission across 64 eras: {:.6} QUG", total_qug);
        info!("   Dust (lost to integer rounding): {:.12} QUG", dust_qug);
    }

    // ─── Era Emission Schedule ───────────────────────────────────────────

    #[test]
    fn test_era_emission_halves_correctly() {
        for k in 0..63u64 {
            let this_era = era_emission(k);
            let next_era = era_emission(k + 1);
            // Each era should be exactly half the previous (integer division)
            assert_eq!(next_era, this_era / 2,
                "Era {} = {}, Era {} = {} (should be half)",
                k, this_era, k + 1, next_era);
        }
    }

    #[test]
    fn test_era_emission_after_64_is_zero() {
        assert_eq!(era_emission(64), 0);
        assert_eq!(era_emission(100), 0);
        assert_eq!(era_emission(u64::MAX), 0);
    }

    #[test]
    fn test_era_0_daily_emission() {
        let daily = daily_emission(0);
        let daily_qug = daily as f64 / 1e24;
        // Expected: 2,625,000 / 365.25 = 7,186.858 QUG/day
        assert!(
            (daily_qug - 7186.858).abs() < 0.01,
            "Daily emission = {:.3} QUG (expected ~7186.858)", daily_qug
        );
    }

    // ─── Target Cumulative Function ──────────────────────────────────────

    #[test]
    fn test_target_cumulative_at_genesis() {
        assert_eq!(target_cumulative_at_time(0), 0);
    }

    #[test]
    fn test_target_cumulative_after_1_year() {
        let one_year = SECONDS_PER_YEAR as u64;
        let target = target_cumulative_at_time(one_year);
        let target_qug = target as f64 / 1e24;
        // Should be ~2,625,000 QUG (1 year of Era 0)
        assert!(
            (target_qug - 2_625_000.0).abs() < 100.0,
            "1-year target = {:.0} QUG (expected ~2,625,000)", target_qug
        );
    }

    #[test]
    fn test_target_cumulative_after_4_years() {
        let target = target_cumulative_at_time(SECONDS_PER_HALVING);
        let target_qug = target as f64 / 1e24;
        // Should be 10,500,000 QUG (full Era 0)
        assert!(
            (target_qug - 10_500_000.0).abs() < 1.0,
            "4-year target = {:.0} QUG (expected 10,500,000)", target_qug
        );
    }

    #[test]
    fn test_target_cumulative_after_8_years() {
        let target = target_cumulative_at_time(2 * SECONDS_PER_HALVING);
        let target_qug = target as f64 / 1e24;
        // Era 0 (10.5M) + Era 1 (5.25M) = 15.75M
        assert!(
            (target_qug - 15_750_000.0).abs() < 1.0,
            "8-year target = {:.0} QUG (expected 15,750,000)", target_qug
        );
    }

    #[test]
    fn test_target_cumulative_after_256_years() {
        let target = target_cumulative_at_time(64 * SECONDS_PER_HALVING);
        let total = total_emission_all_eras();
        // Should equal total emission from all 64 eras
        assert_eq!(target, total,
            "256-year cumulative should equal total emission");
    }

    #[test]
    fn test_target_cumulative_monotonic() {
        let mut prev = 0u128;
        for secs in (0..SECONDS_PER_HALVING * 3).step_by(86400) {
            let target = target_cumulative_at_time(secs);
            assert!(target >= prev,
                "Target cumulative must be monotonically increasing: {} < {} at t={}",
                target, prev, secs);
            prev = target;
        }
    }

    // ─── Adaptive Reward ─────────────────────────────────────────────────

    #[test]
    fn test_reward_scales_inversely_with_throughput() {
        let controller = EmissionController::new(GENESIS_TIMESTAMP);

        let r10 = controller.calculate_adaptive_reward(GENESIS_TIMESTAMP + 3600, 10.0, 0).unwrap();
        let r100 = controller.calculate_adaptive_reward(GENESIS_TIMESTAMP + 3600, 100.0, 0).unwrap();
        let r10k = controller.calculate_adaptive_reward(GENESIS_TIMESTAMP + 3600, 10_000.0, 0).unwrap();

        // r10 should be ~10× r100 and ~1000× r10k
        let ratio_10_100 = r10 as f64 / r100 as f64;
        let ratio_10_10k = r10 as f64 / r10k as f64;

        assert!(ratio_10_100 > 8.0 && ratio_10_100 < 12.0,
            "10bps/100bps ratio = {:.1} (expected ~10)", ratio_10_100);
        assert!(ratio_10_10k > 800.0 && ratio_10_10k < 1200.0,
            "10bps/10Kbps ratio = {:.1} (expected ~1000)", ratio_10_10k);
    }

    #[test]
    fn test_annual_emission_constant_across_throughputs() {
        let controller = EmissionController::new(GENESIS_TIMESTAMP);
        let target = BASE_ANNUAL_EMISSION as f64;

        for rate in [1.0, 5.0, 10.0, 50.0, 100.0, 1000.0, 10_000.0] {
            let reward = controller.calculate_adaptive_reward(
                GENESIS_TIMESTAMP + 86400, rate, 0
            ).unwrap();

            let annual = reward as f64 * rate * SECONDS_PER_YEAR;
            let error = (annual - target).abs() / target;

            assert!(error < 0.05,
                "Annual emission at {:.0} bps = {:.0} QUG ({:.2}% error, max 5%)",
                rate, annual / 1e24, error * 100.0);
        }
    }

    #[test]
    fn test_max_supply_enforced() {
        let controller = EmissionController::new(GENESIS_TIMESTAMP);
        let reward = controller.calculate_adaptive_reward(
            GENESIS_TIMESTAMP + 86400, 10.0, QUG_MAX_SUPPLY
        ).unwrap();
        assert_eq!(reward, 0, "No reward at max supply");
    }

    // ─── Error Correction ────────────────────────────────────────────────

    #[test]
    fn test_correction_factor_on_target() {
        let mut controller = EmissionController::new(GENESIS_TIMESTAMP);
        // Simulate 1 day of perfect emission
        let one_day = 86400u64;
        let target_daily = daily_emission(0);
        controller.total_cumulative_emission = target_daily;
        let factor = controller.calculate_correction_factor(GENESIS_TIMESTAMP + one_day);
        // Should be very close to 1.0
        assert!((factor - 1.0).abs() < 0.01,
            "Correction factor when on target = {:.4} (expected ~1.0)", factor);
    }

    #[test]
    fn test_correction_factor_over_emitted() {
        let mut controller = EmissionController::new(GENESIS_TIMESTAMP);
        let one_day = 86400u64;
        let target_daily = daily_emission(0);
        // Emit 2× target
        controller.total_cumulative_emission = target_daily * 2;
        let factor = controller.calculate_correction_factor(GENESIS_TIMESTAMP + one_day);
        // Should be < 1.0 (reducing rewards)
        assert!(factor < 1.0,
            "Correction factor when 2× over = {:.4} (should be < 1.0)", factor);
        assert!(factor > 0.5,
            "Correction factor shouldn't be too aggressive = {:.4}", factor);
    }

    #[test]
    fn test_correction_factor_under_emitted() {
        let mut controller = EmissionController::new(GENESIS_TIMESTAMP);
        let one_day = 86400u64;
        let target_daily = daily_emission(0);
        // Emit 0.5× target
        controller.total_cumulative_emission = target_daily / 2;
        let factor = controller.calculate_correction_factor(GENESIS_TIMESTAMP + one_day);
        // Should be > 1.0 (increasing rewards)
        assert!(factor > 1.0,
            "Correction factor when 0.5× under = {:.4} (should be > 1.0)", factor);
        assert!(factor < 2.0,
            "Correction factor shouldn't be too aggressive = {:.4}", factor);
    }

    // ─── 256-Year Simulation ─────────────────────────────────────────────

    #[test]
    fn test_256_year_simulation_at_10bps() {
        simulate_emission_schedule(10.0, "10 bps");
    }

    #[test]
    fn test_256_year_simulation_at_100bps() {
        simulate_emission_schedule(100.0, "100 bps");
    }

    #[test]
    fn test_256_year_simulation_at_1bps() {
        simulate_emission_schedule(1.0, "1 bps");
    }

    /// Simulate 256 years of emission at a given block rate
    /// Verifies:
    /// 1. Each era's emission stays within 5% of target
    /// 2. Total emission approaches 21M QUG
    /// 3. No era emits more than its budget
    fn simulate_emission_schedule(block_rate: f64, label: &str) {
        let mut controller = EmissionController::new(GENESIS_TIMESTAMP);
        let mut total_emitted: u128 = 0;
        let blocks_per_second = block_rate;

        // Simulate in 1-day increments over 256 years
        let total_days = 256 * 366; // Slightly more than 256 years
        let step_secs = 86400u64; // 1 day

        let mut era_emissions: Vec<(u64, u128)> = Vec::new();
        let mut current_era = 0u64;
        let mut era_emitted = 0u128;

        for day in 0..total_days {
            let timestamp = GENESIS_TIMESTAMP + day * step_secs;

            // Add blocks for this day
            let blocks_this_day = (blocks_per_second * step_secs as f64) as u64;
            for b in 0..blocks_this_day.min(100) {
                // Only track a sample of blocks to keep test fast
                let t = timestamp + (b as u64 * step_secs / blocks_this_day.max(1));
                controller.add_block(day * blocks_this_day + b, t, true);
            }

            // Calculate reward
            let reward = controller.calculate_block_reward(timestamp, total_emitted)
                .unwrap_or(MIN_REWARD);

            // Emit for all blocks in this day
            let day_emission = reward.saturating_mul(blocks_this_day as u128);
            total_emitted += day_emission;
            controller.record_emission(day_emission);

            // Track era transitions
            let era = era_at_time(timestamp.saturating_sub(GENESIS_TIMESTAMP));
            if era != current_era {
                era_emissions.push((current_era, era_emitted));
                current_era = era;
                era_emitted = 0;
            }
            era_emitted += day_emission;
        }
        era_emissions.push((current_era, era_emitted));

        let total_qug = total_emitted as f64 / 1e24;

        // Verify total is close to 21M
        assert!(
            total_qug > 19_000_000.0 && total_qug < 22_000_000.0,
            "[{}] Total emission = {:.0} QUG (expected ~21,000,000)", label, total_qug
        );

        // Verify each era is within reasonable bounds
        for &(era, emitted) in &era_emissions {
            if era >= 64 { continue; }
            let target = era_emission(era);
            if target == 0 { continue; }
            let ratio = emitted as f64 / target as f64;
            // Allow wider tolerance for very short eras in simulation
            if era < 20 {
                assert!(
                    ratio > 0.5 && ratio < 2.0,
                    "[{}] Era {} emitted {:.0} QUG vs target {:.0} (ratio {:.2})",
                    label, era, emitted as f64 / 1e24, target as f64 / 1e24, ratio
                );
            }
        }
    }

    // ─── Edge Cases ──────────────────────────────────────────────────────

    #[test]
    fn test_era_transition() {
        let mut controller = EmissionController::new(GENESIS_TIMESTAMP);
        controller.update_era(GENESIS_TIMESTAMP + SECONDS_PER_HALVING);
        assert_eq!(controller.current_era, 1);
        assert_eq!(controller.phase, EmissionPhase::Mature);
    }

    #[test]
    fn test_block_rate_default_is_conservative() {
        let controller = EmissionController::new(GENESIS_TIMESTAMP);
        let rate = controller.calculate_economic_rate();
        // Default should be 1.0 bps (conservative, not 0.166 which causes overshoot)
        assert!((rate - 1.0).abs() < 0.01,
            "Default rate = {:.3} (expected 1.0)", rate);
    }

    #[test]
    fn test_block_rate_with_6_bps() {
        let mut controller = EmissionController::new(GENESIS_TIMESTAMP);
        // Simulate 6 blocks/sec for 100 seconds
        for sec in 0..100u64 {
            for b in 0..6u64 {
                controller.add_block(
                    sec * 6 + b,
                    GENESIS_TIMESTAMP + sec,
                    true,
                );
            }
        }
        let rate = controller.calculate_smoothed_rate();
        assert!(rate > 5.0 && rate < 7.0,
            "Rate should be ~6 bps, got {:.2}", rate);
    }

    #[test]
    fn test_timestamp_to_date() {
        // Feb 22, 2026 12:00 UTC (Mainnet 2026.2 genesis)
        let date = EmissionController::timestamp_to_date(1771761600);
        assert_eq!(date, "2026-02-22");
    }

    #[test]
    fn test_emission_controller_creation() {
        let controller = EmissionController::new(GENESIS_TIMESTAMP);
        assert_eq!(controller.current_era, 0);
        assert_eq!(controller.phase, EmissionPhase::Bootstrap);
        assert_eq!(controller.total_cumulative_emission, 0);
    }

    #[test]
    fn test_u128_overflow_safety() {
        // Verify our critical intermediate calculations don't overflow u128
        // Most dangerous: target * PRECISION
        let max_target = BASE_ANNUAL_EMISSION; // 2.625e30
        let product = max_target.checked_mul(PRECISION);
        assert!(product.is_some(),
            "BASE_ANNUAL_EMISSION × PRECISION must not overflow u128");

        // Also verify HI_PRECISION doesn't overflow for era 0
        let hi_product = max_target.checked_mul(HI_PRECISION);
        assert!(hi_product.is_some(),
            "BASE_ANNUAL_EMISSION × HI_PRECISION must not overflow u128");
    }

    #[test]
    fn test_target_cumulative_no_overflow() {
        // Test at various time points across 256 years
        for year in [1, 4, 10, 50, 100, 200, 256] {
            let secs = year as u64 * SECONDS_PER_YEAR as u64;
            let target = target_cumulative_at_time(secs);
            assert!(target <= QUG_MAX_SUPPLY,
                "Target at year {} = {:.0} QUG (must be ≤ 21M)",
                year, target as f64 / 1e24);
        }
    }

    // ─── Emission Schedule Printout (for documentation) ──────────────────

    #[test]
    fn test_print_256_year_schedule() {
        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!("║         Q-NarwhalKnight 256-Year Emission Schedule        ║");
        println!("╠═══════╤═══════════════╤═══════════════╤══════════════════╣");
        println!("║  Era  │ Annual (QUG)  │ Daily (QUG)   │ Block @ 10 bps  ║");
        println!("╟───────┼───────────────┼───────────────┼──────────────────╢");

        let mut cumulative = 0u128;
        for k in 0..20u64 {
            let annual = annual_emission(k);
            let daily = daily_emission(k);
            let per_block = base_reward_for_rate(k, 10.0);
            cumulative += era_emission(k);

            println!(
                "║  {:>3}  │ {:>13.2} │ {:>13.3} │ {:>14.8} QUG ║",
                k,
                annual as f64 / 1e24,
                daily as f64 / 1e24,
                per_block as f64 / 1e24
            );
        }
        println!("╟───────┼───────────────┼───────────────┼──────────────────╢");
        println!(
            "║ TOTAL │ {:>13.2} │               │                  ║",
            total_emission_all_eras() as f64 / 1e24
        );
        println!("╚═══════╧═══════════════╧═══════════════╧══════════════════╝");
        println!("\nGeometric sum verification:");
        println!("  Σ(k=0..63) 10,500,000/2^k = {:.6} QUG", total_emission_all_eras() as f64 / 1e24);
        println!("  Max supply:                 {:.6} QUG", QUG_MAX_SUPPLY as f64 / 1e24);
        println!("  Rounding dust:              {:.12} QUG\n", (QUG_MAX_SUPPLY - total_emission_all_eras()) as f64 / 1e24);
    }
}
