//! # Effective K-Parameter with Observer Dependence
//!
//! Computes K_eff in Lindblad form: a single effective phase-transition
//! parameter that combines the base K-parameter with decoherence rates
//! from multiple independent channels.
//!
//! ## Physics Background
//!
//! The Kristensen K-parameter measures the quantum-to-classical transition.
//! In distributed consensus, this becomes the order-to-disorder transition:
//! is the network in coherent agreement (quantum/ordered) or in conflicting
//! forks (classical/disordered)?
//!
//! Instead of multiplying factors (numerically unstable), we use the
//! Lindblad master equation form:
//!
//!     K_eff = exp(-Gamma_total / Gamma_critical)
//!
//! where Gamma_total = sum of decoherence rates from each channel.
//! This is mathematically equivalent to the product form:
//!
//!     K_eff = Psi * Theta * Delta * Phi * Omega
//!
//! since exp(-sum(gamma_i)) = prod(exp(-gamma_i)) = prod(f_i).
//!
//! ## Observer Dependence (Harlow 2025)
//!
//! Daniel Harlow showed that any observer in a closed universe has an
//! effective Hilbert space of dimension ~ exp(S_Ob). We map this to
//! consensus: a light client with few resources sees a simpler (more
//! classical) network than a full validator with complete state.
//!
//!     Omega_obs = 1 - exp(-S_obs / S_max)
//!
//! Derivation: Haar-random pure state model on H_obs x H_env gives
//! average purity E[Tr(rho^2)] ~ 1/d_obs + 1/d_env, so
//! Omega = 1 - E[Tr(rho^2)] ~ 1 - exp(-S_obs) for S_obs << S_max.
//!
//! ## References
//!
//! - Harlow, Usatyuk & Zhao (2025). arXiv:2501.02359
//! - Almheiri, Dong & Harlow (2015). arXiv:1411.7041
//! - Kristensen & OroBit (2026). Quantum Cosmos Whitepaper v2.0

use crate::{ConsensusTuning, KParameterAnalyzer};
use serde::{Deserialize, Serialize};

/// Default horizon entropy for the consensus network.
/// Analogous to the de Sitter entropy S_max = pi / (G * Lambda).
/// For a blockchain: log2(max_theoretical_validators * max_state_size).
/// Tunable per network; 40.0 ~ 2^40 ~ 1 trillion states.
const DEFAULT_HORIZON_ENTROPY: f64 = 40.0;

/// Default critical decoherence rate (phase boundary at K_eff = 1/e).
const DEFAULT_GAMMA_CRITICAL: f64 = 1.0;

// ─────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────

/// Decoherence rates from each independent channel.
///
/// Each rate gamma_i >= 0 measures how much that channel degrades
/// network coherence.  Negative rates (enhancement) are permitted
/// for the coherence channel when the spectral gap is large.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoherenceRates {
    /// Spectral coherence: gamma = -ln(normalized_spectral_gap).
    /// Large spectral gap → small gamma → good consensus coherence.
    pub coherence: f64,

    /// Thermal noise: gamma = coefficient_of_variation(RTT).
    /// High latency variance → large gamma → network noise.
    pub thermal: f64,

    /// Decoherence (forks): gamma = fork_rate_per_round.
    /// Many forks → large gamma → state has "decohered" into branches.
    pub decoherence: f64,

    /// Topology: gamma = |euler_characteristic| of the DAG.
    /// Complex DAG topology → large gamma → information scrambling.
    pub topology: f64,

    /// Observer: gamma = -ln(Omega_obs).
    /// Light client → large gamma → sees less of the network state.
    pub observer: f64,
}

impl DecoherenceRates {
    /// Total decoherence rate (sum of all channels).
    pub fn gamma_total(&self) -> f64 {
        self.coherence + self.thermal + self.decoherence + self.topology + self.observer
    }
}

/// Result of an effective K-parameter computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KEffectiveResult {
    /// The effective K-parameter in [0, 1].
    pub k_eff: f64,

    /// Total decoherence rate.
    pub gamma_total: f64,

    /// Critical decoherence rate used.
    pub gamma_critical: f64,

    /// Individual channel rates.
    pub rates: DecoherenceRates,

    /// Phase classification based on K_eff.
    pub phase: ConsensusPhase,

    /// Observer entropy used.
    pub observer_entropy: f64,

    /// Observer factor Omega_obs in [0, 1].
    pub omega_obs: f64,

    /// Base K from KParameterAnalyzer (for comparison).
    pub k_base: f64,
}

/// Consensus phase based on K_eff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusPhase {
    /// K_eff > 0.8: strong consensus, fast finality.
    Coherent,

    /// 0.5 < K_eff <= 0.8: healthy consensus, normal operation.
    Ordered,

    /// 1/e < K_eff <= 0.5: weakening consensus, increase monitoring.
    Crossover,

    /// K_eff <= 1/e (~0.368): disordered, forks likely, emergency tuning.
    Disordered,
}

impl std::fmt::Display for ConsensusPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Coherent => write!(f, "Coherent"),
            Self::Ordered => write!(f, "Ordered"),
            Self::Crossover => write!(f, "Crossover"),
            Self::Disordered => write!(f, "Disordered"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Observer factor
// ─────────────────────────────────────────────────────────────────

/// Compute the observer-dependent factor Omega_obs.
///
/// Omega_obs = 1 - exp(-S_obs / S_max)
///
/// # Arguments
/// * `s_observer` - Observer's coarse-grained entropy (e.g., log2(RAM * CPU)).
/// * `s_max` - Horizon entropy of the network (e.g., log2(max_validators * max_state)).
///
/// # Returns
/// Omega_obs in [0, 1].
pub fn observer_factor(s_observer: f64, s_max: f64) -> f64 {
    if s_max <= 0.0 || s_observer <= 0.0 {
        return 0.0;
    }
    (1.0 - (-s_observer / s_max).exp()).clamp(0.0, 1.0)
}

/// Estimate observer entropy from node capabilities.
///
/// S_obs = log2(ram_bytes) + log2(cpu_ops_per_sec) + log2(state_entries)
///
/// A light client with 256MB RAM and no local state has low S_obs.
/// A full validator with 64GB RAM and full UTXO set has high S_obs.
pub fn estimate_observer_entropy(ram_bytes: u64, cpu_ops_per_sec: u64, state_entries: u64) -> f64 {
    let ram_bits = if ram_bytes > 0 {
        (ram_bytes as f64).log2()
    } else {
        0.0
    };
    let cpu_bits = if cpu_ops_per_sec > 0 {
        (cpu_ops_per_sec as f64).log2()
    } else {
        0.0
    };
    let state_bits = if state_entries > 0 {
        (state_entries as f64).log2()
    } else {
        0.0
    };
    ram_bits + cpu_bits + state_bits
}

// ─────────────────────────────────────────────────────────────────
// K_eff analyzer
// ─────────────────────────────────────────────────────────────────

/// Effective K-Parameter analyzer with observer dependence.
///
/// Wraps the existing `KParameterAnalyzer` and `SpectralBFT` to produce
/// a single K_eff value per consensus round, incorporating all decoherence
/// channels including the observer factor from Harlow (2025).
pub struct KEffectiveAnalyzer {
    /// Base K-parameter analyzer.
    k_analyzer: KParameterAnalyzer,

    /// Critical decoherence rate (phase boundary).
    gamma_critical: f64,

    /// Horizon entropy for observer factor.
    horizon_entropy: f64,

    /// This node's observer entropy.
    observer_entropy: f64,

    /// History of K_eff values for trend analysis.
    k_eff_history: Vec<f64>,

    /// Maximum history length.
    max_history: usize,
}

impl Default for KEffectiveAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl KEffectiveAnalyzer {
    /// Create with default parameters.
    pub fn new() -> Self {
        Self {
            k_analyzer: KParameterAnalyzer::new(),
            gamma_critical: DEFAULT_GAMMA_CRITICAL,
            horizon_entropy: DEFAULT_HORIZON_ENTROPY,
            observer_entropy: DEFAULT_HORIZON_ENTROPY, // full validator by default
            k_eff_history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Set the critical decoherence rate (phase boundary tuning).
    pub fn with_gamma_critical(mut self, gamma_critical: f64) -> Self {
        self.gamma_critical = gamma_critical.max(0.001);
        self
    }

    /// Set the network's horizon entropy.
    pub fn with_horizon_entropy(mut self, s_max: f64) -> Self {
        self.horizon_entropy = s_max.max(1.0);
        self
    }

    /// Set this node's observer entropy from capabilities.
    pub fn with_observer(mut self, ram_bytes: u64, cpu_ops: u64, state_entries: u64) -> Self {
        self.observer_entropy = estimate_observer_entropy(ram_bytes, cpu_ops, state_entries);
        self
    }

    /// Set observer entropy directly.
    pub fn with_observer_entropy(mut self, s_obs: f64) -> Self {
        self.observer_entropy = s_obs.max(0.0);
        self
    }

    /// Compute K_eff from raw decoherence rates.
    ///
    /// K_eff = exp(-Gamma_total / Gamma_critical)
    pub fn compute_from_rates(&mut self, rates: DecoherenceRates) -> KEffectiveResult {
        let gamma_total = rates.gamma_total();
        let k_eff = (-gamma_total / self.gamma_critical).exp().clamp(0.0, 1.0);

        // Track history
        self.k_eff_history.push(k_eff);
        if self.k_eff_history.len() > self.max_history {
            self.k_eff_history.remove(0);
        }

        let phase = classify_phase(k_eff);
        let omega = observer_factor(self.observer_entropy, self.horizon_entropy);

        KEffectiveResult {
            k_eff,
            gamma_total,
            gamma_critical: self.gamma_critical,
            rates,
            phase,
            observer_entropy: self.observer_entropy,
            omega_obs: omega,
            k_base: 0.0, // not computed from base analyzer in this path
        }
    }

    /// Compute K_eff from network observables.
    ///
    /// This is the main entry point: feed in the measurable quantities
    /// from the current consensus round and get back K_eff.
    ///
    /// # Arguments
    /// * `spectral_gap` - From SpectralBFT::spectral_gap(). Larger = more coherent.
    /// * `latency_cv` - Coefficient of variation of round-trip times. 0 = no jitter.
    /// * `fork_rate` - Forks detected per round. 0 = no forks.
    /// * `dag_euler_char` - |V - E + F| of the DAG subgraph this round.
    /// * `vertex_energies` - Per-vertex energy values for base K computation.
    /// * `phase_distribution` - Per-vertex phases for entropy variance.
    /// * `round_duration` - Duration of this round in seconds.
    pub fn compute_from_observables(
        &mut self,
        spectral_gap: f64,
        latency_cv: f64,
        fork_rate: f64,
        dag_euler_char: f64,
        vertex_energies: &[f64],
        phase_distribution: &[f64],
        round_duration: f64,
    ) -> KEffectiveResult {
        // Base K from the existing analyzer
        let energy_var = self.k_analyzer.compute_energy_variance(vertex_energies);
        let entropy_var = self.k_analyzer.compute_entropy_variance(phase_distribution);
        let k_base = self
            .k_analyzer
            .compute_k_parameter(energy_var, entropy_var, round_duration);

        // Convert observables to decoherence rates
        //
        // Coherence: normalize spectral gap to [0, ~1] range.
        // A gap of 0 → gamma = large (no coherence).
        // A gap >> 1 → gamma = negative (enhancement).
        let normalized_gap = spectral_gap.max(1e-10);
        let gamma_coherence = -(normalized_gap.min(2.0)).ln();

        // Thermal noise: CV of latency directly maps to noise rate.
        let gamma_thermal = latency_cv.max(0.0);

        // Decoherence: fork rate is a direct decoherence measure.
        let gamma_decoherence = fork_rate.max(0.0);

        // Topology: absolute Euler characteristic.
        let gamma_topology = dag_euler_char.abs();

        // Observer: from Omega_obs.
        let omega = observer_factor(self.observer_entropy, self.horizon_entropy);
        let gamma_observer = if omega > 1e-15 {
            -(omega.ln()) // -ln(Omega) → large when Omega small
        } else {
            50.0 // cap: minimal observer
        };

        let rates = DecoherenceRates {
            coherence: gamma_coherence,
            thermal: gamma_thermal,
            decoherence: gamma_decoherence,
            topology: gamma_topology,
            observer: gamma_observer,
        };

        let gamma_total = rates.gamma_total();
        let k_eff = (-gamma_total / self.gamma_critical).exp().clamp(0.0, 1.0);

        // Track history
        self.k_eff_history.push(k_eff);
        if self.k_eff_history.len() > self.max_history {
            self.k_eff_history.remove(0);
        }

        let phase = classify_phase(k_eff);

        KEffectiveResult {
            k_eff,
            gamma_total,
            gamma_critical: self.gamma_critical,
            rates,
            phase,
            observer_entropy: self.observer_entropy,
            omega_obs: omega,
            k_base,
        }
    }

    /// Get adaptive consensus tuning based on K_eff.
    ///
    /// Maps K_eff to consensus parameters. Unlike the base
    /// KParameterAnalyzer which uses K directly, this accounts
    /// for the observer's perspective.
    pub fn adaptive_tuning(&self, k_eff: f64) -> ConsensusTuning {
        match classify_phase(k_eff) {
            ConsensusPhase::Coherent => ConsensusTuning {
                learning_rate: 0.01,
                max_iterations: 1000,
                spectral_threshold: 0.05,
                convergence_tolerance: 1e-8,
            },
            ConsensusPhase::Ordered => ConsensusTuning {
                learning_rate: 0.1,
                max_iterations: 500,
                spectral_threshold: 0.1,
                convergence_tolerance: 1e-6,
            },
            ConsensusPhase::Crossover => ConsensusTuning {
                learning_rate: 0.5,
                max_iterations: 100,
                spectral_threshold: 0.2,
                convergence_tolerance: 1e-4,
            },
            ConsensusPhase::Disordered => ConsensusTuning {
                learning_rate: 1.0,
                max_iterations: 50,
                spectral_threshold: 0.3,
                convergence_tolerance: 1e-3,
            },
        }
    }

    /// Get the K_eff trend (positive = improving, negative = degrading).
    pub fn k_eff_trend(&self) -> f64 {
        if self.k_eff_history.len() < 2 {
            return 0.0;
        }
        let recent = &self.k_eff_history[self.k_eff_history.len().saturating_sub(5)..];
        if recent.len() < 2 {
            return 0.0;
        }
        let n = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().sum();
        let sum_xy: f64 = recent.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..recent.len()).map(|i| (i as f64).powi(2)).sum();
        let denom = n * sum_x2 - sum_x.powi(2);
        if denom.abs() < 1e-15 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denom
        }
    }

    /// Get the K_eff history.
    pub fn history(&self) -> &[f64] {
        &self.k_eff_history
    }

    /// Get the underlying K-parameter analyzer (for direct access).
    pub fn base_analyzer(&self) -> &KParameterAnalyzer {
        &self.k_analyzer
    }

    /// Prometheus metrics export.
    pub fn prometheus_metrics(&self, result: &KEffectiveResult) -> String {
        format!(
            "# HELP qnk_k_effective Effective K-parameter with observer dependence\n\
             # TYPE qnk_k_effective gauge\n\
             qnk_k_effective {{}}\t{:.6}\n\
             # HELP qnk_k_base Base K-parameter from energy/entropy variance\n\
             # TYPE qnk_k_base gauge\n\
             qnk_k_base {{}}\t{:.6}\n\
             # HELP qnk_gamma_total Total decoherence rate\n\
             # TYPE qnk_gamma_total gauge\n\
             qnk_gamma_total {{}}\t{:.6}\n\
             # HELP qnk_omega_observer Observer factor\n\
             # TYPE qnk_omega_observer gauge\n\
             qnk_omega_observer {{}}\t{:.6}\n\
             # HELP qnk_gamma_coherence Spectral coherence decoherence rate\n\
             # TYPE qnk_gamma_coherence gauge\n\
             qnk_gamma_coherence {{}}\t{:.6}\n\
             # HELP qnk_gamma_thermal Network thermal noise rate\n\
             # TYPE qnk_gamma_thermal gauge\n\
             qnk_gamma_thermal {{}}\t{:.6}\n\
             # HELP qnk_gamma_forks Fork decoherence rate\n\
             # TYPE qnk_gamma_forks gauge\n\
             qnk_gamma_forks {{}}\t{:.6}\n\
             # HELP qnk_gamma_topology DAG topology decoherence rate\n\
             # TYPE qnk_gamma_topology gauge\n\
             qnk_gamma_topology {{}}\t{:.6}\n\
             # HELP qnk_gamma_observer Observer decoherence rate\n\
             # TYPE qnk_gamma_observer gauge\n\
             qnk_gamma_observer {{}}\t{:.6}\n",
            result.k_eff,
            result.k_base,
            result.gamma_total,
            result.omega_obs,
            result.rates.coherence,
            result.rates.thermal,
            result.rates.decoherence,
            result.rates.topology,
            result.rates.observer,
        )
    }
}

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────

fn classify_phase(k_eff: f64) -> ConsensusPhase {
    const INV_E: f64 = 0.367_879_441_171; // 1/e
    if k_eff > 0.8 {
        ConsensusPhase::Coherent
    } else if k_eff > 0.5 {
        ConsensusPhase::Ordered
    } else if k_eff > INV_E {
        ConsensusPhase::Crossover
    } else {
        ConsensusPhase::Disordered
    }
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observer_factor_limits() {
        // Minimal observer → Omega ≈ 0
        assert!((observer_factor(0.0, 40.0) - 0.0).abs() < 1e-10);

        // Maximal observer → Omega ≈ 0.632 (1 - 1/e)
        let omega = observer_factor(40.0, 40.0);
        assert!((omega - (1.0 - (-1.0_f64).exp())).abs() < 1e-6);

        // Very large observer → Omega → 1
        assert!(observer_factor(400.0, 40.0) > 0.999);

        // Negative / zero s_max → 0
        assert_eq!(observer_factor(10.0, 0.0), 0.0);
        assert_eq!(observer_factor(10.0, -1.0), 0.0);
    }

    #[test]
    fn test_estimate_observer_entropy() {
        // Light client: 256MB RAM, 1GHz, no state
        let light = estimate_observer_entropy(256 * 1024 * 1024, 1_000_000_000, 0);
        assert!(light > 0.0);
        assert!(light < 100.0);

        // Full validator: 64GB RAM, 10GHz effective, 10M state entries
        let full = estimate_observer_entropy(64 * 1024 * 1024 * 1024, 10_000_000_000, 10_000_000);
        assert!(full > light);
    }

    #[test]
    fn test_k_eff_lindblad_form() {
        let mut analyzer = KEffectiveAnalyzer::new()
            .with_gamma_critical(1.0)
            .with_horizon_entropy(40.0)
            .with_observer_entropy(40.0); // full validator

        // Zero decoherence → K_eff = 1
        let rates = DecoherenceRates {
            coherence: 0.0,
            thermal: 0.0,
            decoherence: 0.0,
            topology: 0.0,
            observer: 0.0,
        };
        let result = analyzer.compute_from_rates(rates);
        assert!((result.k_eff - 1.0).abs() < 1e-10);
        assert_eq!(result.phase, ConsensusPhase::Coherent);

        // Heavy decoherence → K_eff ≈ 0
        let rates = DecoherenceRates {
            coherence: 2.0,
            thermal: 1.0,
            decoherence: 1.5,
            topology: 0.5,
            observer: 1.0,
        };
        let result = analyzer.compute_from_rates(rates);
        assert!(result.k_eff < 0.01);
        assert_eq!(result.phase, ConsensusPhase::Disordered);
    }

    #[test]
    fn test_k_eff_equivalence_to_product() {
        // Verify Lindblad form = product form
        let factors = [0.9, 0.8, 0.6, 0.95, 0.7];
        let product: f64 = factors.iter().product();

        let rates: Vec<f64> = factors.iter().map(|f| -f.ln()).collect();
        let gamma_total: f64 = rates.iter().sum();
        let k_lindblad = (-gamma_total).exp();

        assert!(
            (k_lindblad - product).abs() < 1e-12,
            "Lindblad {k_lindblad} != product {product}"
        );
    }

    #[test]
    fn test_observer_dependence() {
        // Same network, different observers → different K_eff
        let rates_base = DecoherenceRates {
            coherence: 0.1,
            thermal: 0.2,
            decoherence: 0.15,
            topology: 0.05,
            observer: 0.0, // placeholder, overridden below
        };

        // Light client (low entropy)
        let mut light = KEffectiveAnalyzer::new()
            .with_observer_entropy(5.0)
            .with_horizon_entropy(40.0);

        // Full validator (high entropy)
        let mut full = KEffectiveAnalyzer::new()
            .with_observer_entropy(38.0)
            .with_horizon_entropy(40.0);

        // Compute with identical network rates but different observer gamma
        let omega_light = observer_factor(5.0, 40.0);
        let omega_full = observer_factor(38.0, 40.0);

        let gamma_obs_light = if omega_light > 1e-15 {
            -(omega_light.ln())
        } else {
            50.0
        };
        let gamma_obs_full = if omega_full > 1e-15 {
            -(omega_full.ln())
        } else {
            50.0
        };

        let mut rates_light = rates_base.clone();
        rates_light.observer = gamma_obs_light;
        let mut rates_full = rates_base.clone();
        rates_full.observer = gamma_obs_full;

        let result_light = light.compute_from_rates(rates_light);
        let result_full = full.compute_from_rates(rates_full);

        // Full validator should see higher K_eff (more of the quantum state)
        assert!(
            result_full.k_eff > result_light.k_eff,
            "Full validator K_eff ({}) should exceed light client K_eff ({})",
            result_full.k_eff,
            result_light.k_eff
        );
    }

    #[test]
    fn test_phase_classification() {
        assert_eq!(classify_phase(0.95), ConsensusPhase::Coherent);
        assert_eq!(classify_phase(0.65), ConsensusPhase::Ordered);
        assert_eq!(classify_phase(0.45), ConsensusPhase::Crossover);
        assert_eq!(classify_phase(0.1), ConsensusPhase::Disordered);
    }

    #[test]
    fn test_adaptive_tuning() {
        let analyzer = KEffectiveAnalyzer::new();

        let coherent = analyzer.adaptive_tuning(0.9);
        assert_eq!(coherent.max_iterations, 1000);

        let disordered = analyzer.adaptive_tuning(0.2);
        assert_eq!(disordered.max_iterations, 50);
        assert!(disordered.learning_rate > coherent.learning_rate);
    }

    #[test]
    fn test_k_eff_trend() {
        let mut analyzer = KEffectiveAnalyzer::new();

        // Feed increasing K_eff values
        for i in 0..10 {
            let rates = DecoherenceRates {
                coherence: 1.0 - i as f64 * 0.08,
                thermal: 0.1,
                decoherence: 0.1,
                topology: 0.05,
                observer: 0.1,
            };
            analyzer.compute_from_rates(rates);
        }

        // Trend should be positive (K_eff increasing)
        assert!(analyzer.k_eff_trend() > 0.0);
    }
}
