//! 🎯 K-Parameter Phase Analysis for Quantum Consensus
//!
//! This module implements Kristensen's K-Parameter for distributed consensus:
//!
//! K = 2π √(ΔH · Δs · ℏ) / τ
//!
//! Where:
//! - ΔH: Hamiltonian uncertainty (energy variance in consensus)
//! - Δs: Entropy variance (information entropy of network state)
//! - ℏ: Reduced Planck constant (quantum scale factor)
//! - τ: Characteristic timescale (consensus round duration)
//!
//! Philosophy: Quantum phase transitions in physics provide a mathematical
//! framework for detecting and responding to consensus instabilities.

use crate::{ResonanceVertex, Result, ResonanceError};
use std::collections::HashMap;

/// 🎯 K-Parameter Phase Analyzer
///
/// Analyzes quantum-inspired phase transitions in distributed consensus
/// using Kristensen's unified framework.
pub struct KParameterAnalyzer {
    /// ℏ - Reduced Planck constant (scaled for consensus units)
    planck_constant: f64,

    /// Previous K measurement for change detection
    last_k_value: f64,

    /// Historical K values for trend analysis
    k_history: Vec<f64>,

    /// Threshold for detecting phase transitions
    phase_transition_threshold: f64,

    /// Maximum history length to prevent memory growth
    max_history_length: usize,
}

impl Default for KParameterAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl KParameterAnalyzer {
    /// Create new K-Parameter analyzer with default parameters
    pub fn new() -> Self {
        Self {
            // Use normalized ℏ = 1.0 for dimensionless analysis
            // (Original: 1.0545718e-34 J·s, but we scale to consensus units)
            planck_constant: 1.0,
            last_k_value: 0.0,
            k_history: Vec::new(),
            phase_transition_threshold: 1.0,
            max_history_length: 1000,
        }
    }

    /// Create analyzer with custom Planck constant scaling
    pub fn with_planck_constant(mut self, h_bar: f64) -> Self {
        self.planck_constant = h_bar;
        self
    }

    /// Set phase transition detection threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.phase_transition_threshold = threshold;
        self
    }

    /// 🎯 Compute K-Parameter for consensus round
    ///
    /// K = 2π √(ΔH · Δs · ℏ) / τ
    ///
    /// Returns the K-Parameter value which indicates:
    /// - Low K (<1): Stable consensus state
    /// - Medium K (1-5): Approaching phase transition
    /// - High K (>5): Critical phase transition
    pub fn compute_k_parameter(
        &self,
        energy_variance: f64,     // ΔH - Hamiltonian uncertainty
        entropy_variance: f64,    // Δs - Entropy variance (nats)
        round_duration: f64,      // τ - Characteristic timescale (seconds)
    ) -> f64 {
        if round_duration <= 0.0 {
            return 0.0;
        }

        let product = energy_variance * entropy_variance * self.planck_constant;
        if product < 0.0 {
            return 0.0;
        }

        let numerator = 2.0 * std::f64::consts::PI * product.sqrt();
        numerator / round_duration
    }

    /// 🎯 Compute energy variance across network (ΔH)
    ///
    /// Measures the Hamiltonian uncertainty by computing the standard
    /// deviation of energy across all vertices in the network.
    pub fn compute_energy_variance(
        &self,
        vertex_energies: &[f64],
    ) -> f64 {
        if vertex_energies.is_empty() {
            return 0.0;
        }

        let mean_energy: f64 = vertex_energies.iter().sum::<f64>() / vertex_energies.len() as f64;

        let variance: f64 = vertex_energies.iter()
            .map(|&energy| (energy - mean_energy).powi(2))
            .sum::<f64>() / vertex_energies.len() as f64;

        variance.sqrt() // Standard deviation as uncertainty measure
    }

    /// 🎯 Compute entropy variance (Δs) using Shannon entropy
    ///
    /// Analyzes the information entropy of phase distribution across
    /// the network and computes variance of local entropy measurements.
    pub fn compute_entropy_variance(
        &self,
        phase_distribution: &[f64], // Phases of all vertices
    ) -> f64 {
        if phase_distribution.len() < 10 {
            return 0.0;
        }

        // Compute global entropy
        let global_entropy = self.shannon_entropy(phase_distribution);

        // Compute local entropies using sliding windows
        let window_size = 10.min(phase_distribution.len());
        let local_entropies: Vec<f64> = phase_distribution
            .windows(window_size)
            .map(|window| self.shannon_entropy(window))
            .collect();

        // Variance of local entropies
        self.variance(&local_entropies)
    }

    /// 🎯 Shannon entropy of phase distribution
    ///
    /// H = -Σ p(i) log₂ p(i)
    ///
    /// Bins phases into histogram and computes information entropy.
    fn shannon_entropy(&self, phases: &[f64]) -> f64 {
        if phases.is_empty() {
            return 0.0;
        }

        // Create histogram with 20 bins covering [0, 2π]
        let bins = 20;
        let mut histogram = vec![0; bins];

        for &phase in phases {
            // Normalize phase to [0, 2π]
            let normalized_phase = phase.rem_euclid(2.0 * std::f64::consts::PI);
            let bin = ((normalized_phase / (2.0 * std::f64::consts::PI)) * bins as f64) as usize;
            let bin_idx = bin.min(bins - 1);
            histogram[bin_idx] += 1;
        }

        // Compute Shannon entropy
        let total = phases.len() as f64;
        -histogram.iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total;
                p * p.ln() / std::f64::consts::LN_2 // Convert to nats → bits
            })
            .sum::<f64>()
    }

    /// Compute statistical variance of a dataset
    fn variance(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let variance: f64 = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;

        variance.sqrt() // Return standard deviation
    }

    /// 🎯 Detect quantum phase transitions in consensus
    ///
    /// Analyzes K-Parameter changes to detect phase transitions:
    /// - Stable: K stable, no transition
    /// - Approaching: K increasing, transition approaching
    /// - Critical: Rapid K change, transition occurring
    pub fn detect_phase_transition(&mut self, current_k: f64) -> PhaseTransition {
        // Store in history
        self.k_history.push(current_k);

        // Trim history if too long
        if self.k_history.len() > self.max_history_length {
            self.k_history.remove(0);
        }

        if self.k_history.len() < 2 {
            self.last_k_value = current_k;
            return PhaseTransition::Stable;
        }

        let k_change = (current_k - self.last_k_value).abs();
        self.last_k_value = current_k;

        // Classify based on K change and absolute value
        if k_change > self.phase_transition_threshold * 2.0 || current_k > 10.0 {
            PhaseTransition::Critical
        } else if k_change > self.phase_transition_threshold || current_k > 5.0 {
            PhaseTransition::Approaching
        } else {
            PhaseTransition::Stable
        }
    }

    /// 🎯 Use K-Parameter to adjust consensus parameters
    ///
    /// Dynamically tunes consensus based on quantum phase state:
    /// - Low K: Careful, precise convergence
    /// - Medium K: Balanced approach
    /// - High K: Fast, aggressive convergence
    pub fn adjust_consensus_parameters(&self, k_value: f64) -> ConsensusTuning {
        match k_value {
            k if k < 0.1 => ConsensusTuning {
                learning_rate: 0.01,    // Slow, careful convergence
                max_iterations: 1000,   // More iterations for stability
                spectral_threshold: 0.05, // Sensitive Byzantine detection
                convergence_tolerance: 1e-8,
            },
            k if k < 1.0 => ConsensusTuning {
                learning_rate: 0.1,     // Balanced approach
                max_iterations: 500,
                spectral_threshold: 0.1,
                convergence_tolerance: 1e-6,
            },
            k if k < 5.0 => ConsensusTuning {
                learning_rate: 0.5,     // Fast convergence needed
                max_iterations: 100,
                spectral_threshold: 0.2, // Aggressive Byzantine filtering
                convergence_tolerance: 1e-4,
            },
            _ => ConsensusTuning {
                learning_rate: 1.0,     // Emergency mode
                max_iterations: 50,
                spectral_threshold: 0.3,
                convergence_tolerance: 1e-3,
            }
        }
    }

    /// Get historical K values
    pub fn get_k_history(&self) -> &[f64] {
        &self.k_history
    }

    /// Get current K trend (positive = increasing, negative = decreasing)
    pub fn get_k_trend(&self) -> f64 {
        if self.k_history.len() < 2 {
            return 0.0;
        }

        let recent = &self.k_history[self.k_history.len().saturating_sub(5)..];
        if recent.len() < 2 {
            return 0.0;
        }

        // Linear regression slope
        let n = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().sum();
        let sum_xy: f64 = recent.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..recent.len()).map(|i| (i as f64).powi(2)).sum();

        (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2))
    }
}

/// 🎯 Phase Transition Classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PhaseTransition {
    /// K stable - normal operation
    Stable,

    /// K changing - approaching phase transition
    Approaching,

    /// K critical - phase transition occurring
    Critical,
}

impl std::fmt::Display for PhaseTransition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stable => write!(f, "Stable"),
            Self::Approaching => write!(f, "Approaching"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// 🎯 Consensus Tuning Parameters
///
/// Dynamic parameters adjusted based on K-Parameter analysis
#[derive(Debug, Clone)]
pub struct ConsensusTuning {
    /// Learning rate for gradient descent energy minimization
    pub learning_rate: f64,

    /// Maximum iterations for convergence
    pub max_iterations: usize,

    /// Spectral threshold for Byzantine detection
    pub spectral_threshold: f64,

    /// Convergence tolerance for energy minimization
    pub convergence_tolerance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_parameter_computation() {
        let analyzer = KParameterAnalyzer::new();

        let energy_variance = 1.0;
        let entropy_variance = 2.0;
        let round_duration = 1.0;

        let k = analyzer.compute_k_parameter(energy_variance, entropy_variance, round_duration);

        // K = 2π √(1.0 * 2.0 * 1.0) / 1.0 = 2π √2 ≈ 8.886
        assert!(k > 8.0 && k < 9.0, "K should be approximately 2π√2");
    }

    #[test]
    fn test_energy_variance() {
        let analyzer = KParameterAnalyzer::new();

        let energies = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = analyzer.compute_energy_variance(&energies);

        // Standard deviation should be √2 ≈ 1.414
        assert!(variance > 1.4 && variance < 1.5);
    }

    #[test]
    fn test_shannon_entropy() {
        let analyzer = KParameterAnalyzer::new();

        // Uniform distribution should have maximum entropy
        let uniform_phases: Vec<f64> = (0..100)
            .map(|i| (i as f64 / 100.0) * 2.0 * std::f64::consts::PI)
            .collect();

        let entropy = analyzer.shannon_entropy(&uniform_phases);
        assert!(entropy > 4.0, "Uniform distribution should have high entropy");

        // All same phase should have zero entropy
        let constant_phases = vec![0.0; 100];
        let zero_entropy = analyzer.shannon_entropy(&constant_phases);
        assert!(zero_entropy < 0.1, "Constant distribution should have near-zero entropy");
    }

    #[test]
    fn test_phase_transition_detection() {
        let mut analyzer = KParameterAnalyzer::new();

        // Stable K should be detected as stable
        assert_eq!(analyzer.detect_phase_transition(0.5), PhaseTransition::Stable);
        assert_eq!(analyzer.detect_phase_transition(0.6), PhaseTransition::Stable);

        // Large jump should be detected as critical
        assert_eq!(analyzer.detect_phase_transition(5.0), PhaseTransition::Critical);
    }

    #[test]
    fn test_consensus_tuning() {
        let analyzer = KParameterAnalyzer::new();

        // Low K should give conservative parameters
        let low_k_tuning = analyzer.adjust_consensus_parameters(0.05);
        assert!(low_k_tuning.learning_rate < 0.1);
        assert!(low_k_tuning.max_iterations > 500);

        // High K should give aggressive parameters
        let high_k_tuning = analyzer.adjust_consensus_parameters(10.0);
        assert!(high_k_tuning.learning_rate > 0.5);
        assert!(high_k_tuning.max_iterations < 100);
    }

    #[test]
    fn test_k_trend() {
        let mut analyzer = KParameterAnalyzer::new();

        // Increasing K values
        analyzer.detect_phase_transition(1.0);
        analyzer.detect_phase_transition(2.0);
        analyzer.detect_phase_transition(3.0);

        let trend = analyzer.get_k_trend();
        assert!(trend > 0.0, "Trend should be positive for increasing K");
    }
}
