//! 🎯 K-Parameter Enhanced Energy Functional
//!
//! This module extends the base energy functional with K-Parameter quantum
//! phase analysis for adaptive consensus convergence.

use crate::{
    EnergyFunctional, KParameterAnalyzer, PhaseTransition, ConsensusTuning,
    ResonanceVertex, StringState, Result, ResonanceError,
};
use std::time::Instant;
use num_complex::Complex64;

/// 🎯 K-Parameter Enhanced Energy Functional
///
/// Combines traditional energy minimization with quantum phase analysis
/// to achieve adaptive, self-tuning consensus convergence.
pub struct KEnhancedEnergy {
    /// K-Parameter analyzer for phase transitions
    k_analyzer: KParameterAnalyzer,

    /// Historical K-Parameter values
    k_parameter_history: Vec<f64>,

    /// Round duration tracking for τ computation
    round_start_time: Option<Instant>,
}

impl Default for KEnhancedEnergy {
    fn default() -> Self {
        Self::new()
    }
}

impl KEnhancedEnergy {
    /// Create new K-enhanced energy functional
    pub fn new() -> Self {
        Self {
            k_analyzer: KParameterAnalyzer::new(),
            k_parameter_history: Vec::new(),
            round_start_time: None,
        }
    }

    /// Create with custom K-Parameter analyzer
    pub fn with_analyzer(k_analyzer: KParameterAnalyzer) -> Self {
        Self {
            k_analyzer,
            k_parameter_history: Vec::new(),
            round_start_time: None,
        }
    }

    /// Start timing a consensus round
    pub fn start_round(&mut self) {
        self.round_start_time = Some(Instant::now());
    }

    /// Get elapsed round duration in seconds
    fn get_round_duration(&self) -> f64 {
        self.round_start_time
            .map(|start| start.elapsed().as_secs_f64())
            .unwrap_or(1.0) // Default 1 second if not started
    }

    /// 🎯 Enhanced energy minimization with K-Parameter guidance
    ///
    /// This method:
    /// 1. Computes K-Parameter for current network state
    /// 2. Detects phase transitions
    /// 3. Adjusts consensus parameters dynamically
    /// 4. Runs energy minimization with tuned parameters
    /// 5. Returns (final_energy, k_parameter, phase_analysis)
    pub fn minimize_with_k_guidance(
        &mut self,
        vertices: &mut [ResonanceVertex],
    ) -> Result<(f64, f64, PhaseAnalysis)> {
        if vertices.is_empty() {
            return Err(ResonanceError::InvalidState("Empty vertex set".to_string()));
        }

        // Compute round duration for τ parameter
        let round_duration = self.get_round_duration().max(0.001); // Minimum 1ms

        // Extract string states from vertices
        let strings: Vec<StringState> = vertices.iter()
            .map(|v| v.string_state.clone())
            .collect();

        // Create energy functional from strings
        let mut energy_functional = EnergyFunctional::new(strings.clone());
        energy_functional.rebuild_coupling_matrix();

        // Compute initial energy
        let initial_energy = energy_functional.total_energy();

        // Extract vertex energies for K-Parameter computation
        let vertex_energies: Vec<f64> = vertices.iter()
            .map(|v| self.compute_vertex_energy(v))
            .collect();

        // Extract phases for entropy analysis
        let phases: Vec<f64> = vertices.iter()
            .map(|v| v.string_state.phase.arg())
            .collect();

        // Compute K-Parameter components
        let energy_variance = self.k_analyzer.compute_energy_variance(&vertex_energies);
        let entropy_variance = self.k_analyzer.compute_entropy_variance(&phases);

        // Compute K-Parameter: K = 2π √(ΔH · Δs · ℏ) / τ
        let k_value = self.k_analyzer.compute_k_parameter(
            energy_variance,
            entropy_variance,
            round_duration,
        );

        self.k_parameter_history.push(k_value);

        // Detect phase transition
        let phase_transition = self.k_analyzer.detect_phase_transition(k_value);

        // Adjust parameters based on K-Parameter
        let tuning = self.k_analyzer.adjust_consensus_parameters(k_value);

        tracing::debug!(
            "🎯 K-Parameter analysis: K={:.4}, phase={}, learning_rate={:.3}, iterations={}",
            k_value, phase_transition, tuning.learning_rate, tuning.max_iterations
        );

        // Run full energy minimization with tuned parameters
        let final_energy = energy_functional
            .minimize(tuning.max_iterations, tuning.convergence_tolerance)
            .unwrap_or(initial_energy);

        // Update vertices with minimized string states
        for (i, string_state) in energy_functional.strings.iter().enumerate() {
            if i < vertices.len() {
                vertices[i].string_state = string_state.clone();
            }
        }

        // Create phase analysis result
        let phase_analysis = PhaseAnalysis {
            k_parameter: k_value,
            energy_variance,
            entropy_variance,
            round_duration,
            phase_transition,
            stability: self.compute_stability_metric(),
            recommendation: self.generate_recommendation(k_value, phase_transition),
            tuning_applied: tuning.clone(),
        };

        Ok((final_energy, k_value, phase_analysis))
    }

    /// Compute vertex energy (simplified harmonic oscillator model)
    fn compute_vertex_energy(&self, vertex: &ResonanceVertex) -> f64 {
        // Energy based on amplitude and frequency (E = ½ k x²)
        let amplitude = vertex.string_state.amplitude;
        let frequency = vertex.string_state.frequency;

        // Simple harmonic oscillator energy
        0.5 * frequency * amplitude.powi(2)
    }

    /// 🎯 Compute stability metric from K-Parameter history
    ///
    /// Returns value in [0, 1] where:
    /// - 1.0 = Perfect stability
    /// - 0.0 = Maximum instability
    fn compute_stability_metric(&self) -> f64 {
        if self.k_parameter_history.len() < 5 {
            return 1.0; // Assume stable with insufficient data
        }

        let recent_k: Vec<f64> = self.k_parameter_history
            .iter()
            .rev()
            .take(5)
            .cloned()
            .collect();

        let mean: f64 = recent_k.iter().sum::<f64>() / recent_k.len() as f64;
        let variance: f64 = recent_k.iter()
            .map(|&k| (k - mean).powi(2))
            .sum::<f64>() / recent_k.len() as f64;

        // Inverse relationship with variance
        1.0 / (1.0 + variance)
    }

    /// Generate recommendation based on phase analysis
    fn generate_recommendation(
        &self,
        k_value: f64,
        phase_transition: PhaseTransition,
    ) -> PhaseRecommendation {
        match phase_transition {
            PhaseTransition::Stable if k_value < 1.0 => {
                PhaseRecommendation::NormalOperation
            }
            PhaseTransition::Stable => {
                PhaseRecommendation::IncreaseMonitoring
            }
            PhaseTransition::Approaching => {
                PhaseRecommendation::AdjustParameters
            }
            PhaseTransition::Critical => {
                PhaseRecommendation::EmergencyProtocol
            }
        }
    }

    /// Get K-Parameter history
    pub fn get_k_history(&self) -> &[f64] {
        &self.k_parameter_history
    }

    /// Get current K-Parameter analyzer
    pub fn get_analyzer(&self) -> &KParameterAnalyzer {
        &self.k_analyzer
    }

    /// 🎯 Compute mean energy across vertices
    pub fn mean_energy(&self, vertices: &[ResonanceVertex]) -> f64 {
        if vertices.is_empty() {
            return 0.0;
        }

        let sum: f64 = vertices.iter()
            .map(|v| self.compute_vertex_energy(v))
            .sum();

        sum / vertices.len() as f64
    }

    /// 🎯 Minimize with custom parameters (for tuning)
    pub fn minimize_with_parameters(
        &mut self,
        vertices: &mut [ResonanceVertex],
        max_iterations: usize,
        learning_rate: f64,
    ) -> f64 {
        // Extract string states
        let strings: Vec<StringState> = vertices.iter()
            .map(|v| v.string_state.clone())
            .collect();

        // Create energy functional
        let mut energy_functional = EnergyFunctional::new(strings);
        energy_functional.rebuild_coupling_matrix();

        // Use custom gradient descent with specific learning rate
        let mut energy = energy_functional.total_energy();
        let mut prev_energy = energy;

        for iteration in 0..max_iterations {
            // Perform gradient descent step with custom learning rate
            let gradients: Vec<Complex64> = (0..energy_functional.strings.len())
                .map(|i| energy_functional.compute_gradient(i))
                .collect();

            // Update phases with custom learning rate
            for (i, gradient) in gradients.iter().enumerate() {
                energy_functional.strings[i].update_phase(*gradient, learning_rate);
            }

            energy_functional.update_mean_field();
            energy = energy_functional.total_energy();

            // Check convergence
            if (energy - prev_energy).abs() < 1e-6 {
                tracing::debug!("🎯 Converged after {} iterations", iteration + 1);
                break;
            }

            prev_energy = energy;
        }

        // Update vertices with minimized states
        for (i, string_state) in energy_functional.strings.iter().enumerate() {
            if i < vertices.len() {
                vertices[i].string_state = string_state.clone();
            }
        }

        energy
    }
}

/// 🎯 Phase Analysis Result
#[derive(Debug, Clone)]
pub struct PhaseAnalysis {
    /// Current K-Parameter value
    pub k_parameter: f64,

    /// Energy variance (ΔH)
    pub energy_variance: f64,

    /// Entropy variance (Δs)
    pub entropy_variance: f64,

    /// Round duration (τ)
    pub round_duration: f64,

    /// Phase transition classification
    pub phase_transition: PhaseTransition,

    /// Stability metric [0, 1]
    pub stability: f64,

    /// Operational recommendation
    pub recommendation: PhaseRecommendation,

    /// Tuning parameters that were applied
    pub tuning_applied: ConsensusTuning,
}

impl PhaseAnalysis {
    /// Check if system is in stable state
    pub fn is_stable(&self) -> bool {
        matches!(self.phase_transition, PhaseTransition::Stable) && self.k_parameter < 1.0
    }

    /// Check if emergency action is needed
    pub fn needs_emergency_action(&self) -> bool {
        matches!(self.recommendation, PhaseRecommendation::EmergencyProtocol)
    }

    /// Get human-readable status
    pub fn status_message(&self) -> String {
        format!(
            "K={:.3} | Phase={} | Stability={:.1}% | {}",
            self.k_parameter,
            self.phase_transition,
            self.stability * 100.0,
            self.recommendation
        )
    }
}

/// 🎯 Phase Recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseRecommendation {
    /// K stable - continue as is
    NormalOperation,

    /// K changing - watch closely
    IncreaseMonitoring,

    /// K approaching critical - tune consensus
    AdjustParameters,

    /// K critical - activate emergency measures
    EmergencyProtocol,
}

impl std::fmt::Display for PhaseRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NormalOperation => write!(f, "Normal Operation"),
            Self::IncreaseMonitoring => write!(f, "Increase Monitoring"),
            Self::AdjustParameters => write!(f, "Adjust Parameters"),
            Self::EmergencyProtocol => write!(f, "Emergency Protocol"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn create_test_vertex(amplitude: f64, frequency: f64, phase: f64, round: u64) -> ResonanceVertex {
        let hash = [round as u8; 32];
        let parents = HashSet::new();
        let transactions = vec![];
        let validator = vec![0u8; 32];
        let timestamp = 1000 + round;
        let stake = amplitude * amplitude; // stake = A²
        let network_position = vec![0.0, 0.0, 0.0];
        let entropy = 1.0;

        let mut vertex = ResonanceVertex::new(
            hash,
            round,
            parents,
            transactions,
            validator,
            timestamp,
            stake,
            network_position,
            entropy,
        );

        // Override phase for testing
        vertex.string_state.phase = Complex64::new(phase.cos(), phase.sin());
        vertex.string_state.amplitude = amplitude;
        vertex.string_state.frequency = frequency;

        vertex
    }

    #[test]
    fn test_k_enhanced_minimization() {
        let mut k_energy = KEnhancedEnergy::new();
        k_energy.start_round();

        let mut vertices = vec![
            create_test_vertex(1.0, 1.0, 0.0, 0),
            create_test_vertex(1.0, 1.0, std::f64::consts::PI / 4.0, 1),
            create_test_vertex(1.0, 1.0, std::f64::consts::PI / 2.0, 2),
        ];

        let result = k_energy.minimize_with_k_guidance(&mut vertices);
        assert!(result.is_ok());

        let (final_energy, k_value, analysis) = result.unwrap();
        assert!(final_energy >= 0.0);
        assert!(k_value >= 0.0);
        assert!(analysis.stability > 0.0 && analysis.stability <= 1.0);
    }

    #[test]
    fn test_vertex_energy_computation() {
        let k_energy = KEnhancedEnergy::new();
        let vertex = create_test_vertex(2.0, 3.0, 0.0, 0);

        let energy = k_energy.compute_vertex_energy(&vertex);
        // E = 0.5 * 3.0 * 2.0^2 = 6.0
        assert!((energy - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_stability_metric() {
        let mut k_energy = KEnhancedEnergy::new();

        // Add stable K values
        k_energy.k_parameter_history = vec![1.0, 1.1, 0.9, 1.0, 1.05];
        let stability = k_energy.compute_stability_metric();
        assert!(stability > 0.8, "Should have high stability with low variance");

        // Add unstable K values
        k_energy.k_parameter_history = vec![1.0, 5.0, 0.5, 10.0, 2.0];
        let instability = k_energy.compute_stability_metric();
        assert!(instability < 0.5, "Should have low stability with high variance");
    }

    #[test]
    fn test_phase_analysis_status() {
        let analysis = PhaseAnalysis {
            k_parameter: 0.5,
            energy_variance: 0.1,
            entropy_variance: 0.2,
            round_duration: 1.0,
            phase_transition: PhaseTransition::Stable,
            stability: 0.95,
            recommendation: PhaseRecommendation::NormalOperation,
            tuning_applied: ConsensusTuning {
                learning_rate: 0.1,
                max_iterations: 500,
                spectral_threshold: 0.1,
                convergence_tolerance: 1e-6,
            },
        };

        assert!(analysis.is_stable());
        assert!(!analysis.needs_emergency_action());

        let status = analysis.status_message();
        assert!(status.contains("K=0.500"));
        assert!(status.contains("Stable"));
    }
}
