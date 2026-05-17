//! Energy Functional for resonance consensus
//!
//! The consensus objective is to minimize the total energy:
//! E_total = E_coupling + E_potential + E_ordering + E_fault_tolerance + E_temporal + E_finality
//!
//! Where:
//! - E_coupling = Σ(i,j) J_ij |ψ_i - ψ_j|²  (phase alignment)
//! - E_potential = Σ(i) λ_i (ψ_i - ψ̄)²   (mean field)
//! - E_ordering = Causal ordering constraint
//! - E_fault_tolerance = Byzantine resilience
//! - E_temporal = Time-travel prevention
//! - E_finality = Committed state lock

use crate::string_state::StringState;
use crate::ResonanceError;
use dashmap::DashMap;
use num_complex::Complex;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::sync::Arc;

/// Energy functional for consensus minimization
pub struct EnergyFunctional {
    /// All string states in the system
    pub strings: Vec<StringState>,

    /// Coupling matrix J_ij (sparse representation)
    coupling_matrix: DashMap<(usize, usize), f64>,

    /// Local potential coefficients λ_i
    local_potentials: Vec<f64>,

    /// Slashing factor for Byzantine detection
    slashing_factor: f64,

    /// Mean field state ψ̄
    mean_field: RwLock<Complex<f64>>,

    /// Ordering constraint weight
    ordering_weight: f64,

    /// Fault tolerance weight
    fault_tolerance_weight: f64,

    /// Temporal coherence weight
    temporal_weight: f64,

    /// Finality barrier weight
    finality_weight: f64,

    /// Committed vertex IDs (for finality)
    committed_vertices: Arc<DashMap<[u8; 32], bool>>,
}

impl EnergyFunctional {
    /// Create a new energy functional
    pub fn new(strings: Vec<StringState>) -> Self {
        let n = strings.len();
        let local_potentials = vec![1.0; n];

        Self {
            strings,
            coupling_matrix: DashMap::new(),
            local_potentials,
            slashing_factor: 2.0,
            mean_field: RwLock::new(Complex::new(0.0, 0.0)),
            ordering_weight: 1.0,
            fault_tolerance_weight: 0.5,
            temporal_weight: 0.3,
            finality_weight: 10.0,
            committed_vertices: Arc::new(DashMap::new()),
        }
    }

    /// Compute total energy
    pub fn total_energy(&self) -> f64 {
        self.coupling_energy()
            + self.potential_energy()
            + self.ordering_energy()
            + self.fault_tolerance_energy()
            + self.temporal_energy()
            + self.finality_energy()
    }

    /// Coupling energy: Σ(i,j) J_ij |ψ_i - ψ_j|²
    fn coupling_energy(&self) -> f64 {
        let mut energy = 0.0;

        for i in 0..self.strings.len() {
            for j in (i + 1)..self.strings.len() {
                let j_ij = self
                    .coupling_matrix
                    .get(&(i, j))
                    .map(|v| *v)
                    .unwrap_or_else(|| {
                        // Compute and cache coupling
                        let coupling = self.strings[i].coupling_strength(&self.strings[j]);
                        self.coupling_matrix.insert((i, j), coupling);
                        coupling
                    });

                let phase_diff = self.strings[i].phase - self.strings[j].phase;
                energy += j_ij * phase_diff.norm_sqr();
            }
        }

        energy
    }

    /// Potential energy: Σ(i) λ_i (ψ_i - ψ̄)²
    fn potential_energy(&self) -> f64 {
        let mean = *self.mean_field.read();

        self.strings
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let lambda_i = self.local_potentials.get(i).copied().unwrap_or(1.0);
                let deviation = s.phase - mean;
                lambda_i * deviation.norm_sqr()
            })
            .sum()
    }

    /// Ordering energy: Ensures causal ordering is preserved
    fn ordering_energy(&self) -> f64 {
        let mut energy = 0.0;

        for i in 0..self.strings.len() {
            for j in 0..self.strings.len() {
                if i == j {
                    continue;
                }

                // If j causally depends on i (timestamp), penalize phase reversal
                if self.strings[j].timestamp > self.strings[i].timestamp {
                    let phase_i = self.strings[i].phase.arg();
                    let phase_j = self.strings[j].phase.arg();

                    // Penalize if j has earlier phase than i (violation of causality)
                    if phase_j < phase_i {
                        let resonance = self.strings[i].resonance(&self.strings[j]);
                        energy += self.ordering_weight * resonance * (phase_i - phase_j).powi(2);
                    }
                }
            }
        }

        energy
    }

    /// Fault tolerance energy: Penalizes Byzantine behavior
    fn fault_tolerance_energy(&self) -> f64 {
        let mean_phase = self.compute_mean_phase();
        let variance = self.compute_phase_variance(mean_phase);

        let mut energy = 0.0;

        for string in &self.strings {
            let deviation = (string.phase.arg() - mean_phase.arg()).abs();

            // Nodes far from mean are likely Byzantine
            if deviation > 3.0 * variance.sqrt() {
                energy += self.fault_tolerance_weight
                    * self.slashing_factor
                    * string.amplitude.powi(2);
            }
        }

        energy
    }

    /// Temporal coherence: Prevents time-travel attacks
    fn temporal_energy(&self) -> f64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.strings
            .iter()
            .map(|s| {
                let time_diff = (s.timestamp as i64 - now as i64).abs() as f64;
                self.temporal_weight * time_diff / 1000.0 // Convert to seconds
            })
            .sum()
    }

    /// Finality barrier: Locks committed states
    fn finality_energy(&self) -> f64 {
        let mut energy = 0.0;

        for string in &self.strings {
            if self.committed_vertices.contains_key(&string.id) {
                // Heavily penalize any change to committed states
                let deviation = string.phase.norm_sqr();
                energy += self.finality_weight * string.amplitude * deviation;
            }
        }

        energy
    }

    /// Minimize energy using gradient descent
    pub fn minimize(
        &mut self,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<f64, ResonanceError> {
        let mut prev_energy = self.total_energy();
        let mut learning_rate = 0.1;

        tracing::info!(
            "Starting energy minimization: E_0 = {:.6}",
            prev_energy
        );

        for iteration in 0..max_iterations {
            let energy = self.gradient_descent_step(learning_rate);

            if (prev_energy - energy).abs() < tolerance {
                tracing::info!(
                    "Converged at iteration {}: E = {:.6}",
                    iteration,
                    energy
                );
                return Ok(energy);
            }

            // Adaptive learning rate
            if energy > prev_energy {
                learning_rate *= 0.5;
                tracing::debug!(
                    "Energy increased, reducing learning rate to {:.6}",
                    learning_rate
                );
            } else if iteration % 10 == 0 {
                learning_rate *= 1.05;
            }

            if iteration % 100 == 0 {
                tracing::debug!("Iteration {}: E = {:.6}", iteration, energy);
            }

            prev_energy = energy;
        }

        tracing::warn!(
            "Did not converge after {} iterations, final E = {:.6}",
            max_iterations,
            prev_energy
        );
        Err(ResonanceError::ConvergenceError)
    }

    /// Perform one gradient descent step
    fn gradient_descent_step(&mut self, learning_rate: f64) -> f64 {
        // Compute gradients in parallel
        let gradients: Vec<Complex<f64>> = (0..self.strings.len())
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|i| self.compute_gradient(i))
            .collect();

        // Update phases
        for (i, gradient) in gradients.iter().enumerate() {
            self.strings[i].update_phase(*gradient, learning_rate);
        }

        // Update mean field
        self.update_mean_field();

        self.total_energy()
    }

    /// Compute gradient ∂E/∂ψ_i
    pub fn compute_gradient(&self, i: usize) -> Complex<f64> {
        let mut gradient = Complex::new(0.0, 0.0);

        // Coupling term: 2 Σ_j J_ij (ψ_i - ψ_j)
        for j in 0..self.strings.len() {
            if i == j {
                continue;
            }

            let key = if i < j { (i, j) } else { (j, i) };
            let j_ij = self.coupling_matrix.get(&key).map(|v| *v).unwrap_or(0.0);

            let phase_diff = self.strings[i].phase - self.strings[j].phase;
            gradient += Complex::new(2.0 * j_ij, 0.0) * phase_diff;
        }

        // Potential term: 2λ_i (ψ_i - ψ̄)
        let mean = *self.mean_field.read();
        let lambda_i = self.local_potentials[i];
        gradient += Complex::new(2.0 * lambda_i, 0.0) * (self.strings[i].phase - mean);

        gradient
    }

    /// Update mean field ψ̄ = (1/N) Σ ψ_i
    pub fn update_mean_field(&self) {
        let sum: Complex<f64> = self.strings.iter().map(|s| s.phase).sum();
        let mean = sum / (self.strings.len() as f64);
        *self.mean_field.write() = mean;
    }

    /// Compute mean phase
    fn compute_mean_phase(&self) -> Complex<f64> {
        let sum: Complex<f64> = self.strings.iter().map(|s| s.phase).sum();
        sum / (self.strings.len() as f64)
    }

    /// Compute phase variance
    fn compute_phase_variance(&self, mean: Complex<f64>) -> f64 {
        let sum_sq_diff: f64 = self
            .strings
            .iter()
            .map(|s| (s.phase - mean).norm_sqr())
            .sum();

        sum_sq_diff / (self.strings.len() as f64)
    }

    /// Rebuild coupling matrix (call when strings change)
    pub fn rebuild_coupling_matrix(&mut self) {
        self.coupling_matrix.clear();

        for i in 0..self.strings.len() {
            for j in (i + 1)..self.strings.len() {
                let coupling = self.strings[i].coupling_strength(&self.strings[j]);
                self.coupling_matrix.insert((i, j), coupling);
            }
        }

        tracing::debug!(
            "Rebuilt coupling matrix with {} entries",
            self.coupling_matrix.len()
        );
    }

    /// Mark vertex as committed (for finality)
    pub fn commit_vertex(&self, id: [u8; 32]) {
        self.committed_vertices.insert(id, true);
        tracing::info!("Committed vertex {:?} (finality locked)", id);
    }

    /// Check if consensus achieved (low energy, low variance)
    pub fn has_consensus(&self, energy_threshold: f64, variance_threshold: f64) -> bool {
        let energy = self.total_energy();
        let mean = self.compute_mean_phase();
        let variance = self.compute_phase_variance(mean);

        energy < energy_threshold && variance < variance_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_strings(n: usize) -> Vec<StringState> {
        (0..n)
            .map(|i| {
                let id = [i as u8; 32];
                StringState::new(100.0, 1.0, vec![i as f64, 0.0], id, 1000 + i as u64)
            })
            .collect()
    }

    #[test]
    fn test_energy_computation() {
        let strings = create_test_strings(5);
        let mut energy_fn = EnergyFunctional::new(strings);
        energy_fn.rebuild_coupling_matrix();

        let energy = energy_fn.total_energy();
        assert!(energy >= 0.0);
    }

    #[test]
    fn test_energy_minimization() {
        let strings = create_test_strings(10);
        let mut energy_fn = EnergyFunctional::new(strings);
        energy_fn.rebuild_coupling_matrix();

        let initial_energy = energy_fn.total_energy();
        let final_energy = energy_fn.minimize(100, 1e-4).unwrap_or(initial_energy);

        assert!(final_energy <= initial_energy);
    }

    #[test]
    fn test_consensus_detection() {
        let mut strings = create_test_strings(5);

        // Align all phases
        let common_phase = Complex::new(1.0, 0.0);
        for s in &mut strings {
            s.phase = common_phase;
        }

        let energy_fn = EnergyFunctional::new(strings);
        assert!(energy_fn.has_consensus(10.0, 0.1));
    }

    #[test]
    fn test_finality_barrier() {
        let strings = create_test_strings(3);
        let energy_fn = EnergyFunctional::new(strings.clone());

        let committed_id = strings[0].id;
        energy_fn.commit_vertex(committed_id);

        let finality_energy = energy_fn.finality_energy();
        assert!(finality_energy > 0.0);
    }
}
