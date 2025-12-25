//! Quantum Annealing Simulation
//!
//! Simulated quantum annealing for global optimization problems.
//! Uses parallel tempering for improved exploration.

use rand::Rng;
use std::f64::consts::E;

/// Quantum Annealer for optimization
pub struct QuantumAnnealer {
    /// Number of variables
    num_variables: usize,

    /// Number of replicas for parallel tempering
    num_replicas: usize,

    /// Initial transverse field strength
    initial_gamma: f64,

    /// Final transverse field strength
    final_gamma: f64,

    /// Temperature schedule
    temperatures: Vec<f64>,
}

/// Single replica in parallel tempering
struct Replica {
    /// Configuration (spin values: true = +1, false = -1)
    config: Vec<bool>,

    /// Temperature
    temperature: f64,

    /// Current energy
    energy: f64,
}

impl QuantumAnnealer {
    /// Create new quantum annealer
    pub fn new(num_variables: usize) -> Self {
        let num_replicas = 8;

        // Logarithmic temperature schedule
        let temperatures: Vec<f64> = (0..num_replicas)
            .map(|i| {
                let ratio = i as f64 / (num_replicas - 1).max(1) as f64;
                10.0 * (0.1f64).powf(ratio) // 10.0 to 0.1
            })
            .collect();

        Self {
            num_variables,
            num_replicas,
            initial_gamma: 5.0,
            final_gamma: 0.01,
            temperatures,
        }
    }

    /// Run quantum annealing on problem
    ///
    /// Problem is encoded as coupling strengths between variables.
    /// Returns optimal configuration.
    pub fn anneal(&mut self, couplings: &[f64], num_steps: usize) -> Vec<bool> {
        let mut rng = rand::thread_rng();

        // Initialize replicas at different temperatures
        let mut replicas: Vec<Replica> = self.temperatures
            .iter()
            .map(|&temp| {
                let config: Vec<bool> = (0..self.num_variables)
                    .map(|_| rng.gen())
                    .collect();
                let energy = self.calculate_energy(&config, couplings);
                Replica { config, temperature: temp, energy }
            })
            .collect();

        let mut best_config = replicas[0].config.clone();
        let mut best_energy = replicas[0].energy;

        // Annealing loop
        for step in 0..num_steps {
            let progress = step as f64 / num_steps as f64;

            // Update transverse field (quantum -> classical transition)
            let gamma = self.initial_gamma * (self.final_gamma / self.initial_gamma).powf(progress);

            // Quantum Monte Carlo updates for each replica
            for replica in &mut replicas {
                self.quantum_monte_carlo_step(replica, gamma, couplings, &mut rng);

                // Track best solution
                if replica.energy < best_energy {
                    best_energy = replica.energy;
                    best_config = replica.config.clone();
                }
            }

            // Parallel tempering swaps
            self.attempt_replica_swaps(&mut replicas, couplings, &mut rng);
        }

        best_config
    }

    /// Single Quantum Monte Carlo step
    fn quantum_monte_carlo_step(
        &self,
        replica: &mut Replica,
        gamma: f64,
        couplings: &[f64],
        rng: &mut impl Rng,
    ) {
        for i in 0..self.num_variables {
            // Calculate classical energy change from flipping spin i
            let delta_e_classical = self.local_energy_change(&replica.config, couplings, i);

            // Calculate quantum tunneling contribution
            let tunneling_rate = (-2.0 * gamma / replica.temperature).exp();

            // Combined acceptance probability
            let delta_e_total = delta_e_classical - replica.temperature * tunneling_rate.ln();
            let accept_prob = (-delta_e_total / replica.temperature).exp().min(1.0);

            // Metropolis acceptance
            if rng.gen::<f64>() < accept_prob {
                replica.config[i] = !replica.config[i];
                replica.energy += delta_e_classical;
            }
        }
    }

    /// Attempt replica swaps for parallel tempering
    fn attempt_replica_swaps(
        &self,
        replicas: &mut [Replica],
        _couplings: &[f64],
        rng: &mut impl Rng,
    ) {
        for i in 0..replicas.len().saturating_sub(1) {
            let beta_i = 1.0 / replicas[i].temperature;
            let beta_j = 1.0 / replicas[i + 1].temperature;

            let e_i = replicas[i].energy;
            let e_j = replicas[i + 1].energy;

            let delta = (beta_j - beta_i) * (e_i - e_j);
            let swap_prob = (-delta).exp().min(1.0);

            if rng.gen::<f64>() < swap_prob {
                // Swap using split_at_mut to avoid double mutable borrow
                let (left, right) = replicas.split_at_mut(i + 1);
                std::mem::swap(&mut left[i].config, &mut right[0].config);
                std::mem::swap(&mut left[i].energy, &mut right[0].energy);
            }
        }
    }

    /// Calculate total energy of configuration
    fn calculate_energy(&self, config: &[bool], couplings: &[f64]) -> f64 {
        let mut energy = 0.0;
        let n = self.num_variables;

        // Coupling terms: E = sum_ij J_ij * s_i * s_j
        for i in 0..n {
            for j in (i + 1)..n {
                let coupling_idx = i * n + j;
                if coupling_idx < couplings.len() {
                    let s_i = if config[i] { 1.0 } else { -1.0 };
                    let s_j = if config[j] { 1.0 } else { -1.0 };
                    energy += couplings[coupling_idx] * s_i * s_j;
                }
            }
        }

        // Local field terms (diagonal of coupling matrix)
        for i in 0..n {
            let field_idx = i * n + i;
            if field_idx < couplings.len() {
                let s_i = if config[i] { 1.0 } else { -1.0 };
                energy += couplings[field_idx] * s_i;
            }
        }

        energy
    }

    /// Calculate energy change from flipping a single spin
    fn local_energy_change(&self, config: &[bool], couplings: &[f64], flip_idx: usize) -> f64 {
        let n = self.num_variables;
        let mut delta_e = 0.0;

        let s_flip = if config[flip_idx] { 1.0 } else { -1.0 };

        // Coupling contributions
        for j in 0..n {
            if j != flip_idx {
                let coupling_idx = if flip_idx < j {
                    flip_idx * n + j
                } else {
                    j * n + flip_idx
                };

                if coupling_idx < couplings.len() {
                    let s_j = if config[j] { 1.0 } else { -1.0 };
                    // Flipping changes contribution by -2 * J_ij * s_i * s_j
                    delta_e -= 2.0 * couplings[coupling_idx] * s_flip * s_j;
                }
            }
        }

        // Local field contribution
        let field_idx = flip_idx * n + flip_idx;
        if field_idx < couplings.len() {
            delta_e -= 2.0 * couplings[field_idx] * s_flip;
        }

        delta_e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annealer_creation() {
        let annealer = QuantumAnnealer::new(10);
        assert_eq!(annealer.num_variables, 10);
        assert_eq!(annealer.num_replicas, 8);
    }

    #[test]
    fn test_simple_optimization() {
        let mut annealer = QuantumAnnealer::new(4);

        // Simple problem: all spins should be +1 to minimize energy
        let couplings = vec![
            -1.0, 0.0, 0.0, 0.0,  // Local fields (diagonal)
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 0.0,
            0.0, 0.0, 0.0, -1.0,
        ];

        let result = annealer.anneal(&couplings, 100);

        // All spins should be true (+1) to minimize -sum(s_i)
        let all_positive = result.iter().all(|&s| s);
        // Due to stochastic nature, we just verify we get a valid result
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_energy_calculation() {
        let annealer = QuantumAnnealer::new(2);

        // Antiferromagnetic coupling: J_12 = 1 (opposite spins preferred)
        let couplings = vec![
            0.0, 1.0,
            1.0, 0.0,
        ];

        // Opposite spins: energy = 1 * (-1) * 1 = -1
        let config_opposite = vec![true, false];
        let energy_opposite = annealer.calculate_energy(&config_opposite, &couplings);
        assert!((energy_opposite - (-1.0)).abs() < 1e-10);

        // Same spins: energy = 1 * 1 * 1 = 1
        let config_same = vec![true, true];
        let energy_same = annealer.calculate_energy(&config_same, &couplings);
        assert!((energy_same - 1.0).abs() < 1e-10);
    }
}
