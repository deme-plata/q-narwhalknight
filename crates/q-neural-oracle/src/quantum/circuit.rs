//! Variational Quantum Circuits
//!
//! Implements parametrized quantum circuits for trainable quantum transformations.

use super::state::QuantumState;
use rand::Rng;
use std::f64::consts::PI;

/// Quantum gate types
#[derive(Clone, Debug)]
pub enum QuantumGate {
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    CNOT,
    RotationX(f64),
    RotationY(f64),
    RotationZ(f64),
    ControlledRotationZ(f64),
}

/// Variational Quantum Circuit for feature processing
#[derive(Clone)]
pub struct VariationalQuantumCircuit {
    /// Number of qubits
    num_qubits: usize,

    /// Number of layers
    num_layers: usize,

    /// Trainable rotation angles (3 per qubit per layer + entangling gates)
    parameters: Vec<f64>,

    /// Learning rate for parameter updates
    learning_rate: f64,
}

impl VariationalQuantumCircuit {
    /// Create new variational circuit
    pub fn new(num_qubits: usize, num_layers: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Parameters: 3 rotations per qubit + 1 entangling gate per adjacent pair
        let params_per_layer = num_qubits * 3 + (num_qubits - 1);
        let num_params = num_layers * params_per_layer;

        // Initialize with random angles
        let parameters: Vec<f64> = (0..num_params)
            .map(|_| rng.gen::<f64>() * 2.0 * PI)
            .collect();

        Self {
            num_qubits,
            num_layers,
            parameters,
            learning_rate: 0.01,
        }
    }

    /// Apply circuit to quantum state
    pub fn apply(&self, state: &mut QuantumState) {
        let mut param_idx = 0;

        for _ in 0..self.num_layers {
            // Single-qubit rotations (RX, RY, RZ per qubit)
            for qubit in 0..self.num_qubits {
                state.apply_rotation_x(qubit, self.parameters[param_idx]);
                param_idx += 1;
                state.apply_rotation_y(qubit, self.parameters[param_idx]);
                param_idx += 1;
                state.apply_rotation_z(qubit, self.parameters[param_idx]);
                param_idx += 1;
            }

            // Entangling layer (CNOT ladder + CRZ)
            for qubit in 0..(self.num_qubits - 1) {
                state.apply_cnot(qubit, qubit + 1);
                state.apply_controlled_rotation_z(qubit, qubit + 1, self.parameters[param_idx]);
                param_idx += 1;
            }
        }
    }

    /// Calculate gradient using parameter shift rule
    pub fn gradient(&self, state: &QuantumState, loss_fn: impl Fn(&QuantumState) -> f64) -> Vec<f64> {
        let shift = PI / 2.0;
        let mut gradients = vec![0.0; self.parameters.len()];

        for (i, _) in self.parameters.iter().enumerate() {
            // Forward shift
            let mut params_plus = self.parameters.clone();
            params_plus[i] += shift;
            let loss_plus = self.evaluate_with_params(state, &params_plus, &loss_fn);

            // Backward shift
            let mut params_minus = self.parameters.clone();
            params_minus[i] -= shift;
            let loss_minus = self.evaluate_with_params(state, &params_minus, &loss_fn);

            // Parameter shift gradient
            gradients[i] = (loss_plus - loss_minus) / 2.0;
        }

        gradients
    }

    /// Evaluate circuit with given parameters
    fn evaluate_with_params(
        &self,
        initial_state: &QuantumState,
        params: &[f64],
        loss_fn: &impl Fn(&QuantumState) -> f64,
    ) -> f64 {
        let circuit = VariationalQuantumCircuit {
            num_qubits: self.num_qubits,
            num_layers: self.num_layers,
            parameters: params.to_vec(),
            learning_rate: self.learning_rate,
        };

        let mut state = initial_state.clone();
        circuit.apply(&mut state);
        loss_fn(&state)
    }

    /// Update parameters using gradient descent
    pub fn update_parameters(&mut self, gradients: &[f64]) {
        for (i, grad) in gradients.iter().enumerate() {
            self.parameters[i] -= self.learning_rate * grad;

            // Keep angles in [0, 2π]
            while self.parameters[i] < 0.0 {
                self.parameters[i] += 2.0 * PI;
            }
            while self.parameters[i] > 2.0 * PI {
                self.parameters[i] -= 2.0 * PI;
            }
        }
    }

    /// Get current parameters
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_creation() {
        let circuit = VariationalQuantumCircuit::new(4, 2);
        assert!(!circuit.parameters.is_empty());
    }

    #[test]
    fn test_circuit_application() {
        let circuit = VariationalQuantumCircuit::new(3, 2);
        let mut state = QuantumState::uniform_superposition(3);

        circuit.apply(&mut state);

        // State should still be normalized
        let total_prob: f64 = state.amplitudes()
            .iter()
            .map(|a| a.norm_sqr())
            .sum();
        assert!((total_prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_calculation() {
        let circuit = VariationalQuantumCircuit::new(2, 1);
        let state = QuantumState::uniform_superposition(2);

        // Simple loss: expectation of Z on first qubit
        let loss_fn = |s: &QuantumState| s.expectation_z(0);

        let gradients = circuit.gradient(&state, loss_fn);
        assert_eq!(gradients.len(), circuit.parameters.len());
    }
}
