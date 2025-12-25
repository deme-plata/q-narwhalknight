//! Quantum State Representation
//!
//! Implements amplitude-encoded quantum states for quantum simulation.

use num_complex::Complex64;
use std::f64::consts::SQRT_2;

/// Quantum state representation using amplitude encoding
#[derive(Clone, Debug)]
pub struct QuantumState {
    /// Complex amplitudes for 2^n basis states
    amplitudes: Vec<Complex64>,

    /// Number of qubits
    num_qubits: usize,

    /// Fidelity estimate (quality of quantum state)
    fidelity_estimate: f64,
}

impl QuantumState {
    /// Create a new quantum state with given number of qubits, initialized to |0⟩
    pub fn new(num_qubits: usize) -> Self {
        let num_states = 1 << num_qubits;
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); num_states];
        amplitudes[0] = Complex64::new(1.0, 0.0); // |00...0⟩

        Self {
            amplitudes,
            num_qubits,
            fidelity_estimate: 1.0,
        }
    }

    /// Create uniform superposition of all basis states
    pub fn uniform_superposition(num_qubits: usize) -> Self {
        let num_states = 1 << num_qubits;
        let amplitude = Complex64::new(1.0 / (num_states as f64).sqrt(), 0.0);

        Self {
            amplitudes: vec![amplitude; num_states],
            num_qubits,
            fidelity_estimate: 1.0,
        }
    }

    /// Encode classical data as quantum amplitudes
    pub fn from_classical_data(data: &[f64]) -> Self {
        // Determine number of qubits needed
        let num_qubits = (data.len() as f64).log2().ceil() as usize;
        let num_states = 1 << num_qubits;

        // Normalize data to unit sphere
        let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm = norm.max(1e-10);

        // Create amplitudes from normalized data
        let mut amplitudes: Vec<Complex64> = data
            .iter()
            .map(|&x| Complex64::new(x / norm, 0.0))
            .collect();

        // Pad to power of 2
        amplitudes.resize(num_states, Complex64::new(0.0, 0.0));

        // Renormalize to ensure valid quantum state
        let total_norm: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        for amp in &mut amplitudes {
            *amp /= total_norm;
        }

        Self {
            amplitudes,
            num_qubits,
            fidelity_estimate: 1.0,
        }
    }

    /// Apply Hadamard gate to specified qubit
    pub fn apply_hadamard(&mut self, qubit: usize) {
        let h_factor = 1.0 / SQRT_2;
        let num_states = self.amplitudes.len();
        let mask = 1 << qubit;

        for i in 0..num_states {
            if i & mask == 0 {
                let j = i | mask;
                let a = self.amplitudes[i];
                let b = self.amplitudes[j];

                self.amplitudes[i] = Complex64::new(h_factor, 0.0) * (a + b);
                self.amplitudes[j] = Complex64::new(h_factor, 0.0) * (a - b);
            }
        }
    }

    /// Apply CNOT gate (controlled-NOT)
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        let num_states = self.amplitudes.len();
        let control_mask = 1 << control;
        let target_mask = 1 << target;

        for i in 0..num_states {
            if i & control_mask != 0 {
                let j = i ^ target_mask;
                if i < j {
                    self.amplitudes.swap(i, j);
                }
            }
        }
    }

    /// Apply rotation around X axis
    pub fn apply_rotation_x(&mut self, qubit: usize, angle: f64) {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        let num_states = self.amplitudes.len();
        let mask = 1 << qubit;

        for i in 0..num_states {
            if i & mask == 0 {
                let j = i | mask;
                let a = self.amplitudes[i];
                let b = self.amplitudes[j];

                self.amplitudes[i] = Complex64::new(cos_half, 0.0) * a
                    + Complex64::new(0.0, -sin_half) * b;
                self.amplitudes[j] = Complex64::new(0.0, -sin_half) * a
                    + Complex64::new(cos_half, 0.0) * b;
            }
        }
    }

    /// Apply rotation around Y axis
    pub fn apply_rotation_y(&mut self, qubit: usize, angle: f64) {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        let num_states = self.amplitudes.len();
        let mask = 1 << qubit;

        for i in 0..num_states {
            if i & mask == 0 {
                let j = i | mask;
                let a = self.amplitudes[i];
                let b = self.amplitudes[j];

                self.amplitudes[i] = Complex64::new(cos_half, 0.0) * a
                    - Complex64::new(sin_half, 0.0) * b;
                self.amplitudes[j] = Complex64::new(sin_half, 0.0) * a
                    + Complex64::new(cos_half, 0.0) * b;
            }
        }
    }

    /// Apply rotation around Z axis
    pub fn apply_rotation_z(&mut self, qubit: usize, angle: f64) {
        let phase_0 = Complex64::from_polar(1.0, -angle / 2.0);
        let phase_1 = Complex64::from_polar(1.0, angle / 2.0);
        let num_states = self.amplitudes.len();
        let mask = 1 << qubit;

        for i in 0..num_states {
            if i & mask == 0 {
                self.amplitudes[i] *= phase_0;
            } else {
                self.amplitudes[i] *= phase_1;
            }
        }
    }

    /// Apply controlled rotation around Z axis
    pub fn apply_controlled_rotation_z(&mut self, control: usize, target: usize, angle: f64) {
        let phase = Complex64::from_polar(1.0, angle);
        let num_states = self.amplitudes.len();
        let control_mask = 1 << control;
        let target_mask = 1 << target;

        for i in 0..num_states {
            // Only apply phase when both control and target are |1⟩
            if (i & control_mask != 0) && (i & target_mask != 0) {
                self.amplitudes[i] *= phase;
            }
        }
    }

    /// Measure quantum state (returns measurement result)
    pub fn measure(&self) -> MeasurementResult {
        use rand::distributions::{Distribution, WeightedIndex};
        use rand::thread_rng;

        // Calculate probability distribution
        let probabilities: Vec<f64> = self.amplitudes
            .iter()
            .map(|a| a.norm_sqr())
            .collect();

        // Sample from distribution
        let dist = WeightedIndex::new(&probabilities).unwrap();
        let measured_state = dist.sample(&mut thread_rng());

        MeasurementResult {
            measured_state,
            probability: probabilities[measured_state],
        }
    }

    /// Calculate expectation value of Z operator on specified qubit
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        let mask = 1 << qubit;
        let mut expectation = 0.0;

        for (i, amp) in self.amplitudes.iter().enumerate() {
            let eigenvalue = if i & mask == 0 { 1.0 } else { -1.0 };
            expectation += eigenvalue * amp.norm_sqr();
        }

        expectation
    }

    /// Calculate entanglement entropy
    pub fn entanglement_entropy(&self) -> f64 {
        if self.num_qubits < 2 {
            return 0.0;
        }

        // Partition at half
        let partition_size = self.num_qubits / 2;
        let subsystem_dim = 1 << partition_size;
        let env_dim = 1 << (self.num_qubits - partition_size);

        // Calculate reduced density matrix
        let mut reduced_density = vec![vec![Complex64::new(0.0, 0.0); subsystem_dim]; subsystem_dim];

        for i in 0..subsystem_dim {
            for j in 0..subsystem_dim {
                for k in 0..env_dim {
                    let idx_i = i + k * subsystem_dim;
                    let idx_j = j + k * subsystem_dim;
                    reduced_density[i][j] += self.amplitudes[idx_i] * self.amplitudes[idx_j].conj();
                }
            }
        }

        // Calculate von Neumann entropy: S = -Tr(ρ log ρ)
        // Approximate by calculating eigenvalues (use diagonal for simplicity)
        let mut entropy = 0.0;
        for i in 0..subsystem_dim {
            let eigenvalue = reduced_density[i][i].re;
            if eigenvalue > 1e-10 {
                entropy -= eigenvalue * eigenvalue.ln();
            }
        }

        entropy
    }

    /// Get fidelity estimate
    pub fn fidelity(&self) -> f64 {
        self.fidelity_estimate
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get amplitudes (for expert systems)
    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amplitudes
    }

    /// Create from raw amplitudes (for GPU compatibility)
    pub fn from_amplitudes(amplitudes: Vec<Complex64>, num_qubits: usize) -> Self {
        Self {
            amplitudes,
            num_qubits,
            fidelity_estimate: 1.0,
        }
    }

    /// Apply sigmoid-like activation (measurement and rescaling)
    pub fn apply_sigmoid_activation(&self) -> QuantumState {
        let mut result = self.clone();

        // Apply sigmoid to probability amplitudes
        for amp in &mut result.amplitudes {
            let magnitude = amp.norm();
            let sigmoid_mag = 1.0 / (1.0 + (-magnitude).exp());
            if magnitude > 1e-10 {
                *amp = *amp * (sigmoid_mag / magnitude);
            }
        }

        // Renormalize
        let norm: f64 = result.amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        for amp in &mut result.amplitudes {
            *amp /= norm;
        }

        result.fidelity_estimate *= 0.99; // Slight fidelity loss
        result
    }

    /// Apply tanh activation
    pub fn apply_tanh_activation(&self) -> QuantumState {
        let mut result = self.clone();

        for amp in &mut result.amplitudes {
            let magnitude = amp.norm();
            let tanh_mag = magnitude.tanh();
            if magnitude > 1e-10 {
                *amp = *amp * (tanh_mag / magnitude);
            }
        }

        // Renormalize
        let norm: f64 = result.amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        for amp in &mut result.amplitudes {
            *amp /= norm;
        }

        result.fidelity_estimate *= 0.99;
        result
    }

    /// Hadamard product (element-wise)
    pub fn hadamard_product(&self, other: &QuantumState) -> QuantumState {
        assert_eq!(self.amplitudes.len(), other.amplitudes.len());

        let mut result_amplitudes: Vec<Complex64> = self.amplitudes
            .iter()
            .zip(other.amplitudes.iter())
            .map(|(a, b)| a * b)
            .collect();

        // Renormalize
        let norm: f64 = result_amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for amp in &mut result_amplitudes {
                *amp /= norm;
            }
        }

        QuantumState {
            amplitudes: result_amplitudes,
            num_qubits: self.num_qubits,
            fidelity_estimate: self.fidelity_estimate * other.fidelity_estimate,
        }
    }

    /// Quantum addition (superposition)
    pub fn quantum_add(&self, other: &QuantumState) -> QuantumState {
        assert_eq!(self.amplitudes.len(), other.amplitudes.len());

        let mut result_amplitudes: Vec<Complex64> = self.amplitudes
            .iter()
            .zip(other.amplitudes.iter())
            .map(|(a, b)| a + b)
            .collect();

        // Renormalize
        let norm: f64 = result_amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        for amp in &mut result_amplitudes {
            *amp /= norm;
        }

        QuantumState {
            amplitudes: result_amplitudes,
            num_qubits: self.num_qubits,
            fidelity_estimate: (self.fidelity_estimate + other.fidelity_estimate) / 2.0,
        }
    }

    /// Tensor product (combine quantum states)
    pub fn tensor_product(&self, other: &QuantumState) -> QuantumState {
        let new_num_qubits = self.num_qubits + other.num_qubits;
        let mut new_amplitudes = Vec::with_capacity(self.amplitudes.len() * other.amplitudes.len());

        for a in &self.amplitudes {
            for b in &other.amplitudes {
                new_amplitudes.push(a * b);
            }
        }

        QuantumState {
            amplitudes: new_amplitudes,
            num_qubits: new_num_qubits,
            fidelity_estimate: self.fidelity_estimate * other.fidelity_estimate,
        }
    }

    /// Apply dropout (zero random amplitudes)
    pub fn apply_dropout(&mut self, rate: f64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for amp in &mut self.amplitudes {
            if rng.gen::<f64>() < rate {
                *amp = Complex64::new(0.0, 0.0);
            }
        }

        // Renormalize
        let norm: f64 = self.amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for amp in &mut self.amplitudes {
                *amp /= norm;
            }
        }

        self.fidelity_estimate *= 1.0 - rate * 0.1;
    }
}

/// Measurement result
#[derive(Clone, Debug)]
pub struct MeasurementResult {
    /// Index of measured basis state
    pub measured_state: usize,

    /// Probability of this measurement
    pub probability: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_superposition() {
        let state = QuantumState::uniform_superposition(3);
        assert_eq!(state.num_qubits(), 3);

        // All amplitudes should be equal
        let expected_amp = 1.0 / (8.0f64).sqrt();
        for amp in state.amplitudes() {
            assert!((amp.re - expected_amp).abs() < 1e-10);
            assert!(amp.im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_hadamard_creates_superposition() {
        let mut state = QuantumState::new(1);
        state.apply_hadamard(0);

        // Should be (|0⟩ + |1⟩) / sqrt(2)
        let expected = 1.0 / SQRT_2;
        assert!((state.amplitudes()[0].re - expected).abs() < 1e-10);
        assert!((state.amplitudes()[1].re - expected).abs() < 1e-10);
    }

    #[test]
    fn test_from_classical_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let state = QuantumState::from_classical_data(&data);

        // State should be normalized
        let total_prob: f64 = state.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        assert!((total_prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_measurement_probabilities_sum_to_one() {
        let state = QuantumState::uniform_superposition(3);
        let probabilities: f64 = state.amplitudes()
            .iter()
            .map(|a| a.norm_sqr())
            .sum();
        assert!((probabilities - 1.0).abs() < 1e-10);
    }
}
