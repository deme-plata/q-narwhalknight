//! Quantum Computing Layer
//!
//! Provides simulated quantum computing for feature extraction and optimization.
//! Uses amplitude encoding and variational quantum circuits.
//!
//! ## Phases
//! - Phase 1: CPU simulation up to 25 qubits
//! - Phase 2: GPU-accelerated simulation up to 128 qubits

mod state;
mod circuit;
mod annealing;
pub mod gpu_simulation;

pub use state::QuantumState;
pub use circuit::{VariationalQuantumCircuit, QuantumGate};
pub use annealing::QuantumAnnealer;
pub use gpu_simulation::{GpuQuantumState, GpuQuantumSimulator, Observable};

use num_complex::Complex64;
use tracing::debug;

/// Quantum Layer - main interface for quantum operations
pub struct QuantumLayer {
    /// Number of qubits
    num_qubits: usize,

    /// Variational quantum circuit for feature processing
    vqc: VariationalQuantumCircuit,

    /// Quantum annealer for optimization
    annealer: QuantumAnnealer,
}

impl QuantumLayer {
    /// Create new quantum layer
    pub fn new(num_qubits: usize, circuit_layers: usize) -> Self {
        debug!("🔮 Creating quantum layer: {} qubits, {} layers", num_qubits, circuit_layers);

        Self {
            num_qubits,
            vqc: VariationalQuantumCircuit::new(num_qubits, circuit_layers),
            annealer: QuantumAnnealer::new(num_qubits),
        }
    }

    /// Encode classical features as quantum state
    pub fn encode_features(&self, features: &[f64]) -> QuantumState {
        // Amplitude encoding: classical vector -> quantum amplitudes
        let state = QuantumState::from_classical_data(features);

        // Apply variational circuit for feature transformation
        let mut processed = state;
        self.vqc.apply(&mut processed);

        processed
    }

    /// Run quantum annealing for optimization
    pub fn optimize(&mut self, problem: &[f64], num_steps: usize) -> Vec<bool> {
        self.annealer.anneal(problem, num_steps)
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}
