//! String State representation for resonance consensus
//!
//! A StringState represents a transaction as a vibrating string in multi-dimensional space.
//! The wavefunction ψ(x,t) = A·e^(i(kx - ωt + φ))·sin(nπx/L) encodes:
//! - Amplitude A: Proportional to sqrt(stake_weight)
//! - Frequency ω: Transaction priority (2π·priority)
//! - Phase φ: Temporal alignment
//! - Mode n: Harmonic layer

use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// String state representing a vibrating transaction in consensus space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringState {
    /// Amplitude A = sqrt(stake_weight)
    pub amplitude: f64,

    /// Frequency ω = 2π·priority
    pub frequency: f64,

    /// Phase e^(iφ) for temporal alignment
    pub phase: Complex<f64>,

    /// Harmonic mode number n (layer in hypergraph)
    pub mode: u32,

    /// Position in n-dimensional consensus space
    pub position: Vec<f64>,

    /// Velocity vector dx/dt
    pub velocity: Vec<f64>,

    /// Unique identifier (transaction hash)
    pub id: [u8; 32],

    /// Unix timestamp in milliseconds
    pub timestamp: u64,

    /// Arbitrary metadata (ZK-proofs, oracles, etc.)
    pub metadata: HashMap<String, Vec<u8>>,
}

impl StringState {
    /// Create a new string state from transaction parameters
    pub fn new(
        stake_weight: f64,
        priority: f64,
        position: Vec<f64>,
        id: [u8; 32],
        timestamp: u64,
    ) -> Self {
        let amplitude = stake_weight.sqrt();
        let frequency = 2.0 * PI * priority;
        let phase = Complex::new(0.0, 0.0); // Start with zero phase
        let velocity = vec![1.0; position.len()]; // Unit velocity initially

        Self {
            amplitude,
            frequency,
            phase,
            mode: 1, // Ground state
            position,
            velocity,
            id,
            timestamp,
            metadata: HashMap::new(),
        }
    }

    /// Compute wavefunction ψ(x,t) at given position and time
    ///
    /// ψ(x,t) = A·e^(i(kx - ωt + φ))·sin(nπx/L)
    pub fn wavefunction(&self, x: &[f64], t: f64) -> Complex<f64> {
        // Wave number k = 2π·frequency / velocity
        let velocity_magnitude: f64 = self.velocity.iter().sum::<f64>();
        let k = 2.0 * PI * self.frequency / velocity_magnitude;

        // Angular frequency ω
        let omega = 2.0 * PI * self.frequency;

        // Spatial phase: k·(x - x₀)
        let spatial_phase: f64 = x
            .iter()
            .zip(&self.position)
            .map(|(xi, xi0)| k * (xi - xi0))
            .sum();

        // Temporal phase: ωt
        let temporal_phase = omega * t;

        // Total phase: kx - ωt + φ
        let total_phase = spatial_phase - temporal_phase + self.phase.arg();

        // Standing wave envelope: sin(nπx/L)
        let length = self.position.len() as f64;
        let standing_wave: f64 = x
            .iter()
            .enumerate()
            .map(|(_i, &xi)| ((self.mode as f64 * PI * xi) / length).sin())
            .product();

        // ψ = A·e^(iθ)·sin_envelope
        self.amplitude * Complex::from_polar(standing_wave.abs(), total_phase)
    }

    /// Compute coupling strength J_ij with another string
    ///
    /// J_ij = (A_i·A_j)^(1/2) · Re(φ_i·φ_j*) / (1 + ||x_i - x_j||)
    pub fn coupling_strength(&self, other: &StringState) -> f64 {
        // Spatial distance in n-D space
        let distance: f64 = self
            .position
            .iter()
            .zip(&other.position)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // Stake coupling factor
        let stake_factor = (self.amplitude * other.amplitude).sqrt();

        // Phase coherence: Re(φ_i · φ_j*)
        let phase_coherence = (self.phase * other.phase.conj()).re;

        // J_ij = stake · coherence / (1 + distance)
        stake_factor * phase_coherence / (1.0 + distance)
    }

    /// Compute resonance score with another string (for ordering)
    pub fn resonance(&self, other: &StringState) -> f64 {
        let coupling = self.coupling_strength(other);
        let frequency_match = (-0.5 * (self.frequency - other.frequency).powi(2)).exp();
        coupling * frequency_match
    }

    /// Update phase based on gradient of energy functional
    pub fn update_phase(&mut self, gradient: Complex<f64>, learning_rate: f64) {
        let delta_phase = gradient * Complex::new(-learning_rate, 0.0);
        self.phase += delta_phase;

        // Normalize to unit circle
        self.phase /= self.phase.norm();
    }

    /// Check if this string is in-phase with another (aligned for consensus)
    pub fn in_phase_with(&self, other: &StringState, threshold: f64) -> bool {
        let phase_diff = (self.phase - other.phase).norm();
        phase_diff < threshold
    }

    /// Compute energy contribution from this string
    pub fn kinetic_energy(&self) -> f64 {
        let velocity_sq: f64 = self.velocity.iter().map(|v| v.powi(2)).sum();
        0.5 * self.amplitude.powi(2) * velocity_sq
    }

    /// Set metadata field
    pub fn set_metadata(&mut self, key: String, value: Vec<u8>) {
        self.metadata.insert(key, value);
    }

    /// Get metadata field
    pub fn get_metadata(&self, key: &str) -> Option<&Vec<u8>> {
        self.metadata.get(key)
    }

    /// Compute hash of string state for verification
    pub fn compute_hash(&self) -> [u8; 32] {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(&self.amplitude.to_le_bytes());
        hasher.update(&self.frequency.to_le_bytes());
        hasher.update(&self.phase.re.to_le_bytes());
        hasher.update(&self.phase.im.to_le_bytes());
        hasher.update(&self.mode.to_le_bytes());
        hasher.update(&self.id);
        hasher.update(&self.timestamp.to_le_bytes());

        let hash = hasher.finalize();
        let mut result = [0u8; 32];
        result.copy_from_slice(hash.as_bytes());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_state_creation() {
        let position = vec![0.0, 0.0, 0.0];
        let id = [1u8; 32];
        let state = StringState::new(100.0, 1.0, position, id, 1000);

        assert!((state.amplitude - 10.0).abs() < 1e-10);
        assert!((state.frequency - 2.0 * PI).abs() < 1e-10);
        assert_eq!(state.mode, 1);
    }

    #[test]
    fn test_wavefunction_computation() {
        let position = vec![0.0, 0.0];
        let id = [2u8; 32];
        let state = StringState::new(100.0, 1.0, position.clone(), id, 1000);

        let psi = state.wavefunction(&position, 0.0);
        assert!(psi.norm() > 0.0);
    }

    #[test]
    fn test_coupling_strength() {
        let id1 = [3u8; 32];
        let id2 = [4u8; 32];

        let state1 = StringState::new(100.0, 1.0, vec![0.0, 0.0], id1, 1000);
        let mut state2 = StringState::new(100.0, 1.0, vec![1.0, 1.0], id2, 1001);

        // Set same phase for maximum coupling
        state2.phase = state1.phase;

        let coupling = state1.coupling_strength(&state2);
        assert!(coupling > 0.0);
    }

    #[test]
    fn test_phase_alignment() {
        let id1 = [5u8; 32];
        let id2 = [6u8; 32];

        let state1 = StringState::new(100.0, 1.0, vec![0.0], id1, 1000);
        let mut state2 = StringState::new(100.0, 1.0, vec![0.0], id2, 1000);

        state2.phase = state1.phase;
        assert!(state1.in_phase_with(&state2, 0.1));

        state2.phase = Complex::new(0.0, 1.0);
        assert!(!state1.in_phase_with(&state2, 0.1));
    }

    #[test]
    fn test_resonance_score() {
        let id1 = [7u8; 32];
        let id2 = [8u8; 32];

        let state1 = StringState::new(100.0, 1.0, vec![0.0], id1, 1000);
        let mut state2 = StringState::new(100.0, 1.0, vec![0.1], id2, 1001);

        state2.phase = state1.phase;
        state2.frequency = state1.frequency;

        let resonance = state1.resonance(&state2);
        assert!(resonance > 0.0);
    }
}
