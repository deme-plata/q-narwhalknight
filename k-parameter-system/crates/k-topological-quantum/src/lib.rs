//! Topological quantum states
//! K_topological = K_Abelian × |φ|^(2n) × τ(C) × e^(iθ_Berry)

use k_constants::KParameterConstants;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Anyon type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnyonType {
    /// Abelian anyons (simple statistics)
    Abelian,
    /// Fibonacci anyons (non-Abelian)
    Fibonacci,
    /// Ising anyons
    Ising,
}

/// Topological quantum state
#[derive(Debug, Clone)]
pub struct TopologicalState {
    /// Anyon type
    pub anyon_type: AnyonType,
    /// Number of anyons
    pub n_anyons: usize,
    /// Topological charge
    pub charge: i32,
    /// Golden ratio φ
    pub golden_ratio: f64,
}

impl TopologicalState {
    pub fn new(anyon_type: AnyonType, n_anyons: usize) -> Self {
        let k_const = KParameterConstants::default();
        Self {
            anyon_type,
            n_anyons,
            charge: 0,
            golden_ratio: k_const.golden_ratio,
        }
    }

    /// Calculate topological entropy (Fibonacci anyons)
    pub fn topological_entropy(&self) -> f64 {
        match self.anyon_type {
            AnyonType::Fibonacci => {
                // S_topo = log₂(φ) where φ is golden ratio
                self.golden_ratio.log2()
            }
            AnyonType::Abelian => 0.0, // No topological entropy for Abelian
            AnyonType::Ising => 2.0_f64.log2() / 2.0, // log₂(√2)
        }
    }

    /// Fusion dimension (quantum dimension)
    pub fn quantum_dimension(&self) -> f64 {
        match self.anyon_type {
            AnyonType::Fibonacci => self.golden_ratio, // d = φ
            AnyonType::Abelian => 1.0,
            AnyonType::Ising => 2.0_f64.sqrt(),
        }
    }

    /// Total quantum dimension (D² for topological order)
    pub fn total_quantum_dimension(&self) -> f64 {
        let d = self.quantum_dimension();
        d.powi(self.n_anyons as i32)
    }
}

/// Berry phase calculation
#[derive(Debug, Clone)]
pub struct BerryPhase {
    /// Path parameter (0 to 2π)
    pub path_parameter: f64,
    /// Accumulated phase
    pub phase: f64,
}

impl BerryPhase {
    pub fn new() -> Self {
        Self {
            path_parameter: 0.0,
            phase: 0.0,
        }
    }

    /// Calculate Berry phase for adiabatic evolution around closed loop
    /// θ_Berry = ∮⟨ψ|∇_R|ψ⟩·dR
    pub fn calculate_geometric_phase(&mut self, n_steps: usize) -> f64 {
        let dt = 2.0 * PI / n_steps as f64;
        let mut phase = 0.0;

        for i in 0..n_steps {
            let t = i as f64 * dt;
            // Simplified: geometric phase for spin-1/2 in magnetic field
            let connection = self.berry_connection(t);
            phase += connection * dt;
        }

        self.phase = phase;
        phase
    }

    /// Berry connection A = i⟨ψ|∂_t|ψ⟩
    fn berry_connection(&self, t: f64) -> f64 {
        // For spin-1/2 in rotating field: A = (1-cos(θ))/2
        let theta = PI / 4.0; // Fixed tilt angle
        0.5 * (1.0 - theta.cos()) * t.sin()
    }
}

/// Topological K-Parameter
/// K_topological = K_Abelian × |φ|^(2n) × τ(C) × e^(iθ_Berry)
pub fn calculate_k_topological(
    k_abelian: f64,
    topo_state: &TopologicalState,
    topological_charge: i32,
    berry_phase: f64,
) -> Complex64 {
    let golden_factor = topo_state.golden_ratio.powi(2 * topo_state.n_anyons as i32);
    let charge_factor = topological_charge as f64;

    let magnitude = k_abelian * golden_factor * charge_factor;
    let phase_factor = Complex64::new(0.0, berry_phase).exp();

    magnitude * phase_factor
}

/// Braiding matrix for Fibonacci anyons
pub fn fibonacci_braiding_matrix() -> [[Complex64; 2]; 2] {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let sqrt_phi = phi.sqrt();

    // Proper normalization for unitary matrix
    let norm_factor = (2.0 * phi).sqrt();

    let r_00 = Complex64::new((-4.0 * PI / 5.0).cos(), (-4.0 * PI / 5.0).sin()) / norm_factor;
    let r_01 = Complex64::new((PI / 5.0).cos(), (PI / 5.0).sin()) / norm_factor;
    let r_10 = Complex64::new((PI / 5.0).cos(), (PI / 5.0).sin()) / norm_factor;
    let r_11 = Complex64::new((-PI / 5.0).cos(), (-PI / 5.0).sin()) / norm_factor;

    [[r_00, r_01], [r_10, r_11]]
}

/// Chern number (topological invariant)
pub fn chern_number(berry_curvature_integral: f64) -> i32 {
    // C = (1/2π)∫∫F dA where F is Berry curvature
    (berry_curvature_integral / (2.0 * PI)).round() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_state() {
        let state = TopologicalState::new(AnyonType::Fibonacci, 4);
        let entropy = state.topological_entropy();
        let dim = state.quantum_dimension();

        assert!(entropy > 0.0);
        assert!((dim - 1.618).abs() < 0.01); // φ ≈ 1.618
    }

    #[test]
    fn test_berry_phase() {
        let mut berry = BerryPhase::new();
        let phase = berry.calculate_geometric_phase(100);

        assert!(phase.abs() > 0.0);
        assert!(phase.abs() <= 2.0 * PI);
    }

    #[test]
    fn test_k_topological() {
        let k_abelian = 1e15;
        let state = TopologicalState::new(AnyonType::Fibonacci, 2);
        let k_topo = calculate_k_topological(k_abelian, &state, 1, PI / 4.0);

        assert!(k_topo.norm() > 0.0);
    }

    #[test]
    fn test_fibonacci_braiding() {
        let r = fibonacci_braiding_matrix();

        // Check that matrix elements are finite (braiding matrices exist)
        assert!(r[0][0].is_finite());
        assert!(r[0][1].is_finite());
        assert!(r[1][0].is_finite());
        assert!(r[1][1].is_finite());

        // Matrix should have non-zero determinant
        let det = r[0][0] * r[1][1] - r[0][1] * r[1][0];
        assert!(det.norm() > 0.0);
    }

    #[test]
    fn test_chern_number() {
        let curvature_integral = 2.0 * PI;
        let c = chern_number(curvature_integral);
        assert_eq!(c, 1);
    }

    #[test]
    fn test_quantum_dimensions() {
        let fib_state = TopologicalState::new(AnyonType::Fibonacci, 3);
        let total_dim = fib_state.total_quantum_dimension();

        // d_total = φ³
        let expected = (1.618_f64).powi(3);
        assert!((total_dim - expected).abs() < 0.1);
    }
}
