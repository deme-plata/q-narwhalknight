//! Biological quantum coherence
//! K_bio = K_quantum × Φ_bio × e^(-Γ_dephasing·t) × T_func(T_body)

use k_constants::PhysicalConstants;

/// Biological quantum system parameters
#[derive(Debug, Clone)]
pub struct BiologicalQuantumSystem {
    /// Quantum efficiency factor Φ_bio
    pub phi_bio: f64,
    /// Dephasing rate Γ (s⁻¹)
    pub gamma_dephasing: f64,
    /// Body temperature (K)
    pub temperature: f64,
}

impl Default for BiologicalQuantumSystem {
    fn default() -> Self {
        Self {
            phi_bio: 0.95,            // High quantum efficiency
            gamma_dephasing: 1e12,    // THz dephasing rate
            temperature: 310.0,       // Human body temp ~37°C
        }
    }
}

/// Calculate biological K-Parameter
/// K_bio = K_quantum × Φ_bio × e^(-Γ_dephasing·t) × T_func(T_body)
pub fn calculate_k_bio(
    k_quantum: f64,
    bio_system: &BiologicalQuantumSystem,
    time: f64,
) -> f64 {
    let coherence_decay = (-bio_system.gamma_dephasing * time).exp();
    let temperature_factor = temperature_function(bio_system.temperature);

    k_quantum * bio_system.phi_bio * coherence_decay * temperature_factor
}

/// Temperature dependence function
fn temperature_function(temp: f64) -> f64 {
    // Gaussian centered at optimal biological temperature
    let t_optimal = 310.0; // K
    let sigma: f64 = 20.0; // K
    (-(temp - t_optimal).powi(2) / (2.0 * sigma.powi(2))).exp()
}

/// Photosynthetic quantum coherence (FMO complex)
#[derive(Debug, Clone)]
pub struct PhotosyntheticCoherence {
    /// Number of chromophores
    pub n_chromophores: usize,
    /// Excitation transfer rate (s⁻¹)
    pub transfer_rate: f64,
    /// Quantum yield
    pub quantum_yield: f64,
}

impl Default for PhotosyntheticCoherence {
    fn default() -> Self {
        Self {
            n_chromophores: 7,     // FMO complex
            transfer_rate: 1e12,   // ps timescale
            quantum_yield: 0.95,   // High efficiency
        }
    }
}

impl PhotosyntheticCoherence {
    /// Calculate excitation transfer efficiency with quantum coherence
    pub fn transfer_efficiency(&self, coherence_time: f64) -> f64 {
        let coherent_contribution = (self.transfer_rate * coherence_time).tanh();
        self.quantum_yield * (0.5 + 0.5 * coherent_contribution)
    }

    /// Energy transfer time through chromophore network
    pub fn transfer_time(&self) -> f64 {
        // Simplified: t ∝ N/k_transfer
        self.n_chromophores as f64 / self.transfer_rate
    }
}

/// Avian magnetoreception (radical pair mechanism)
#[derive(Debug, Clone)]
pub struct Magnetoreception {
    /// Magnetic field strength (T)
    pub b_field: f64,
    /// Hyperfine coupling constant (T)
    pub hyperfine_coupling: f64,
    /// Radical pair lifetime (s)
    pub lifetime: f64,
}

impl Default for Magnetoreception {
    fn default() -> Self {
        Self {
            b_field: 5e-5,           // Earth's magnetic field ~50 μT
            hyperfine_coupling: 1e-3, // mT scale
            lifetime: 1e-6,          // μs scale
        }
    }
}

impl Magnetoreception {
    /// Singlet-triplet mixing angle
    pub fn mixing_angle(&self, constants: &PhysicalConstants) -> f64 {
        // θ ∝ (g·μ_B·B) / (hyperfine)
        let g_factor = 2.0; // Electron g-factor
        let mu_b = 9.274e-24; // Bohr magneton (J/T)
        let zeeman = g_factor * mu_b * self.b_field / constants.hbar;
        (zeeman / self.hyperfine_coupling).atan()
    }

    /// Magnetic field sensitivity
    pub fn sensitivity(&self) -> f64 {
        // Sensitivity inversely proportional to hyperfine coupling
        1.0 / self.hyperfine_coupling
    }
}

/// Olfactory quantum tunneling
pub fn olfactory_tunneling_probability(barrier_height: f64, barrier_width: f64, mass: f64, hbar: f64) -> f64 {
    // WKB approximation: T ≈ exp(-2∫√(2m(V-E))dx/ℏ)
    let kappa = (2.0 * mass * barrier_height / hbar.powi(2)).sqrt();
    let tunneling_exponent = -2.0 * kappa * barrier_width;
    tunneling_exponent.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_bio_calculation() {
        let k_quantum = 1e15;
        let bio_system = BiologicalQuantumSystem::default();
        let time = 1e-12; // 1 ps

        let k_bio = calculate_k_bio(k_quantum, &bio_system, time);
        assert!(k_bio > 0.0);
        assert!(k_bio < k_quantum); // Should be suppressed by coherence decay
    }

    #[test]
    fn test_photosynthetic_efficiency() {
        let fmo = PhotosyntheticCoherence::default();
        let efficiency = fmo.transfer_efficiency(1e-12);

        assert!(efficiency > 0.5);
        assert!(efficiency <= 1.0);
    }

    #[test]
    fn test_magnetoreception() {
        let constants = PhysicalConstants::default();
        let mag = Magnetoreception::default();

        let angle = mag.mixing_angle(&constants);
        assert!(angle > 0.0);
        assert!(angle < std::f64::consts::PI / 2.0);

        let sensitivity = mag.sensitivity();
        assert!(sensitivity > 0.0);
    }

    #[test]
    fn test_olfactory_tunneling() {
        let hbar = 1.0545718e-34;
        let mass = 9.109e-31; // Electron mass
        let barrier_height = 1.6e-19; // 1 eV
        let barrier_width = 1e-10; // 1 Angstrom

        let prob = olfactory_tunneling_probability(barrier_height, barrier_width, mass, hbar);
        assert!(prob > 0.0);
        assert!(prob < 1.0);
    }

    #[test]
    fn test_temperature_function() {
        let t_normal = temperature_function(310.0);
        let t_cold = temperature_function(280.0);

        assert!((t_normal - 1.0).abs() < 0.1);
        assert!(t_cold < t_normal);
    }
}
