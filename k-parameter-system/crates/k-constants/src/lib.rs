//! Physical constants and fundamental parameters
//! Based on Extended K-Parameter Kristensen Framework (OroBit Research Consortium, 2025)

use std::f64::consts::PI;

/// Fundamental physical constants
#[derive(Debug, Clone, Copy)]
pub struct PhysicalConstants {
    /// Gravitational constant (m³/kg·s²)
    pub g: f64,
    /// Speed of light (m/s)
    pub c: f64,
    /// Reduced Planck constant (J·s)
    pub hbar: f64,
    /// Boltzmann constant (J/K)
    pub k_b: f64,
}

impl Default for PhysicalConstants {
    fn default() -> Self {
        Self {
            g: 6.67430e-11,      // CODATA 2018
            c: 2.99792458e8,     // Exact (defined)
            hbar: 1.0545718e-34, // CODATA 2018
            k_b: 1.380649e-23,   // CODATA 2019 (exact)
        }
    }
}

/// Planck scale quantities
#[derive(Debug, Clone, Copy)]
pub struct PlanckScales {
    /// Planck length (m)
    pub length: f64,
    /// Planck time (s)
    pub time: f64,
    /// Planck mass (kg)
    pub mass: f64,
    /// Planck energy (J)
    pub energy: f64,
    /// Planck temperature (K)
    pub temperature: f64,
}

impl PlanckScales {
    /// Calculate Planck scales from fundamental constants
    pub fn from_constants(c: &PhysicalConstants) -> Self {
        let length = (c.hbar * c.g / c.c.powi(3)).sqrt();
        let time = (c.hbar * c.g / c.c.powi(5)).sqrt();
        let mass = (c.hbar * c.c / c.g).sqrt();
        let energy = (c.hbar * c.c.powi(5) / c.g).sqrt();
        let temperature = energy / c.k_b;

        Self {
            length,
            time,
            mass,
            energy,
            temperature,
        }
    }
}

/// K-Parameter framework constants
#[derive(Debug, Clone, Copy)]
pub struct KParameterConstants {
    /// Gravitational enhancement factor α_GN
    pub alpha_gn: f64,
    /// Dark matter coupling λ_DM
    pub lambda_dm: f64,
    /// Dark energy coupling β_DE
    pub beta_de: f64,
    /// Biological quantum enhancement factor Φ_bio
    pub phi_bio: f64,
    /// Golden ratio φ for topological states
    pub golden_ratio: f64,
}

impl Default for KParameterConstants {
    fn default() -> Self {
        Self {
            alpha_gn: 1.0,              // Gravitational enhancement
            lambda_dm: 0.1,             // Dark matter coupling
            beta_de: 0.7,               // Dark energy coupling (w = -0.7)
            phi_bio: 0.95,              // Biological quantum efficiency
            golden_ratio: (1.0 + 5.0_f64.sqrt()) / 2.0, // φ = 1.618...
        }
    }
}

/// Cosmological parameters
#[derive(Debug, Clone, Copy)]
pub struct CosmologicalParameters {
    /// Hubble constant H₀ (km/s/Mpc)
    pub h0: f64,
    /// Dark energy density parameter Ω_Λ
    pub omega_lambda: f64,
    /// Matter density parameter Ω_m
    pub omega_matter: f64,
    /// Baryon density parameter Ω_b
    pub omega_baryon: f64,
    /// Cosmological constant Λ (m⁻²)
    pub lambda: f64,
}

impl Default for CosmologicalParameters {
    fn default() -> Self {
        Self {
            h0: 67.4,           // Planck 2018
            omega_lambda: 0.685, // Dark energy
            omega_matter: 0.315, // Total matter
            omega_baryon: 0.049, // Baryonic matter
            lambda: 1.11e-52,   // Cosmological constant
        }
    }
}

/// Calculate standard K-Parameter
/// K = 2π√(ΔH·ΔS·ℏ/τ)
pub fn calculate_k_standard(delta_h: f64, delta_s: f64, tau: f64, hbar: f64) -> f64 {
    2.0 * PI * (delta_h * delta_s * hbar / tau).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_constants() {
        let c = PhysicalConstants::default();
        assert!((c.c - 2.99792458e8).abs() < 1e-6);
        assert!(c.hbar > 0.0);
        assert!(c.g > 0.0);
    }

    #[test]
    fn test_planck_scales() {
        let c = PhysicalConstants::default();
        let p = PlanckScales::from_constants(&c);

        // Planck length should be ~1.616e-35 m
        assert!((p.length - 1.616e-35).abs() < 1e-37);
        assert!(p.time > 0.0);
        assert!(p.mass > 0.0);
    }

    #[test]
    fn test_k_standard_calculation() {
        let hbar = 1.0545718e-34;
        let k = calculate_k_standard(1e-20, 1e-20, 1e-15, hbar);
        assert!(k > 0.0);
    }

    #[test]
    fn test_golden_ratio() {
        let k_const = KParameterConstants::default();
        assert!((k_const.golden_ratio - 1.618033988749).abs() < 1e-10);
    }
}
