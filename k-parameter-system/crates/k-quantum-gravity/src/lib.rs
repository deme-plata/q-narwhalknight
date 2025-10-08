//! Quantum gravity corrections and gravitational K-Parameter enhancements
//! Implements Schwarzschild metric, string theory corrections, loop quantum gravity

use k_constants::{PhysicalConstants, PlanckScales};
use nalgebra::{Matrix4, Vector4};
use std::f64::consts::PI;

/// Schwarzschild metric components
#[derive(Debug, Clone)]
pub struct SchwarzschildMetric {
    /// Mass of the gravitating body (kg)
    pub mass: f64,
    /// Radial coordinate (m)
    pub radius: f64,
    /// Metric tensor g_μν
    pub metric_tensor: Matrix4<f64>,
}

impl SchwarzschildMetric {
    /// Construct Schwarzschild metric at given radius
    pub fn new(mass: f64, radius: f64, constants: &PhysicalConstants) -> Self {
        let rs = 2.0 * constants.g * mass / constants.c.powi(2); // Schwarzschild radius
        let f = 1.0 - rs / radius;

        let mut metric = Matrix4::zeros();
        metric[(0, 0)] = -f; // g_tt
        metric[(1, 1)] = 1.0 / f; // g_rr
        metric[(2, 2)] = radius.powi(2); // g_θθ
        metric[(3, 3)] = radius.powi(2); // g_φφ (simplified, should multiply by sin²θ)

        Self {
            mass,
            radius,
            metric_tensor: metric,
        }
    }

    /// Calculate Schwarzschild radius (event horizon)
    pub fn schwarzschild_radius(&self, constants: &PhysicalConstants) -> f64 {
        2.0 * constants.g * self.mass / constants.c.powi(2)
    }

    /// Check if radius is within event horizon
    pub fn is_within_horizon(&self, constants: &PhysicalConstants) -> bool {
        self.radius < self.schwarzschild_radius(constants)
    }
}

/// Quantum gravity corrections (string theory and loop quantum gravity)
#[derive(Debug, Clone)]
pub struct QuantumGravityCorrections {
    /// String length parameter l_s (m)
    pub string_length: f64,
    /// String coupling constant g_s
    pub string_coupling: f64,
    /// Loop quantum gravity immirzi parameter γ
    pub immirzi_parameter: f64,
}

impl Default for QuantumGravityCorrections {
    fn default() -> Self {
        Self {
            string_length: 1e-35,  // Near Planck scale
            string_coupling: 0.1,   // Weak coupling regime
            immirzi_parameter: 0.2375, // Standard LQG value
        }
    }
}

impl QuantumGravityCorrections {
    /// String theory α' corrections to spacetime
    pub fn string_alpha_prime_correction(&self, radius: f64) -> f64 {
        let alpha_prime = self.string_length.powi(2);
        1.0 + alpha_prime / radius.powi(2)
    }

    /// Loop quantum gravity area gap correction
    pub fn lqg_area_gap_correction(&self, area: f64, planck: &PlanckScales) -> f64 {
        let a_min = 4.0 * PI * self.immirzi_parameter * planck.length.powi(2);
        (1.0 - (-area / a_min).exp()).sqrt()
    }

    /// Quantum foam fluctuation amplitude
    pub fn quantum_foam_amplitude(&self, scale: f64, planck: &PlanckScales) -> f64 {
        (planck.length / scale).powi(3)
    }
}

/// Gravitational K-Parameter enhancement
/// K_gravity = K_standard × √(1-2GM/rc²) × (1 + α_GN·M/(r·c²))
pub fn calculate_k_gravity(
    k_standard: f64,
    mass: f64,
    radius: f64,
    alpha_gn: f64,
    constants: &PhysicalConstants,
) -> f64 {
    let rs = 2.0 * constants.g * mass / constants.c.powi(2);
    let redshift_factor = (1.0 - rs / radius).sqrt();
    let enhancement = 1.0 + alpha_gn * constants.g * mass / (radius * constants.c.powi(2));

    k_standard * redshift_factor * enhancement
}

/// Black hole evaporation (Hawking radiation)
#[derive(Debug, Clone)]
pub struct BlackHoleEvaporation {
    /// Initial black hole mass (kg)
    pub initial_mass: f64,
    /// Current black hole mass (kg)
    pub current_mass: f64,
    /// Evaporation time elapsed (s)
    pub time_elapsed: f64,
}

impl BlackHoleEvaporation {
    pub fn new(initial_mass: f64) -> Self {
        Self {
            initial_mass,
            current_mass: initial_mass,
            time_elapsed: 0.0,
        }
    }

    /// Calculate Hawking temperature
    pub fn hawking_temperature(&self, constants: &PhysicalConstants) -> f64 {
        constants.hbar * constants.c.powi(3) / (8.0 * PI * constants.g * constants.k_b * self.current_mass)
    }

    /// Calculate evaporation rate dM/dt
    pub fn evaporation_rate(&self, constants: &PhysicalConstants) -> f64 {
        // dM/dt = -ħc⁴/(15360πG²M²)
        -constants.hbar * constants.c.powi(4) / (15360.0 * PI * constants.g.powi(2) * self.current_mass.powi(2))
    }

    /// Evolve black hole mass over time step dt
    pub fn evolve(&mut self, dt: f64, constants: &PhysicalConstants) {
        let dm = self.evaporation_rate(constants) * dt;
        self.current_mass += dm;
        self.time_elapsed += dt;

        // Prevent negative mass
        if self.current_mass < 0.0 {
            self.current_mass = 0.0;
        }
    }

    /// Calculate total evaporation time
    pub fn total_evaporation_time(&self, constants: &PhysicalConstants) -> f64 {
        5120.0 * PI * constants.g.powi(2) * self.initial_mass.powi(3) / (constants.hbar * constants.c.powi(4))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schwarzschild_metric() {
        let constants = PhysicalConstants::default();
        let mass = 1.989e30; // Solar mass
        let radius = 1e9; // 1000 km

        let metric = SchwarzschildMetric::new(mass, radius, &constants);
        let rs = metric.schwarzschild_radius(&constants);

        // Solar Schwarzschild radius ~3 km
        assert!((rs - 2953.0).abs() < 100.0);
        assert!(!metric.is_within_horizon(&constants));
    }

    #[test]
    fn test_k_gravity_enhancement() {
        let constants = PhysicalConstants::default();
        let k_standard = 1e15;
        let mass = 1.989e30; // Solar mass
        let radius = 1e10; // Far from horizon

        let k_grav = calculate_k_gravity(k_standard, mass, radius, 1.0, &constants);
        assert!(k_grav > 0.0);
    }

    #[test]
    fn test_hawking_evaporation() {
        let constants = PhysicalConstants::default();
        let mut bh = BlackHoleEvaporation::new(1e15); // ~Moon mass

        let temp = bh.hawking_temperature(&constants);
        assert!(temp > 0.0);

        let t_evap = bh.total_evaporation_time(&constants);
        assert!(t_evap > 0.0);
    }

    #[test]
    fn test_quantum_corrections() {
        let qgc = QuantumGravityCorrections::default();
        let planck = PlanckScales::from_constants(&PhysicalConstants::default());

        let correction = qgc.string_alpha_prime_correction(1e-34);
        assert!(correction > 1.0);

        let foam_amp = qgc.quantum_foam_amplitude(1e-34, &planck);
        assert!(foam_amp > 0.0);
    }
}
