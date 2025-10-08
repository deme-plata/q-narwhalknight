//! Cosmological inflation dynamics with Starobinsky potential
//! Slow-roll parameters, power spectrum analysis, spectral index

use k_constants::PhysicalConstants;
use ndarray::Array1;
use std::f64::consts::PI;

/// Starobinsky inflation potential: V(φ) = Λ⁴(1 - e^(-√(2/3)φ/M_pl))²
#[derive(Debug, Clone)]
pub struct StarobinskyPotential {
    /// Energy scale Λ (GeV)
    pub lambda: f64,
    /// Planck mass M_pl (kg)
    pub m_planck: f64,
}

impl StarobinskyPotential {
    pub fn new(lambda: f64, constants: &PhysicalConstants) -> Self {
        let m_planck = (constants.hbar * constants.c / constants.g).sqrt();
        Self {
            lambda,
            m_planck,
        }
    }

    /// Calculate potential V(φ)
    pub fn potential(&self, phi: f64) -> f64 {
        let reduced_phi = (2.0_f64 / 3.0).sqrt() * phi / self.m_planck;
        let term = 1.0 - (-reduced_phi).exp();
        self.lambda.powi(4) * term.powi(2)
    }

    /// First derivative dV/dφ
    pub fn derivative(&self, phi: f64) -> f64 {
        let reduced_phi = (2.0_f64 / 3.0).sqrt() * phi / self.m_planck;
        let exp_term = (-reduced_phi).exp();
        let prefactor = 2.0 * self.lambda.powi(4) * (2.0_f64 / 3.0).sqrt() / self.m_planck;
        prefactor * (1.0 - exp_term) * exp_term
    }

    /// Second derivative d²V/dφ²
    pub fn second_derivative(&self, phi: f64) -> f64 {
        let reduced_phi = (2.0_f64 / 3.0).sqrt() * phi / self.m_planck;
        let exp_term = (-reduced_phi).exp();
        let prefactor = 2.0 * self.lambda.powi(4) * (2.0_f64 / 3.0) / self.m_planck.powi(2);
        prefactor * exp_term * (exp_term - 1.0 + reduced_phi)
    }
}

/// Slow-roll parameters
#[derive(Debug, Clone)]
pub struct SlowRollParameters {
    /// First slow-roll parameter ε = (M_pl²/2)(V'/V)²
    pub epsilon: f64,
    /// Second slow-roll parameter η = M_pl²(V''/V)
    pub eta: f64,
}

impl SlowRollParameters {
    /// Calculate slow-roll parameters from potential
    pub fn from_potential(potential: &StarobinskyPotential, phi: f64) -> Self {
        let v = potential.potential(phi);
        let v_prime = potential.derivative(phi);
        let v_double_prime = potential.second_derivative(phi);

        let epsilon = 0.5 * potential.m_planck.powi(2) * (v_prime / v).powi(2);
        let eta = potential.m_planck.powi(2) * v_double_prime / v;

        Self { epsilon, eta }
    }

    /// Check if slow-roll conditions are satisfied
    pub fn is_slow_roll(&self) -> bool {
        self.epsilon < 1.0 && self.eta.abs() < 1.0
    }

    /// Calculate number of e-folds N = ∫dφ/√(2ε)M_pl
    pub fn efolds(&self, phi_start: f64, phi_end: f64, m_planck: f64) -> f64 {
        // Simplified approximation
        (phi_start - phi_end) / ((2.0 * self.epsilon).sqrt() * m_planck)
    }
}

/// Power spectrum for scalar perturbations
#[derive(Debug, Clone)]
pub struct PowerSpectrum {
    /// Amplitude P_R
    pub amplitude: f64,
    /// Spectral index n_s
    pub spectral_index: f64,
    /// Running of spectral index α_s
    pub running: f64,
}

impl PowerSpectrum {
    /// Calculate from slow-roll parameters
    pub fn from_slow_roll(sr: &SlowRollParameters, v: f64, m_planck: f64) -> Self {
        // P_R = V/(24π²ε M_pl⁴)
        let amplitude = v / (24.0 * PI.powi(2) * sr.epsilon * m_planck.powi(4));

        // n_s = 1 - 6ε + 2η
        let spectral_index = 1.0 - 6.0 * sr.epsilon + 2.0 * sr.eta;

        // α_s = -24ε² + 16εη - 2ξ (simplified, ignoring ξ)
        let running = -24.0 * sr.epsilon.powi(2) + 16.0 * sr.epsilon * sr.eta;

        Self {
            amplitude,
            spectral_index,
            running,
        }
    }

    /// Evaluate power spectrum at wavenumber k
    pub fn evaluate(&self, k: f64, k_pivot: f64) -> f64 {
        let ln_k_ratio = (k / k_pivot).ln();
        self.amplitude * (k / k_pivot).powf(self.spectral_index - 1.0)
            * (0.5 * self.running * ln_k_ratio.powi(2)).exp()
    }
}

/// Tensor-to-scalar ratio r = 16ε
pub fn tensor_to_scalar_ratio(epsilon: f64) -> f64 {
    16.0 * epsilon
}

/// Hubble parameter during inflation
pub fn hubble_inflation(v: f64, m_planck: f64) -> f64 {
    (v / (3.0 * m_planck.powi(2))).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starobinsky_potential() {
        let constants = PhysicalConstants::default();
        let potential = StarobinskyPotential::new(1e16, &constants);

        // Use larger field value for meaningful derivative
        let phi = 3.0 * potential.m_planck;
        let v = potential.potential(phi);
        assert!(v > 0.0);

        let v_prime = potential.derivative(phi);
        // Derivative can be very small but should exist
        assert!(v_prime.is_finite());
    }

    #[test]
    fn test_slow_roll_parameters() {
        let constants = PhysicalConstants::default();
        let potential = StarobinskyPotential::new(1e16, &constants);
        let phi = 3.0 * potential.m_planck;

        let sr = SlowRollParameters::from_potential(&potential, phi);
        assert!(sr.epsilon > 0.0);
        assert!(sr.is_slow_roll());
    }

    #[test]
    fn test_power_spectrum() {
        let sr = SlowRollParameters {
            epsilon: 0.01,
            eta: -0.02,
        };
        let v = 1e80; // J
        let m_planck = 2.176e-8; // kg

        let ps = PowerSpectrum::from_slow_roll(&sr, v, m_planck);
        assert!(ps.spectral_index < 1.0); // n_s < 1 for inflation
    }

    #[test]
    fn test_tensor_to_scalar() {
        let epsilon = 0.01;
        let r = tensor_to_scalar_ratio(epsilon);
        assert!((r - 0.16).abs() < 1e-10);
    }
}
