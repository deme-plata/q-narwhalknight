//! Physical constants for Higgs field simulation

use serde::{Deserialize, Serialize};

/// Physical constants relevant to Higgs field simulation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhysicalConstants {
    /// Vacuum expectation value (VEV) in GeV
    pub vev_gev: f64,

    /// Vacuum expectation value squared
    pub vacuum_expectation_value_sq: f64,

    /// Higgs self-coupling λ
    pub higgs_self_coupling: f64,

    /// Higgs mass in GeV
    pub higgs_mass_gev: f64,

    /// Reduced Planck constant ℏ in GeV·s
    pub hbar_gev_s: f64,

    /// Speed of light in nm/fs
    pub speed_of_light_nm_per_fs: f64,

    /// Fine structure constant α
    pub fine_structure: f64,

    /// Conversion factor: GeV to eV
    pub gev_to_ev: f64,

    /// Conversion factor: femtosecond to attosecond
    pub fs_to_as: f64,
}

impl PhysicalConstants {
    /// Create constants for Standard Model Higgs field
    pub fn standard_model() -> Self {
        let vev_gev: f64 = 246.22; // GeV
        let higgs_mass_gev: f64 = 125.1; // GeV (measured at LHC)

        // Calculate self-coupling from mass and VEV: λ = 2m²/v²
        let higgs_self_coupling = 2.0 * higgs_mass_gev.powi(2) / vev_gev.powi(2);

        Self {
            vev_gev,
            vacuum_expectation_value_sq: vev_gev * vev_gev,
            higgs_self_coupling,
            higgs_mass_gev,
            hbar_gev_s: 6.582119569e-25,       // GeV·s
            speed_of_light_nm_per_fs: 299.792458, // nm/fs
            fine_structure: 1.0 / 137.036,
            gev_to_ev: 1e9,
            fs_to_as: 1000.0,
        }
    }

    /// Create constants for testing with reduced energy scales
    pub fn test_scale(scale_factor: f64) -> Self {
        let mut constants = Self::standard_model();
        constants.vev_gev *= scale_factor;
        constants.vacuum_expectation_value_sq *= scale_factor * scale_factor;
        constants.higgs_mass_gev *= scale_factor;
        constants
    }

    /// Get vacuum expectation value (convenience method)
    pub fn vacuum_expectation_value(&self) -> f64 {
        self.vev_gev
    }

    /// Calculate Compton wavelength of Higgs boson in femtometers
    pub fn higgs_compton_wavelength_fm(&self) -> f64 {
        // λ_C = ℏ/(mc) in natural units
        let hbar_gev_fm = 0.1973; // ℏc in GeV·fm
        hbar_gev_fm / self.higgs_mass_gev
    }

    /// Calculate oscillation frequency of Higgs field around VEV in Hz
    pub fn higgs_oscillation_frequency_hz(&self) -> f64 {
        // ω = m/ℏ
        self.higgs_mass_gev * self.gev_to_ev * 1.602e-19 / (1.055e-34)
    }

    /// Calculate critical laser intensity for nonlinear effects (W/cm²)
    pub fn critical_intensity_w_per_cm2(&self) -> f64 {
        // I_crit ~ (m²c³)/(e²ℏ)
        // Rough estimate for when laser-matter coupling becomes strong
        1e16 // Typical atomic unit of intensity
    }

    /// Convert energy from GeV to eV
    pub fn gev_to_electron_volts(&self, energy_gev: f64) -> f64 {
        energy_gev * self.gev_to_ev
    }

    /// Convert time from attoseconds to seconds
    pub fn as_to_seconds(&self, time_as: f64) -> f64 {
        time_as * 1e-18
    }

    /// Convert time from femtoseconds to attoseconds
    pub fn fs_to_attoseconds(&self, time_fs: f64) -> f64 {
        time_fs * self.fs_to_as
    }

    /// Calculate electric field strength in V/m from intensity in W/cm²
    pub fn intensity_to_field(&self, intensity_w_per_cm2: f64) -> f64 {
        // I = (1/2) * ε₀ * c * E²
        // E = sqrt(2I/(ε₀c))
        const EPSILON_0: f64 = 8.854e-12; // F/m
        const C: f64 = 2.998e8; // m/s

        let intensity_si = intensity_w_per_cm2 * 1e4; // Convert W/cm² to W/m²
        (2.0 * intensity_si / (EPSILON_0 * C)).sqrt()
    }

    /// Calculate ponderomotive energy Up in eV from laser parameters
    pub fn ponderomotive_energy_ev(&self, intensity_w_per_cm2: f64, wavelength_nm: f64) -> f64 {
        // Up [eV] = 9.33e-14 * I [W/cm²] * λ² [μm²]
        let lambda_um = wavelength_nm * 1e-3;
        9.33e-14 * intensity_w_per_cm2 * lambda_um * lambda_um
    }
}

impl Default for PhysicalConstants {
    fn default() -> Self {
        Self::standard_model()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_model_constants() {
        let constants = PhysicalConstants::standard_model();

        // Check VEV
        assert!((constants.vev_gev - 246.22).abs() < 1e-6);

        // Check Higgs mass
        assert!((constants.higgs_mass_gev - 125.1).abs() < 1e-6);

        // Self-coupling should be ~0.13
        assert!(constants.higgs_self_coupling > 0.1 && constants.higgs_self_coupling < 0.2);
    }

    #[test]
    fn test_compton_wavelength() {
        let constants = PhysicalConstants::standard_model();
        let lambda_c = constants.higgs_compton_wavelength_fm();

        // Should be ~1.6 fm for 125 GeV Higgs
        assert!(lambda_c > 1.0 && lambda_c < 2.0);
    }

    #[test]
    fn test_oscillation_frequency() {
        let constants = PhysicalConstants::standard_model();
        let freq = constants.higgs_oscillation_frequency_hz();

        // Should be ~3e25 Hz
        assert!(freq > 1e25 && freq < 1e26);
    }

    #[test]
    fn test_intensity_conversions() {
        let constants = PhysicalConstants::standard_model();

        let intensity = 1e14; // W/cm²
        let field = constants.intensity_to_field(intensity);

        // Should be ~2.7e11 V/m
        assert!(field > 2e11 && field < 3e11);
    }

    #[test]
    fn test_ponderomotive_energy() {
        let constants = PhysicalConstants::standard_model();

        let up = constants.ponderomotive_energy_ev(1e14, 800.0);

        // Should be ~6 eV for 800 nm, 1e14 W/cm²
        assert!(up > 5.0 && up < 7.0);
    }

    #[test]
    fn test_scale_factor() {
        let constants = PhysicalConstants::test_scale(0.1);

        assert!((constants.vev_gev - 24.622).abs() < 1e-3);
        assert!((constants.higgs_mass_gev - 12.51).abs() < 1e-3);
    }
}
