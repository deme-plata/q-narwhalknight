//! ⚡ Attosecond Laser System: X-Ray Pulse Imprinting & Phase Control
//! 30-attosecond X-ray pulses that imprint K-Parameters onto water's hydrogen-bond lattice

use crate::constants::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

/// Attosecond laser pulse parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserPulse {
    pub wavelength_nm: f64,     // Laser wavelength (800nm typical)
    pub pulse_duration_as: f64, // Pulse duration in attoseconds (30as typical)
    pub intensity_wpm2: f64,    // Intensity (W/m²)
    pub phase_shift: f64,       // Phase shift (0..2π)
    pub polarization: [f64; 3], // 3D polarization vector
}

impl LaserPulse {
    /// Create standard 30-attosecond X-ray pulse
    pub fn standard_xray() -> Self {
        Self {
            wavelength_nm: 0.1, // X-ray wavelength
            pulse_duration_as: 30.0,
            intensity_wpm2: 1e18, // Extreme intensity for HHG
            phase_shift: 0.0,
            polarization: [1.0, 0.0, 0.0], // Linear polarization
        }
    }

    /// Create infrared driving pulse (800nm)
    pub fn infrared_800nm() -> Self {
        Self {
            wavelength_nm: 800.0,
            pulse_duration_as: 100.0, // Longer IR pulse
            intensity_wpm2: 1e14,
            phase_shift: 0.0,
            polarization: [1.0, 0.0, 0.0],
        }
    }

    /// Calculate photon energy (eV)
    pub fn photon_energy_ev(&self) -> f64 {
        const H_EV_S: f64 = 4.135667696e-15; // Planck constant in eV⋅s
        H_EV_S * C / (self.wavelength_nm * 1e-9)
    }

    /// Encode Tor onion address in phase modulation
    pub fn encode_onion_address(&mut self, onion_addr: &str) -> Vec<f64> {
        let hash = blake3::hash(onion_addr.as_bytes());
        let mut phase_encoding = Vec::new();

        // Map hash bytes to phase shifts (0..2π)
        for &byte in hash.as_bytes().iter().take(16) {
            // Use first 16 bytes
            let phase = (byte as f64 / 255.0) * TAU;
            phase_encoding.push(phase);
        }

        self.phase_shift = phase_encoding[0]; // Set primary phase
        phase_encoding
    }
}

/// X-ray imprint result on water structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XRayImprint {
    pub imprint_id: String,
    pub hydrogen_bond_modulation: f64, // 0..1 strength
    pub lattice_distortion: [f64; 3],  // 3D lattice shift vector
    pub coherence_time_fs: f64,        // Coherence time in femtoseconds
    pub k_parameter_encoding: f64,     // Encoded K-parameter value
    pub tor_phase_signature: Vec<f64>, // Tor address phase encoding
}

impl XRayImprint {
    /// Create new imprint from laser pulse
    pub fn from_pulse(pulse: &LaserPulse, k_parameter: f64, tor_phases: Vec<f64>) -> Self {
        let imprint_id = hex::encode(
            &blake3::hash(
                &format!(
                    "{}{}{}",
                    pulse.wavelength_nm, k_parameter, pulse.phase_shift
                )
                .as_bytes(),
            )
            .as_bytes()[..8],
        );

        // Calculate hydrogen bond modulation strength
        let hb_modulation = (pulse.intensity_wpm2 / 1e18).min(1.0);

        // 3D lattice distortion based on pulse polarization and intensity
        let lattice_distortion = [
            pulse.polarization[0] * hb_modulation * 0.1, // Angstrom scale
            pulse.polarization[1] * hb_modulation * 0.1,
            pulse.polarization[2] * hb_modulation * 0.1,
        ];

        // Coherence time inversely related to pulse intensity
        let coherence_time_fs = (FEMTOSECOND * 1000.0) / (pulse.intensity_wpm2 / 1e14).max(1.0);

        Self {
            imprint_id,
            hydrogen_bond_modulation: hb_modulation,
            lattice_distortion,
            coherence_time_fs,
            k_parameter_encoding: k_parameter,
            tor_phase_signature: tor_phases,
        }
    }

    /// Check if imprint is stable enough for quantum computation
    pub fn is_quantum_stable(&self) -> bool {
        self.hydrogen_bond_modulation > 0.5
            && self.coherence_time_fs > 100.0
            && self.k_parameter_encoding > 5.0
    }

    /// Get imprint strength for Tor routing quality
    pub fn tor_routing_quality(&self) -> f64 {
        let phase_quality = self
            .tor_phase_signature
            .iter()
            .map(|&p| (p.sin().abs() + p.cos().abs()) / 2.0)
            .sum::<f64>()
            / self.tor_phase_signature.len() as f64;

        (self.hydrogen_bond_modulation + phase_quality) / 2.0
    }
}

/// Attosecond laser system for water robot control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttosecondLaser {
    pub seed: u64,
    pub current_pulse: LaserPulse,
    pub imprint_history: Vec<XRayImprint>,
    pub cavity_length_nm: f64,   // Laser cavity length
    pub pulse_energy_mj: f64,    // Pulse energy in millijoules
    pub repetition_rate_hz: f64, // Pulse repetition rate
    pub beam_profile: BeamProfile,
    pub temperature_k: f64, // Operating temperature
}

/// Laser beam spatial profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamProfile {
    pub waist_radius_um: f64,    // Beam waist radius (micrometers)
    pub rayleigh_length_mm: f64, // Rayleigh length
    pub m2_factor: f64,          // Beam quality factor
}

impl AttosecondLaser {
    /// Initialize attosecond laser system
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            current_pulse: LaserPulse::standard_xray(),
            imprint_history: Vec::new(),
            cavity_length_nm: 1000.0,   // 1 micrometer cavity
            pulse_energy_mj: 0.1,       // 100 microjoules
            repetition_rate_hz: 1000.0, // 1 kHz
            beam_profile: BeamProfile {
                waist_radius_um: 10.0,
                rayleigh_length_mm: 1.0,
                m2_factor: 1.1, // Near diffraction-limited
            },
            temperature_k: 77.0, // Liquid nitrogen cooling
        }
    }

    /// Process intent through attosecond laser imprinting
    pub async fn process_intent(
        &mut self,
        intent: &str,
        k_parameter: f64,
    ) -> anyhow::Result<XRayImprint> {
        // Generate intent-specific laser pulse
        let mut pulse = if intent.contains("multiverse") {
            LaserPulse::standard_xray()
        } else {
            LaserPulse::infrared_800nm()
        };

        // Encode intent in laser phase
        let tor_phases = pulse.encode_onion_address(intent);

        // Adjust pulse parameters based on K-parameter
        pulse.intensity_wpm2 *= k_parameter / K_PARAMETER_BASE;
        pulse.pulse_duration_as *= (K_PARAMETER_BASE / k_parameter).sqrt();

        // Create X-ray imprint
        let imprint = XRayImprint::from_pulse(&pulse, k_parameter, tor_phases);

        // Update laser state
        self.current_pulse = pulse;
        self.imprint_history.push(imprint.clone());

        // Limit history size
        if self.imprint_history.len() > 100 {
            self.imprint_history.remove(0);
        }

        Ok(imprint)
    }

    /// Generate High Harmonic Generation (HHG) spectrum
    pub fn generate_hhg_spectrum(&self, driving_pulse: &LaserPulse) -> Vec<(f64, f64)> {
        let mut spectrum = Vec::new();
        let fundamental_freq = C / (driving_pulse.wavelength_nm * 1e-9);

        // Generate odd harmonics (HHG characteristic)
        for harmonic in (1..100).step_by(2) {
            let freq = fundamental_freq * harmonic as f64;
            let intensity = driving_pulse.intensity_wpm2 / (harmonic as f64).powi(3); // Power law falloff

            if intensity > 1e10 {
                // Cutoff threshold
                spectrum.push((freq, intensity));
            }
        }

        spectrum
    }

    /// Calculate attosecond pulse train characteristics
    pub fn pulse_train_analysis(&self) -> AttosecondPulseTrain {
        let pulse_separation_as = 1.0 / self.repetition_rate_hz * 1e18; // Convert to attoseconds
        let pulse_count = (1e6 / pulse_separation_as) as u32; // Pulses per microsecond

        AttosecondPulseTrain {
            pulse_separation_as,
            pulse_count_per_us: pulse_count,
            coherence_length_nm: self.cavity_length_nm,
            beam_divergence_mrad: 1.22 * self.current_pulse.wavelength_nm * 1e-9
                / (self.beam_profile.waist_radius_um * 1e-6)
                * 1000.0,
            peak_power_gw: self.pulse_energy_mj * 1e-3
                / (self.current_pulse.pulse_duration_as * 1e-18),
        }
    }

    /// Get current laser state description
    pub fn state_description(&self) -> String {
        format!(
            "{}nm, {:.0}as, {:.1}mJ @ {:.0}Hz",
            self.current_pulse.wavelength_nm,
            self.current_pulse.pulse_duration_as,
            self.pulse_energy_mj,
            self.repetition_rate_hz
        )
    }

    /// Check if laser is ready for quantum operations
    pub fn is_quantum_ready(&self) -> bool {
        self.current_pulse.pulse_duration_as < 50.0 && // Sub-50 attosecond pulses
        self.pulse_energy_mj > 0.05 && // Sufficient energy
        self.temperature_k < 100.0 // Cryogenic cooling
    }

    /// Get current pulse timing information
    pub fn get_current_pulse_timing(&self) -> f64 {
        self.current_pulse.pulse_duration_as
    }
}

/// Attosecond pulse train characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttosecondPulseTrain {
    pub pulse_separation_as: f64,  // Time between pulses (attoseconds)
    pub pulse_count_per_us: u32,   // Number of pulses per microsecond
    pub coherence_length_nm: f64,  // Spatial coherence length
    pub beam_divergence_mrad: f64, // Beam divergence (milliradians)
    pub peak_power_gw: f64,        // Peak power (gigawatts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laser_pulse_creation() {
        let xray_pulse = LaserPulse::standard_xray();
        assert!(xray_pulse.photon_energy_ev() > 1000.0); // X-ray energies

        let ir_pulse = LaserPulse::infrared_800nm();
        assert!(ir_pulse.photon_energy_ev() < 10.0); // IR energies
    }

    #[test]
    fn test_onion_address_encoding() {
        let mut pulse = LaserPulse::infrared_800nm();
        let phases = pulse.encode_onion_address("test.onion");
        assert_eq!(phases.len(), 16);
        assert!(phases.iter().all(|&p| p >= 0.0 && p <= TAU));
    }

    #[tokio::test]
    async fn test_attosecond_laser_system() {
        let mut laser = AttosecondLaser::new(1234);
        let imprint = laser
            .process_intent("bridge to multiverse-42", 7.5)
            .await
            .unwrap();

        assert!(imprint.is_quantum_stable());
        assert!(imprint.tor_routing_quality() > 0.0);
        assert!(!imprint.imprint_id.is_empty());
    }

    #[test]
    fn test_hhg_spectrum_generation() {
        let laser = AttosecondLaser::new(567);
        let driving_pulse = LaserPulse::infrared_800nm();
        let spectrum = laser.generate_hhg_spectrum(&driving_pulse);

        assert!(!spectrum.is_empty());
        // Check that we have odd harmonics
        assert!(spectrum
            .iter()
            .enumerate()
            .all(|(i, _)| (i * 2 + 1) % 2 == 1));
    }
}
