//! Attosecond laser pulse interaction with Higgs field

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Laser pulse parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LaserPulse {
    /// Central wavelength in nanometers
    pub wavelength_nm: f64,

    /// Pulse duration (FWHM) in attoseconds
    pub duration_as: f64,

    /// Peak intensity in W/cm²
    pub peak_intensity: f64,

    /// Carrier-envelope phase in radians
    pub cep_phase: f64,

    /// Pulse arrival time in attoseconds
    pub arrival_time_as: f64,

    /// Beam waist radius in nanometers (for Gaussian beam)
    pub beam_waist_nm: f64,

    /// Focus position [x, y, z] in nanometers
    pub focus_position: [f64; 3],

    /// Pulse envelope type
    pub envelope: PulseEnvelope,

    /// Polarization
    pub polarization: Polarization,
}

/// Pulse envelope shapes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PulseEnvelope {
    /// Gaussian envelope: exp(-(t/τ)²)
    Gaussian,

    /// Sech² envelope: sech²(t/τ)
    Sech2,

    /// Flat-top with cos² rise/fall
    FlatTop { rise_time_as: f64 },
}

/// Polarization states
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Polarization {
    /// Linear polarization along x-axis
    LinearX,

    /// Linear polarization along y-axis
    LinearY,

    /// Circular polarization (left-handed)
    CircularLeft,

    /// Circular polarization (right-handed)
    CircularRight,

    /// Elliptical polarization
    Elliptical { ellipticity: f64, angle: f64 },
}

impl LaserPulse {
    /// Create a new Gaussian laser pulse
    pub fn gaussian(
        wavelength_nm: f64,
        duration_as: f64,
        peak_intensity: f64,
        beam_waist_nm: f64,
    ) -> Self {
        Self {
            wavelength_nm,
            duration_as,
            peak_intensity,
            cep_phase: 0.0,
            arrival_time_as: 0.0,
            beam_waist_nm,
            focus_position: [0.0, 0.0, 0.0],
            envelope: PulseEnvelope::Gaussian,
            polarization: Polarization::LinearX,
        }
    }

    /// Create an attosecond XUV pulse (from HHG)
    pub fn xuv_attosecond(duration_as: f64, photon_energy_ev: f64, peak_intensity: f64) -> Self {
        // Convert photon energy to wavelength: λ = hc/E
        const HC_EV_NM: f64 = 1239.84; // Planck constant * speed of light in eV·nm
        let wavelength_nm = HC_EV_NM / photon_energy_ev;

        Self {
            wavelength_nm,
            duration_as,
            peak_intensity,
            cep_phase: 0.0,
            arrival_time_as: 0.0,
            beam_waist_nm: 100.0, // Typical XUV focus
            focus_position: [0.0, 0.0, 0.0],
            envelope: PulseEnvelope::Gaussian,
            polarization: Polarization::LinearX,
        }
    }

    /// Calculate temporal envelope at given time
    pub fn temporal_envelope(&self, time_as: f64) -> f64 {
        let t = time_as - self.arrival_time_as;
        let tau = self.duration_as / 2.355; // Convert FWHM to 1/e²

        match self.envelope {
            PulseEnvelope::Gaussian => (-((t / tau).powi(2))).exp(),

            PulseEnvelope::Sech2 => {
                let x = t / tau;
                1.0 / (x.cosh().powi(2))
            }

            PulseEnvelope::FlatTop { rise_time_as } => {
                if t.abs() < self.duration_as / 2.0 - rise_time_as {
                    1.0
                } else if t.abs() < self.duration_as / 2.0 {
                    let phase = PI * (t.abs() - (self.duration_as / 2.0 - rise_time_as))
                        / (2.0 * rise_time_as);
                    phase.cos().powi(2)
                } else {
                    0.0
                }
            }
        }
    }

    /// Calculate spatial intensity profile (Gaussian beam)
    pub fn spatial_intensity(&self, position: &[f64; 3]) -> f64 {
        let dx = position[0] - self.focus_position[0];
        let dy = position[1] - self.focus_position[1];
        let r_sq = dx * dx + dy * dy;

        let w0_sq = self.beam_waist_nm * self.beam_waist_nm;

        (-2.0 * r_sq / w0_sq).exp()
    }

    /// Calculate total intensity at given time
    pub fn intensity_at_time(&self, time_as: f64) -> f64 {
        let envelope = self.temporal_envelope(time_as);
        self.peak_intensity * envelope * envelope
    }

    /// Calculate intensity at given position (time-integrated)
    pub fn intensity_at_position(&self, position: &[f64; 3]) -> f64 {
        self.spatial_intensity(position)
    }

    /// Calculate electric field amplitude at given time and position
    pub fn electric_field(&self, time_as: f64, position: &[f64; 3]) -> [f64; 3] {
        let envelope = self.temporal_envelope(time_as);
        let spatial = self.spatial_intensity(position).sqrt();

        // Angular frequency
        let omega = 2.0 * PI * 299.792458 / self.wavelength_nm; // c = 299.792458 nm/fs
        let time_fs = time_as * 1e-3;

        // Carrier wave with CEP
        let carrier = (omega * time_fs + self.cep_phase).cos();

        // Field amplitude (in arbitrary units, to be calibrated)
        let amplitude = (self.peak_intensity / 1e14).sqrt(); // Normalize by reference intensity

        let field_magnitude = amplitude * envelope * spatial * carrier;

        match &self.polarization {
            Polarization::LinearX => [field_magnitude, 0.0, 0.0],

            Polarization::LinearY => [0.0, field_magnitude, 0.0],

            Polarization::CircularLeft => {
                let ex = field_magnitude * (omega * time_fs + self.cep_phase).cos();
                let ey = field_magnitude * (omega * time_fs + self.cep_phase).sin();
                [ex, ey, 0.0]
            }

            Polarization::CircularRight => {
                let ex = field_magnitude * (omega * time_fs + self.cep_phase).cos();
                let ey = -field_magnitude * (omega * time_fs + self.cep_phase).sin();
                [ex, ey, 0.0]
            }

            Polarization::Elliptical { ellipticity, angle } => {
                let theta = angle;
                let eps = ellipticity;

                let ex = field_magnitude
                    * (theta.cos() * (omega * time_fs + self.cep_phase).cos()
                        - eps * theta.sin() * (omega * time_fs + self.cep_phase).sin());

                let ey = field_magnitude
                    * (theta.sin() * (omega * time_fs + self.cep_phase).cos()
                        + eps * theta.cos() * (omega * time_fs + self.cep_phase).sin());

                [ex, ey, 0.0]
            }
        }
    }

    /// Calculate ponderomotive potential Up = e²E²/(4mω²)
    pub fn ponderomotive_potential(&self, time_as: f64) -> f64 {
        let intensity = self.intensity_at_time(time_as);

        // Up [eV] ≈ 9.33e-14 * I [W/cm²] * λ² [μm²]
        let lambda_um = self.wavelength_nm * 1e-3;
        9.33e-14 * intensity * lambda_um * lambda_um
    }

    /// Check if pulse is present at given time (above 1% of peak)
    pub fn is_active(&self, time_as: f64) -> bool {
        self.temporal_envelope(time_as) > 0.01
    }

    /// Get pulse duration window (start and end times where intensity > 1% of peak)
    pub fn time_window(&self) -> (f64, f64) {
        // For Gaussian: 1% at ~3σ
        let half_width = 3.0 * self.duration_as / 2.355;
        (
            self.arrival_time_as - half_width,
            self.arrival_time_as + half_width,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_pulse() {
        let pulse = LaserPulse::gaussian(800.0, 50.0, 1e14, 100.0);

        // At peak, envelope should be 1.0
        assert!((pulse.temporal_envelope(0.0) - 1.0).abs() < 1e-10);

        // At FWHM, envelope should be ~0.5
        let at_fwhm = pulse.temporal_envelope(pulse.duration_as / 2.0);
        assert!((at_fwhm - 0.5).abs() < 0.05);

        // Intensity scales as envelope squared
        let intensity_peak = pulse.intensity_at_time(0.0);
        assert!((intensity_peak - 1e14).abs() < 1.0);
    }

    #[test]
    fn test_xuv_pulse() {
        let pulse = LaserPulse::xuv_attosecond(100.0, 50.0, 1e12);

        // 50 eV photon → ~24.8 nm wavelength
        assert!((pulse.wavelength_nm - 24.8).abs() < 0.1);

        // Should be active near arrival time
        assert!(pulse.is_active(0.0));
        assert!(!pulse.is_active(500.0));
    }

    #[test]
    fn test_spatial_profile() {
        let pulse = LaserPulse::gaussian(800.0, 50.0, 1e14, 100.0);

        // At focus center, intensity should be peak
        assert!((pulse.spatial_intensity(&[0.0, 0.0, 0.0]) - 1.0).abs() < 1e-10);

        // At beam waist, intensity should be ~1/e²
        let at_waist = pulse.spatial_intensity(&[100.0, 0.0, 0.0]);
        assert!((at_waist - (1.0 / (2.0_f64).exp())).abs() < 0.05);
    }

    #[test]
    fn test_ponderomotive_potential() {
        let pulse = LaserPulse::gaussian(800.0, 50.0, 1e14, 100.0);

        let up = pulse.ponderomotive_potential(0.0);

        // For 1e14 W/cm², 800 nm → Up ~ 6 eV
        assert!(up > 5.0 && up < 7.0);
    }

    #[test]
    fn test_time_window() {
        let pulse = LaserPulse::gaussian(800.0, 100.0, 1e14, 100.0);

        let (start, end) = pulse.time_window();

        // Window should be symmetric around arrival time
        assert!((start + end) / 2.0 < 1.0);

        // Window should be ~6σ wide
        let width = end - start;
        assert!(width > 200.0 && width < 300.0);
    }
}
