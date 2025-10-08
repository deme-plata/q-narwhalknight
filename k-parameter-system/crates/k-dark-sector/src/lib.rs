//! Dark sector quantum interactions (dark matter and dark energy coupling)
//! K_dark = K_standard × e^(-λ_DM·ρ_DM·t) × (1 + β_DE·Λ·t²)

/// Dark matter properties
#[derive(Debug, Clone)]
pub struct DarkMatter {
    /// Dark matter density (kg/m³)
    pub density: f64,
    /// Dark matter coupling constant λ_DM
    pub lambda_dm: f64,
    /// Dark matter velocity dispersion (m/s)
    pub velocity_dispersion: f64,
}

impl Default for DarkMatter {
    fn default() -> Self {
        Self {
            density: 1.25e-27,      // ~0.3 GeV/cm³
            lambda_dm: 0.1,         // Coupling strength
            velocity_dispersion: 2.2e5, // ~220 km/s
        }
    }
}

/// Dark energy properties
#[derive(Debug, Clone)]
pub struct DarkEnergy {
    /// Cosmological constant Λ (m⁻²)
    pub lambda: f64,
    /// Dark energy coupling β_DE
    pub beta_de: f64,
    /// Equation of state parameter w
    pub w: f64,
}

impl Default for DarkEnergy {
    fn default() -> Self {
        Self {
            lambda: 1.11e-52,   // Cosmological constant
            beta_de: 0.7,       // Coupling strength
            w: -1.0,            // Vacuum energy
        }
    }
}

/// Calculate dark sector K-Parameter
/// K_dark = K_standard × e^(-λ_DM·ρ_DM·t) × (1 + β_DE·Λ·t²)
pub fn calculate_k_dark(
    k_standard: f64,
    dark_matter: &DarkMatter,
    dark_energy: &DarkEnergy,
    time: f64,
) -> f64 {
    let dm_suppression = (-dark_matter.lambda_dm * dark_matter.density * time).exp();
    let de_enhancement = 1.0 + dark_energy.beta_de * dark_energy.lambda * time.powi(2);

    k_standard * dm_suppression * de_enhancement
}

/// Dark matter halo profile (NFW profile)
pub fn nfw_density(radius: f64, scale_radius: f64, characteristic_density: f64) -> f64 {
    let x = radius / scale_radius;
    characteristic_density / (x * (1.0 + x).powi(2))
}

/// Dark energy density evolution with scale factor
pub fn dark_energy_density(scale_factor: f64, rho_lambda: f64, w: f64) -> f64 {
    rho_lambda * scale_factor.powf(-3.0 * (1.0 + w))
}

/// Quintessence potential (dark energy scalar field)
pub fn quintessence_potential(phi: f64, v0: f64, alpha: f64) -> f64 {
    v0 * (-alpha * phi).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_dark_calculation() {
        let k_standard = 1e15;
        let dm = DarkMatter::default();
        let de = DarkEnergy::default();
        let time = 1e12; // ~31,000 years

        let k_dark = calculate_k_dark(k_standard, &dm, &de, time);
        assert!(k_dark > 0.0);
    }

    #[test]
    fn test_nfw_profile() {
        let density = nfw_density(10e3, 20e3, 1e7); // 10 kpc radius, 20 kpc scale
        assert!(density > 0.0);
    }

    #[test]
    fn test_dark_energy_evolution() {
        let rho = dark_energy_density(0.5, 1e-26, -1.0);
        assert!(rho > 0.0);
    }
}
