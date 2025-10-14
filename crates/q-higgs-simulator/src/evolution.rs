//! Field evolution physics using Klein-Gordon equation with Mexican hat potential

use crate::field::ScalarField3D;
use crate::laser_interaction::LaserPulse;
use crate::constants::PhysicalConstants;
use anyhow::Result;
use ndarray::Array3;
use rayon::prelude::*;

/// Field evolution engine using finite-difference time-domain (FDTD) method
pub struct FieldEvolution {
    /// Physical constants
    pub constants: PhysicalConstants,

    /// Previous field values for second-order time stepping
    prev_field: Option<Array3<f64>>,

    /// Time derivative of field (for velocity Verlet integration)
    field_velocity: Option<Array3<f64>>,
}

impl FieldEvolution {
    /// Create new field evolution engine
    pub fn new(constants: PhysicalConstants) -> Self {
        Self {
            constants,
            prev_field: None,
            field_velocity: None,
        }
    }

    /// Evolve field one time step using Klein-Gordon equation
    ///
    /// Klein-Gordon equation: ∂²φ/∂t² = ∇²φ - dV/dφ
    /// where V(φ) = λ/4 * (φ² - v²)² is the Mexican hat potential
    pub fn evolve_step(&mut self, field: &mut ScalarField3D, dt_as: f64) -> Result<()> {
        let dt_sec = dt_as * 1e-18; // Convert attoseconds to seconds

        // Initialize velocity field if first step
        if self.field_velocity.is_none() {
            self.field_velocity = Some(Array3::zeros(field.data.dim()));
        }

        // Calculate spatial Laplacian ∇²φ
        let laplacian = field.laplacian();

        // Calculate potential derivative dV/dφ for each point
        let potential_derivative = self.calculate_potential_derivative(&field.data);

        // Update field using velocity Verlet algorithm
        // This is more stable than simple Euler or leapfrog for this problem

        // Update positions using indexed parallel iteration
        let dt_sec_copy = dt_sec;
        {
            let velocity = self.field_velocity.as_ref().unwrap();
            ndarray::Zip::from(field.data.view_mut())
                .and(velocity.view())
                .and(laplacian.view())
                .and(potential_derivative.view())
                .par_for_each(|phi, &vel, &lap, &dpot| {
                    let acceleration = lap - dpot;
                    *phi += vel * dt_sec_copy + 0.5 * acceleration * dt_sec_copy * dt_sec_copy;
                });
        }

        // Recalculate forces at new position
        let laplacian_new = field.laplacian();
        let potential_derivative_new = self.calculate_potential_derivative(&field.data);

        // Update velocities
        let velocity = self.field_velocity.as_mut().unwrap();
        ndarray::Zip::from(velocity.view_mut())
            .and(laplacian.view())
            .and(potential_derivative.view())
            .and(laplacian_new.view())
            .and(potential_derivative_new.view())
            .par_for_each(|vel, &lap_old, &dpot_old, &lap_new, &dpot_new| {
                let accel_old = lap_old - dpot_old;
                let accel_new = lap_new - dpot_new;
                *vel += 0.5 * (accel_old + accel_new) * dt_sec_copy;
            });

        Ok(())
    }

    /// Calculate dV/dφ for Mexican hat potential
    ///
    /// V(φ) = λ/4 * (φ² - v²)²
    /// dV/dφ = λ * φ * (φ² - v²)
    fn calculate_potential_derivative(&self, field_data: &Array3<f64>) -> Array3<f64> {
        let lambda = self.constants.higgs_self_coupling;
        let v_sq = self.constants.vacuum_expectation_value_sq;

        field_data.mapv(|phi| lambda * phi * (phi * phi - v_sq))
    }

    /// Apply laser-field interaction
    ///
    /// Laser modifies the effective mass: m²_eff = m² + α|E|²
    /// This creates local potential wells that can trap field excitations
    pub fn apply_laser_interaction(
        &mut self,
        field: &mut ScalarField3D,
        laser: &LaserPulse,
        time_as: f64,
    ) -> Result<()> {
        let time_sec = time_as * 1e-18;

        // Calculate laser intensity at this time
        let intensity = laser.intensity_at_time(time_sec);

        // Coupling strength (to be calibrated experimentally)
        let alpha = 1e-6; // Field-laser coupling (placeholder)

        // Modify field according to laser-induced potential shift
        let dx = field.dx_nm;

        ndarray::Zip::indexed(field.data.view_mut()).par_for_each(|(i, j, k), phi| {
            // Get laser intensity at this spatial position
            let position = [
                i as f64 * dx,
                j as f64 * dx,
                k as f64 * dx,
            ];

            let local_intensity = laser.intensity_at_position(&position);

            // Laser creates an effective potential shift: ΔV = α|E|²φ²/2
            let potential_shift = alpha * local_intensity * intensity;

            // This acts as a restoring force: F = -dV/dφ = -α|E|²φ
            let force = -potential_shift * (*phi);

            // Apply small perturbation (time step handled in evolve_step)
            *phi += force * 1e-20; // Small coupling for stability
        });

        Ok(())
    }

    /// Reset evolution state (clears history)
    pub fn reset(&mut self) {
        self.prev_field = None;
        self.field_velocity = None;
    }

    /// Get current field velocity (for diagnostics)
    pub fn get_velocity(&self) -> Option<&Array3<f64>> {
        self.field_velocity.as_ref()
    }
}

/// Helper trait to convert Vec to Array3
trait IntoArray3<T> {
    fn into_shape(self, dim: (usize, usize, usize)) -> Result<Array3<T>, ()>;
}

impl IntoArray3<f64> for Vec<f64> {
    fn into_shape(self, dim: (usize, usize, usize)) -> Result<Array3<f64>, ()> {
        Array3::from_shape_vec(dim, self).map_err(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::PhysicalConstants;

    #[test]
    fn test_evolution_creation() {
        let constants = PhysicalConstants::standard_model();
        let evolution = FieldEvolution::new(constants);
        assert!(evolution.field_velocity.is_none());
    }

    #[test]
    fn test_potential_derivative() {
        let constants = PhysicalConstants::standard_model();
        let evolution = FieldEvolution::new(constants.clone());

        // At vacuum expectation value, dV/dφ should be zero
        let field_data = Array3::from_elem((8, 8, 8), constants.vacuum_expectation_value());
        let derivative = evolution.calculate_potential_derivative(&field_data);

        // Should be close to zero (numerical precision)
        assert!(derivative.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn test_vacuum_stability() {
        let constants = PhysicalConstants::standard_model();
        let mut evolution = FieldEvolution::new(constants.clone());
        let mut field = ScalarField3D::new(32, 100.0);

        // Initialize at vacuum expectation value
        field.fill_uniform(constants.vacuum_expectation_value());

        // Evolve for 1000 attoseconds
        for _ in 0..100 {
            evolution.evolve_step(&mut field, 10.0).unwrap();
        }

        // Field should remain close to VEV (stable vacuum)
        let mean = field.mean_value();
        assert!((mean - constants.vacuum_expectation_value()).abs() < 1.0);
    }

    #[test]
    fn test_small_perturbation_evolution() {
        let constants = PhysicalConstants::standard_model();
        let mut evolution = FieldEvolution::new(constants.clone());
        let mut field = ScalarField3D::new(32, 100.0);

        // Initialize with small perturbation around VEV
        field.fill_uniform(constants.vacuum_expectation_value());
        field.add_wave_perturbation(1.0, 50.0);

        let initial_energy = field.total_energy();

        // Evolve for 1000 attoseconds
        for _ in 0..100 {
            evolution.evolve_step(&mut field, 10.0).unwrap();
        }

        let final_energy = field.total_energy();

        // Energy should be approximately conserved (within 10% due to numerical error)
        let energy_change = (final_energy - initial_energy).abs() / initial_energy;
        assert!(energy_change < 0.1, "Energy change: {:.2}%", energy_change * 100.0);
    }
}
