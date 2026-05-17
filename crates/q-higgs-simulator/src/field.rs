//! 3D scalar field data structures and operations

use ndarray::{Array3, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// 3D scalar field with uniform grid spacing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScalarField3D {
    /// Field values at grid points
    pub data: Array3<f64>,

    /// Grid resolution (number of points per dimension)
    pub resolution: usize,

    /// Physical size of simulation box in nanometers
    pub size_nm: f64,

    /// Grid spacing in nanometers
    pub dx_nm: f64,
}

impl ScalarField3D {
    /// Create a new scalar field initialized to zero
    pub fn new(resolution: usize, size_nm: f64) -> Self {
        let data = Array3::zeros((resolution, resolution, resolution));
        let dx_nm = size_nm / (resolution as f64);

        Self {
            data,
            resolution,
            size_nm,
            dx_nm,
        }
    }

    /// Create field from existing data array
    pub fn from_array(data: Array3<f64>, size_nm: f64) -> Self {
        let resolution = data.shape()[0];
        let dx_nm = size_nm / (resolution as f64);

        Self {
            data,
            resolution,
            size_nm,
            dx_nm,
        }
    }

    /// Fill entire field with uniform value
    pub fn fill_uniform(&mut self, value: f64) {
        self.data.fill(value);
    }

    /// Add a sine wave perturbation
    pub fn add_wave_perturbation(&mut self, amplitude: f64, wavelength_nm: f64) {
        let k = 2.0 * PI / wavelength_nm;
        let dx = self.dx_nm;

        ndarray::Zip::indexed(self.data.view_mut()).par_for_each(|(i, j, k_idx), value| {
            let x = i as f64 * dx;
            let y = j as f64 * dx;
            let z = k_idx as f64 * dx;

            *value += amplitude * (k * x).sin() * (k * y).sin() * (k * z).sin();
        });
    }

    /// Add a Gaussian perturbation centered in the box
    pub fn add_gaussian_perturbation(&mut self, amplitude: f64, width_nm: f64) {
        let center = self.size_nm / 2.0;
        let dx = self.dx_nm;

        ndarray::Zip::indexed(self.data.view_mut()).par_for_each(|(i, j, k), value| {
            let x = i as f64 * dx;
            let y = j as f64 * dx;
            let z = k as f64 * dx;

            let r_sq = (x - center).powi(2) + (y - center).powi(2) + (z - center).powi(2);
            *value += amplitude * (-r_sq / (2.0 * width_nm * width_nm)).exp();
        });
    }

    /// Calculate Laplacian using finite differences (parallel)
    pub fn laplacian(&self) -> Array3<f64> {
        let mut result = Array3::zeros(self.data.dim());
        let dx_sq = self.dx_nm * self.dx_nm;
        let res = self.resolution;
        let data = &self.data;

        ndarray::Zip::indexed(result.view_mut()).par_for_each(|(i, j, k), lap_value| {
            if i > 0 && i < res - 1 && j > 0 && j < res - 1 && k > 0 && k < res - 1 {
                let center = data[[i, j, k]];

                // Second derivatives in each direction
                let d2_dx2 = (data[[i + 1, j, k]] - 2.0 * center + data[[i - 1, j, k]]) / dx_sq;
                let d2_dy2 = (data[[i, j + 1, k]] - 2.0 * center + data[[i, j - 1, k]]) / dx_sq;
                let d2_dz2 = (data[[i, j, k + 1]] - 2.0 * center + data[[i, j, k - 1]]) / dx_sq;

                *lap_value = d2_dx2 + d2_dy2 + d2_dz2;
            }
        });

        result
    }

    /// Calculate gradient at each point (returns 3 component fields)
    pub fn gradient(&self) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let mut grad_x = Array3::zeros(self.data.dim());
        let mut grad_y = Array3::zeros(self.data.dim());
        let mut grad_z = Array3::zeros(self.data.dim());

        let dx_2 = 2.0 * self.dx_nm;
        let res = self.resolution;
        let data = &self.data;

        ndarray::Zip::indexed(grad_x.view_mut())
            .and(grad_y.view_mut())
            .and(grad_z.view_mut())
            .par_for_each(|(i, j, k), gx, gy, gz| {
                if i > 0 && i < res - 1 {
                    *gx = (data[[i + 1, j, k]] - data[[i - 1, j, k]]) / dx_2;
                }
                if j > 0 && j < res - 1 {
                    *gy = (data[[i, j + 1, k]] - data[[i, j - 1, k]]) / dx_2;
                }
                if k > 0 && k < res - 1 {
                    *gz = (data[[i, j, k + 1]] - data[[i, j, k - 1]]) / dx_2;
                }
            });

        (grad_x, grad_y, grad_z)
    }

    /// Calculate total energy density (kinetic + potential)
    pub fn total_energy(&self) -> f64 {
        let (grad_x, grad_y, grad_z) = self.gradient();
        let dx_cubed = self.dx_nm.powi(3);

        ndarray::Zip::from(&self.data)
            .and(&grad_x)
            .and(&grad_y)
            .and(&grad_z)
            .par_map_collect(|&phi, &gx, &gy, &gz| {
                let kinetic = 0.5 * (gx * gx + gy * gy + gz * gz);
                let potential = self.mexican_hat_potential(phi);
                (kinetic + potential) * dx_cubed
            })
            .sum()
    }

    /// Mexican hat potential: V(φ) = λ/4 * (φ² - v²)²
    #[inline]
    fn mexican_hat_potential(&self, phi: f64) -> f64 {
        const LAMBDA: f64 = 0.13; // Higgs self-coupling
        const V_SQ: f64 = 246.22 * 246.22; // VEV squared (GeV²)

        0.25 * LAMBDA * (phi * phi - V_SQ).powi(2)
    }

    /// Get maximum field value
    pub fn max_value(&self) -> f64 {
        self.data
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    /// Get minimum field value
    pub fn min_value(&self) -> f64 {
        self.data
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    /// Get average field value
    pub fn mean_value(&self) -> f64 {
        self.data.mean().unwrap_or(0.0)
    }

    /// Get field value at specific grid point
    pub fn get(&self, i: usize, j: usize, k: usize) -> Option<f64> {
        self.data.get([i, j, k]).copied()
    }

    /// Set field value at specific grid point
    pub fn set(&mut self, i: usize, j: usize, k: usize, value: f64) {
        if let Some(cell) = self.data.get_mut([i, j, k]) {
            *cell = value;
        }
    }

    /// Convert linear index to 3D coordinates
    pub fn index_to_coords(&self, idx: usize) -> (usize, usize, usize) {
        let res = self.resolution;
        let k = idx % res;
        let j = (idx / res) % res;
        let i = idx / (res * res);
        (i, j, k)
    }

    /// Extract a 2D slice at fixed z
    pub fn slice_xy(&self, z_index: usize) -> Option<ndarray::Array2<f64>> {
        if z_index < self.resolution {
            Some(self.data.index_axis(Axis(2), z_index).to_owned())
        } else {
            None
        }
    }

    /// Extract a 2D slice at fixed y
    pub fn slice_xz(&self, y_index: usize) -> Option<ndarray::Array2<f64>> {
        if y_index < self.resolution {
            Some(self.data.index_axis(Axis(1), y_index).to_owned())
        } else {
            None
        }
    }

    /// Extract a 2D slice at fixed x
    pub fn slice_yz(&self, x_index: usize) -> Option<ndarray::Array2<f64>> {
        if x_index < self.resolution {
            Some(self.data.index_axis(Axis(0), x_index).to_owned())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_creation() {
        let field = ScalarField3D::new(64, 100.0);
        assert_eq!(field.resolution, 64);
        assert_eq!(field.size_nm, 100.0);
        assert!((field.dx_nm - 100.0 / 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_fill() {
        let mut field = ScalarField3D::new(32, 50.0);
        field.fill_uniform(246.22);
        assert_eq!(field.mean_value(), 246.22);
    }

    #[test]
    fn test_wave_perturbation() {
        let mut field = ScalarField3D::new(32, 50.0);
        field.fill_uniform(246.22);
        field.add_wave_perturbation(10.0, 25.0);

        // Field should now have values above and below 246.22
        assert!(field.max_value() > 246.22);
        assert!(field.min_value() < 246.22);
    }

    #[test]
    fn test_laplacian() {
        let mut field = ScalarField3D::new(32, 50.0);
        field.fill_uniform(246.22);

        let lap = field.laplacian();

        // Laplacian of constant field should be ~0
        assert!(lap.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_slicing() {
        let field = ScalarField3D::new(32, 50.0);

        let slice_xy = field.slice_xy(16);
        assert!(slice_xy.is_some());
        assert_eq!(slice_xy.unwrap().shape(), &[32, 32]);

        let slice_invalid = field.slice_xy(100);
        assert!(slice_invalid.is_none());
    }
}
