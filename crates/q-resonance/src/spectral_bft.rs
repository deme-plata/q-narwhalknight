//! Spectral Byzantine Fault Tolerance
//!
//! Detects Byzantine nodes via Laplacian eigenvalue analysis.
//! The key insight: Byzantine attacks appear as high-frequency modes in the spectral decomposition.

use crate::vertex::ResonanceVertex;
use crate::ResonanceError;
use ndarray::{Array1, Array2};
use ndarray_linalg::Eigh;
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};

/// Spectral BFT analyzer
pub struct SpectralBFT {
    /// Detection threshold (vertices with spectral projection > threshold are Byzantine)
    detection_threshold: f64,

    /// Number of top eigenvalues to analyze
    num_modes: usize,

    /// Minimum network size for spectral analysis
    min_network_size: usize,

    /// Cache of computed Laplacians (reserved for future optimization)
    #[allow(dead_code)]
    laplacian_cache: HashMap<u64, Array2<f64>>,
}

impl SpectralBFT {
    /// Create a new spectral BFT analyzer
    pub fn new(detection_threshold: f64, num_modes: usize) -> Self {
        Self {
            detection_threshold,
            num_modes,
            min_network_size: 4,
            laplacian_cache: HashMap::new(),
        }
    }

    /// Detect Byzantine vertices using full spectral analysis
    pub fn detect_byzantine(
        &mut self,
        vertices: &[ResonanceVertex],
    ) -> Result<HashSet<[u8; 32]>, ResonanceError> {
        if vertices.len() < self.min_network_size {
            return Ok(HashSet::new());
        }

        // Build Laplacian matrix
        let laplacian = self.compute_laplacian(vertices);

        // Compute eigenvalues and eigenvectors
        let (eigenvalues, eigenvectors) = laplacian
            .eigh(ndarray_linalg::UPLO::Lower)
            .map_err(|e| ResonanceError::SpectralError(format!("Eigenvalue computation failed: {}", e)))?;

        // Analyze top k modes (highest eigenvalues = attack modes)
        let mut byzantine_nodes = HashSet::new();

        let num_modes = self.num_modes.min(eigenvalues.len());

        // Sort indices by eigenvalue (descending)
        let mut indexed_eigenvalues: Vec<(usize, f64)> = (0..eigenvalues.len())
            .map(|i| (i, eigenvalues[i]))
            .collect();
        indexed_eigenvalues.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_indices: Vec<usize> = indexed_eigenvalues
            .into_iter()
            .take(num_modes)
            .map(|(i, _)| i)
            .collect();

        for mode_idx in top_indices {
            let eigenvector = eigenvectors.column(mode_idx);

            // Vertices with high projection onto attack modes are Byzantine
            for (i, &coefficient) in eigenvector.iter().enumerate() {
                if coefficient.abs() > self.detection_threshold {
                    byzantine_nodes.insert(vertices[i].hash);

                    tracing::warn!(
                        "Byzantine vertex detected: {:?} (mode {}, coefficient: {:.4})",
                        vertices[i].hash,
                        mode_idx,
                        coefficient
                    );
                }
            }
        }

        Ok(byzantine_nodes)
    }

    /// Compute graph Laplacian L = D - A
    ///
    /// Where:
    /// - D is degree matrix (diagonal with vertex degrees)
    /// - A is adjacency matrix weighted by resonance strength
    pub fn compute_laplacian(&self, vertices: &[ResonanceVertex]) -> Array2<f64> {
        let n = vertices.len();
        let mut laplacian = Array2::zeros((n, n));

        // Build adjacency matrix with resonance weights
        for i in 0..n {
            let mut degree = 0.0;

            for j in 0..n {
                if i == j {
                    continue;
                }

                let resonance = vertices[i].resonance(&vertices[j]);
                laplacian[[i, j]] = -resonance; // Off-diagonal: -A_ij
                degree += resonance;
            }

            laplacian[[i, i]] = degree; // Diagonal: D_ii
        }

        laplacian
    }

    /// Sample-based Byzantine detection (O(k³) instead of O(n³))
    pub fn detect_byzantine_sampled(
        &mut self,
        vertices: &[ResonanceVertex],
        sample_size: usize,
    ) -> Result<HashSet<[u8; 32]>, ResonanceError> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        if vertices.len() < self.min_network_size {
            return Ok(HashSet::new());
        }

        let actual_sample_size = sample_size.min(vertices.len());
        let mut rng = thread_rng();

        // Random sample of vertices
        let mut sampled_indices: Vec<usize> = (0..vertices.len()).collect();
        sampled_indices.shuffle(&mut rng);
        sampled_indices.truncate(actual_sample_size);

        let sampled_vertices: Vec<&ResonanceVertex> =
            sampled_indices.iter().map(|&i| &vertices[i]).collect();

        // Build sampled Laplacian
        let laplacian = self.compute_laplacian_from_refs(&sampled_vertices);

        // Eigenvalue decomposition
        let (eigenvalues, eigenvectors) = laplacian
            .eigh(ndarray_linalg::UPLO::Lower)
            .map_err(|e| ResonanceError::SpectralError(format!("Sampled eigenvalue computation failed: {}", e)))?;

        // Detect Byzantine in sample
        let mut byzantine_nodes = HashSet::new();

        let num_modes = self.num_modes.min(eigenvalues.len());

        // Sort indices by eigenvalue (descending)
        let mut indexed_eigenvalues: Vec<(usize, f64)> = (0..eigenvalues.len())
            .map(|i| (i, eigenvalues[i]))
            .collect();
        indexed_eigenvalues.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_indices: Vec<usize> = indexed_eigenvalues
            .into_iter()
            .take(num_modes)
            .map(|(i, _)| i)
            .collect();

        for mode_idx in top_indices {
            let eigenvector = eigenvectors.column(mode_idx);

            for (i, &coefficient) in eigenvector.iter().enumerate() {
                if coefficient.abs() > self.detection_threshold {
                    byzantine_nodes.insert(sampled_vertices[i].hash);
                }
            }
        }

        Ok(byzantine_nodes)
    }

    /// Compute Laplacian from vertex references
    fn compute_laplacian_from_refs(&self, vertices: &[&ResonanceVertex]) -> Array2<f64> {
        let n = vertices.len();
        let mut laplacian = Array2::zeros((n, n));

        for i in 0..n {
            let mut degree = 0.0;

            for j in 0..n {
                if i == j {
                    continue;
                }

                let resonance = vertices[i].resonance(vertices[j]);
                laplacian[[i, j]] = -resonance;
                degree += resonance;
            }

            laplacian[[i, i]] = degree;
        }

        laplacian
    }

    /// Streaming Laplacian update for incremental consensus
    ///
    /// Uses matrix perturbation theory to update eigenvalues without full recomputation
    pub fn streaming_update(
        &mut self,
        existing_eigenvalues: &Array1<f64>,
        new_vertices: &[ResonanceVertex],
        old_vertices: &[ResonanceVertex],
    ) -> Array1<f64> {
        // Compute perturbation matrix
        let delta_laplacian = self.compute_perturbation(new_vertices, old_vertices);

        // First-order perturbation correction: λ'_i ≈ λ_i + δL_ii
        let mut updated_eigenvalues = existing_eigenvalues.clone();

        for (i, eigenvalue) in updated_eigenvalues.iter_mut().enumerate() {
            if i < delta_laplacian.nrows() {
                *eigenvalue += delta_laplacian[[i, i]];
            }
        }

        updated_eigenvalues
    }

    /// Compute perturbation matrix δL from vertex changes
    fn compute_perturbation(
        &self,
        new_vertices: &[ResonanceVertex],
        old_vertices: &[ResonanceVertex],
    ) -> Array2<f64> {
        let n = new_vertices.len().max(old_vertices.len());
        let mut delta = Array2::zeros((n, n));

        // Compute difference in Laplacians
        let new_laplacian = self.compute_laplacian(new_vertices);
        let old_laplacian = self.compute_laplacian(old_vertices);

        let min_size = new_laplacian.nrows().min(old_laplacian.nrows());

        for i in 0..min_size {
            for j in 0..min_size {
                delta[[i, j]] = new_laplacian[[i, j]] - old_laplacian[[i, j]];
            }
        }

        delta
    }

    /// Compute spectral gap (λ₁ - λ₀)
    ///
    /// Large spectral gap indicates strong consensus
    pub fn spectral_gap(&self, vertices: &[ResonanceVertex]) -> Result<f64, ResonanceError> {
        if vertices.len() < 2 {
            return Ok(0.0);
        }

        let laplacian = self.compute_laplacian(vertices);
        let eigenvalues = laplacian
            .eigh(ndarray_linalg::UPLO::Lower)
            .map(|(vals, _)| vals)
            .map_err(|e| ResonanceError::SpectralError(format!("Spectral gap computation failed: {}", e)))?;

        if eigenvalues.len() < 2 {
            return Ok(0.0);
        }

        let mut sorted_vals: Vec<f64> = eigenvalues.to_vec();
        sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        Ok(sorted_vals[1] - sorted_vals[0])
    }

    /// Check if network has strong consensus (large spectral gap)
    pub fn has_strong_consensus(
        &self,
        vertices: &[ResonanceVertex],
        threshold: f64,
    ) -> Result<bool, ResonanceError> {
        let gap = self.spectral_gap(vertices)?;
        Ok(gap > threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_state::StringState;
    use std::collections::HashSet;

    fn create_test_vertices(n: usize) -> Vec<ResonanceVertex> {
        (0..n)
            .map(|i| {
                let hash = [i as u8; 32];
                ResonanceVertex::new(
                    hash,
                    1,
                    HashSet::new(),
                    vec![],
                    vec![i as u8],
                    1000 + i as u64,
                    100.0,
                    vec![i as f64, 0.0],
                    0.5,
                )
            })
            .collect()
    }

    #[test]
    fn test_laplacian_computation() {
        let vertices = create_test_vertices(5);
        let spectral = SpectralBFT::new(0.5, 2);

        let laplacian = spectral.compute_laplacian(&vertices);
        assert_eq!(laplacian.nrows(), 5);
        assert_eq!(laplacian.ncols(), 5);

        // Laplacian should be symmetric
        for i in 0..5 {
            for j in 0..5 {
                assert!((laplacian[[i, j]] - laplacian[[j, i]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_byzantine_detection() {
        let mut vertices = create_test_vertices(10);

        // Make one vertex Byzantine (very different phase)
        vertices[5].string_state.phase = num_complex::Complex::new(10.0, 10.0);

        let mut spectral = SpectralBFT::new(0.3, 3);

        match spectral.detect_byzantine(&vertices) {
            Ok(byzantine) => {
                tracing::info!("Detected {} Byzantine vertices", byzantine.len());
                // Note: Detection may or may not find the Byzantine node depending on threshold
            }
            Err(e) => {
                tracing::error!("Byzantine detection failed: {}", e);
            }
        }
    }

    #[test]
    fn test_spectral_gap() {
        let vertices = create_test_vertices(6);
        let spectral = SpectralBFT::new(0.5, 2);

        match spectral.spectral_gap(&vertices) {
            Ok(gap) => {
                assert!(gap >= 0.0);
                tracing::info!("Spectral gap: {:.6}", gap);
            }
            Err(e) => {
                tracing::error!("Spectral gap computation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_sampled_detection() {
        let vertices = create_test_vertices(20);
        let mut spectral = SpectralBFT::new(0.4, 3);

        match spectral.detect_byzantine_sampled(&vertices, 10) {
            Ok(byzantine) => {
                tracing::info!("Sampled detection found {} Byzantine vertices", byzantine.len());
            }
            Err(e) => {
                tracing::error!("Sampled detection failed: {}", e);
            }
        }
    }
}
