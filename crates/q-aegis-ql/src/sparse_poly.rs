//! Sparse polynomial operations for AEGIS-QL
//!
//! Optimized operations on sparse polynomials to reduce
//! complexity from O(n²) to O(k·n) where k is the sparsity.

use super::{SparsePolynomial, POLY_DEGREE};

/// Convert sparse polynomial to dense representation
pub fn to_dense(sparse: &SparsePolynomial, degree: usize) -> Vec<u32> {
    let mut dense = vec![0u32; degree];
    for (i, &idx) in sparse.indices.iter().enumerate() {
        if idx < degree && i < sparse.coefficients.len() {
            dense[idx] = sparse.coefficients[i];
        }
    }
    dense
}

/// Sparse polynomial multiplication (O(k·n) instead of O(n²))
pub fn sparse_multiply(a: &SparsePolynomial, b: &[u32], modulus: u32) -> Vec<u32> {
    let degree = b.len();
    let mut result = vec![0u32; degree];

    // Only iterate over non-zero coefficients of sparse polynomial
    for (i, &coeff_a) in a.coefficients.iter().enumerate() {
        if coeff_a == 0 || i >= a.indices.len() {
            continue;
        }

        let idx_a = a.indices[i];
        if idx_a >= degree {
            continue;
        }

        // Multiply with all coefficients of dense polynomial
        for (j, &coeff_b) in b.iter().enumerate() {
            let result_idx = (idx_a + j) % degree;
            let product = ((coeff_a as u64) * (coeff_b as u64)) % (modulus as u64);
            result[result_idx] = ((result[result_idx] as u64 + product) % (modulus as u64)) as u32;
        }
    }

    result
}

/// Count non-zero coefficients
pub fn count_nonzero(poly: &[u32]) -> usize {
    poly.iter().filter(|&&x| x != 0).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_dense() {
        let sparse = SparsePolynomial {
            coefficients: vec![1, 2, 3],
            indices: vec![0, 5, 10],
            degree: 16,
        };

        let dense = to_dense(&sparse, 16);
        assert_eq!(dense[0], 1);
        assert_eq!(dense[5], 2);
        assert_eq!(dense[10], 3);
        assert_eq!(count_nonzero(&dense), 3);
    }
}
