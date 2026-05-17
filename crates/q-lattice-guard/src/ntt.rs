//! Number Theoretic Transform (NTT) for efficient polynomial operations
//!
//! NTT is the finite field analog of FFT, enabling O(n log n) polynomial
//! multiplication in the ring R_q = Z_q[X]/(X^n + 1).

use crate::{params::RlweParams, Scalar};

/// NTT operator for polynomial operations
#[derive(Clone, Debug)]
pub struct NttOperator {
    /// Ring dimension (must be power of 2)
    dimension: usize,
    /// Field modulus
    modulus: Scalar,
    /// Primitive 2n-th root of unity
    root: Scalar,
    /// Inverse of root
    root_inv: Scalar,
    /// Inverse of dimension mod q
    n_inv: Scalar,
    /// Precomputed powers of root (twiddle factors)
    powers: Vec<Scalar>,
    /// Precomputed powers of root_inv
    powers_inv: Vec<Scalar>,
}

impl NttOperator {
    /// Create new NTT operator for given parameters
    pub fn new(params: &RlweParams) -> Self {
        let dimension = params.dimension;
        let modulus = params.modulus;
        let root = params.ntt_root;

        // Compute root inverse
        let root_inv = params.mod_inverse(root).expect("Root must be invertible");

        // Compute n inverse
        let n_inv = params.mod_inverse(dimension as Scalar).expect("Dimension must be invertible");

        // Precompute twiddle factors
        let mut powers = vec![1u64; dimension];
        let mut powers_inv = vec![1u64; dimension];

        for i in 1..dimension {
            powers[i] = ((powers[i - 1] as u128 * root as u128) % modulus as u128) as Scalar;
            powers_inv[i] = ((powers_inv[i - 1] as u128 * root_inv as u128) % modulus as u128) as Scalar;
        }

        Self {
            dimension,
            modulus,
            root,
            root_inv,
            n_inv,
            powers,
            powers_inv,
        }
    }

    /// Forward NTT: Convert from coefficient form to evaluation form
    pub fn forward(&self, poly: &[Scalar]) -> Vec<Scalar> {
        assert_eq!(poly.len(), self.dimension);
        let mut result = poly.to_vec();
        self.ntt_internal(&mut result, &self.powers);
        result
    }

    /// Inverse NTT: Convert from evaluation form to coefficient form
    pub fn inverse(&self, evals: &[Scalar]) -> Vec<Scalar> {
        assert_eq!(evals.len(), self.dimension);
        let mut result = evals.to_vec();
        self.ntt_internal(&mut result, &self.powers_inv);

        // Scale by 1/n
        for coeff in result.iter_mut() {
            *coeff = ((*coeff as u128 * self.n_inv as u128) % self.modulus as u128) as Scalar;
        }

        result
    }

    /// Internal NTT using Cooley-Tukey butterfly
    fn ntt_internal(&self, data: &mut [Scalar], twiddles: &[Scalar]) {
        let n = data.len();
        if n <= 1 {
            return;
        }

        // Bit-reversal permutation
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                data.swap(i, j);
            }
        }

        // Cooley-Tukey butterfly
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;

            for i in (0..n).step_by(len) {
                for j in 0..half {
                    let u = data[i + j];
                    let v = ((data[i + j + half] as u128 * twiddles[j * step] as u128)
                        % self.modulus as u128) as Scalar;

                    data[i + j] = (u + v) % self.modulus;
                    data[i + j + half] = if u >= v { u - v } else { self.modulus - v + u };
                }
            }

            len *= 2;
        }
    }

    /// Pointwise multiplication in NTT domain
    pub fn pointwise_mul(&self, a: &[Scalar], b: &[Scalar]) -> Vec<Scalar> {
        assert_eq!(a.len(), self.dimension);
        assert_eq!(b.len(), self.dimension);

        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| ((x as u128 * y as u128) % self.modulus as u128) as Scalar)
            .collect()
    }

    /// Multiply two polynomials using NTT
    ///
    /// Much faster than schoolbook multiplication: O(n log n) vs O(n²)
    pub fn mul(&self, a: &[Scalar], b: &[Scalar]) -> Vec<Scalar> {
        let a_ntt = self.forward(a);
        let b_ntt = self.forward(b);
        let c_ntt = self.pointwise_mul(&a_ntt, &b_ntt);
        self.inverse(&c_ntt)
    }

    /// Add two polynomials
    pub fn add(&self, a: &[Scalar], b: &[Scalar]) -> Vec<Scalar> {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x + y) % self.modulus)
            .collect()
    }

    /// Subtract two polynomials
    pub fn sub(&self, a: &[Scalar], b: &[Scalar]) -> Vec<Scalar> {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| if x >= y { x - y } else { self.modulus - y + x })
            .collect()
    }

    /// Compute L2 norm squared of polynomial (coefficient form)
    pub fn norm_squared(&self, poly: &[Scalar]) -> u128 {
        poly.iter().map(|&x| {
            // Center the coefficient: if x > q/2, treat as negative
            let centered = if x > self.modulus / 2 {
                (self.modulus - x) as i128
            } else {
                x as i128
            };
            (centered * centered) as u128
        }).sum()
    }

    /// Check if error is within bound
    pub fn check_error_bound(&self, error: &[Scalar], bound: Scalar) -> bool {
        for &e in error {
            let centered = if e > self.modulus / 2 {
                self.modulus - e
            } else {
                e
            };
            if centered > bound {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntt_roundtrip() {
        let params = RlweParams::pq128();
        let ntt = NttOperator::new(&params);

        // Random polynomial
        let mut rng = rand::thread_rng();
        let poly: Vec<Scalar> = (0..params.dimension)
            .map(|_| rand::Rng::gen::<Scalar>(&mut rng) % params.modulus)
            .collect();

        // Forward then inverse should give back original
        let ntt_form = ntt.forward(&poly);
        let recovered = ntt.inverse(&ntt_form);

        for i in 0..params.dimension {
            assert_eq!(poly[i], recovered[i], "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_ntt_multiplication() {
        let params = RlweParams::pq128();
        let ntt = NttOperator::new(&params);

        // Simple test: (1 + x) * (1 - x) = 1 - x²
        let mut a = vec![0u64; params.dimension];
        let mut b = vec![0u64; params.dimension];

        a[0] = 1;
        a[1] = 1;  // 1 + x

        b[0] = 1;
        b[1] = params.modulus - 1;  // 1 - x = 1 + (-1)*x

        let result = ntt.mul(&a, &b);

        // Result should be 1 - x² (with reduction by X^n + 1)
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 0);  // No x term
        assert_eq!(result[2], params.modulus - 1);  // -x² term
    }

    #[test]
    fn test_pointwise_mul() {
        let params = RlweParams::pq128();
        let ntt = NttOperator::new(&params);

        let a = vec![2u64; params.dimension];
        let b = vec![3u64; params.dimension];

        let result = ntt.pointwise_mul(&a, &b);

        for val in result {
            assert_eq!(val, 6);
        }
    }
}
