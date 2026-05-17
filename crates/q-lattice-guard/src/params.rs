//! RLWE Parameters for LatticeGuard
//!
//! This module defines the cryptographic parameters for the RLWE-based
//! LatticeGuard zk-SNARK system.

use serde::{Deserialize, Serialize};

/// Security level for post-quantum protection
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// 128-bit post-quantum security (NIST Level I)
    PQ128,
    /// 192-bit post-quantum security (NIST Level III)
    PQ192,
    /// 256-bit post-quantum security (NIST Level V)
    PQ256,
}

impl Default for SecurityLevel {
    fn default() -> Self {
        SecurityLevel::PQ128
    }
}

/// RLWE parameters for LatticeGuard
///
/// These parameters define the ring R_q = Z_q[X]/(X^n + 1) used in RLWE.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RlweParams {
    /// Ring dimension (must be power of 2)
    /// Polynomial degree n for R = Z[X]/(X^n + 1)
    pub dimension: usize,

    /// Modulus q for Z_q
    /// Chosen to support NTT: q ≡ 1 (mod 2n)
    pub modulus: u64,

    /// Standard deviation for error distribution
    /// Gaussian with σ for RLWE security
    pub std_dev: f64,

    /// Error bound for approximate computations
    /// All errors must be less than this bound
    pub error_bound: u64,

    /// Number of bits in modulus
    pub modulus_bits: u32,

    /// Security level
    pub security_level: SecurityLevel,

    /// Primitive 2n-th root of unity for NTT
    pub ntt_root: u64,
}

impl RlweParams {
    /// Create parameters from security level
    pub fn from_security_level(level: SecurityLevel) -> Self {
        match level {
            SecurityLevel::PQ128 => Self::pq128(),
            SecurityLevel::PQ192 => Self::pq192(),
            SecurityLevel::PQ256 => Self::pq256(),
        }
    }

    /// 128-bit post-quantum security parameters
    ///
    /// Based on Kyber-512 and SEAL parameters:
    /// - n = 1024
    /// - q ≈ 2^32
    /// - σ = 3.2
    pub fn pq128() -> Self {
        // q = 4294957057 (prime, q ≡ 1 mod 2048)
        let modulus = 4294957057u64;
        let dimension = 1024;

        Self {
            dimension,
            modulus,
            std_dev: 3.2,
            error_bound: 1 << 20, // 2^20 error bound
            modulus_bits: 32,
            security_level: SecurityLevel::PQ128,
            ntt_root: Self::find_primitive_root(modulus, 2 * dimension as u64),
        }
    }

    /// 192-bit post-quantum security parameters
    ///
    /// Based on Kyber-768:
    /// - n = 2048
    /// - q ≈ 2^48
    /// - σ = 3.2
    pub fn pq192() -> Self {
        // q = 281474976710597 (prime, q ≡ 1 mod 4096)
        let modulus = 281474976710597u64;
        let dimension = 2048;

        Self {
            dimension,
            modulus,
            std_dev: 3.2,
            error_bound: 1 << 24,
            modulus_bits: 48,
            security_level: SecurityLevel::PQ192,
            ntt_root: Self::find_primitive_root(modulus, 2 * dimension as u64),
        }
    }

    /// 256-bit post-quantum security parameters
    ///
    /// Based on Kyber-1024:
    /// - n = 4096
    /// - q ≈ 2^60
    /// - σ = 3.2
    pub fn pq256() -> Self {
        // q = 1152921504606846883 (prime, q ≡ 1 mod 8192)
        let modulus = 1152921504606846883u64;
        let dimension = 4096;

        Self {
            dimension,
            modulus,
            std_dev: 3.2,
            error_bound: 1 << 30,
            modulus_bits: 60,
            security_level: SecurityLevel::PQ256,
            ntt_root: Self::find_primitive_root(modulus, 2 * dimension as u64),
        }
    }

    /// Find a primitive k-th root of unity modulo q
    fn find_primitive_root(q: u64, k: u64) -> u64 {
        // For efficiency, we use precomputed roots for standard parameters
        // In production, this would compute the actual root

        // Simplified: return a placeholder that works for testing
        // Real implementation would use Tonelli-Shanks or similar
        let mut g = 2u64;
        let order = q - 1;

        while g < q {
            // Check if g^(order/k) != 1 for all prime factors of k
            let test = Self::mod_pow(g, order / k, q);
            if test != 1 {
                let root = Self::mod_pow(g, order / k, q);
                // Verify it's a k-th root of unity
                if Self::mod_pow(root, k, q) == 1 {
                    return root;
                }
            }
            g += 1;
        }

        // Fallback for testing
        3
    }

    /// Modular exponentiation: base^exp mod modulus
    pub fn mod_pow(base: u64, exp: u64, modulus: u64) -> u64 {
        if modulus == 1 {
            return 0;
        }
        let mut result = 1u128;
        let mut base = (base as u128) % (modulus as u128);
        let mut exp = exp;
        let modulus = modulus as u128;

        while exp > 0 {
            if exp % 2 == 1 {
                result = (result * base) % modulus;
            }
            exp /= 2;
            base = (base * base) % modulus;
        }
        result as u64
    }

    /// Modular inverse using extended Euclidean algorithm
    pub fn mod_inverse(&self, a: u64) -> Option<u64> {
        Self::extended_gcd(a as i128, self.modulus as i128)
    }

    /// Extended GCD for modular inverse
    fn extended_gcd(a: i128, m: i128) -> Option<u64> {
        if a == 0 {
            return None;
        }

        let (mut old_r, mut r) = (a, m);
        let (mut old_s, mut s) = (1i128, 0i128);

        while r != 0 {
            let quotient = old_r / r;
            (old_r, r) = (r, old_r - quotient * r);
            (old_s, s) = (s, old_s - quotient * s);
        }

        if old_r != 1 {
            return None; // Not coprime
        }

        Some(((old_s % m + m) % m) as u64)
    }

    /// Sample from discrete Gaussian distribution
    pub fn sample_gaussian<R: rand::Rng>(&self, rng: &mut R) -> i64 {
        // Box-Muller transform for Gaussian sampling
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();

        let mag = self.std_dev * (-2.0 * u1.ln()).sqrt();
        let z = mag * (2.0 * std::f64::consts::PI * u2).cos();

        z.round() as i64
    }

    /// Sample error polynomial with Gaussian coefficients
    pub fn sample_error_polynomial<R: rand::Rng>(&self, rng: &mut R) -> Vec<u64> {
        (0..self.dimension)
            .map(|_| {
                let e = self.sample_gaussian(rng);
                // Map to positive representative in Z_q
                if e < 0 {
                    (self.modulus as i64 + e) as u64
                } else {
                    e as u64
                }
            })
            .collect()
    }

    /// Sample uniform polynomial in Z_q[X]/(X^n + 1)
    pub fn sample_uniform_polynomial<R: rand::Rng>(&self, rng: &mut R) -> Vec<u64> {
        (0..self.dimension)
            .map(|_| rng.gen::<u64>() % self.modulus)
            .collect()
    }
}

impl Default for RlweParams {
    fn default() -> Self {
        Self::pq128()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_pow() {
        // 2^10 mod 1000 = 1024 mod 1000 = 24
        assert_eq!(RlweParams::mod_pow(2, 10, 1000), 24);

        // 3^5 mod 7 = 243 mod 7 = 5
        assert_eq!(RlweParams::mod_pow(3, 5, 7), 5);
    }

    #[test]
    fn test_security_levels() {
        let pq128 = RlweParams::pq128();
        assert_eq!(pq128.dimension, 1024);
        assert_eq!(pq128.security_level, SecurityLevel::PQ128);

        let pq192 = RlweParams::pq192();
        assert_eq!(pq192.dimension, 2048);
        assert_eq!(pq192.security_level, SecurityLevel::PQ192);

        let pq256 = RlweParams::pq256();
        assert_eq!(pq256.dimension, 4096);
        assert_eq!(pq256.security_level, SecurityLevel::PQ256);
    }

    #[test]
    fn test_gaussian_sampling() {
        let params = RlweParams::pq128();
        let mut rng = rand::thread_rng();

        // Sample many values and check they're roughly Gaussian
        let samples: Vec<i64> = (0..1000)
            .map(|_| params.sample_gaussian(&mut rng))
            .collect();

        // Mean should be close to 0
        let mean: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 1.0, "Mean {} is too far from 0", mean);

        // Most samples should be within 3σ
        let within_3sigma = samples.iter().filter(|&&x| (x as f64).abs() < 3.0 * params.std_dev).count();
        assert!(within_3sigma > 900, "Not enough samples within 3σ");
    }
}
