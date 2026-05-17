//! Ring Learning With Errors (RLWE) Encryption
//!
//! This module provides RLWE-based encryption primitives for LatticeGuard.
//! Security is based on the hardness of the RLWE problem.

use crate::{errors::LatticeGuardError, params::RlweParams, Polynomial, Scalar};
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};

/// RLWE ciphertext: (a, b) where b = a*s + e + Δ*m
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RlweCiphertext {
    /// First component (uniform random polynomial)
    pub a: Vec<Scalar>,
    /// Second component (encrypted message)
    pub b: Vec<Scalar>,
}

impl RlweCiphertext {
    /// Create new ciphertext
    pub fn new(a: Vec<Scalar>, b: Vec<Scalar>) -> Self {
        Self { a, b }
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.a.len()
    }

    /// Add two ciphertexts homomorphically
    pub fn add(&self, other: &RlweCiphertext, modulus: Scalar) -> RlweCiphertext {
        let a = self
            .a
            .iter()
            .zip(other.a.iter())
            .map(|(&x, &y)| (x + y) % modulus)
            .collect();
        let b = self
            .b
            .iter()
            .zip(other.b.iter())
            .map(|(&x, &y)| (x + y) % modulus)
            .collect();
        RlweCiphertext { a, b }
    }

    /// Scalar multiplication (constant * ciphertext)
    pub fn scalar_mul(&self, scalar: Scalar, modulus: Scalar) -> RlweCiphertext {
        let a = self.a.iter().map(|&x| (x * scalar) % modulus).collect();
        let b = self.b.iter().map(|&x| (x * scalar) % modulus).collect();
        RlweCiphertext { a, b }
    }
}

/// RLWE secret key
#[derive(Clone, Serialize, Deserialize)]
pub struct RlweSecretKey {
    /// Secret polynomial with small coefficients
    pub s: Vec<Scalar>,
}

impl RlweSecretKey {
    /// Generate new secret key with ternary distribution {-1, 0, 1}
    pub fn generate<R: Rng + CryptoRng>(params: &RlweParams, rng: &mut R) -> Self {
        let s: Vec<Scalar> = (0..params.dimension)
            .map(|_| {
                match rng.gen_range(0..3) {
                    0 => params.modulus - 1, // -1 mod q
                    1 => 0,
                    _ => 1,
                }
            })
            .collect();

        Self { s }
    }

    /// Encrypt a polynomial message
    pub fn encrypt<R: Rng + CryptoRng>(
        &self,
        message: &Polynomial,
        params: &RlweParams,
        rng: &mut R,
    ) -> Result<RlweCiphertext, LatticeGuardError> {
        // Sample random polynomial a
        let a = params.sample_uniform_polynomial(rng);

        // Sample error polynomial e
        let e = params.sample_error_polynomial(rng);

        // Compute b = a*s + e + Δ*m
        // where Δ = q/t is the scaling factor (here we use q/2 for binary messages)
        let delta = params.modulus / 2;

        // Polynomial multiplication in ring R_q = Z_q[X]/(X^n + 1)
        let as_product = ring_mul(&a, &self.s, params.modulus, params.dimension);

        let mut b = vec![0u64; params.dimension];
        for i in 0..params.dimension {
            let mut val = as_product[i];
            val = (val + e[i]) % params.modulus;

            // Add scaled message
            if i < message.coefficients.len() {
                val = (val + (message.coefficients[i] * delta) % params.modulus) % params.modulus;
            }

            b[i] = val;
        }

        Ok(RlweCiphertext::new(a, b))
    }

    /// Decrypt a ciphertext
    pub fn decrypt(
        &self,
        ciphertext: &RlweCiphertext,
        params: &RlweParams,
    ) -> Result<Polynomial, LatticeGuardError> {
        // Compute m' = b - a*s = Δ*m + e
        let as_product = ring_mul(&ciphertext.a, &self.s, params.modulus, params.dimension);

        let mut coefficients = vec![0u64; params.dimension];
        let delta = params.modulus / 2;

        for i in 0..params.dimension {
            // Compute b[i] - (a*s)[i] mod q
            let diff = if ciphertext.b[i] >= as_product[i] {
                ciphertext.b[i] - as_product[i]
            } else {
                params.modulus - as_product[i] + ciphertext.b[i]
            };

            // Round to nearest multiple of delta
            // If diff > q/2, treat as negative
            let centered = if diff > params.modulus / 2 {
                0 // Rounds to 0
            } else {
                (diff + delta / 2) / delta // Rounds to nearest integer
            };

            coefficients[i] = centered;
        }

        Ok(Polynomial::new(coefficients))
    }

    /// Generate public key from secret key
    pub fn public_key<R: Rng + CryptoRng>(
        &self,
        params: &RlweParams,
        rng: &mut R,
    ) -> Result<RlwePublicKey, LatticeGuardError> {
        // Public key is an RLWE encryption of zero
        let zero = Polynomial::new(vec![0]);
        let ct = self.encrypt(&zero, params, rng)?;

        Ok(RlwePublicKey { a: ct.a, b: ct.b })
    }
}

impl std::fmt::Debug for RlweSecretKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Don't print the actual secret
        f.debug_struct("RlweSecretKey")
            .field("dimension", &self.s.len())
            .finish()
    }
}

/// RLWE public key
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RlwePublicKey {
    /// First component
    pub a: Vec<Scalar>,
    /// Second component (a*s + e for secret s)
    pub b: Vec<Scalar>,
}

impl RlwePublicKey {
    /// Encrypt using public key
    pub fn encrypt<R: Rng + CryptoRng>(
        &self,
        message: &Polynomial,
        params: &RlweParams,
        rng: &mut R,
    ) -> Result<RlweCiphertext, LatticeGuardError> {
        // Sample random polynomial u with small coefficients
        let u: Vec<Scalar> = (0..params.dimension)
            .map(|_| match rng.gen_range(0..3) {
                0 => params.modulus - 1,
                1 => 0,
                _ => 1,
            })
            .collect();

        // Sample error polynomials e1, e2
        let e1 = params.sample_error_polynomial(rng);
        let e2 = params.sample_error_polynomial(rng);

        // c1 = a*u + e1
        let au = ring_mul(&self.a, &u, params.modulus, params.dimension);
        let c1: Vec<Scalar> = au
            .iter()
            .zip(e1.iter())
            .map(|(&x, &e)| (x + e) % params.modulus)
            .collect();

        // c2 = b*u + e2 + Δ*m
        let bu = ring_mul(&self.b, &u, params.modulus, params.dimension);
        let delta = params.modulus / 2;

        let mut c2 = vec![0u64; params.dimension];
        for i in 0..params.dimension {
            let mut val = (bu[i] + e2[i]) % params.modulus;
            if i < message.coefficients.len() {
                val = (val + (message.coefficients[i] * delta) % params.modulus) % params.modulus;
            }
            c2[i] = val;
        }

        Ok(RlweCiphertext::new(c1, c2))
    }
}

/// RLWE keypair
#[derive(Clone, Serialize, Deserialize)]
pub struct RlweKeypair {
    /// Secret key
    pub secret_key: RlweSecretKey,
    /// Public key
    pub public_key: RlwePublicKey,
}

impl RlweKeypair {
    /// Generate new keypair
    pub fn generate<R: Rng + CryptoRng>(
        params: &RlweParams,
        rng: &mut R,
    ) -> Result<Self, LatticeGuardError> {
        let secret_key = RlweSecretKey::generate(params, rng);
        let public_key = secret_key.public_key(params, rng)?;

        Ok(Self {
            secret_key,
            public_key,
        })
    }
}

impl std::fmt::Debug for RlweKeypair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RlweKeypair")
            .field("public_key", &self.public_key)
            .finish()
    }
}

/// Ring multiplication in R_q = Z_q[X]/(X^n + 1)
///
/// Uses schoolbook multiplication with reduction by X^n + 1.
/// For production, this should use NTT for O(n log n) performance.
pub fn ring_mul(a: &[Scalar], b: &[Scalar], modulus: Scalar, n: usize) -> Vec<Scalar> {
    let mut result = vec![0u64; n];

    for i in 0..n {
        for j in 0..n {
            let product = ((a[i] as u128) * (b[j] as u128)) % (modulus as u128);
            let idx = i + j;

            if idx < n {
                result[idx] = (result[idx] + product as u64) % modulus;
            } else {
                // Reduce by X^n + 1: X^n ≡ -1
                let reduced_idx = idx - n;
                // Subtract (equivalent to adding -1 * product)
                if result[reduced_idx] >= product as u64 {
                    result[reduced_idx] -= product as u64;
                } else {
                    result[reduced_idx] = modulus - (product as u64 - result[reduced_idx]);
                }
            }
        }
    }

    result
}

/// Ring addition in R_q
pub fn ring_add(a: &[Scalar], b: &[Scalar], modulus: Scalar) -> Vec<Scalar> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x + y) % modulus)
        .collect()
}

/// Ring subtraction in R_q
pub fn ring_sub(a: &[Scalar], b: &[Scalar], modulus: Scalar) -> Vec<Scalar> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            if x >= y {
                x - y
            } else {
                modulus - y + x
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let params = RlweParams::pq128();
        let mut rng = rand::thread_rng();

        let keypair = RlweKeypair::generate(&params, &mut rng).unwrap();
        assert_eq!(keypair.secret_key.s.len(), params.dimension);
        assert_eq!(keypair.public_key.a.len(), params.dimension);
    }

    #[test]
    fn test_encrypt_decrypt_zero() {
        let params = RlweParams::pq128();
        let mut rng = rand::thread_rng();

        let keypair = RlweKeypair::generate(&params, &mut rng).unwrap();
        let message = Polynomial::new(vec![0; params.dimension]);

        let ciphertext = keypair.secret_key.encrypt(&message, &params, &mut rng).unwrap();
        let decrypted = keypair.secret_key.decrypt(&ciphertext, &params).unwrap();

        // All coefficients should decrypt to 0 (or very small due to noise)
        for coeff in decrypted.coefficients.iter().take(10) {
            assert!(*coeff < 2, "Expected near-zero, got {}", coeff);
        }
    }

    #[test]
    fn test_ring_mul_simple() {
        // Simple test: (1 + x) * (1 + x) = 1 + 2x + x^2
        let a = vec![1, 1, 0, 0]; // 1 + x in Z_q[X]/(X^4 + 1)
        let b = vec![1, 1, 0, 0];
        let modulus = 1000u64;

        let result = ring_mul(&a, &b, modulus, 4);

        // Result should be 1 + 2x + x^2 = [1, 2, 1, 0]
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 1);
        assert_eq!(result[3], 0);
    }

    #[test]
    fn test_homomorphic_addition() {
        let params = RlweParams::pq128();
        let mut rng = rand::thread_rng();

        let sk = RlweSecretKey::generate(&params, &mut rng);

        // Encrypt two messages
        let m1 = Polynomial::new(vec![1; params.dimension]);
        let m2 = Polynomial::new(vec![1; params.dimension]);

        let c1 = sk.encrypt(&m1, &params, &mut rng).unwrap();
        let c2 = sk.encrypt(&m2, &params, &mut rng).unwrap();

        // Add ciphertexts
        let c_sum = c1.add(&c2, params.modulus);

        // Decrypt and check (should be close to 2)
        let decrypted = sk.decrypt(&c_sum, &params).unwrap();

        // First few coefficients should be close to 2
        for coeff in decrypted.coefficients.iter().take(5) {
            assert!(
                *coeff >= 1 && *coeff <= 3,
                "Expected ~2, got {}",
                coeff
            );
        }
    }
}
