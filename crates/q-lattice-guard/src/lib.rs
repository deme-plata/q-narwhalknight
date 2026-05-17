//! # LatticeGuard: A Novel Lattice-Based Post-Quantum zk-SNARK
//!
//! LatticeGuard provides post-quantum secure zero-knowledge proofs based on
//! the Ring Learning With Errors (RLWE) assumption, which is believed to be
//! resistant to quantum attacks.
//!
//! ## Key Features
//!
//! - **Post-Quantum Security**: Based on RLWE/RSIS hardness assumptions
//! - **Practical Performance**: 10-50KB proofs, 10-100ms verification
//! - **Approximate Arithmetic**: Novel proof system with bounded error
//! - **Compatible Interface**: Implements standard SNARK traits
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                    LatticeGuard Protocol                      │
//! ├──────────────────────────────────────────────────────────────┤
//! │  1. RLWE-based Polynomial Commitment                         │
//! │     └─> Commitment = RLWE encryption of polynomial           │
//! │                                                              │
//! │  2. Approximate Product Proofs                               │
//! │     └─> Proves c ≈ a * b with bounded error                  │
//! │                                                              │
//! │  3. Lattice-Based Fiat-Shamir                                │
//! │     └─> Challenge generation via SIS-based hash              │
//! │                                                              │
//! │  4. Zero-Knowledge via Noise Flooding                        │
//! │     └─> Statistical ZK from RLWE noise distribution          │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Security Levels
//!
//! | Level | Dimension | Modulus Bits | Post-Quantum Security |
//! |-------|-----------|--------------|----------------------|
//! | PQ128 | 1024      | 32           | ~128 bits            |
//! | PQ192 | 2048      | 48           | ~192 bits            |
//! | PQ256 | 4096      | 64           | ~256 bits            |

pub mod approximate_product;
pub mod commitment;
pub mod errors;
pub mod ntt;
pub mod params;
pub mod prover;
pub mod rlwe;
pub mod transcript;
pub mod verifier;

// Re-exports for convenience
pub use approximate_product::{ApproximateProductProof, ApproximateProductProver};
pub use commitment::{LatticeCommitment, OpeningProof};
pub use errors::LatticeGuardError;
pub use params::{RlweParams, SecurityLevel};
pub use prover::{LatticeGuardProof, LatticeGuardProver};
pub use rlwe::{RlweCiphertext, RlweKeypair, RlwePublicKey, RlweSecretKey};
pub use transcript::LatticeTranscript;
pub use verifier::LatticeGuardVerifier;

use serde::{Deserialize, Serialize};
use std::fmt;
use tracing::{info, warn};

/// Scalar type for field elements (using 64-bit for efficiency)
pub type Scalar = u64;

/// Polynomial representation in coefficient form
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Polynomial {
    /// Coefficients in ascending order of degree
    pub coefficients: Vec<Scalar>,
}

impl Polynomial {
    /// Create a new polynomial from coefficients
    pub fn new(coefficients: Vec<Scalar>) -> Self {
        Self { coefficients }
    }

    /// Create zero polynomial of given degree
    pub fn zero(degree: usize) -> Self {
        Self {
            coefficients: vec![0; degree + 1],
        }
    }

    /// Get degree of polynomial
    pub fn degree(&self) -> usize {
        self.coefficients.len().saturating_sub(1)
    }

    /// Evaluate polynomial at a point (Horner's method)
    pub fn evaluate(&self, x: Scalar, modulus: Scalar) -> Scalar {
        let mut result = 0u64;
        for coeff in self.coefficients.iter().rev() {
            result = (result.wrapping_mul(x).wrapping_add(*coeff)) % modulus;
        }
        result
    }
}

/// Challenge polynomial with bounded coefficients
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Challenge {
    /// Challenge polynomial
    pub polynomial: Polynomial,
    /// Bound on coefficient magnitudes
    pub bound: u64,
}

/// Structured Reference String for LatticeGuard
#[derive(Clone, Serialize, Deserialize)]
pub struct LatticeGuardSRS {
    /// RLWE parameters
    pub params: RlweParams,

    /// Powers of tau encoded as RLWE samples
    pub powers_of_tau: Vec<RlweCiphertext>,

    /// Verification key for fast verification
    pub verifying_key: VerifyingKey,

    /// Maximum supported circuit size
    pub max_constraints: usize,
}

impl LatticeGuardSRS {
    /// Generate a new SRS for given parameters and circuit size
    /// Optimized with parallel processing for large constraint counts
    pub fn generate<R: rand::Rng + rand::CryptoRng>(
        params: RlweParams,
        max_constraints: usize,
        rng: &mut R,
    ) -> Result<Self, LatticeGuardError> {
        info!(
            "Generating LatticeGuard SRS: dim={}, max_constraints={}",
            params.dimension, max_constraints
        );

        // Generate secret trapdoor (never revealed)
        let secret = rlwe::RlweSecretKey::generate(&params, rng);

        // Generate tau and all its powers first (this is fast)
        let tau = rng.gen::<u64>() % params.modulus;
        let modulus = params.modulus;

        // Pre-compute all tau powers (sequential but fast - just multiplication)
        let tau_powers: Vec<u64> = {
            let mut powers = Vec::with_capacity(max_constraints);
            let mut tau_power = 1u64;
            for _ in 0..max_constraints {
                powers.push(tau_power);
                tau_power = tau_power.wrapping_mul(tau) % modulus;
            }
            powers
        };

        info!("  Pre-computed {} tau powers, starting parallel encryption...", tau_powers.len());

        // Parallel RLWE encryption using rayon
        #[cfg(feature = "parallel")]
        let powers_of_tau: Vec<RlweCiphertext> = {
            use rayon::prelude::*;
            use rand::SeedableRng;
            use rand_chacha::ChaCha20Rng;

            // Get a seed from the main RNG for reproducibility
            let base_seed: [u8; 32] = rng.gen();

            tau_powers
                .par_iter()
                .enumerate()
                .map(|(i, &tau_power)| {
                    // Create a deterministic RNG for this iteration
                    let mut seed = base_seed;
                    seed[0] ^= (i & 0xff) as u8;
                    seed[1] ^= ((i >> 8) & 0xff) as u8;
                    seed[2] ^= ((i >> 16) & 0xff) as u8;
                    seed[3] ^= ((i >> 24) & 0xff) as u8;
                    let mut thread_rng = ChaCha20Rng::from_seed(seed);

                    // Encrypt tau^i under RLWE
                    let plaintext = Polynomial::new(vec![tau_power]);
                    secret.encrypt(&plaintext, &params, &mut thread_rng)
                        .expect("RLWE encryption failed")
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let powers_of_tau: Vec<RlweCiphertext> = {
            let mut result = Vec::with_capacity(max_constraints);
            for &tau_power in &tau_powers {
                let plaintext = Polynomial::new(vec![tau_power]);
                let ciphertext = secret.encrypt(&plaintext, &params, rng)?;
                result.push(ciphertext);
            }
            result
        };

        info!("  Parallel encryption complete, generating verifying key...");

        // Generate verifying key
        let verifying_key = VerifyingKey {
            public_key: secret.public_key(&params, rng)?,
            params: params.clone(),
        };

        info!("  SRS generation complete!");

        Ok(Self {
            params,
            powers_of_tau,
            verifying_key,
            max_constraints,
        })
    }

    /// Save SRS to disk for reuse across restarts
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), LatticeGuardError> {
        use std::io::Write;

        info!("💾 Saving SRS to {:?} ({} constraints)...", path, self.max_constraints);

        let encoded = bincode::serialize(self)
            .map_err(|e| LatticeGuardError::SerializationError(e.to_string()))?;

        // Use atomic write (write to temp, then rename)
        let temp_path = path.with_extension("tmp");
        let mut file = std::fs::File::create(&temp_path)
            .map_err(|e| LatticeGuardError::IoError(e.to_string()))?;

        file.write_all(&encoded)
            .map_err(|e| LatticeGuardError::IoError(e.to_string()))?;

        file.sync_all()
            .map_err(|e| LatticeGuardError::IoError(e.to_string()))?;

        std::fs::rename(&temp_path, path)
            .map_err(|e| LatticeGuardError::IoError(e.to_string()))?;

        info!("✅ SRS saved: {} bytes", encoded.len());
        Ok(())
    }

    /// Load SRS from disk
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, LatticeGuardError> {
        info!("📂 Loading SRS from {:?}...", path);

        let encoded = std::fs::read(path)
            .map_err(|e| LatticeGuardError::IoError(e.to_string()))?;

        let srs: Self = bincode::deserialize(&encoded)
            .map_err(|e| LatticeGuardError::SerializationError(e.to_string()))?;

        info!("✅ SRS loaded: {} constraints, {} bytes",
            srs.max_constraints, encoded.len());

        Ok(srs)
    }

    /// Generate or load SRS with caching
    /// This is the recommended way to get an SRS for production use
    pub fn generate_or_load<R: rand::Rng + rand::CryptoRng>(
        params: RlweParams,
        max_constraints: usize,
        cache_path: &std::path::Path,
        rng: &mut R,
    ) -> Result<Self, LatticeGuardError> {
        // Generate a deterministic filename based on parameters
        let cache_file = cache_path.join(format!(
            "lattice_guard_srs_dim{}_mod{}_constraints{}.bin",
            params.dimension, params.modulus, max_constraints
        ));

        // Try to load from cache
        if cache_file.exists() {
            match Self::load_from_file(&cache_file) {
                Ok(srs) => {
                    // Verify parameters match
                    if srs.params.dimension == params.dimension
                        && srs.params.modulus == params.modulus
                        && srs.max_constraints >= max_constraints
                    {
                        info!("🎯 Using cached SRS (saved {}s generation time)",
                            max_constraints / 1000); // Rough estimate
                        return Ok(srs);
                    }
                    warn!("⚠️  Cached SRS has different parameters, regenerating...");
                }
                Err(e) => {
                    warn!("⚠️  Failed to load cached SRS: {}, regenerating...", e);
                }
            }
        }

        // Generate new SRS
        let srs = Self::generate(params, max_constraints, rng)?;

        // Save to cache for next time
        if let Err(e) = std::fs::create_dir_all(cache_path) {
            warn!("⚠️  Failed to create cache directory: {}", e);
        } else if let Err(e) = srs.save_to_file(&cache_file) {
            warn!("⚠️  Failed to cache SRS: {}", e);
        }

        Ok(srs)
    }
}

impl fmt::Debug for LatticeGuardSRS {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LatticeGuardSRS")
            .field("params", &self.params)
            .field("max_constraints", &self.max_constraints)
            .field("powers_of_tau_count", &self.powers_of_tau.len())
            .finish()
    }
}

/// Verifying key for proof verification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyingKey {
    /// Public key for RLWE verification
    pub public_key: RlwePublicKey,
    /// Parameters
    pub params: RlweParams,
}

/// Arithmetic circuit representation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArithmeticCircuit {
    /// Number of constraints (multiplication gates)
    pub num_constraints: usize,
    /// Number of public inputs
    pub num_public_inputs: usize,
    /// Number of private witness elements
    pub num_witness: usize,
    /// Constraint matrices (a, b, c for R1CS)
    pub constraints: Vec<R1CSConstraint>,
}

/// Single R1CS constraint: a * b = c
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1CSConstraint {
    /// Linear combination for left input
    pub a: Vec<(usize, Scalar)>,
    /// Linear combination for right input
    pub b: Vec<(usize, Scalar)>,
    /// Linear combination for output
    pub c: Vec<(usize, Scalar)>,
}

impl ArithmeticCircuit {
    /// Create new empty circuit
    pub fn new(num_public_inputs: usize, num_witness: usize) -> Self {
        Self {
            num_constraints: 0,
            num_public_inputs,
            num_witness,
            constraints: Vec::new(),
        }
    }

    /// Add a multiplication constraint
    pub fn add_multiplication_gate(
        &mut self,
        a: Vec<(usize, Scalar)>,
        b: Vec<(usize, Scalar)>,
        c: Vec<(usize, Scalar)>,
    ) {
        self.constraints.push(R1CSConstraint { a, b, c });
        self.num_constraints += 1;
    }
}

/// Main LatticeGuard proof system
pub struct LatticeGuard {
    /// RLWE parameters
    params: RlweParams,
    /// Prover instance
    prover: LatticeGuardProver,
    /// Verifier instance
    verifier: LatticeGuardVerifier,
}

impl LatticeGuard {
    /// Create new LatticeGuard instance with given security level
    pub fn new(security_level: SecurityLevel) -> Result<Self, LatticeGuardError> {
        let params = RlweParams::from_security_level(security_level);
        let prover = LatticeGuardProver::new(params.clone())?;
        let verifier = LatticeGuardVerifier::new(params.clone())?;

        Ok(Self {
            params,
            prover,
            verifier,
        })
    }

    /// Generate a proof for the given circuit and witness
    pub fn prove<R: rand::Rng + rand::CryptoRng>(
        &self,
        circuit: &ArithmeticCircuit,
        witness: &[Scalar],
        public_inputs: &[Scalar],
        srs: &LatticeGuardSRS,
        rng: &mut R,
    ) -> Result<LatticeGuardProof, LatticeGuardError> {
        self.prover.generate_proof(circuit, witness, public_inputs, srs, rng)
    }

    /// Verify a proof
    pub fn verify(
        &self,
        circuit: &ArithmeticCircuit,
        public_inputs: &[Scalar],
        proof: &LatticeGuardProof,
        srs: &LatticeGuardSRS,
    ) -> Result<bool, LatticeGuardError> {
        self.verifier.verify(circuit, public_inputs, proof, srs)
    }

    /// Get the parameters
    pub fn params(&self) -> &RlweParams {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_evaluation() {
        // p(x) = 3 + 2x + x^2
        let poly = Polynomial::new(vec![3, 2, 1]);

        // p(0) = 3
        assert_eq!(poly.evaluate(0, 1000), 3);

        // p(1) = 3 + 2 + 1 = 6
        assert_eq!(poly.evaluate(1, 1000), 6);

        // p(2) = 3 + 4 + 4 = 11
        assert_eq!(poly.evaluate(2, 1000), 11);
    }

    #[test]
    fn test_arithmetic_circuit() {
        let mut circuit = ArithmeticCircuit::new(1, 2);

        // Add constraint: a * b = c
        circuit.add_multiplication_gate(
            vec![(0, 1)],  // a = x_0
            vec![(1, 1)],  // b = x_1
            vec![(2, 1)],  // c = x_2
        );

        assert_eq!(circuit.num_constraints, 1);
    }
}
